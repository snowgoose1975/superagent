"""School chat-completions gateway and per-session development ledger."""

from __future__ import annotations

import json
import math
import os
import socket
import threading
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

from evaluation_d.task_budget import COURSE_BUDGET_LIMIT_RMB, TaskBalanceExhausted, TaskBudgetError

_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
_MAX_OUTPUT_TOKENS = 4096
_DEFAULT_OUTPUT_TOKENS = 1024
_DEFAULT_INPUT_LIMIT = 16_384
_MAX_INPUT_BYTES = 131_072
_MAX_INPUT_TOKENS = 131_072
_CHAT_TEMPLATE_OVERHEAD_BYTES = 256
_SECRET_CONFIG_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "auth_header",
    "access_token",
    "secret",
    "key",
}


class GatewayError(RuntimeError):
    """Error containing no response body, authentication headers, or raw transport exception."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Prevent forwarding Authorization to a redirect target."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _urllib_transport(
    request: urllib.request.Request, timeout: float
) -> tuple[int, Mapping[str, str], bytes]:
    opener = urllib.request.build_opener(_NoRedirect())
    try:
        response = opener.open(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        # Discard HTTP error bodies and return redirects as failures.
        return int(exc.code), exc.headers or {}, b""
    with response:
        body = response.read(_MAX_RESPONSE_BYTES + 1)
        return int(response.status), response.headers or {}, body


def _safe_config(config: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(config, (str, Path)):
        try:
            value = json.loads(Path(config).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            raise GatewayError("CONFIG_INVALID", "Cannot read the school gateway configuration.") from None
    else:
        value = config
    if not isinstance(value, Mapping):
        raise GatewayError("CONFIG_INVALID", "School gateway configuration must be a JSON object.")

    def reject_secrets(item: Any) -> bool:
        if isinstance(item, Mapping):
            for key, nested in item.items():
                if isinstance(key, str) and key.casefold() in _SECRET_CONFIG_KEYS:
                    return True
                if reject_secrets(nested):
                    return True
        elif isinstance(item, list):
            return any(reject_secrets(element) for element in item)
        return False

    if reject_secrets(value):
        raise GatewayError("CONFIG_INVALID", "Gateway configuration must not contain secrets or authentication fields.")
    for optional_label in ("pricing_profile", "pricing_tier"):
        label = value.get(optional_label)
        if label is not None and (
            not isinstance(label, str) or not label.strip() or "\n" in label or "\r" in label
        ):
            raise GatewayError("CONFIG_INVALID", f"{optional_label} must be a nonempty single-line string.")
    required = {
        "url",
        "model",
        "input_cny_per_million",
        "output_cny_per_million",
    }
    if not required.issubset(value):
        raise GatewayError("CONFIG_INVALID", "School gateway configuration is missing required fields.")
    # Keep only approved configuration fields so extra input cannot reach the ledger.
    return {
        "url": value["url"],
        "model": value["model"],
        "input_cny_per_million": value["input_cny_per_million"],
        "output_cny_per_million": value["output_cny_per_million"],
        "pricing_profile": value.get("pricing_profile"),
        "pricing_tier": value.get("pricing_tier"),
        "max_tokens": value.get("max_tokens", _DEFAULT_OUTPUT_TOKENS),
        "max_input_tokens": value.get("max_input_tokens", _DEFAULT_INPUT_LIMIT),
        "max_input_bytes": value.get(
            "max_input_bytes", value.get("max_input_tokens", _DEFAULT_INPUT_LIMIT)
        ),
        "timeout": value.get("timeout", 90),
    }


def _decimal(value: Any, field: str) -> Decimal:
    if isinstance(value, bool):
        raise GatewayError("CONFIG_INVALID", f"{field} must be a nonnegative decimal.")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise GatewayError("CONFIG_INVALID", f"{field} must be a nonnegative decimal.") from None
    if not result.is_finite() or result < 0:
        raise GatewayError("CONFIG_INVALID", f"{field} must be a nonnegative decimal.")
    return result


def _strict_int(value: Any, field: str, *, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise GatewayError("CONFIG_INVALID", f"{field} is out of range.")
    return value


def _validate_url(value: Any) -> str:
    if not isinstance(value, str):
        raise GatewayError("CONFIG_INVALID", "The gateway URL must use HTTPS.")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        raise GatewayError("CONFIG_INVALID", "The gateway URL format is invalid.") from None
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or (port is not None and port != 443)
    ):
        raise GatewayError("CONFIG_INVALID", "Use an HTTPS gateway URL without credentials, query parameters, or fragments.")
    return value


def _usage(payload: Any) -> tuple[int, int] | None:
    if not isinstance(payload, dict):
        return None
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None
    input_tokens = usage.get("prompt_tokens")
    output_tokens = usage.get("completion_tokens")
    if (
        type(input_tokens) is not int
        or input_tokens < 0
        or type(output_tokens) is not int
        or output_tokens < 0
    ):
        return None
    if "total_tokens" in usage:
        total_tokens = usage["total_tokens"]
        if type(total_tokens) is not int or total_tokens != input_tokens + output_tokens:
            return None
    return input_tokens, output_tokens


def _duplicate_free_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _reject_json_constant(_value: str) -> None:
    raise ValueError("non-standard JSON constant")


class SchoolGateway:
    """Reserve and account for each school API request before returning a result."""

    def __init__(
        self,
        config: Mapping[str, Any] | str | Path,
        api_key: str,
        ledger_root: str | Path,
        max_calls: int = 1,
        budget_rmb: str | Decimal = "0.10",
        transport=None,
        task_budget_ledger: Any | None = None,
    ) -> None:
        self._config = _safe_config(config)
        self._url = _validate_url(self._config["url"])
        model = self._config["model"]
        if not isinstance(model, str) or not model.strip() or "\n" in model or "\r" in model:
            raise GatewayError("CONFIG_INVALID", "model must be a nonempty single-line string.")
        self._model = model
        self._input_price = _decimal(self._config["input_cny_per_million"], "input_cny_per_million")
        self._output_price = _decimal(
            self._config["output_cny_per_million"], "output_cny_per_million"
        )
        self._max_tokens = _strict_int(
            self._config["max_tokens"], "max_tokens", minimum=1, maximum=_MAX_OUTPUT_TOKENS
        )
        self._max_input_tokens = _strict_int(
            self._config["max_input_tokens"],
            "max_input_tokens",
            minimum=1,
            maximum=_MAX_INPUT_TOKENS,
        )
        self._max_input_bytes = _strict_int(
            self._config["max_input_bytes"],
            "max_input_bytes",
            minimum=1,
            maximum=_MAX_INPUT_BYTES,
        )
        timeout = self._config["timeout"]
        if type(timeout) not in (int, float) or not 0 < timeout <= 300:
            raise GatewayError("CONFIG_INVALID", "timeout must be greater than zero and at most 300 seconds.")
        self._configured_timeout = float(timeout)
        self._timeout = self._configured_timeout
        if (
            not isinstance(api_key, str)
            or not api_key.strip()
            or any(c in api_key for c in "\r\n\0")
        ):
            raise GatewayError("KEY_REQUIRED", "Supply a valid API key at runtime.")
        if type(max_calls) is not int or max_calls < 1:
            raise GatewayError("CONFIG_INVALID", "max_calls must be a positive integer.")
        self._api_key = api_key
        self._max_calls = max_calls
        self._budget = _decimal(budget_rmb, "budget_rmb")
        if not 0 < self._budget <= COURSE_BUDGET_LIMIT_RMB:
            raise GatewayError("CONFIG_INVALID", "budget_rmb must be greater than zero and at most CNY 20.")
        self._transport = transport if transport is not None else _urllib_transport
        self._is_injected = transport is not None
        self._task_budget_ledger = task_budget_ledger
        task_price_profile = (
            task_budget_ledger.snapshot().get("price_profile")
            if task_budget_ledger is not None
            else None
        )
        self._pricing_profile = self._config.get("pricing_profile")
        self._pricing_tier = self._config.get("pricing_tier")
        self._task_budget_profile = None
        if task_budget_ledger is not None:
            if not isinstance(task_price_profile, dict):
                raise GatewayError("CONFIG_INVALID", "Attached task ledger has no pricing profile.")
            self._task_budget_profile = task_price_profile["profile_id"]
            self._pricing_profile = self._task_budget_profile
            self._pricing_tier = task_price_profile["source_basis"].get("price_tier")
            self._input_price = Decimal(task_price_profile["input_cny_per_million"])
            self._output_price = Decimal(task_price_profile["output_cny_per_million"])

        self._known_input_tokens = 0
        self._known_output_tokens = 0
        self._known_cost = Decimal("0")
        self._reserved = Decimal("0")
        self._usage_unknown = False
        self._usage_pending = False
        self._blocked = False
        self._blocked_reason: str | None = None
        self._task_budget_error: str | None = None
        self._last_completion: dict[str, Any] | None = None
        self._call_attempts = 0
        self._real_llm_calls = 0
        self._mock_calls = 0
        self._lock = threading.Lock()

        root = Path(ledger_root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        # Use short session IDs to limit metadata path lengths on Windows.
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self._session_id = f"api-{stamp}-{uuid.uuid4().hex[:8]}"
        self._session_dir = root / self._session_id
        self._session_dir.mkdir(parents=False, exist_ok=False)
        self._calls_path = self._session_dir / "calls.jsonl"
        self._metadata_path = self._session_dir / "metadata.json"
        self._calls_path.open("x", encoding="utf-8", newline="\n").close()
        self._persist_metadata()

    def __repr__(self) -> str:
        return (
            f"SchoolGateway(model={self._model!r}, call_attempts={self._call_attempts}, "
            f"blocked={self._blocked}, transport={'injected' if self._is_injected else 'urllib'})"
        )

    @property
    def remaining_budget_rmb(self) -> float:
        remaining = self._remaining_decimal()
        if self._task_budget_ledger is not None:
            try:
                task_remaining = Decimal(self._task_budget_ledger.snapshot()["remaining_rmb"])
            except (TaskBudgetError, OSError):
                return 0.0
            remaining = min(remaining, task_remaining)
        return float(remaining)

    @property
    def can_call(self) -> bool:
        return (
            not self._blocked
            and not self._usage_pending
            and self._call_attempts < self._max_calls
            and self._remaining_decimal() >= self._worst_case_cost()
        )

    @property
    def paid_limit_reason(self) -> str | None:
        if self._usage_pending or self._blocked:
            return None
        if self._call_attempts >= self._max_calls:
            return "CALL_LIMIT"
        if self._remaining_decimal() < self._worst_case_cost():
            return "BUDGET_EXCEEDED"
        return None

    def recover_empty_reply(self) -> bool:
        """Reopen only a request whose unusable reply was fully accounted for."""
        with self._lock:
            if (
                self._blocked_reason != "INVALID_RESPONSE"
                or self._usage_unknown or self._usage_pending or self._reserved
            ):
                return False
            self._blocked = False
            self._blocked_reason = None
            self._persist_metadata()
            return True

    def set_request_timeout_cap(self, seconds: float | None) -> None:
        """Bound the next synchronous request by the remaining session window."""
        if seconds is None:
            self._timeout = self._configured_timeout
            return
        if type(seconds) not in (int, float):
            raise GatewayError("TIMEOUT_INVALID", "Insufficient time remains for a school API request.")
        try:
            cap = float(seconds)
        except (OverflowError, ValueError):
            raise GatewayError("TIMEOUT_INVALID", "Insufficient time remains for a school API request.") from None
        if not math.isfinite(cap) or cap <= 0:
            raise GatewayError("TIMEOUT_INVALID", "Insufficient time remains for a school API request.")
        self._timeout = min(self._configured_timeout, cap)

    def _remaining_decimal(self) -> Decimal:
        return self._budget - self._known_cost - self._reserved

    def _pricing_log_fields(self) -> dict[str, str | None]:
        return {
            "pricing_profile": self._pricing_profile,
            "pricing_tier": self._pricing_tier,
            "task_budget_profile": self._task_budget_profile,
        }

    def _worst_case_cost(self) -> Decimal:
        return (
            Decimal(self._max_input_tokens) * self._input_price
            + Decimal(self._max_tokens) * self._output_price
        ) / Decimal(1_000_000)

    def metadata(self) -> dict[str, Any]:
        usage_status = (
            "unknown"
            if self._usage_unknown or self._usage_pending
            else ("known" if self._call_attempts else "not_called")
        )
        task_budget = None
        task_budget_error = self._task_budget_error
        if self._task_budget_ledger is not None:
            try:
                task_budget = self._task_budget_ledger.snapshot()
            except (TaskBudgetError, OSError):
                task_budget_error = task_budget_error or "snapshot_failed"
        return {
            "session_id": self._session_id,
            "model": self._model,
            "url": self._url,
            **self._pricing_log_fields(),
            "real_llm_calls": self._real_llm_calls,
            "mock_calls": self._mock_calls,
            "call_attempts": self._call_attempts,
            "max_calls": self._max_calls,
            "blocked": self._blocked,
            "can_call": self.can_call and task_budget_error is None,
            "blocked_reason": self._blocked_reason,
            "last_completion": self._last_completion,
            "usage_status": usage_status,
            "input_tokens": None
            if self._usage_unknown or self._usage_pending
            else self._known_input_tokens,
            "output_tokens": None
            if self._usage_unknown or self._usage_pending
            else self._known_output_tokens,
            "estimated_cost_rmb": (
                None
                if self._usage_unknown or self._usage_pending
                else format(self._known_cost, "f")
            ),
            "reserved_rmb": format(self._reserved, "f"),
            "budget_rmb": format(self._budget, "f"),
            "remaining_budget_rmb": format(self._remaining_decimal(), "f"),
            "max_input_tokens": self._max_input_tokens,
            "max_input_bytes": self._max_input_bytes,
            "max_output_tokens": self._max_tokens,
            "transport": "injected" if self._is_injected else "urllib",
            "ledger_path": str(self._session_dir.resolve()),
            "task_budget": task_budget,
            "task_budget_error": task_budget_error,
        }

    def _persist_metadata(self) -> None:
        value = self.metadata()
        temporary = self._session_dir / f"metadata.{uuid.uuid4().hex[:8]}.tmp"
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self._metadata_path)

    def _append_call(self, record: dict[str, Any]) -> None:
        with self._calls_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _block(self, reason: str) -> None:
        self._blocked = True
        self._blocked_reason = reason
        self._persist_metadata()

    def _unknown(self, attempt: int, code: str, available_before: Decimal) -> None:
        if self._task_budget_ledger is not None:
            try:
                self._task_budget_ledger.mark_unknown(
                    f"{self._session_id}:{attempt}", reason_code=code
                )
            except Exception:
                # A still-pending task reservation remains fully charged if its
                # status update fails; the local session is nevertheless blocked.
                self._task_budget_error = "mark_unknown_failed"
        self._usage_unknown = True
        self._usage_pending = False
        self._reserved = available_before
        self._blocked = True
        self._blocked_reason = code
        self._append_call(
            {
                "attempt": attempt,
                "status": "unknown",
                "error_code": code,
                "task_budget_error": self._task_budget_error,
                "input_tokens": None,
                "output_tokens": None,
                "estimated_cost_rmb": None,
                "reserved_rmb": format(available_before, "f"),
                **self._pricing_log_fields(),
            }
        )
        self._persist_metadata()

    def _charge(self, input_tokens: int, output_tokens: int) -> Decimal:
        return (
            Decimal(input_tokens) * self._input_price + Decimal(output_tokens) * self._output_price
        ) / Decimal(1_000_000)

    def _record_known(
        self, attempt: int, input_tokens: int, output_tokens: int, *, status: str
    ) -> Decimal:
        cost = self._charge(input_tokens, output_tokens)
        if self._task_budget_ledger is not None:
            try:
                self._task_budget_ledger.settle(
                    f"{self._session_id}:{attempt}",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
            except Exception:
                self._task_budget_error = "settlement_failed"
                self._blocked = True
                self._blocked_reason = "task_ledger_settle_failed"
        self._known_input_tokens += input_tokens
        self._known_output_tokens += output_tokens
        self._known_cost += cost
        self._reserved = Decimal("0")
        self._usage_pending = False
        self._append_call(
            {
                "attempt": attempt,
                "status": status,
                "usage_status": "known",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "estimated_cost_rmb": format(cost, "f"),
                "reserved_rmb": "0",
                "completion": self._last_completion,
                "task_budget_error": self._task_budget_error,
                **self._pricing_log_fields(),
            }
        )
        self._persist_metadata()
        if self._task_budget_error:
            raise GatewayError(
                "TASK_LEDGER_ERROR",
                "School API usage is known but ledger settlement failed. The session is blocked.",
            )
        return cost

    def _safe_transport(self, request: urllib.request.Request) -> tuple[int, bytes]:
        try:
            response = self._transport(request, self._timeout)
        except (TimeoutError, socket.timeout):
            raise GatewayError("TIMEOUT", "The school API request timed out. Usage is unknown and the session is blocked.") from None
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise GatewayError(
                    "TIMEOUT", "The school API request timed out. Usage is unknown and the session is blocked."
                ) from None
            raise GatewayError(
                "TRANSPORT_ERROR", "The school API connection failed. Usage is unknown and the session is blocked."
            ) from None
        except Exception:
            raise GatewayError(
                "TRANSPORT_ERROR", "The school API transport failed. Usage is unknown and the session is blocked."
            ) from None
        if not isinstance(response, tuple) or len(response) != 3:
            raise GatewayError("TRANSPORT_ERROR", "The school API transport returned an invalid result.")
        status, _headers, body = response
        if type(status) is not int or not isinstance(body, (bytes, bytearray)):
            raise GatewayError("TRANSPORT_ERROR", "The school API transport returned an invalid result.")
        if len(body) > _MAX_RESPONSE_BYTES:
            raise GatewayError(
                "RESPONSE_TOO_LARGE", "The school API response exceeded 2 MiB. Usage is unknown and the session is blocked."
            )
        return status, bytes(body)

    def call_llm(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        # Serialize reservations and requests to enforce the call limit and available balance.
        with self._lock:
            return self._call_llm(messages)

    def _call_llm(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        if not self.can_call:
            if not self._blocked:
                if self._usage_pending:
                    code, message = "SESSION_BLOCKED", "The previous request has unknown usage. The gateway cannot make another request."
                    self._block("usage_pending")
                elif self._call_attempts >= self._max_calls:
                    code, message = "CALL_LIMIT", "The school API call limit has been reached."
                else:
                    code, message = "BUDGET_EXCEEDED", "The remaining budget cannot cover the request reservation."
                    self._block("budget_exceeded")
            else:
                code, message = "SESSION_BLOCKED", "The school API session is blocked and cannot make another request."
            raise GatewayError(code, message)
        if not isinstance(messages, list) or not messages:
            raise GatewayError("INPUT_INVALID", "messages must be a nonempty list.")
        if any(
            not isinstance(message, dict)
            or not isinstance(message.get("role"), str)
            or not isinstance(message.get("content"), str)
            for message in messages
        ):
            raise GatewayError("INPUT_INVALID", "Each message must have string role and content fields.")
        try:
            serialized_messages = json.dumps(
                messages, ensure_ascii=False, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        except (TypeError, ValueError, UnicodeError):
            raise GatewayError("INPUT_INVALID", "messages must be serializable as JSON.") from None
        input_bound_bytes = len(serialized_messages) + _CHAT_TEMPLATE_OVERHEAD_BYTES
        if input_bound_bytes > self._max_input_bytes:
            self._block("input_too_large")
            raise GatewayError("INPUT_TOO_LARGE", "Messages exceeded the UTF-8 input limit. No request was sent.")

        available_before = self._remaining_decimal()
        reserve = self._worst_case_cost()
        if reserve > available_before:
            self._block("budget_exceeded")
            raise GatewayError("BUDGET_EXCEEDED", "The remaining budget cannot cover the request reservation.")

        self._call_attempts += 1
        attempt = self._call_attempts
        if self._task_budget_ledger is not None:
            try:
                self._task_budget_ledger.reserve(
                    f"{self._session_id}:{attempt}",
                    session_id=self._session_id,
                    max_input_tokens=self._max_input_tokens,
                    max_output_tokens=self._max_tokens,
                )
            except TaskBalanceExhausted:
                self._call_attempts -= 1
                self._block("task_budget_exhausted")
                raise GatewayError(
                    "TASK_BALANCE_EXHAUSTED", "The task budget cannot cover the reservation. No request was sent.",
                ) from None
            except Exception:
                self._call_attempts -= 1
                self._task_budget_error = "reservation_failed"
                self._block("task_budget_reservation_failed")
                raise GatewayError(
                    "TASK_BUDGET_EXCEEDED",
                    "Task budget reservation failed. No request was sent.",
                ) from None
        if self._is_injected:
            self._mock_calls += 1
        else:
            self._real_llm_calls += 1
        self._reserved = reserve
        self._usage_pending = True
        self._append_call(
            {
                "attempt": attempt,
                "status": "reserved",
                "input_upper_bound_bytes": input_bound_bytes,
                "max_input_bytes": self._max_input_bytes,
                "max_input_tokens": self._max_input_tokens,
                "max_output_tokens": self._max_tokens,
                "reserved_rmb": format(reserve, "f"),
                **self._pricing_log_fields(),
            }
        )
        self._persist_metadata()

        request_body = json.dumps(
            {
                "model": self._model,
                "messages": messages,
                "max_tokens": self._max_tokens,
                "temperature": 0,
                "stream": False,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            self._url,
            data=request_body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )

        try:
            status, body = self._safe_transport(request)
        except KeyboardInterrupt:
            self._unknown(attempt, "INTERRUPTED", available_before)
            raise
        except GatewayError as exc:
            self._unknown(attempt, exc.code, available_before)
            raise exc from None
        if not 200 <= status < 300:
            self._unknown(attempt, "HTTP_ERROR", available_before)
            raise GatewayError("HTTP_ERROR", "The school API returned an HTTP error. Usage is unknown and the session is blocked.")
        try:
            payload = json.loads(
                body.decode("utf-8"),
                object_pairs_hook=_duplicate_free_object,
                parse_constant=_reject_json_constant,
            )
        except (UnicodeError, json.JSONDecodeError, ValueError):
            self._unknown(attempt, "INVALID_JSON", available_before)
            raise GatewayError(
                "INVALID_JSON", "The school API returned invalid JSON. Usage is unknown and the session is blocked."
            ) from None

        usage = _usage(payload)
        if usage is None:
            self._unknown(attempt, "USAGE_UNKNOWN", available_before)
            raise GatewayError("USAGE_UNKNOWN", "The school API did not return valid usage. The session is blocked.")
        input_tokens, output_tokens = usage
        # Record the reported usage even when it exceeds the reservation, then block the session.
        usage_exceeds_reservation = (
            input_tokens > self._max_input_tokens or output_tokens > self._max_tokens
        )
        choices = payload.get("choices") if isinstance(payload, dict) else None
        text_value: Any = None
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            message = choices[0].get("message")
            if isinstance(message, dict):
                text_value = message.get("content")
        valid_content = isinstance(text_value, str) and bool(text_value.strip())
        finish_reason = (
            choices[0].get("finish_reason")
            if isinstance(choices, list) and choices and isinstance(choices[0], dict)
            else None
        )
        self._last_completion = {
            "attempt": attempt,
            "has_usable_text": valid_content,
            "output_limit_reached": output_tokens >= self._max_tokens,
            "finish_reason": finish_reason if finish_reason in (
                "stop", "length", "content_filter", "tool_calls", "function_call",
            ) else None,
        }
        details = payload["usage"].get("completion_tokens_details")
        reasoning_tokens = details.get("reasoning_tokens") if isinstance(details, dict) else None
        if type(reasoning_tokens) is int and 0 <= reasoning_tokens <= output_tokens:
            self._last_completion["reasoning_tokens"] = reasoning_tokens
        if usage_exceeds_reservation or not valid_content:
            self._record_known(attempt, input_tokens, output_tokens, status="failed_known")
            error_code = "USAGE_EXCEEDS_LIMIT" if usage_exceeds_reservation else "INVALID_RESPONSE"
            self._block(error_code)
            if usage_exceeds_reservation:
                raise GatewayError(
                    error_code, "Provider usage exceeded the reservation. The usage was recorded and the session is blocked."
                )
            raise GatewayError(error_code, "The school API reply contained no usable text. Usage was recorded.")

        cost = self._record_known(attempt, input_tokens, output_tokens, status="complete")
        return {
            "text": text_value,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_rmb": float(cost),
        }
