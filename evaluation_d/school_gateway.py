"""学校 chat/completions 网关与独立开发调用账本。"""

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

_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
_MAX_OUTPUT_TOKENS = 1024
_MAX_INPUT_BYTES = 16384
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
    """不暴露服务端正文、认证头或底层异常文本的安全错误。"""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """禁止把 Authorization 转发到重定向目标。"""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _urllib_transport(
    request: urllib.request.Request, timeout: float
) -> tuple[int, Mapping[str, str], bytes]:
    opener = urllib.request.build_opener(_NoRedirect())
    try:
        response = opener.open(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        # HTTPError body 不读取、不记录；特别是 3xx 只作为失败返回，不跟随。
        return int(exc.code), exc.headers or {}, b""
    with response:
        body = response.read(_MAX_RESPONSE_BYTES + 1)
        return int(response.status), response.headers or {}, body


def _safe_config(config: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(config, (str, Path)):
        try:
            value = json.loads(Path(config).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            raise GatewayError("CONFIG_INVALID", "学校网关配置无法读取") from None
    else:
        value = config
    if not isinstance(value, Mapping):
        raise GatewayError("CONFIG_INVALID", "学校网关配置必须是 JSON 对象")

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
        raise GatewayError("CONFIG_INVALID", "网关配置禁止包含密钥或认证字段")
    required = {
        "url",
        "model",
        "input_cny_per_million",
        "output_cny_per_million",
    }
    if not required.issubset(value):
        raise GatewayError("CONFIG_INVALID", "学校网关配置缺少必需字段")
    # 只复制白名单配置字段，开发账本不保存原配置中的额外数据。
    return {
        "url": value["url"],
        "model": value["model"],
        "input_cny_per_million": value["input_cny_per_million"],
        "output_cny_per_million": value["output_cny_per_million"],
        "max_tokens": value.get("max_tokens", 1024),
        "max_input_tokens": value.get("max_input_tokens", _MAX_INPUT_BYTES),
        "timeout": value.get("timeout", 90),
    }


def _decimal(value: Any, field: str) -> Decimal:
    if isinstance(value, bool):
        raise GatewayError("CONFIG_INVALID", f"{field} 必须是非负十进制数")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise GatewayError("CONFIG_INVALID", f"{field} 必须是非负十进制数") from None
    if not result.is_finite() or result < 0:
        raise GatewayError("CONFIG_INVALID", f"{field} 必须是非负十进制数")
    return result


def _strict_int(value: Any, field: str, *, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise GatewayError("CONFIG_INVALID", f"{field} 超出允许范围")
    return value


def _validate_url(value: Any) -> str:
    if not isinstance(value, str):
        raise GatewayError("CONFIG_INVALID", "网关 URL 必须为 HTTPS URL")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        raise GatewayError("CONFIG_INVALID", "网关 URL 格式无效") from None
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or (port is not None and port != 443)
    ):
        raise GatewayError("CONFIG_INVALID", "网关仅允许无凭据、无查询参数的 HTTPS URL")
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
    """默认最多一次学校 API 调用；失败/未知 usage 后永久封锁本实例。"""

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
            raise GatewayError("CONFIG_INVALID", "model 必须是非空单行字符串")
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
            maximum=_MAX_INPUT_BYTES,
        )
        timeout = self._config["timeout"]
        if type(timeout) not in (int, float) or not 0 < timeout <= 300:
            raise GatewayError("CONFIG_INVALID", "timeout 必须在 0 到 300 秒之间")
        self._configured_timeout = float(timeout)
        self._timeout = self._configured_timeout
        if (
            not isinstance(api_key, str)
            or not api_key.strip()
            or any(c in api_key for c in "\r\n\0")
        ):
            raise GatewayError("KEY_REQUIRED", "必须通过运行时参数提供有效 API key")
        if type(max_calls) is not int or max_calls < 1:
            raise GatewayError("CONFIG_INVALID", "max_calls 必须为正整数")
        self._api_key = api_key
        self._max_calls = max_calls
        self._budget = _decimal(budget_rmb, "budget_rmb")
        if self._budget <= 0:
            raise GatewayError("CONFIG_INVALID", "budget_rmb 必须大于零")
        self._transport = transport if transport is not None else _urllib_transport
        self._is_injected = transport is not None
        self._task_budget_ledger = task_budget_ledger

        self._known_input_tokens = 0
        self._known_output_tokens = 0
        self._known_cost = Decimal("0")
        self._reserved = Decimal("0")
        self._usage_unknown = False
        self._usage_pending = False
        self._blocked = False
        self._blocked_reason: str | None = None
        self._call_attempts = 0
        self._real_llm_calls = 0
        self._mock_calls = 0
        self._lock = threading.Lock()

        root = Path(ledger_root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        # 使用短编号，避免 Windows 深目录中的元数据路径超过文件 API 限制。
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
        return float(self._budget - self._known_cost - self._reserved)

    @property
    def can_call(self) -> bool:
        return (
            not self._blocked
            and not self._usage_pending
            and self._call_attempts < self._max_calls
            and self._remaining_decimal() >= self._worst_case_cost()
        )

    def set_request_timeout_cap(self, seconds: float | None) -> None:
        """Bound the next synchronous request by the remaining session window."""
        if seconds is None:
            self._timeout = self._configured_timeout
            return
        if type(seconds) not in (int, float):
            raise GatewayError("TIMEOUT_INVALID", "剩余时间不足以发起学校 API 请求")
        try:
            cap = float(seconds)
        except (OverflowError, ValueError):
            raise GatewayError("TIMEOUT_INVALID", "剩余时间不足以发起学校 API 请求") from None
        if not math.isfinite(cap) or cap <= 0:
            raise GatewayError("TIMEOUT_INVALID", "剩余时间不足以发起学校 API 请求")
        self._timeout = min(self._configured_timeout, cap)

    def _remaining_decimal(self) -> Decimal:
        return self._budget - self._known_cost - self._reserved

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
        return {
            "session_id": self._session_id,
            "model": self._model,
            "url": self._url,
            "real_llm_calls": self._real_llm_calls,
            "mock_calls": self._mock_calls,
            "call_attempts": self._call_attempts,
            "max_calls": self._max_calls,
            "blocked": self._blocked,
            "can_call": self.can_call,
            "blocked_reason": self._blocked_reason,
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
            "max_output_tokens": self._max_tokens,
            "transport": "injected" if self._is_injected else "urllib",
            "ledger_path": str(self._session_dir.resolve()),
            "task_budget": (
                self._task_budget_ledger.snapshot()
                if self._task_budget_ledger is not None
                else None
            ),
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
                self._blocked_reason = "task_ledger_update_failed"
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
                "input_tokens": None,
                "output_tokens": None,
                "estimated_cost_rmb": None,
                "reserved_rmb": format(available_before, "f"),
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
        task_ledger_failed = False
        if self._task_budget_ledger is not None:
            try:
                self._task_budget_ledger.settle(
                    f"{self._session_id}:{attempt}",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
            except Exception:
                task_ledger_failed = True
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
            }
        )
        self._persist_metadata()
        if task_ledger_failed:
            self._block("task_ledger_settle_failed")
            raise GatewayError(
                "TASK_LEDGER_ERROR",
                "学校 API usage 已知，但课程预算账本无法结算；本会话已停止",
            )
        return cost

    def _safe_transport(self, request: urllib.request.Request) -> tuple[int, bytes]:
        try:
            response = self._transport(request, self._timeout)
        except (TimeoutError, socket.timeout):
            raise GatewayError("TIMEOUT", "学校 API 请求超时；usage 未知，已封锁本会话") from None
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise GatewayError(
                    "TIMEOUT", "学校 API 请求超时；usage 未知，已封锁本会话"
                ) from None
            raise GatewayError(
                "TRANSPORT_ERROR", "学校 API 连接失败；usage 未知，已封锁本会话"
            ) from None
        except Exception:
            raise GatewayError(
                "TRANSPORT_ERROR", "学校 API 传输失败；usage 未知，已封锁本会话"
            ) from None
        if not isinstance(response, tuple) or len(response) != 3:
            raise GatewayError("TRANSPORT_ERROR", "学校 API transport 返回格式无效")
        status, _headers, body = response
        if type(status) is not int or not isinstance(body, (bytes, bytearray)):
            raise GatewayError("TRANSPORT_ERROR", "学校 API transport 返回格式无效")
        if len(body) > _MAX_RESPONSE_BYTES:
            raise GatewayError(
                "RESPONSE_TOO_LARGE", "学校 API 响应超过 2 MiB；usage 未知，已封锁本会话"
            )
        return status, bytes(body)

    def call_llm(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        # 同一实例串行预留，避免并发调用绕过 max_calls 或重复花费余额。
        with self._lock:
            return self._call_llm(messages)

    def _call_llm(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        if not self.can_call:
            if not self._blocked:
                if self._usage_pending:
                    code, message = "SESSION_BLOCKED", "上次请求仍有未知用量，不能继续调用"
                    self._block("usage_pending")
                elif self._call_attempts >= self._max_calls:
                    code, message = "CALL_LIMIT", "学校 API 调用次数已达上限"
                else:
                    code, message = "BUDGET_EXCEEDED", "剩余预算不足以预留本次请求上限"
                    self._block("budget_exceeded")
            else:
                code, message = "SESSION_BLOCKED", "学校 API 会话已封锁，不能继续调用"
            raise GatewayError(code, message)
        if not isinstance(messages, list) or not messages:
            raise GatewayError("INPUT_INVALID", "messages 必须是非空消息列表")
        if any(
            not isinstance(message, dict)
            or not isinstance(message.get("role"), str)
            or not isinstance(message.get("content"), str)
            for message in messages
        ):
            raise GatewayError("INPUT_INVALID", "每条 message 必须含字符串 role 和 content")
        try:
            serialized_messages = json.dumps(
                messages, ensure_ascii=False, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        except (TypeError, ValueError, UnicodeError):
            raise GatewayError("INPUT_INVALID", "messages 必须是可编码的 JSON 消息列表") from None
        input_bound_bytes = len(serialized_messages) + _CHAT_TEMPLATE_OVERHEAD_BYTES
        if input_bound_bytes > self._max_input_tokens:
            self._block("input_too_large")
            raise GatewayError("INPUT_TOO_LARGE", "messages 超过 UTF-8 输入字节上限；未发送请求")

        available_before = self._remaining_decimal()
        reserve = self._worst_case_cost()
        if reserve > available_before:
            self._block("budget_exceeded")
            raise GatewayError("BUDGET_EXCEEDED", "剩余预算不足以预留本次请求上限")

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
            except Exception:
                self._call_attempts -= 1
                self._block("task_budget_reservation_failed")
                raise GatewayError(
                    "TASK_BUDGET_EXCEEDED",
                    "课程预算无法预留本次学校 API 请求上限；未发送请求",
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
                "max_input_tokens": self._max_input_tokens,
                "max_output_tokens": self._max_tokens,
                "reserved_rmb": format(reserve, "f"),
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
        except GatewayError as exc:
            self._unknown(attempt, exc.code, available_before)
            raise exc from None
        if not 200 <= status < 300:
            self._unknown(attempt, "HTTP_ERROR", available_before)
            raise GatewayError("HTTP_ERROR", "学校 API 返回非成功状态；usage 未知，已封锁本会话")
        try:
            payload = json.loads(
                body.decode("utf-8"),
                object_pairs_hook=_duplicate_free_object,
                parse_constant=_reject_json_constant,
            )
        except (UnicodeError, json.JSONDecodeError, ValueError):
            self._unknown(attempt, "INVALID_JSON", available_before)
            raise GatewayError(
                "INVALID_JSON", "学校 API 响应不是有效 JSON；usage 未知，已封锁本会话"
            ) from None

        usage = _usage(payload)
        if usage is None:
            self._unknown(attempt, "USAGE_UNKNOWN", available_before)
            raise GatewayError("USAGE_UNKNOWN", "学校 API 未返回完整有效 usage；已封锁本会话")
        input_tokens, output_tokens = usage
        cost = self._charge(input_tokens, output_tokens)

        # 若供应商报告值超过预留的 token 上限，仍如实登记已知费用，再停止本会话。
        usage_exceeds_reservation = (
            input_tokens > self._max_input_tokens or output_tokens > self._max_tokens
        )
        choices = payload.get("choices") if isinstance(payload, dict) else None
        text_value: Any = None
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            message = choices[0].get("message")
            if isinstance(message, dict):
                text_value = message.get("content")
        valid_content = isinstance(text_value, str)
        if usage_exceeds_reservation or not valid_content:
            self._record_known(attempt, input_tokens, output_tokens, status="failed_known")
            error_code = "USAGE_EXCEEDS_LIMIT" if usage_exceeds_reservation else "INVALID_RESPONSE"
            self._block(error_code)
            if usage_exceeds_reservation:
                raise GatewayError(
                    error_code, "供应商 usage 超过调用预留上限；已登记费用并封锁本会话"
                )
            raise GatewayError(error_code, "学校 API 响应缺少有效文本；已登记 usage 并封锁本会话")

        self._record_known(attempt, input_tokens, output_tokens, status="complete")
        return {
            "text": text_value,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_rmb": float(cost),
        }
