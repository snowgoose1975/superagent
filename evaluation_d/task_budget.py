"""Atomic task-wide budget ledger for explicitly selected formal price profiles.

This module is deliberately separate from ``SchoolGateway``'s historical
per-session development ledger. It does not infer which provider/model a school
endpoint actually bills as; callers must explicitly opt into a named profile.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterator


class TaskBudgetError(RuntimeError):
    """Budget ledger could not safely reserve or settle a task expense."""


COURSE_PRICE_PROFILE_2026_10_21: dict[str, Any] = {
    "profile_id": "deepseek-v4.1-flash-peak-2026-10-21",
    "provider": "DeepSeek (published reference price)",
    "model_identity_verified_for_school_endpoint": False,
    "effective_date": "2026-10-21",
    "input_cny_per_million": "2.015760",
    "output_cny_per_million": "8.063040",
    "source_basis": {
        "usd_input_per_million": "0.30",
        "usd_output_per_million": "1.20",
        "price_tier": "published peak, cache-miss standard rates",
        "fx_cny_per_usd": "6.7192",
        "fx_date": "2026-10-21",
    },
    "sources": [
        "https://api-docs.deepseek.com/quick_start/pricing/",
        "https://api-docs.deepseek.com/updates/",
        "https://www.shibor.sh.cn/chinese/ccprnoticecontent/index.html?searchDate=2026-10-21",
    ],
    "billing_caveat": (
        "Conservative conversion of DeepSeek's published peak/cache-miss rates; "
        "not confirmation of the school endpoint's provider, model alias, or actual billing."
    ),
}

_PROFILE_BY_ID = {
    COURSE_PRICE_PROFILE_2026_10_21["profile_id"]: COURSE_PRICE_PROFILE_2026_10_21
}
_BUDGET_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_RESERVATION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$")
_LEDGER_SCHEMA_VERSION = "1.0"
_MILLION = Decimal(1_000_000)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _decimal(value: Any, field: str, *, positive: bool = False) -> Decimal:
    if isinstance(value, bool):
        raise TaskBudgetError(f"{field} must be a finite non-negative decimal")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise TaskBudgetError(f"{field} must be a finite non-negative decimal") from None
    if not result.is_finite() or result < 0 or (positive and result == 0):
        raise TaskBudgetError(f"{field} must be a finite non-negative decimal")
    return result


def _canonical_decimal(value: Decimal) -> str:
    """Normalize equivalent user spellings without using exponent notation."""
    return format(value.normalize(), "f")


def _token_count(value: Any, field: str) -> int:
    if type(value) is not int or value < 0:
        raise TaskBudgetError(f"{field} must be a non-negative integer")
    return value


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    """Cross-process advisory lock, retained as a harmless stable sidecar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt

            stream.seek(0)
            if stream.read(1) == b"":
                stream.seek(0)
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            while True:
                try:
                    msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
                    break
                except OSError:
                    time.sleep(0.025)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class TaskBudgetLedger:
    """A restart-safe, concurrent budget ledger for one explicit task budget ID.

    Pending and unknown reservations continue to consume their full worst-case
    amount. A task budget ID is an accounting namespace only; it must never be
    used for model/planner routing.
    """

    def __init__(
        self,
        root: str | Path,
        budget_id: str,
        *,
        budget_rmb: str | Decimal,
        profile: str | dict[str, Any],
    ) -> None:
        if not isinstance(budget_id, str) or not _BUDGET_ID_RE.fullmatch(budget_id):
            raise TaskBudgetError("budget_id must be a simple safe identifier")
        self.budget_id = budget_id
        self._limit = _decimal(budget_rmb, "budget_rmb", positive=True)
        if isinstance(profile, str):
            try:
                profile_value = _PROFILE_BY_ID[profile]
            except KeyError:
                raise TaskBudgetError("unknown official price profile") from None
        elif isinstance(profile, dict):
            profile_value = profile
        else:
            raise TaskBudgetError("profile must be a known profile ID or profile object")
        self._profile = self._validate_profile(profile_value)
        root_path = Path(root).expanduser().resolve()
        root_path.mkdir(parents=True, exist_ok=True)
        self.path = root_path / f"{budget_id}.json"
        self._lock_path = root_path / f".{budget_id}.lock"
        self._thread_lock = threading.RLock()
        with self._locked():
            if self.path.exists():
                data = self._read()
                self._validate_existing(data)
            else:
                self._write(self._new_ledger())

    @staticmethod
    def _validate_profile(profile: dict[str, Any]) -> dict[str, Any]:
        required = {
            "profile_id",
            "provider",
            "model_identity_verified_for_school_endpoint",
            "effective_date",
            "input_cny_per_million",
            "output_cny_per_million",
            "source_basis",
            "sources",
            "billing_caveat",
        }
        if not isinstance(profile, dict) or not required.issubset(profile):
            raise TaskBudgetError("price profile is missing required provenance fields")
        if not isinstance(profile["profile_id"], str) or not profile["profile_id"]:
            raise TaskBudgetError("price profile ID is invalid")
        if type(profile["model_identity_verified_for_school_endpoint"]) is not bool:
            raise TaskBudgetError("price profile identity verification flag is invalid")
        for field in ("input_cny_per_million", "output_cny_per_million"):
            _decimal(profile[field], field, positive=True)
        if not isinstance(profile["sources"], list) or not all(
            isinstance(item, str) and item.startswith("https://") for item in profile["sources"]
        ):
            raise TaskBudgetError("price profile sources must be HTTPS URLs")
        # Copy only the profile fields; arbitrary caller data is not persisted.
        return {
            "profile_id": profile["profile_id"],
            "provider": str(profile["provider"]),
            "model_identity_verified_for_school_endpoint": profile[
                "model_identity_verified_for_school_endpoint"
            ],
            "effective_date": str(profile["effective_date"]),
            "input_cny_per_million": format(
                _decimal(profile["input_cny_per_million"], "input_cny_per_million", positive=True),
                "f",
            ),
            "output_cny_per_million": format(
                _decimal(
                    profile["output_cny_per_million"], "output_cny_per_million", positive=True
                ),
                "f",
            ),
            "source_basis": profile["source_basis"],
            "sources": list(profile["sources"]),
            "billing_caveat": str(profile["billing_caveat"]),
        }

    @contextmanager
    def _locked(self) -> Iterator[None]:
        with self._thread_lock:
            with _file_lock(self._lock_path):
                yield

    def _new_ledger(self) -> dict[str, Any]:
        return {
            "schema_version": _LEDGER_SCHEMA_VERSION,
            "budget_id": self.budget_id,
            "created_at": _now(),
            "updated_at": _now(),
            "budget_rmb": _canonical_decimal(self._limit),
            "price_profile": self._profile,
            "reservations": {},
        }

    def _read(self) -> dict[str, Any]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            raise TaskBudgetError("task budget ledger cannot be read; refusing to reset it") from None
        if not isinstance(value, dict):
            raise TaskBudgetError("task budget ledger is malformed; refusing to reset it")
        return value

    def _validate_existing(self, data: dict[str, Any]) -> None:
        if (
            data.get("schema_version") != _LEDGER_SCHEMA_VERSION
            or data.get("budget_id") != self.budget_id
            or data.get("budget_rmb") != _canonical_decimal(self._limit)
            or data.get("price_profile") != self._profile
            or not isinstance(data.get("reservations"), dict)
        ):
            raise TaskBudgetError(
                "task budget ledger settings differ from this request; refusing to reset or reprice"
            )
        for reservation_id, item in data["reservations"].items():
            if not isinstance(reservation_id, str) or not isinstance(item, dict):
                raise TaskBudgetError("task budget ledger reservation is malformed")
            if item.get("status") not in {"pending", "unknown", "settled"}:
                raise TaskBudgetError("task budget ledger reservation status is invalid")
            _decimal(item.get("reserved_rmb"), "reserved_rmb")
            if item["status"] == "settled":
                _decimal(item.get("actual_cost_rmb"), "actual_cost_rmb")

    def _write(self, data: dict[str, Any]) -> None:
        data["updated_at"] = _now()
        fd, temp_name = tempfile.mkstemp(prefix=f".{self.budget_id}.", suffix=".tmp", dir=self.path.parent)
        temporary = Path(temp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            _fsync_directory(self.path.parent)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def _cost(self, input_tokens: int, output_tokens: int) -> Decimal:
        input_rate = Decimal(self._profile["input_cny_per_million"])
        output_rate = Decimal(self._profile["output_cny_per_million"])
        return (Decimal(input_tokens) * input_rate + Decimal(output_tokens) * output_rate) / _MILLION

    @staticmethod
    def _totals(data: dict[str, Any]) -> tuple[Decimal, Decimal, Decimal]:
        settled = Decimal("0")
        reserved = Decimal("0")
        for item in data["reservations"].values():
            if item["status"] == "settled":
                settled += Decimal(item["actual_cost_rmb"])
            else:
                # Both in-flight and unknown requests stay fully reserved.
                reserved += Decimal(item["reserved_rmb"])
        return settled, reserved, Decimal(data["budget_rmb"]) - settled - reserved

    def reserve(
        self,
        reservation_id: str,
        *,
        session_id: str,
        max_input_tokens: int,
        max_output_tokens: int,
    ) -> dict[str, str]:
        """Reserve a worst-case amount before a request is sent.

        Reservation IDs are one-shot. A caller restart must not reuse an old
        pending/unknown ID to trigger another API request; the existing reserve
        continues to count against the task limit.
        """
        if not isinstance(reservation_id, str) or not _RESERVATION_ID_RE.fullmatch(reservation_id):
            raise TaskBudgetError("reservation_id must be a safe identifier")
        if not isinstance(session_id, str) or not session_id or len(session_id) > 256:
            raise TaskBudgetError("session_id must be a non-empty short string")
        input_bound = _token_count(max_input_tokens, "max_input_tokens")
        output_bound = _token_count(max_output_tokens, "max_output_tokens")
        if input_bound + output_bound == 0:
            raise TaskBudgetError("request reservation must include a positive token bound")
        amount = self._cost(input_bound, output_bound)
        with self._locked():
            data = self._read()
            self._validate_existing(data)
            previous = data["reservations"].get(reservation_id)
            if previous is not None:
                raise TaskBudgetError("reservation ID already exists and cannot be reused")
            _settled, _reserved, remaining = self._totals(data)
            if amount > remaining:
                raise TaskBudgetError("task budget cannot cover the worst-case request reservation")
            data["reservations"][reservation_id] = {
                "session_id": session_id,
                "status": "pending",
                "created_at": _now(),
                "max_input_tokens": input_bound,
                "max_output_tokens": output_bound,
                "reserved_rmb": format(amount, "f"),
                "actual_input_tokens": None,
                "actual_output_tokens": None,
                "actual_cost_rmb": None,
            }
            self._write(data)
        return {"reservation_id": reservation_id, "reserved_rmb": format(amount, "f")}

    def settle(self, reservation_id: str, *, input_tokens: int, output_tokens: int) -> dict[str, str]:
        """Replace a pending worst-case reservation with known actual usage."""
        input_count = _token_count(input_tokens, "input_tokens")
        output_count = _token_count(output_tokens, "output_tokens")
        with self._locked():
            data = self._read()
            self._validate_existing(data)
            try:
                item = data["reservations"][reservation_id]
            except KeyError:
                raise TaskBudgetError("reservation ID is unknown") from None
            if item["status"] == "settled":
                if (
                    item["actual_input_tokens"] == input_count
                    and item["actual_output_tokens"] == output_count
                ):
                    return {"actual_cost_rmb": item["actual_cost_rmb"], "status": "settled"}
                raise TaskBudgetError("settled reservation cannot be changed")
            if item["status"] != "pending":
                raise TaskBudgetError("unknown reservation cannot be settled")
            cost = self._cost(input_count, output_count)
            item.update(
                {
                    "status": "settled",
                    "settled_at": _now(),
                    "actual_input_tokens": input_count,
                    "actual_output_tokens": output_count,
                    "actual_cost_rmb": format(cost, "f"),
                }
            )
            self._write(data)
        return {"actual_cost_rmb": format(cost, "f"), "status": "settled"}

    def mark_unknown(self, reservation_id: str, *, reason_code: str = "USAGE_UNKNOWN") -> dict[str, str]:
        """Keep the full reservation charged when usage could not be established."""
        if not isinstance(reason_code, str) or not re.fullmatch(r"[A-Z0-9_]{1,64}", reason_code):
            raise TaskBudgetError("reason_code is invalid")
        with self._locked():
            data = self._read()
            self._validate_existing(data)
            try:
                item = data["reservations"][reservation_id]
            except KeyError:
                raise TaskBudgetError("reservation ID is unknown") from None
            if item["status"] == "unknown":
                return {"reserved_rmb": item["reserved_rmb"], "status": "unknown"}
            if item["status"] != "pending":
                raise TaskBudgetError("settled reservation cannot be marked unknown")
            item.update({"status": "unknown", "unknown_at": _now(), "reason_code": reason_code})
            self._write(data)
            return {"reserved_rmb": item["reserved_rmb"], "status": "unknown"}

    def snapshot(self) -> dict[str, Any]:
        """Return a consistent, safe accounting summary and immutable provenance."""
        with self._locked():
            data = self._read()
            self._validate_existing(data)
            settled, reserved, remaining = self._totals(data)
            reservations = data["reservations"]
            return {
                "schema_version": data["schema_version"],
                "budget_id": data["budget_id"],
                "ledger_path": str(self.path),
                "budget_rmb": data["budget_rmb"],
                "settled_cost_rmb": format(settled, "f"),
                "reserved_rmb": format(reserved, "f"),
                "remaining_rmb": format(remaining, "f"),
                "pending_count": sum(item["status"] == "pending" for item in reservations.values()),
                "unknown_count": sum(item["status"] == "unknown" for item in reservations.values()),
                "settled_count": sum(item["status"] == "settled" for item in reservations.values()),
                "price_profile": data["price_profile"],
            }

