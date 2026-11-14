"""学校 API 网关只用注入的 HTTP 假响应验证，不发送真实请求。"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

import pytest
from evaluation_d.school_gateway import GatewayError, SchoolGateway, _NoRedirect
from evaluation_d.task_budget import TaskBudgetLedger


def _config(**overrides: Any) -> dict[str, Any]:
    config = {
        "url": "https://aigc-api.hkust-gz.edu.cn/api/v2/chat/completions",
        "model": "DeepSeek-V4.1-Flash",
        "input_cny_per_million": "0.50",
        "output_cny_per_million": "2.00",
        "max_tokens": 32,
        "max_input_tokens": 4096,
        "timeout": 90,
    }
    config.update(overrides)
    return config


def _payload(*, usage: dict[str, Any] | None = None, text: Any = "计划文本") -> bytes:
    body: dict[str, Any] = {"choices": [{"message": {"content": text}}]}
    if usage is not None:
        body["usage"] = usage
    return json.dumps(body, ensure_ascii=False).encode("utf-8")


class FakeTransport:
    def __init__(
        self,
        response: tuple[int, dict[str, str], bytes] | None = None,
        error: Exception | None = None,
    ):
        self.response = response
        self.error = error
        self.requests: list[tuple[urllib.request.Request, float]] = []

    def __call__(self, request: urllib.request.Request, timeout: float):
        self.requests.append((request, timeout))
        if self.error is not None:
            raise self.error
        return self.response


def _ledger_text(gateway: SchoolGateway) -> str:
    directory = Path(gateway.metadata()["ledger_path"])
    return "\n".join(path.read_text(encoding="utf-8") for path in directory.iterdir())


def test_interrupted_request_keeps_reservation_and_cannot_call_again(tmp_path):
    def interrupt(request, timeout):
        raise KeyboardInterrupt

    gateway = SchoolGateway(
        _config(),
        "synthetic-interrupt-key",
        tmp_path / "sessions",
        max_calls=2,
        transport=interrupt,
    )
    with pytest.raises(KeyboardInterrupt):
        gateway.call_llm([{"role": "user", "content": "synthetic request"}])
    assert gateway.metadata()["usage_status"] == "unknown"
    assert gateway.metadata()["estimated_cost_rmb"] is None
    assert gateway.can_call is False
    with pytest.raises(GatewayError, match="不能继续调用"):
        gateway.call_llm([{"role": "user", "content": "do not send"}])
    assert gateway.metadata()["call_attempts"] == 1


def test_success_records_reservation_then_real_usage_and_safe_request(tmp_path):
    transport = FakeTransport(
        (
            200,
            {"Content-Type": "application/json"},
            _payload(usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}),
        )
    )
    api_key = "runtime-key-do-not-persist"
    messages = [{"role": "user", "content": "测试问题：只在内存传递"}]
    gateway = SchoolGateway(_config(), api_key, tmp_path / "sessions", transport=transport)
    result = gateway.call_llm(messages)

    assert result["text"] == "计划文本"
    assert result["input_tokens"] == 100
    assert result["output_tokens"] == 20
    assert type(result["cost_rmb"]) is float
    assert result["cost_rmb"] == pytest.approx(0.00009)
    assert not gateway.can_call
    metadata = gateway.metadata()
    assert metadata["real_llm_calls"] == 0
    assert metadata["mock_calls"] == metadata["call_attempts"] == 1
    assert metadata["usage_status"] == "known"
    assert metadata["input_tokens"] == 100 and metadata["output_tokens"] == 20
    assert metadata["estimated_cost_rmb"] == "0.00009"
    assert metadata["reserved_rmb"] == "0"
    assert metadata["remaining_budget_rmb"] == "0.09991"

    request, timeout = transport.requests[0]
    assert request.full_url == _config()["url"]
    assert request.get_method() == "POST"
    assert request.get_header("Content-type") == "application/json"


def test_request_timeout_cap_only_reduces_configured_timeout(tmp_path):
    transport = FakeTransport(
        (
            200,
            {"Content-Type": "application/json"},
            _payload(usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}),
        )
    )
    gateway = SchoolGateway(
        _config(),
        "synthetic-timeout-key",
        tmp_path / "sessions",
        max_calls=3,
        transport=transport,
    )
    message = [{"role": "user", "content": "synthetic timeout cap test"}]

    gateway.set_request_timeout_cap(600)
    gateway.call_llm(message)
    gateway.set_request_timeout_cap(30)
    gateway.call_llm(message)
    gateway.set_request_timeout_cap(None)
    gateway.call_llm(message)

    assert [timeout for _, timeout in transport.requests] == [90, 30, 90]
    assert request.get_header("Authorization") == f"Bearer {api_key}"
    assert timeout == 90
    sent = json.loads(request.data)
    assert sent["messages"] == messages
    assert sent["max_tokens"] == 32
    assert sent["temperature"] == 0
    assert sent["stream"] is False

    calls = Path(metadata["ledger_path"], "calls.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["status"] for line in calls] == ["reserved", "complete"]
    ledger = _ledger_text(gateway)
    assert api_key not in repr(gateway)
    assert api_key not in ledger
    assert messages[0]["content"] not in ledger
    assert "计划文本" not in ledger
    assert "Authorization" not in ledger


def test_missing_usage_is_unknown_reserves_all_remaining_budget_and_blocks(tmp_path):
    transport = FakeTransport((200, {}, _payload()))
    gateway = SchoolGateway(_config(), "secret", tmp_path / "sessions", transport=transport)
    with pytest.raises(GatewayError) as error:
        gateway.call_llm([{"role": "user", "content": "hello"}])
    assert error.value.code == "USAGE_UNKNOWN"
    assert gateway.can_call is False
    metadata = gateway.metadata()
    assert metadata["call_attempts"] == 1
    assert metadata["usage_status"] == "unknown"
    assert metadata["input_tokens"] is None and metadata["output_tokens"] is None
    assert metadata["estimated_cost_rmb"] is None
    assert metadata["reserved_rmb"] == metadata["budget_rmb"] == "0.10"
    assert metadata["remaining_budget_rmb"] == "0.00"
    with pytest.raises(GatewayError) as blocked:
        gateway.call_llm([{"role": "user", "content": "second call"}])
    assert blocked.value.code == "SESSION_BLOCKED"
    assert len(transport.requests) == 1


def test_timeout_is_unknown_without_retry(tmp_path):
    transport = FakeTransport(error=TimeoutError("secret server details"))
    gateway = SchoolGateway(_config(), "secret-key", tmp_path / "sessions", transport=transport)
    with pytest.raises(GatewayError) as error:
        gateway.call_llm([{"role": "user", "content": "hello"}])
    assert error.value.code == "TIMEOUT"
    assert "secret server details" not in str(error.value)
    assert gateway.metadata()["usage_status"] == "unknown"
    assert gateway.metadata()["real_llm_calls"] == 0
    assert gateway.metadata()["mock_calls"] == 1
    assert len(transport.requests) == 1


def test_call_limit_blocks_second_call_without_reservation(tmp_path):
    transport = FakeTransport(
        (200, {}, _payload(usage={"prompt_tokens": 3, "completion_tokens": 2}))
    )
    gateway = SchoolGateway(
        _config(), "secret", tmp_path / "sessions", transport=transport, max_calls=1
    )
    gateway.call_llm([{"role": "user", "content": "first"}])
    calls_path = Path(gateway.metadata()["ledger_path"], "calls.jsonl")
    before = calls_path.read_text(encoding="utf-8")
    with pytest.raises(GatewayError) as error:
        gateway.call_llm([{"role": "user", "content": "second"}])
    assert error.value.code == "CALL_LIMIT"
    assert gateway.metadata()["call_attempts"] == 1
    assert calls_path.read_text(encoding="utf-8") == before
    assert len(transport.requests) == 1


def test_budget_and_input_preflight_do_not_count_requests(tmp_path):
    expensive = _config(
        input_cny_per_million="1000",
        output_cny_per_million="1000",
        max_input_tokens=4096,
        max_tokens=1024,
    )
    transport = FakeTransport(
        (200, {}, _payload(usage={"prompt_tokens": 1, "completion_tokens": 1}))
    )
    gateway = SchoolGateway(
        expensive, "secret", tmp_path / "budget", budget_rmb="0.01", transport=transport
    )
    assert gateway.can_call is False
    with pytest.raises(GatewayError) as budget_error:
        gateway.call_llm([{"role": "user", "content": "small"}])
    assert budget_error.value.code == "BUDGET_EXCEEDED"
    assert gateway.metadata()["call_attempts"] == 0
    assert gateway.metadata()["real_llm_calls"] == 0
    assert not transport.requests

    size_limited = SchoolGateway(
        _config(max_input_tokens=32), "secret", tmp_path / "large", transport=transport
    )
    with pytest.raises(GatewayError) as input_error:
        size_limited.call_llm([{"role": "user", "content": "x" * 100}])
    assert input_error.value.code == "INPUT_TOO_LARGE"
    assert size_limited.metadata()["call_attempts"] == 0
    assert not transport.requests


def test_redirect_is_not_followed_and_usage_remains_unknown(tmp_path):
    transport = FakeTransport((302, {"Location": "https://attacker.invalid/collect"}, b""))
    gateway = SchoolGateway(_config(), "secret", tmp_path / "sessions", transport=transport)
    with pytest.raises(GatewayError) as error:
        gateway.call_llm([{"role": "user", "content": "hello"}])
    assert error.value.code == "HTTP_ERROR"
    assert gateway.metadata()["usage_status"] == "unknown"
    assert len(transport.requests) == 1
    request = urllib.request.Request("https://safe.example/", data=b"x")
    assert (
        _NoRedirect().redirect_request(
            request,
            None,
            302,
            "Found",
            {"Location": "https://attacker.invalid/"},
            "https://attacker.invalid/",
        )
        is None
    )


def test_invalid_response_with_valid_usage_is_charged_before_failure(tmp_path):
    transport = FakeTransport(
        (
            200,
            {},
            _payload(
                usage={"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
                text={"not": "text"},
            ),
        )
    )
    gateway = SchoolGateway(_config(), "secret", tmp_path / "sessions", transport=transport)
    with pytest.raises(GatewayError) as error:
        gateway.call_llm([{"role": "user", "content": "hello"}])
    assert error.value.code == "INVALID_RESPONSE"
    metadata = gateway.metadata()
    assert metadata["usage_status"] == "known"
    assert metadata["input_tokens"] == 11 and metadata["output_tokens"] == 7
    assert metadata["estimated_cost_rmb"] == "0.0000195"
    assert metadata["blocked"] is True and metadata["can_call"] is False
    assert '"status": "failed_known"' in _ledger_text(gateway)


@pytest.mark.parametrize(
    "url",
    [
        "http://aigc-api.hkust-gz.edu.cn/api/v2/chat/completions",
        "https://user:pass@example.com/v1/chat/completions",
        "https://example.com/v1/chat/completions?key=secret",
        "https://example.com:8443/v1/chat/completions",
    ],
)
def test_rejects_unsafe_urls_without_echoing_url(tmp_path, url):
    with pytest.raises(GatewayError) as error:
        SchoolGateway(_config(url=url), "secret", tmp_path / "sessions", transport=FakeTransport())
    assert error.value.code == "CONFIG_INVALID"
    assert url not in str(error.value)
    assert not (tmp_path / "sessions").exists()


def test_config_key_is_rejected_and_runtime_key_never_persisted(tmp_path):
    with pytest.raises(GatewayError) as error:
        SchoolGateway(_config(api_key="must-not-be-accepted"), "runtime-secret", tmp_path / "bad")
    assert error.value.code == "CONFIG_INVALID"
    assert "must-not-be-accepted" not in str(error.value)
    assert not (tmp_path / "bad").exists()

    transport = FakeTransport(
        (200, {}, _payload(usage={"prompt_tokens": 1, "completion_tokens": 1}))
    )
    gateway = SchoolGateway(_config(), "runtime-secret", tmp_path / "good", transport=transport)
    gateway.call_llm([{"role": "user", "content": "do not log this"}])
    assert "runtime-secret" not in _ledger_text(gateway)
    assert "do not log this" not in _ledger_text(gateway)
    assert "runtime-secret" not in repr(gateway)


def test_explicit_task_budget_reserves_then_settles_known_usage(tmp_path):
    task_ledger = TaskBudgetLedger(
        tmp_path / "task-budgets",
        "course-final",
        budget_rmb="20",
        profile="deepseek-v4.1-flash-peak-2026-10-21",
    )
    transport = FakeTransport(
        (200, {}, _payload(usage={"prompt_tokens": 100, "completion_tokens": 20}))
    )
    gateway = SchoolGateway(
        _config(),
        "synthetic-only-key",
        tmp_path / "sessions",
        transport=transport,
        task_budget_ledger=task_ledger,
    )
    gateway.call_llm([{"role": "user", "content": "mock only"}])
    summary = task_ledger.snapshot()
    assert summary["settled_count"] == 1
    assert summary["pending_count"] == summary["unknown_count"] == 0
    assert summary["settled_cost_rmb"] == "0.0003628368"
    metadata = gateway.metadata()
    assert metadata["task_budget"]["budget_id"] == "course-final"
    assert "synthetic-only-key" not in _ledger_text(gateway)


def test_explicit_task_budget_keeps_ambiguous_request_reserved(tmp_path):
    task_ledger = TaskBudgetLedger(
        tmp_path / "task-budgets",
        "course-unknown",
        budget_rmb="20",
        profile="deepseek-v4.1-flash-peak-2026-10-21",
    )
    gateway = SchoolGateway(
        _config(),
        "synthetic-only-key",
        tmp_path / "sessions",
        transport=FakeTransport(error=TimeoutError("synthetic timeout")),
        task_budget_ledger=task_ledger,
    )
    with pytest.raises(GatewayError, match="usage 未知"):
        gateway.call_llm([{"role": "user", "content": "mock only"}])
    summary = task_ledger.snapshot()
    assert summary["unknown_count"] == 1
    assert summary["reserved_rmb"] != "0"
