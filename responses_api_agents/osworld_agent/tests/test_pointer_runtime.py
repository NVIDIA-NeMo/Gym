# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Gym's task-local Pointer client wrapper."""

from __future__ import annotations

import builtins
import sys
from enum import Enum
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from responses_api_agents.osworld_agent import pointer_runtime


class FakeAPIStatusError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status_code: int = 429,
        body: object = None,
        retry_after: object = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body
        self.response = SimpleNamespace(headers={"retry-after": retry_after})


class FakeMessages:
    def __init__(self) -> None:
        self.create = Mock()
        self.count_tokens = Mock()


class FakeAnthropic:
    instances: list["FakeAnthropic"] = []
    __hash__ = None

    def __init__(self, **options: object) -> None:
        self.options = options
        self.closed = False
        self.beta = SimpleNamespace(messages=FakeMessages(), marker="beta")
        self.messages = FakeMessages()
        self.instances.append(self)

    def close(self) -> None:
        self.closed = True


class FakeProvider(Enum):
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"


class FakeContextManager:
    def __init__(self, client: FakeAnthropic) -> None:
        self.client = client
        self._counting_client = FakeAnthropic(label="old-counting")
        self.reject_next_client = False
        self.summary_errors = 0

    def __setattr__(self, name: str, value: object) -> None:
        if (
            name == "client"
            and getattr(self, "reject_next_client", False)
            and isinstance(value, pointer_runtime._ClientProxy)
        ):
            self.reject_next_client = False
            raise RuntimeError("install rejected")
        object.__setattr__(self, name, value)

    def summarize_history(self, client: object) -> object:
        try:
            return client.messages.create()
        except Exception as error:
            self.summary_errors += 1
            return f"[Summary failed: {error}]"


class FakeLLMClient:
    def __init__(self, name: str) -> None:
        self.name = name
        self.provider = FakeProvider.ANTHROPIC
        self.logger = Mock()
        self.client = FakeAnthropic(label=f"old-{name}")
        self.context_manager = FakeContextManager(self.client)
        self.use_summary = False
        self.summary_before_main = False
        self.generic_errors = 0
        self.compactions = 0

    def _refresh_client(self) -> None:
        raise AssertionError("credential refresh is not expected in these tests")

    def generate(self) -> object:
        last_error: Exception | None = None
        if self.summary_before_main:
            self.context_manager.summarize_history(self.client)
        for _ in range(5):
            try:
                if self.use_summary:
                    return self.context_manager.summarize_history(self.client)
                return self.client.beta.messages.create()
            except Exception as error:
                last_error = error
                self.generic_errors += 1
                message = str(error).lower()
                if "413" in message or "too large" in message or "too long" in message:
                    self.compactions += 1
        raise RuntimeError(f"All API attempts exhausted: {last_error}")


class FakePointerAgent:
    def __init__(self) -> None:
        for role in pointer_runtime._POINTER_ROLES:
            setattr(self, role, SimpleNamespace(llm=FakeLLMClient(role)))

    def predict(self, _obs: object = None) -> object:
        return self.gate.llm.generate()


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, delay: float) -> None:
        self.sleeps.append(delay)
        self.now += delay


CLIENT_OPTIONS = {
    "api_key": "test-key",  # pragma: allowlist secret
    "base_url": "https://inference-api.nvidia.com",
    "max_retries": 0,
    "timeout": 120.0,
}


def _install_fake_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("anthropic")
    module.APIStatusError = FakeAPIStatusError
    module.Anthropic = FakeAnthropic
    monkeypatch.setitem(sys.modules, "anthropic", module)
    FakeAnthropic.instances.clear()


def _budget_error(
    *,
    status_code: int = 429,
    error_type: str = "budget_exceeded",
    retry_after: object = None,
) -> FakeAPIStatusError:
    return FakeAPIStatusError(
        "sensitive provider response",
        status_code=status_code,
        body={"error": {"type": error_type, "message": "sensitive body"}},
        retry_after=retry_after,
    )


def _hardened_agent(
    monkeypatch: pytest.MonkeyPatch,
    *,
    deadline: float | None = None,
) -> tuple[FakePointerAgent, list[FakeAnthropic]]:
    _install_fake_anthropic(monkeypatch)
    agent = FakePointerAgent()
    old_clients = list(FakeAnthropic.instances)
    pointer_runtime.harden_pointer_agent(
        agent,
        CLIENT_OPTIONS,
        deadline_monotonic=deadline,
    )
    return agent, old_clients


def _install_clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    clock = FakeClock()
    monkeypatch.setattr(pointer_runtime.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(pointer_runtime.time, "sleep", clock.sleep)
    monkeypatch.setattr(pointer_runtime.random, "random", lambda: 0.0)
    return clock


def test_budget_detector_requires_typed_structured_429(monkeypatch) -> None:
    _install_fake_anthropic(monkeypatch)

    assert pointer_runtime.is_budget_exceeded(_budget_error()) is True
    assert pointer_runtime.is_budget_exceeded(_budget_error(error_type="rate_limit_error")) is False
    assert pointer_runtime.is_budget_exceeded(_budget_error(status_code=500)) is False
    assert pointer_runtime.is_budget_exceeded(FakeAPIStatusError("bad body", body="budget_exceeded")) is False
    assert pointer_runtime.is_budget_exceeded(RuntimeError("429 budget_exceeded")) is False

    original_import = builtins.__import__

    def import_without_anthropic(name, *args, **kwargs):
        if name == "anthropic":
            raise ImportError
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_anthropic)
    assert pointer_runtime.is_budget_exceeded(_budget_error()) is False


def test_hardening_replaces_only_instance_clients_and_is_idempotent(monkeypatch) -> None:
    _install_fake_anthropic(monkeypatch)
    agent = FakePointerAgent()
    old_predict = FakePointerAgent.predict
    old_generate = FakeLLMClient.generate
    old_clients = list(FakeAnthropic.instances)

    pointer_runtime.harden_pointer_agent(agent, CLIENT_OPTIONS)

    replacements = FakeAnthropic.instances[len(old_clients) :]
    assert len(replacements) == 4
    assert all(client.closed for client in old_clients)
    for role, raw_client in zip(pointer_runtime._POINTER_ROLES, replacements, strict=True):
        llm = getattr(agent, role).llm
        assert llm.client._client is raw_client
        assert llm.client.options == CLIENT_OPTIONS
        assert llm.client.beta.marker == "beta"
        assert raw_client.options == CLIENT_OPTIONS
        assert llm.context_manager.client is llm.client
        assert llm.context_manager._counting_client is llm.client
    assert FakePointerAgent.predict is old_predict
    assert FakeLLMClient.generate is old_generate

    pointer_runtime.harden_pointer_agent(agent, CLIENT_OPTIONS)
    assert len(FakeAnthropic.instances) == len(old_clients) + 4


def test_hardening_preserves_client_wrapper_across_refresh(monkeypatch) -> None:
    _install_fake_anthropic(monkeypatch)
    agent = FakePointerAgent()
    agent.gate.llm.name = "feasibility_gate"
    wrapped: list[tuple[str, Any]] = []

    class ClientWrapper:
        def __init__(self, target: Any, label: str) -> None:
            self.target = target
            self.label = label
            self.beta = target.beta
            self.messages = target.messages

        def __getattr__(self, name: str) -> Any:
            return getattr(self.target, name)

    def wrap_client(client: Any, label: str) -> ClientWrapper:
        wrapped.append((label, client))
        return ClientWrapper(client, label)

    pointer_runtime.harden_pointer_agent(agent, CLIENT_OPTIONS, client_wrapper=wrap_client)

    gate = agent.gate.llm
    first_client = gate.client
    assert first_client._client.label == "feasibility_gate"
    assert [label for label, _client in wrapped] == ["feasibility_gate", "planner", "executor", "verifier"]

    gate._refresh_client()

    assert gate.client is gate.context_manager.client
    assert gate.client is gate.context_manager._counting_client
    assert gate.client._client.label == "feasibility_gate"
    assert wrapped[-1][0] == "feasibility_gate"
    assert first_client._client.target.closed is True


def test_hardening_validates_every_role_before_creating_clients(monkeypatch) -> None:
    _install_fake_anthropic(monkeypatch)
    agent = FakePointerAgent()
    old_clients = list(FakeAnthropic.instances)
    agent.verifier.llm.provider = FakeProvider.BEDROCK

    with pytest.raises(RuntimeError, match="Pointer verifier LLM"):
        pointer_runtime.harden_pointer_agent(agent, CLIENT_OPTIONS)

    assert FakeAnthropic.instances == old_clients
    assert all(not client.closed for client in old_clients)


def test_hardening_requires_sdk_retries_disabled(monkeypatch) -> None:
    _install_fake_anthropic(monkeypatch)
    agent = FakePointerAgent()
    count = len(FakeAnthropic.instances)

    with pytest.raises(ValueError, match="max_retries=0"):
        pointer_runtime.harden_pointer_agent(agent, {**CLIENT_OPTIONS, "max_retries": 1})

    assert len(FakeAnthropic.instances) == count


def test_client_construction_failure_closes_partial_replacements(monkeypatch) -> None:
    _install_fake_anthropic(monkeypatch)
    agent = FakePointerAgent()
    old_clients = {role: getattr(agent, role).llm.client for role in pointer_runtime._POINTER_ROLES}
    original = pointer_runtime._new_client
    created: list[pointer_runtime._ClientProxy] = []

    def fail_third(*args, **kwargs):
        if len(created) == 2:
            raise RuntimeError("constructor failed")
        created.append(original(*args, **kwargs))
        return created[-1]

    monkeypatch.setattr(pointer_runtime, "_new_client", fail_third)
    with pytest.raises(RuntimeError, match="constructor failed"):
        pointer_runtime.harden_pointer_agent(agent, CLIENT_OPTIONS)

    assert all(client._client.closed for client in created)
    assert all(getattr(agent, role).llm.client is old_clients[role] for role in pointer_runtime._POINTER_ROLES)
    assert not hasattr(agent, pointer_runtime._RUNTIME_STATE)


def test_install_failure_rolls_back_every_role(monkeypatch) -> None:
    _install_fake_anthropic(monkeypatch)
    agent = FakePointerAgent()
    old_clients = {role: getattr(agent, role).llm.client for role in pointer_runtime._POINTER_ROLES}
    old_context_clients = {
        role: getattr(agent, role).llm.context_manager.client for role in pointer_runtime._POINTER_ROLES
    }
    del agent.gate.llm.context_manager._counting_client
    agent.planner.llm.context_manager.reject_next_client = True

    with pytest.raises(RuntimeError, match="install rejected"):
        pointer_runtime.harden_pointer_agent(agent, CLIENT_OPTIONS)

    for role in pointer_runtime._POINTER_ROLES:
        llm = getattr(agent, role).llm
        assert llm.client is old_clients[role]
        assert llm.context_manager.client is old_context_clients[role]
    assert not hasattr(agent.gate.llm.context_manager, "_counting_client")
    assert all(client.closed for client in FakeAnthropic.instances[-4:])
    assert not hasattr(agent, pointer_runtime._RUNTIME_STATE)


def test_quota_recovery_continues_beyond_upstream_retry_limit(monkeypatch) -> None:
    clock = _install_clock(monkeypatch)
    agent, old_clients = _hardened_agent(monkeypatch)
    raw_client = FakeAnthropic.instances[len(old_clients)]
    success = object()
    raw_client.beta.messages.create.side_effect = [*[_budget_error() for _ in range(5)], success]

    assert agent.predict() is success
    assert clock.sleeps == [30.0, 60.0, 120.0, 150.0, 150.0]
    assert agent.gate.llm.generic_errors == 0
    assert "sensitive" not in str(agent.gate.llm.logger.warning.call_args_list)


def test_quota_recovery_ignores_retry_log_failure(monkeypatch, caplog) -> None:
    clock = _install_clock(monkeypatch)
    agent, old_clients = _hardened_agent(monkeypatch)
    raw_client = FakeAnthropic.instances[len(old_clients)]
    agent.gate.llm.logger.warning.side_effect = RuntimeError("logger unavailable")
    raw_client.beta.messages.create.side_effect = [_budget_error(), "ok"]

    with caplog.at_level("WARNING"):
        assert agent.predict() == "ok"

    assert clock.sleeps == [30.0]
    assert "Pointer quota retry logging failed" in caplog.text


def test_retry_logging_runs_outside_provider_exception_context(monkeypatch) -> None:
    clock = _install_clock(monkeypatch)
    agent, old_clients = _hardened_agent(monkeypatch)
    raw_client = FakeAnthropic.instances[len(old_clients)]
    active_exceptions: list[BaseException | None] = []

    def capture_exception_context(*_args: object) -> None:
        active_exceptions.append(sys.exception())

    agent.gate.llm.logger.warning.side_effect = capture_exception_context
    raw_client.beta.messages.create.side_effect = [_budget_error(), "ok"]

    assert agent.predict() == "ok"
    assert clock.sleeps == [30.0]
    assert active_exceptions == [None]


@pytest.mark.parametrize("delay_source", ["logger", "sleep"])
def test_retry_delay_cannot_cross_deadline_then_issue_request(monkeypatch, delay_source: str) -> None:
    clock = _install_clock(monkeypatch)
    agent, old_clients = _hardened_agent(monkeypatch, deadline=20.0)
    raw_client = FakeAnthropic.instances[len(old_clients)]
    raw_client.beta.messages.create.side_effect = _budget_error()

    if delay_source == "logger":
        agent.gate.llm.logger.warning.side_effect = lambda *_args: setattr(clock, "now", 25.0)
    else:
        monkeypatch.setattr(
            pointer_runtime.time, "sleep", lambda delay: setattr(clock, "now", clock.now + delay + 5.0)
        )

    with pytest.raises(RuntimeError, match="Provider quota retry window exhausted"):
        agent.predict()

    assert raw_client.beta.messages.create.call_count == 1


def test_provider_call_time_is_included_when_deadline_expires(monkeypatch) -> None:
    clock = _install_clock(monkeypatch)
    agent, old_clients = _hardened_agent(monkeypatch, deadline=50.0)
    raw_client = FakeAnthropic.instances[len(old_clients)]
    calls = 0

    def budget_error_after_provider_time() -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            clock.now = 55.0
        raise _budget_error()

    raw_client.beta.messages.create.side_effect = budget_error_after_provider_time

    with pytest.raises(RuntimeError, match="Cumulative wait: 55.0s"):
        agent.predict()

    assert calls == 2
    assert clock.sleeps == [30.0]


def test_quota_recovery_exhaustion_bypasses_upstream_retries(monkeypatch) -> None:
    clock = _install_clock(monkeypatch)
    agent, old_clients = _hardened_agent(monkeypatch)
    raw_client = FakeAnthropic.instances[len(old_clients)]
    raw_client.beta.messages.create.side_effect = _budget_error()

    with pytest.raises(RuntimeError, match="Provider quota retry window exhausted") as exc_info:
        agent.predict()

    assert clock.sleeps == [30.0, 60.0, 120.0, 150.0, 150.0, 150.0, 150.0, 90.0]
    assert sum(clock.sleeps) == 900.0
    assert "Cumulative wait: 900.0s" in str(exc_info.value)
    assert "sensitive" not in str(exc_info.value)
    assert exc_info.value.__suppress_context__ is True
    assert agent.gate.llm.generic_errors == 0
    assert raw_client.beta.messages.create.call_count == 8


def test_quota_recovery_stops_at_rollout_deadline(monkeypatch) -> None:
    clock = _install_clock(monkeypatch)
    agent, old_clients = _hardened_agent(monkeypatch, deadline=100.0)
    raw_client = FakeAnthropic.instances[len(old_clients)]
    raw_client.beta.messages.create.side_effect = _budget_error()

    with pytest.raises(RuntimeError, match="Cumulative wait: 100.0s"):
        agent.predict()

    assert clock.sleeps == [30.0, 60.0, 10.0]
    assert raw_client.beta.messages.create.call_count == 3


def test_retry_after_is_bounded_and_not_partially_honored(monkeypatch) -> None:
    clock = _install_clock(monkeypatch)
    agent, old_clients = _hardened_agent(monkeypatch)
    raw_client = FakeAnthropic.instances[len(old_clients)]
    raw_client.beta.messages.create.side_effect = [_budget_error(retry_after=500), "ok"]
    assert agent.predict() == "ok"
    assert clock.sleeps == [300.0]

    pointer_runtime.close_pointer_agent(agent)
    clock = _install_clock(monkeypatch)
    agent, old_clients = _hardened_agent(monkeypatch)
    raw_client = FakeAnthropic.instances[len(old_clients)]
    raw_client.beta.messages.create.side_effect = [_budget_error(retry_after="invalid"), "ok"]
    assert agent.predict() == "ok"
    assert clock.sleeps == [30.0]

    pointer_runtime.close_pointer_agent(agent)
    clock = _install_clock(monkeypatch)
    agent, old_clients = _hardened_agent(monkeypatch)
    raw_client = FakeAnthropic.instances[len(old_clients)]
    raw_client.beta.messages.create.side_effect = _budget_error(retry_after=200)
    with pytest.raises(RuntimeError, match="Cumulative wait: 800.0s"):
        agent.predict()
    assert clock.sleeps == [200.0, 200.0, 200.0, 200.0]


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("content_length_limit"),
        RuntimeError("content length exceeded"),
        FakeAPIStatusError("opaque", status_code=413),
        FakeAPIStatusError("opaque", body={"error": {"type": "content_length_limit"}}),
    ],
)
def test_content_length_errors_use_upstream_compaction(monkeypatch, error: Exception) -> None:
    _install_clock(monkeypatch)
    agent, old_clients = _hardened_agent(monkeypatch)
    raw_client = FakeAnthropic.instances[len(old_clients)]
    raw_client.beta.messages.create.side_effect = [error, "ok"]

    assert agent.predict() == "ok"
    assert agent.gate.llm.compactions == 1
    assert agent.gate.llm.generic_errors == 1


def test_stable_summary_endpoint_shares_quota_recovery(monkeypatch) -> None:
    clock = _install_clock(monkeypatch)
    agent, old_clients = _hardened_agent(monkeypatch)
    llm = agent.gate.llm
    llm.use_summary = True
    raw_client = FakeAnthropic.instances[len(old_clients)]
    raw_client.messages.create.side_effect = [_budget_error(), _budget_error(), "summary"]

    assert agent.predict() == "summary"
    assert clock.sleeps == [30.0, 60.0]
    assert llm.context_manager.summary_errors == 0


def test_stable_summary_failure_does_not_expose_provider_error(monkeypatch) -> None:
    agent, old_clients = _hardened_agent(monkeypatch)
    llm = agent.gate.llm
    llm.use_summary = True
    raw_client = FakeAnthropic.instances[len(old_clients)]
    raw_client.messages.create.side_effect = RuntimeError("sensitive provider body with api_key=secret-value")

    result = agent.predict()

    assert pointer_runtime.PROVIDER_FAILURE_MESSAGE in result
    assert "sensitive" not in result
    assert "secret-value" not in result
    assert llm.context_manager.summary_errors == 1


def test_quota_window_is_shared_by_summary_and_main_call(monkeypatch) -> None:
    clock = _install_clock(monkeypatch)
    agent, old_clients = _hardened_agent(monkeypatch)
    llm = agent.gate.llm
    llm.summary_before_main = True
    raw_client = FakeAnthropic.instances[len(old_clients)]
    raw_client.messages.create.side_effect = [*[_budget_error() for _ in range(5)], "summary"]
    raw_client.beta.messages.create.side_effect = _budget_error()

    with pytest.raises(RuntimeError, match="Cumulative wait: 900.0s"):
        agent.predict()

    assert sum(clock.sleeps) == 900.0
    assert raw_client.messages.create.call_count == 6
    assert raw_client.beta.messages.create.call_count == 3


def test_credential_refresh_preserves_proxy_options_and_controller(monkeypatch) -> None:
    clock = _install_clock(monkeypatch)
    agent, old_clients = _hardened_agent(monkeypatch)
    llm = agent.gate.llm
    old_proxy = llm.client
    controller = old_proxy._controller

    llm._refresh_client()

    assert llm.client is llm.context_manager.client
    assert llm.client is llm.context_manager._counting_client
    assert llm.client._controller is controller
    assert llm.client._client.options == CLIENT_OPTIONS
    assert old_proxy._client.closed is True
    llm.client._client.beta.messages.create.side_effect = [_budget_error(), "ok"]
    assert agent.predict() == "ok"
    assert clock.sleeps == [30.0]


def test_failed_credential_refresh_restores_current_proxy(monkeypatch) -> None:
    agent, old_clients = _hardened_agent(monkeypatch)
    llm = agent.gate.llm
    old_proxy = llm.client
    llm.context_manager.reject_next_client = True

    with pytest.raises(RuntimeError, match=pointer_runtime.CLIENT_REFRESH_FAILURE_MESSAGE) as exc_info:
        llm._refresh_client()

    assert llm.client is old_proxy
    assert llm.context_manager.client is old_proxy
    assert llm.context_manager._counting_client is old_proxy
    assert FakeAnthropic.instances[-1].closed is True
    assert exc_info.value.__suppress_context__ is True
    assert "install rejected" not in str(exc_info.value)


def test_failed_credential_refresh_construction_is_sanitized(monkeypatch) -> None:
    agent, _old_clients = _hardened_agent(monkeypatch)
    llm = agent.gate.llm

    monkeypatch.setattr(
        pointer_runtime,
        "_new_client",
        Mock(side_effect=RuntimeError("sensitive endpoint and credential")),
    )

    with pytest.raises(RuntimeError, match=pointer_runtime.CLIENT_REFRESH_FAILURE_MESSAGE) as exc_info:
        llm._refresh_client()

    assert "sensitive" not in str(exc_info.value)
    assert exc_info.value.__suppress_context__ is True


@pytest.mark.parametrize("interruption_type", [KeyboardInterrupt, SystemExit])
def test_credential_refresh_preserves_process_interruptions(monkeypatch, interruption_type) -> None:
    agent, _old_clients = _hardened_agent(monkeypatch)
    llm = agent.gate.llm
    interruption = interruption_type()
    monkeypatch.setattr(pointer_runtime, "_new_client", Mock(side_effect=interruption))

    with pytest.raises(interruption_type) as exc_info:
        llm._refresh_client()

    assert exc_info.value is interruption
    assert exc_info.value.__suppress_context__ is True


def test_swap_interruption_takes_priority_over_rollback_failure(monkeypatch) -> None:
    agent, _old_clients = _hardened_agent(monkeypatch)
    llm = agent.gate.llm
    context_type = type(llm.context_manager)
    original_setattr = context_type.__setattr__
    interruption = KeyboardInterrupt()
    assignment_errors = [interruption, RuntimeError("sensitive rollback failure")]

    def fail_client_assignment(self, name, value):
        if name == "client" and isinstance(value, pointer_runtime._ClientProxy) and assignment_errors:
            raise assignment_errors.pop(0)
        original_setattr(self, name, value)

    monkeypatch.setattr(context_type, "__setattr__", fail_client_assignment)

    with pytest.raises(KeyboardInterrupt) as exc_info:
        llm._refresh_client()

    assert exc_info.value is interruption
    assert exc_info.value.__suppress_context__ is True
    assert assignment_errors == []
    assert FakeAnthropic.instances[-1].closed is True


def test_non_budget_errors_and_count_tokens_delegate_unchanged(monkeypatch) -> None:
    agent, old_clients = _hardened_agent(monkeypatch)
    raw_client = FakeAnthropic.instances[len(old_clients)]
    error = RuntimeError("ordinary failure")
    raw_client.beta.messages.create.side_effect = error

    with pytest.raises(RuntimeError) as exc_info:
        agent.gate.llm.client.beta.messages.create()
    assert exc_info.value is error

    raw_client.beta.messages.count_tokens.return_value = SimpleNamespace(input_tokens=42)
    result = agent.gate.llm.context_manager._counting_client.beta.messages.count_tokens()
    assert result.input_tokens == 42
    raw_client.beta.messages.create.side_effect = None
    raw_client.beta.messages.create.return_value = "direct success"
    assert agent.gate.llm.client.beta.messages.create() == "direct success"


def test_upstream_exhaustion_does_not_expose_provider_error(monkeypatch) -> None:
    agent, old_clients = _hardened_agent(monkeypatch)
    raw_client = FakeAnthropic.instances[len(old_clients)]
    raw_client.beta.messages.create.side_effect = RuntimeError("sensitive provider response")

    with pytest.raises(RuntimeError, match=pointer_runtime.PROVIDER_FAILURE_MESSAGE) as exc_info:
        agent.predict()

    assert "sensitive" not in str(exc_info.value)
    assert exc_info.value.__suppress_context__ is True


def test_private_quota_signal_cannot_escape_predict_scope(monkeypatch) -> None:
    clock = _install_clock(monkeypatch)
    agent, old_clients = _hardened_agent(monkeypatch, deadline=0.0)
    client = agent.gate.llm.client
    raw_client = FakeAnthropic.instances[len(old_clients)]
    raw_client.beta.messages.create.side_effect = _budget_error()

    with pytest.raises(RuntimeError, match="Provider quota retry window exhausted"):
        client.beta.messages.create()
    assert client._controller.active is False

    client._controller.deadline = 1000.0
    raw_client.beta.messages.create.side_effect = _budget_error()
    monkeypatch.setattr(pointer_runtime.time, "sleep", Mock(side_effect=KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt) as exc_info:
        agent.predict()
    assert exc_info.value.__suppress_context__ is True
    assert client._controller.active is False
    assert clock.sleeps == []


def test_reentrant_generation_fails_closed(monkeypatch) -> None:
    agent, _ = _hardened_agent(monkeypatch)
    controller = agent.gate.llm.client._controller
    controller.begin()
    try:
        with pytest.raises(RuntimeError, match="generation re-entered"):
            controller.begin()
    finally:
        controller.end()


def test_new_client_and_cleanup_failure_paths_are_best_effort(monkeypatch, caplog) -> None:
    _install_fake_anthropic(monkeypatch)
    controller = pointer_runtime._RetryController("gate", Mock(), None)

    monkeypatch.setattr(
        pointer_runtime,
        "_ClientProxy",
        Mock(side_effect=RuntimeError("proxy failed")),
    )
    with pytest.raises(RuntimeError, match="proxy failed"):
        pointer_runtime._new_client(CLIENT_OPTIONS, controller, "gate", None)
    assert FakeAnthropic.instances[-1].closed is True

    pointer_runtime._close_client(object())
    bad_client = SimpleNamespace(close=Mock(side_effect=RuntimeError("close failed")))
    with caplog.at_level("WARNING"):
        pointer_runtime._close_client(bad_client)
    assert "Pointer Anthropic client cleanup failed" in caplog.text


def test_missing_predict_is_rejected_before_client_creation(monkeypatch) -> None:
    _install_fake_anthropic(monkeypatch)
    agent = FakePointerAgent()
    agent.predict = None
    existing_count = len(FakeAnthropic.instances)

    with pytest.raises(RuntimeError, match="missing predict"):
        pointer_runtime.harden_pointer_agent(agent, CLIENT_OPTIONS)

    assert len(FakeAnthropic.instances) == existing_count


def test_cleanup_closes_current_clients_and_is_idempotent(monkeypatch) -> None:
    agent, old_clients = _hardened_agent(monkeypatch)
    replacements = FakeAnthropic.instances[len(old_clients) :]

    pointer_runtime.close_pointer_agent(agent)
    pointer_runtime.close_pointer_agent(agent)

    assert all(client.closed for client in replacements)
    with pytest.raises(RuntimeError, match="already closed"):
        pointer_runtime.harden_pointer_agent(agent, CLIENT_OPTIONS)


def test_cleanup_handles_unhardened_partial_agent(monkeypatch) -> None:
    _install_fake_anthropic(monkeypatch)
    agent = FakePointerAgent()
    remaining_clients = [
        client
        for role in ("gate", "planner", "executor")
        for client in (
            getattr(agent, role).llm.client,
            getattr(agent, role).llm.context_manager._counting_client,
        )
    ]
    del agent.verifier

    pointer_runtime.close_pointer_agent(agent)

    assert all(client.closed for client in remaining_clients)
