# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import signal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_correlation import current_rollout_id
from nemo_gym.rollout_observability import AgentEpisode, AgentObservationBundle, ObservationGap
from responses_api_agents.nooa_agent import sandbox_entrypoint as entrypoint
from responses_api_agents.nooa_agent.config import NOOAInvocationConfig
from responses_api_agents.nooa_agent.runner import NOOARunFailure, NOOARunRequest, NOOARunResult
from responses_api_agents.nooa_agent.tests.test_gym_llm import model_response


def payload() -> entrypoint.SandboxInput:
    return entrypoint.SandboxInput(
        invocation=NOOAInvocationConfig(agent_class="test:Agent", invocation_adapter="test:invoke"),
        request=NOOARunRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="question", temperature=0.2),
            model_url_path="/ng-rollout/task-a1/train/v1/responses",
            rollout_id="task-a1",
            task_id="task",
            model_cookies={"model": "old"},
            resource_cookies={"resource": "old"},
        ),
        model_base_url="http://model:8000",
        model_server_name="policy",
        max_policy_calls=3,
        context_window=262144,
    )


def run_result() -> NOOARunResult:
    return NOOARunResult(
        episode=AgentEpisode(
            response=NeMoGymResponse.model_validate(model_response()),
            observations=AgentObservationBundle(source="nooa", gaps=[ObservationGap(code="test", detail="evidence")]),
        ),
        return_value="answer",
        model_cookies={"model": "new"},
        resource_cookies={"resource": "new"},
    )


@pytest.mark.parametrize("field", ["provider_credentials", "verifier_metadata", "global_config"])
def test_launch_payload_rejects_outer_state(field: str) -> None:
    data = payload().model_dump(mode="json")
    data[field] = {"secret": "not-for-agent"}
    with pytest.raises(ValidationError, match=field):
        entrypoint.SandboxInput.model_validate(data)


@pytest.mark.asyncio
async def test_model_route_applied_once_and_cookie_jar_forwarded(monkeypatch) -> None:
    call = AsyncMock()
    monkeypatch.setattr(entrypoint, "request", call)
    p = payload()
    client = entrypoint._ModelClient(str(p.model_base_url))
    await client.post(
        server_name="policy",
        url_path=p.request.model_url_path,
        json=p.request.responses_create_params,
        cookies=p.request.model_cookies,
        headers={"x-session-id": "child-invocation"},
    )
    assert call.await_args.args == ("POST", "http://model:8000/ng-rollout/task-a1/train/v1/responses")
    assert call.await_args.kwargs["json"]["temperature"] == 0.2
    assert call.await_args.kwargs["cookies"] == {"model": "old"}
    assert call.await_args.kwargs["headers"] == {"x-session-id": "child-invocation"}


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "transient", "fatal", "cancelled"])
@pytest.mark.parametrize("limit", [None, 3])
async def test_child_preserves_evidence_and_classifies_outcome(monkeypatch, outcome: str, limit: int | None) -> None:
    p = payload()
    p.max_policy_calls = limit
    result = run_result()

    async def run(request):
        assert current_rollout_id() == "task-a1"
        request.model_cookies.update(result.model_cookies)
        request.resource_cookies.update(result.resource_cookies)
        if outcome == "cancelled":
            error = asyncio.CancelledError()
            error.nooa_result = result
            raise error
        if outcome != "success":
            cause = ConnectionError("offline") if outcome == "transient" else ValueError("broken adapter")
            raise NOOARunFailure(cause, result) from cause
        return result

    runner_factory = MagicMock(return_value=MagicMock(run=run))
    monkeypatch.setattr(entrypoint, "InProcessNOOARunner", runner_factory)
    artifact = await entrypoint.execute(p)
    assert runner_factory.call_args.kwargs["max_policy_calls"] == limit
    assert runner_factory.call_args.kwargs["context_window"] == 262144
    assert artifact.observations.gaps[0].code == "test"
    assert artifact.model_cookies == {"model": "new"}
    assert artifact.resource_cookies == {"resource": "new"}
    assert artifact.response is not None
    assert current_rollout_id() is None
    if outcome == "success":
        assert artifact.error is None
        assert artifact.response.output[-1].content[0].text == "answer"
    else:
        assert artifact.error.kind == outcome
        assert artifact.termination_reason == ("cancelled" if outcome == "cancelled" else "infrastructure_error")
    assert entrypoint.SandboxResult.model_validate_json(artifact.model_dump_json()).response == artifact.response


def test_launch_policy_call_limit_is_optional_and_survives_json() -> None:
    data = payload().model_dump(mode="json")
    data.pop("max_policy_calls")
    p = entrypoint.SandboxInput.model_validate(data)
    assert p.max_policy_calls is None
    assert entrypoint.SandboxInput.model_validate_json(p.model_dump_json()).max_policy_calls is None
    for invalid in (0, -1):
        with pytest.raises(ValidationError, match="max_policy_calls"):
            entrypoint.SandboxInput.model_validate(data | {"max_policy_calls": invalid})


def test_sandbox_context_window_is_optional_positive_and_survives_json_boundary() -> None:
    p = payload()
    assert entrypoint.SandboxInput.model_validate_json(p.model_dump_json()).context_window == 262144
    data = p.model_dump(mode="json")
    data.pop("context_window")
    assert entrypoint.SandboxInput.model_validate(data).context_window is None
    for invalid in (0, -1):
        with pytest.raises(ValidationError, match="context_window"):
            entrypoint.SandboxInput.model_validate(data | {"context_window": invalid})


@pytest.mark.asyncio
async def test_child_setup_failure_is_not_empty_success(monkeypatch) -> None:
    monkeypatch.setattr(entrypoint, "InProcessNOOARunner", MagicMock(side_effect=ValueError("invalid config")))
    artifact = await entrypoint.execute(payload())
    assert artifact.error.kind == "fatal"
    assert artifact.response is None
    with pytest.raises(RuntimeError, match="no response"):
        artifact.run_result()


@pytest.mark.asyncio
@pytest.mark.parametrize("fails", [False, True])
async def test_main_initializes_shared_transport_and_always_closes_it(monkeypatch, tmp_path, fails: bool) -> None:
    events = []

    async def shutdown():
        events.append("close")

    async def execute(p):
        assert events == ["open"]
        events.append("execute")
        if fails:
            raise RuntimeError("unexpected bootstrap failure")
        return entrypoint.SandboxResult(
            observations=AgentObservationBundle(source="nooa"), model_cookies={}, resource_cookies={}
        )

    client = MagicMock(close=shutdown)

    def initialize(config):
        events.append("open")
        return client

    monkeypatch.setattr(entrypoint, "set_global_aiohttp_client", initialize)
    monkeypatch.setattr(entrypoint, "execute", execute)
    monkeypatch.setattr(asyncio.get_running_loop(), "add_signal_handler", MagicMock())
    source, destination = tmp_path / "input.json", tmp_path / "result.json"
    paths = {"stop_path": tmp_path / "stop", "completion_path": tmp_path / "completion.json"}
    source.write_text(payload().model_dump_json())
    if fails:
        with pytest.raises(RuntimeError, match="bootstrap failure"):
            await entrypoint._main(source, destination, **paths)
        assert not destination.exists()
        assert not paths["completion_path"].exists()
    else:
        await entrypoint._main(source, destination, **paths)
        assert entrypoint.SandboxResult.model_validate_json(destination.read_text()).observations.source == "nooa"
        assert not destination.with_suffix(".tmp").exists()
        assert json.loads(paths["completion_path"].read_text()) == {"task_completed": True}
    assert events == ["open", "execute", "close"]


async def wait_for_file(path: Path) -> None:
    async with asyncio.timeout(2):
        while not path.exists():
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_method", ["file", "signal"])
async def test_completed_worker_holds_services_until_stop(monkeypatch, tmp_path: Path, stop_method: str) -> None:
    source, destination = tmp_path / "input.json", tmp_path / "result.json"
    stop, completion = tmp_path / "stop", tmp_path / "completion.json"
    source.write_text(payload().model_dump_json())
    result = entrypoint.SandboxResult(
        response=run_result().episode.response,
        observations=AgentObservationBundle(source="nooa"),
        model_cookies={},
        resource_cookies={},
    )
    client = MagicMock(close=AsyncMock())
    monkeypatch.setattr(entrypoint, "set_global_aiohttp_client", MagicMock(return_value=client))
    monkeypatch.setattr(entrypoint, "execute", AsyncMock(return_value=result))
    register_signal = MagicMock()
    monkeypatch.setattr(asyncio.get_running_loop(), "add_signal_handler", register_signal)
    worker = asyncio.create_task(entrypoint._main(source, destination, stop_path=stop, completion_path=completion))
    try:
        await wait_for_file(completion)
        assert json.loads(completion.read_text()) == {"task_completed": True}
        assert entrypoint.SandboxResult.model_validate_json(destination.read_text()) == result
        client.close.assert_awaited_once()
        assert not worker.done()
        if stop_method == "file":
            stop.touch()
        else:
            assert register_signal.call_args.args[0] == signal.SIGTERM
            register_signal.call_args.args[1]()
        await asyncio.wait_for(worker, timeout=2)
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["stop-file", "signal", "fatal"])
async def test_failure_or_interrupted_execution_never_waits_for_verification(
    monkeypatch, tmp_path: Path, outcome: str
) -> None:
    source, destination = tmp_path / "input.json", tmp_path / "result.json"
    stop, completion = tmp_path / "stop", tmp_path / "completion.json"
    source.write_text(payload().model_dump_json())
    started = asyncio.Event()

    async def run(request):
        started.set()
        result = run_result()
        if outcome == "fatal":
            raise NOOARunFailure(ValueError("failed invocation"), result)
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError as error:
            error.nooa_result = result
            raise

    monkeypatch.setattr(entrypoint, "InProcessNOOARunner", MagicMock(return_value=MagicMock(run=run)))
    client = MagicMock(close=AsyncMock())
    monkeypatch.setattr(entrypoint, "set_global_aiohttp_client", MagicMock(return_value=client))
    register_signal = MagicMock()
    monkeypatch.setattr(asyncio.get_running_loop(), "add_signal_handler", register_signal)
    worker = asyncio.create_task(entrypoint._main(source, destination, stop_path=stop, completion_path=completion))
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        if outcome == "stop-file":
            stop.touch()
        elif outcome == "signal":
            register_signal.call_args.args[1]()
        # Do not cancel on timeout: cancellation itself would checkpoint the
        # worker and could hide a broken durable-stop watcher.
        done, _ = await asyncio.wait({worker}, timeout=2)
        assert worker in done
        await worker
        result = entrypoint.SandboxResult.model_validate_json(destination.read_text())
        assert result.error.kind == ("fatal" if outcome == "fatal" else "cancelled")
        assert result.observations.gaps[0].code == "test"
        assert result.response is not None
        assert json.loads(completion.read_text()) == {"task_completed": True}
        client.close.assert_awaited_once()
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)


@pytest.mark.asyncio
async def test_stop_before_worker_start_fences_execution(monkeypatch, tmp_path: Path) -> None:
    stop = tmp_path / "stop"
    stop.touch()
    execute = AsyncMock()
    monkeypatch.setattr(entrypoint, "execute", execute)
    await entrypoint._main(
        tmp_path / "missing-input.json",
        tmp_path / "result.json",
        stop_path=stop,
        completion_path=tmp_path / "completion.json",
    )
    execute.assert_not_awaited()
    assert not (tmp_path / "completion.json").exists()
