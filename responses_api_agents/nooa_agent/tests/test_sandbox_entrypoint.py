# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import signal
import stat
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_correlation import current_rollout_id
from nemo_gym.rollout_observability import AgentEpisode, AgentObservationBundle, ObservationGap
from responses_api_agents.nooa_agent import sandbox_entrypoint as entrypoint
from responses_api_agents.nooa_agent.config import NOOAInvocationConfig
from responses_api_agents.nooa_agent.runner import NOOARunFailure, NOOARunRequest, NOOARunResult
from responses_api_agents.nooa_agent.termination import NOOATerminationReason
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
@pytest.mark.parametrize(
    "outcome",
    [
        "success",
        NOOATerminationReason.POLICY_BUDGET_EXCEEDED,
        NOOATerminationReason.MAX_ITERATIONS_EXCEEDED,
        NOOATerminationReason.MAX_OUTPUT_TOKENS_EXCEEDED,
        "transient",
        "fatal",
        "cancelled",
    ],
)
@pytest.mark.parametrize("limit", [None, 3])
async def test_child_preserves_evidence_and_classifies_outcome(monkeypatch, outcome: str, limit: int | None) -> None:
    p = payload()
    p.max_policy_calls = limit
    result = run_result()
    result.termination_reason = outcome if isinstance(outcome, NOOATerminationReason) else None
    gradable = outcome == "success" or result.termination_reason is not None

    async def run(request):
        assert current_rollout_id() == "task-a1"
        request.model_cookies.update(result.model_cookies)
        request.resource_cookies.update(result.resource_cookies)
        if outcome == "cancelled":
            error = asyncio.CancelledError()
            error.nooa_result = result
            raise error
        if not gradable:
            cause = ConnectionError("offline") if outcome == "transient" else ValueError("broken adapter")
            raise NOOARunFailure(cause, result) from cause
        return result

    runner_factory = MagicMock(return_value=MagicMock(run=run))
    monkeypatch.setattr(entrypoint, "InProcessNOOARunner", runner_factory)
    artifact = await entrypoint.execute(p)
    assert runner_factory.call_args.kwargs["max_policy_calls"] == limit
    assert artifact.observations.gaps[0].code == "test"
    assert artifact.model_cookies == {"model": "new"}
    assert artifact.resource_cookies == {"resource": "new"}
    assert artifact.response is not None
    assert current_rollout_id() is None
    if gradable:
        assert artifact.error is None
        assert artifact.response.output[-1].content[0].text == "answer"
        assert artifact.termination_reason == result.termination_reason
        if outcome != "success":
            assert artifact.response.status == "incomplete"
            assert artifact.response.error is None
            assert outcome in {gap.code for gap in artifact.observations.gaps}
    else:
        assert artifact.error.kind == outcome
        assert artifact.termination_reason is (
            NOOATerminationReason.CANCELLED if outcome == "cancelled" else NOOATerminationReason.INFRASTRUCTURE_ERROR
        )
    restored = entrypoint.SandboxResult.model_validate_json(artifact.model_dump_json())
    assert restored == artifact
    assert restored.run_result().termination_reason == artifact.termination_reason


@pytest.mark.parametrize("reason", [None, *NOOATerminationReason])
def test_termination_reason_round_trips_as_a_string_and_restores_the_enum(
    reason: NOOATerminationReason | None,
) -> None:
    artifact = entrypoint.SandboxResult(
        response=run_result().episode.response,
        observations=AgentObservationBundle(source="nooa"),
        model_cookies={},
        resource_cookies={},
        termination_reason=reason,
    )
    wire = artifact.model_dump(mode="json")
    assert wire["termination_reason"] == (None if reason is None else reason.value)
    if reason is not None:
        assert type(wire["termination_reason"]) is str
    restored = entrypoint.SandboxResult.model_validate_json(artifact.model_dump_json())
    assert restored.termination_reason is reason
    assert restored.run_result().termination_reason is reason


@pytest.mark.parametrize("reason", ["timeout", "agent_run_timeout", "unknown", "transient", "fatal", "legitimate"])
def test_worker_result_rejects_unsupported_termination_reasons(reason: str) -> None:
    with pytest.raises(ValidationError, match="termination_reason"):
        entrypoint.SandboxResult.model_validate(
            {
                "observations": {"source": "nooa"},
                "model_cookies": {},
                "resource_cookies": {},
                "termination_reason": reason,
            }
        )


def test_launch_policy_call_limit_is_optional_and_survives_json() -> None:
    data = payload().model_dump(mode="json")
    data.pop("max_policy_calls")
    p = entrypoint.SandboxInput.model_validate(data)
    assert p.max_policy_calls is None
    assert entrypoint.SandboxInput.model_validate_json(p.model_dump_json()).max_policy_calls is None
    for invalid in (0, -1):
        with pytest.raises(ValidationError, match="max_policy_calls"):
            entrypoint.SandboxInput.model_validate(data | {"max_policy_calls": invalid})


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

    async def execute(p, *, on_progress):
        assert events == ["open"]
        events.append("execute")
        on_progress(run_result())
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
    source.write_text(payload().model_dump_json())
    if fails:
        with pytest.raises(RuntimeError, match="bootstrap failure"):
            await entrypoint._main(source, destination)
        assert not destination.exists()
    else:
        await entrypoint._main(source, destination)
        assert entrypoint.SandboxResult.model_validate_json(destination.read_text()).observations.source == "nooa"
        assert sorted(path.name for path in tmp_path.iterdir()) == ["checkpoint.json", "input.json", "result.json"]
    assert events == ["open", "execute", "close"]
    checkpoint = entrypoint.SandboxCheckpoint.model_validate_json((tmp_path / "checkpoint.json").read_text())
    assert checkpoint.partial is True
    assert checkpoint.response.status == "incomplete"


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform == "win32", reason="The sandbox worker requires POSIX signal handlers")
async def test_sigterm_writes_cancelled_result_from_real_main(tmp_path) -> None:
    source, destination = tmp_path / "input.json", tmp_path / "result.json"
    source.write_text(payload().model_dump_json())
    script = textwrap.dedent(
        """
        import asyncio
        import sys
        from pathlib import Path
        from responses_api_agents.nooa_agent import sandbox_entrypoint as entrypoint

        class WaitingRunner:
            def __init__(self, **kwargs):
                pass

            async def run(self, request):
                print("ready", flush=True)
                await asyncio.Event().wait()

        entrypoint.InProcessNOOARunner = WaitingRunner
        asyncio.run(entrypoint._main(Path(sys.argv[1]), Path(sys.argv[2])))
        """
    )
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        script,
        str(source),
        str(destination),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        assert await asyncio.wait_for(process.stdout.readline(), timeout=30) == b"ready\n"
        process.send_signal(signal.SIGTERM)
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10)
        assert process.returncode == 0, (stdout.decode(errors="replace"), stderr.decode(errors="replace"))
        result = entrypoint.SandboxResult.model_validate_json(destination.read_text())
        assert result.error.kind == "cancelled"
        assert result.response is None
        assert sorted(path.name for path in tmp_path.iterdir()) == ["input.json", "result.json"]
    finally:
        if process.returncode is None:
            process.kill()
            await process.communicate()


def test_worker_artifacts_are_private_and_readable_by_the_worker_uid(tmp_path) -> None:
    path = tmp_path / "result.json"
    result = entrypoint.SandboxResult(
        observations=AgentObservationBundle(source="nooa"), model_cookies={}, resource_cookies={}
    )
    entrypoint._write_json(path, result)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert entrypoint.SandboxResult.model_validate_json(path.read_text()) == result


def test_checkpoint_is_explicit_partial_evidence_and_excludes_python_return_values(tmp_path) -> None:
    result = run_result()
    result.return_value = ModuleType("runtime_only")
    path = tmp_path / "checkpoint.json"
    entrypoint.write_checkpoint(path, result)
    checkpoint = entrypoint.SandboxCheckpoint.model_validate_json(path.read_text())
    assert checkpoint.partial is True
    assert checkpoint.response.status == "incomplete"
    assert checkpoint.response.output == result.episode.response.output
    assert checkpoint.termination_reason is None
    assert checkpoint.error is None
    assert "partial_checkpoint" in {gap.code for gap in checkpoint.observations.gaps}
    assert "runtime_only" not in path.read_text()
    with pytest.raises(ValidationError, match="partial"):
        entrypoint.SandboxResult.model_validate_json(path.read_text())
    with pytest.raises(ValidationError, match="partial"):
        entrypoint.SandboxCheckpoint.model_validate(checkpoint.model_dump(exclude={"partial"}))


def test_interrupted_checkpoint_replacement_preserves_last_valid_evidence(monkeypatch, tmp_path) -> None:
    path = tmp_path / "checkpoint.json"
    result = run_result()
    entrypoint.write_checkpoint(path, result)
    previous = path.read_bytes()

    def interrupt(source, destination):
        assert source.parent == destination.parent
        assert entrypoint.SandboxCheckpoint.model_validate_json(source.read_text()).partial
        raise InterruptedError("worker interrupted before publication")

    monkeypatch.setattr(Path, "replace", interrupt)
    result.model_cookies["model"] = "newer"
    with pytest.raises(InterruptedError, match="before publication"):
        entrypoint.write_checkpoint(path, result)
    assert path.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [path]


def test_concurrent_atomic_writes_never_publish_partial_json(tmp_path) -> None:
    path = tmp_path / "checkpoint.json"
    entrypoint.write_checkpoint(path, run_result())

    def publish(index: int) -> None:
        result = run_result()
        result.model_cookies["sequence"] = str(index)
        entrypoint.write_checkpoint(path, result)
        assert entrypoint.SandboxCheckpoint.model_validate_json(path.read_text()).partial

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(publish, range(32)))
    assert list(tmp_path.iterdir()) == [path]
