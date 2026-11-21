# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native lifecycle delegation and the added dataset/routing boundary, without services."""

import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from environment_servers.harbor.app import HarborEnvironmentServer
from environment_servers.harbor.tests.conftest import FakeSandbox, episode, make_client, make_server
from environment_servers.harbor.verifier import (
    EXIT_RECORD_NAME,
    MISSING_REWARD_KIND,
    VERIFIER_UNAVAILABLE_KIND,
    parse_reward_file,
)
from environment_servers.unified_video_harbor.app import (
    UnifiedVideoEpisodeRequest,
    UnifiedVideoHarborConfig,
    UnifiedVideoHarborServer,
    _VideoTaskHarbor,
)
from nemo_gym.core import NgFailureEvent
from nemo_gym.data.tasks.harbor import HarborMCPServer
from nemo_gym.sandbox import SandboxExecResult


def unified(tmp_path, monkeypatch):
    base, task, sandbox, created = make_server(tmp_path, monkeypatch)
    config = UnifiedVideoHarborConfig.model_validate(
        base.config.model_dump()
        | {
            "harness_agents": {
                name: {"type": "responses_api_agents", "name": "agent"} for name in ("opencode", "codex", "hermes")
            },
            "task_mcp_grants": "required",
        }
    )
    return UnifiedVideoHarborServer(config=config, server_client=make_client()), task


def test_native_lifecycle_methods_are_inherited():
    assert UnifiedVideoHarborServer.run_request is HarborEnvironmentServer.run_request
    for name in ("_prepare_workdir", "_task_tool_accesses", "_seed"):
        assert getattr(_VideoTaskHarbor, name) is getattr(HarborEnvironmentServer, name)


@pytest.mark.parametrize("harness", ["opencode", "codex", "hermes"])
def test_checkpoint_route_and_native_required_http_grant(tmp_path, monkeypatch, harness):
    server, task = unified(tmp_path, monkeypatch)
    original = episode(task).model_dump(mode="json") | {"harness": harness}
    request = UnifiedVideoEpisodeRequest.model_validate(original)
    restored = UnifiedVideoEpisodeRequest.model_validate(
        {"episode_id": original["episode_id"], "task": request.task.model_dump(mode="json")}
    )
    assert restored.harness == harness
    assert server.agent_server_for_episode(restored).name == "agent"
    route = server._route(restored)
    assert route._active is server._active
    assert route.config.max_concurrent_episodes == server.config.max_concurrent_episodes
    grants = route._task_tool_accesses(
        task, [HarborMCPServer(name="claw", transport="streamable-http", url="http://127.0.0.1:17654/mcp")]
    )
    assert len(grants) == 1 and grants[0].required
    assert str(grants[0].connection.url) == "http://127.0.0.1:17654/mcp"
    bad = restored.model_dump(mode="json")
    bad["harness"] = "other"
    with pytest.raises(ValueError, match="checkpoint harness disagree"):
        UnifiedVideoEpisodeRequest.model_validate(bad)


@pytest.mark.asyncio
@pytest.mark.parametrize("separate", [False, True])
@pytest.mark.parametrize("reward", [0.0, 0.2, 1.0])
async def test_native_measured_grades_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reward: float, separate: bool
) -> None:
    server, task = unified(tmp_path, monkeypatch)
    task.task_data["video_source"] = "claweval_agent"
    (tmp_path / "reward.json").write_text(json.dumps({"reward": reward, "details": {"reason": "graded output"}}))
    rewards, problem = parse_reward_file(tmp_path, precedence="json", non_numeric="ignore")
    assert rewards == {"reward": reward} and problem is None
    result = server._score(
        task,
        SimpleNamespace(network_mode=None),
        SandboxExecResult(stdout="", stderr="", return_code=0),
        tmp_path,
        asyncio.get_running_loop().time(),
        timed_out=False,
        rewards=rewards,
        problem=problem,
        separate=separate,
    )
    assert result["reward"] == reward and result["verifier_rewards"] == {"reward": reward}


@pytest.mark.asyncio
@pytest.mark.parametrize("separate", [False, True])
async def test_unavailable_claw_grade_is_not_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, separate: bool
) -> None:
    server, task = unified(tmp_path, monkeypatch)
    values = dict(
        settings=SimpleNamespace(network_mode=None),
        result=SandboxExecResult(stdout="", stderr="", return_code=1),
        logs_dir=tmp_path,
        started=asyncio.get_running_loop().time(),
        timed_out=False,
        rewards=None,
        problem="reward.json not written",
        separate=separate,
    )
    task.task_data["video_source"] = "claweval_agent"
    with pytest.raises(NgFailureEvent) as caught:
        server._score(task, **values)
    assert caught.value.stage == "verification"


@pytest.mark.asyncio
@pytest.mark.parametrize("separate", [False, True])
async def test_native_verifier_exit_trust_distinction_is_forwarded(tmp_path, monkeypatch, separate):
    server, task = unified(tmp_path, monkeypatch)
    # Exercise Harbor's unchanged fallback policy: only a separate verifier owns
    # a trusted EX_TEMPFAIL record; an agent can forge that record in shared mode.
    (tmp_path / EXIT_RECORD_NAME).write_text("75 100 101\n")
    values = dict(
        settings=SimpleNamespace(network_mode=None),
        result=SandboxExecResult(stdout="", stderr="", return_code=75),
        logs_dir=tmp_path,
        started=asyncio.get_running_loop().time(),
        timed_out=False,
        rewards=None,
        problem="no reward.json or reward.txt was written",
        separate=separate,
    )
    expected = HarborEnvironmentServer._score(server, task, **values)
    actual = server._score(task, **values)
    assert {key: value for key, value in actual.items() if key != "verifier_seconds"} == {
        key: value for key, value in expected.items() if key != "verifier_seconds"
    }
    assert actual["mask_sample"] is separate
    assert actual["failure_kind"] == (VERIFIER_UNAVAILABLE_KIND if separate else MISSING_REWARD_KIND)
    assert actual["verifier_return_code"] == 75


@pytest.mark.asyncio
@pytest.mark.parametrize("harness", ["opencode", "codex", "hermes"])
async def test_actual_native_session_lifecycle_adds_only_explicit_identity(tmp_path, monkeypatch, harness):
    server, task = unified(tmp_path, monkeypatch)
    request = UnifiedVideoEpisodeRequest.model_validate(episode(task).model_dump(mode="json") | {"harness": harness})
    route = server._route(request)
    sandbox = FakeSandbox()

    async def create(task, workdir, *, episode_key=None):
        sandbox.created_in(workdir)
        return sandbox

    monkeypatch.setattr(route, "_create_sandbox", create)
    response = await server.run_request(request)
    assert response.failure is None, response.failure
    assert response.result.harness == harness
    assert response.result.harness_agent_ref.name == "agent"
    assert response.result.task_source == "agent"
    assert response.result.reward == 1.0
    assert response.result.mask_sample is False
    assert response.result.verifier_mode == "shared" and response.result.verifier_return_code == 0
    assert sandbox.stopped and not route._sandboxes and not route._active
    paths = route.server_client.paths()
    assert "/v1/agent_sessions" in paths
    assert any(path.endswith("/v1/responses") for path in paths)
    assert any(path.endswith("/close") for path in paths)


def test_typed_failure_mask_is_authoritative_without_inventing_a_grade():
    from environment_servers.single_agent_turn_legacy.app import SingleAgentTurnLegacyEnvironmentServer
    from environment_servers.single_agent_turn_legacy.tests.test_app import _agent_response, _environment_server
    from nemo_gym.core import EpisodeId, TaskId
    from nemo_gym.server_runtime import SingleAgentTurnFailure, SingleAgentTurnResponse, SingleAgentTurnResult

    server, _ = _environment_server()
    for terminal in (False, True):
        response = SingleAgentTurnResponse(
            episode_id=EpisodeId(rollout_id="failure"),
            task_id=TaskId(taskset="source", task_id="task"),
            failure=SingleAgentTurnFailure(stage="verification", terminal=terminal, failure_reason="unavailable"),
        )
        result = SingleAgentTurnLegacyEnvironmentServer._legacy_result(server, response)
        assert result["mask_sample"] is True and "reward" not in result
        assert result["_ng_failure_terminal"] is terminal
    for reward in (0.0, 0.2, 1.0):
        response = SingleAgentTurnResponse(
            episode_id=EpisodeId(rollout_id="valid"),
            task_id=TaskId(taskset="source", task_id="task"),
            result=SingleAgentTurnResult(
                reward=reward, mask_sample=False, response=_agent_response(), responses_create_params={"input": "task"}
            ),
        )
        result = SingleAgentTurnLegacyEnvironmentServer._legacy_result(server, response)
        assert result["reward"] == reward and result["mask_sample"] is False


@pytest.mark.parametrize("failure_kind", ["provider_unavailable", None])
def test_flat_failure_journal_preserves_kind_and_legacy_mask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_kind: str | None
) -> None:
    from environment_servers.harbor.app import HarborEpisodeResponse
    from environment_servers.single_agent_turn_legacy.app import SingleAgentTurnLegacyEnvironmentServer
    from nemo_gym.data.tasks.harbor import materialize_task
    from nemo_gym.server_runtime import SingleAgentTurnFailure

    server, task = unified(tmp_path, monkeypatch)
    request = UnifiedVideoEpisodeRequest.model_validate(episode(task).model_dump(mode="json") | {"harness": "hermes"})
    response = HarborEpisodeResponse(
        episode_id=request.episode_id,
        task_id=request.task.task_id,
        failure=SingleAgentTurnFailure(
            stage="verification",
            terminal=False,
            failure_kind=failure_kind,
            failure_reason="provider temporarily unavailable",
        ),
    )
    baseline = SingleAgentTurnLegacyEnvironmentServer._legacy_result(server, response)

    def decoded_row(self: UnifiedVideoHarborServer, row: dict) -> UnifiedVideoEpisodeRequest:
        assert self is server
        return request

    async def failed_request(
        self: UnifiedVideoHarborServer, body: UnifiedVideoEpisodeRequest
    ) -> HarborEpisodeResponse:
        assert self is server
        assert body is request
        return response

    monkeypatch.setattr(UnifiedVideoHarborServer, "_episode_request_from_row", decoded_row)
    monkeypatch.setattr(UnifiedVideoHarborServer, "run_request", failed_request)
    row = {
        "_unified_video_task": materialize_task(task, request.task.task_id.taskset).model_dump(
            mode="json", exclude_unset=True
        )
    }
    result = asyncio.run(server.run_legacy(row))
    assert result["mask_sample"] is baseline["mask_sample"] is True
    assert "reward" not in result and "reward" not in baseline
    assert result["_ng_failure_kind"] == failure_kind
    assert result["_ng_failure_stage"] == "verification"
    recorded = json.loads(
        (server.config.artifacts_dir / "harness_rollouts" / f"{request.episode_id.capture_key}.json").read_text()
    )
    assert recorded["failure_kind"] == (failure_kind or "verification")
    assert recorded["mask_sample"] is True
    assert recorded["reward"] is None
    assert set(recorded) == {
        "capture_key",
        "harness",
        "task_source",
        "logical_rollout_id",
        "task_id",
        "rollout_index",
        "reward",
        "mask_sample",
        "failure_kind",
        "failure_reason",
        "verifier_logs_dir",
        "completed_epoch",
    }


@pytest.mark.parametrize("override", [{"instructions": "changed"}, {"model": "other"}, {"tools": []}])
def test_flat_row_refuses_unpinned_task_parameter_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, override: dict[str, object]
) -> None:
    server, task = unified(tmp_path, monkeypatch)
    # This legacy-boundary fixture must name its actual configured frontend before testing parameter rejection.
    server.config.agent_server = server.config.agent_server.model_copy(update={"name": "claweval_agent"})
    binding = episode(task).task.model_dump(mode="json")
    binding["task_id"]["taskset"] = "claweval_agent"
    metadata = {"prompt_sha256": hashlib.sha256(b"Create hello.txt").hexdigest()}
    binding["task_input"]["task_data"]["verifier_metadata"] = metadata
    row = {
        "task_source": "claweval_agent",
        "verifier_metadata": metadata,
        "_unified_video_task": binding,
        "responses_create_params": {"input": [{"role": "user", "content": "Create hello.txt"}], **override},
    }
    with pytest.raises(ValueError, match="only the prompt and native model sampling"):
        server._episode_request_from_row(row)
    row["responses_create_params"] = {
        "input": [{"role": "user", "content": "Create hello.txt"}],
        "max_output_tokens": 8192,
        "temperature": 1.0,
    }
    request = server._episode_request_from_row(row)
    assert request.model_sampling.max_output_tokens == 8192 and request.model_sampling.temperature == 1.0
    assert request.task.task_input.responses_create_params.max_output_tokens is None
    assert (
        request.task.task_input.responses_create_params.metadata
        == episode(task).task.task_input.responses_create_params.metadata
    )


def test_flat_avb_dispatch_requires_stock_harbor_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    server, _ = unified(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="stock Harbor task rows"):
        server._episode_request_from_row({"task_source": "agentic_vbench_agent"})


@pytest.mark.parametrize("field", ["avb_judge_protocol", "avb_judge_model_server", "avb_revision_tasksets"])
def test_removed_avb_configuration_is_not_silently_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    server, _ = unified(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        UnifiedVideoHarborConfig.model_validate(server.config.model_dump() | {field: None})
