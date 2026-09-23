# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The shared checkpoint control plane every Gym server exposes.

Capabilities let the NemoGym actor fail setup on a missing requirement
instead of failing the first checkpoint at its deadline. The control fence
makes every control route idempotent by (checkpoint_id, operation), rejects
stale coordinators, and keeps phase transitions crash-consistent: a failed
operation restores the entry phase so a retry or abort is still possible.
"""

import asyncio

import pytest
from fastapi import FastAPI
from omegaconf import OmegaConf
from pydantic import ValidationError
from starlette.testclient import TestClient

from nemo_gym._checkpoint import (
    CHECKPOINT_CONTROL_TOKEN_ENV,
    CONTROL_SCHEMA_VERSION,
    CONTROL_URL_PREFIX,
    CheckpointConflictError,
    CheckpointControlRequest,
    CheckpointPhase,
    ControlFence,
    Deadline,
    InvalidPhaseError,
    StaleCheckpointError,
    checkpoint_control_auth_token,
    multi_process_capability_from_num_workers,
)
from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import BaseServerConfig
from nemo_gym.server_utils import ServerClient


# --- deadline ---


def test_deadline_remaining_clamps_at_zero() -> None:
    deadline = Deadline(deadline_ts=1000.0)
    assert deadline.remaining(now=900.0) == 100.0
    assert deadline.remaining(now=1000.0) == 0.0
    # Expired means zero budget for draining, not an error.
    assert deadline.remaining(now=2000.0) == 0.0
    assert deadline.expired(now=2000.0)
    assert not deadline.expired(now=900.0)


def test_checkpoint_auth_is_independent_of_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CHECKPOINT_CONTROL_TOKEN_ENV, "checkpoint-secret")
    assert checkpoint_control_auth_token({"token_id_capture": {"enabled": False}}) == "checkpoint-secret"


def test_control_request_rejects_invalid_identity_and_deadline() -> None:
    with pytest.raises(ValidationError):
        CheckpointControlRequest(checkpoint_id="../bad", deadline_ts=1000.0)
    with pytest.raises(ValidationError):
        CheckpointControlRequest(checkpoint_id="ckpt-1", deadline_ts=float("inf"))


# --- multi-process declaration ---


def test_multi_process_capability_from_num_workers() -> None:
    assert multi_process_capability_from_num_workers(None).mode == "single_worker"
    assert multi_process_capability_from_num_workers(1).mode == "single_worker"
    unmanaged = multi_process_capability_from_num_workers(4)
    assert unmanaged.mode == "unmanaged"
    assert unmanaged.num_workers == 4


# --- control fence ---


def _prepare_kwargs(**overrides):
    kwargs = dict(
        allowed_phases=frozenset({CheckpointPhase.IDLE}),
        phase_during=CheckpointPhase.PREPARING,
        phase_after=CheckpointPhase.PREPARED,
    )
    kwargs.update(overrides)
    return kwargs


@pytest.mark.asyncio
async def test_fence_records_and_replays_results() -> None:
    fence = ControlFence()
    runs = 0

    async def run() -> dict:
        nonlocal runs
        runs += 1
        return {"state": "prepared"}

    first = await fence.run_operation("ckpt-1", "pause", run=run, **_prepare_kwargs())
    replay = await fence.run_operation("ckpt-1", "pause", run=run, **_prepare_kwargs())
    assert first == replay == {"state": "prepared"}
    assert runs == 1
    assert fence.phase == CheckpointPhase.PREPARED
    assert fence.active_checkpoint_id == "ckpt-1"


@pytest.mark.asyncio
async def test_fence_coalesces_concurrent_duplicates() -> None:
    fence = ControlFence()
    runs = 0
    release = asyncio.Event()

    async def run() -> dict:
        nonlocal runs
        runs += 1
        await release.wait()
        return {"state": "prepared"}

    first = asyncio.create_task(fence.run_operation("ckpt-1", "pause", run=run, **_prepare_kwargs()))
    await asyncio.sleep(0)
    second = asyncio.create_task(fence.run_operation("ckpt-1", "pause", run=run, **_prepare_kwargs()))
    await asyncio.sleep(0)
    release.set()
    assert await first == await second == {"state": "prepared"}
    assert runs == 1


@pytest.mark.asyncio
async def test_fence_serializes_resume_before_delayed_commit() -> None:
    fence = ControlFence()

    async def prepare() -> dict:
        return {"state": "prepared"}

    await fence.run_operation("ckpt-1", "pause", run=prepare, **_prepare_kwargs())
    resume_started = asyncio.Event()
    release_resume = asyncio.Event()
    commit_started = asyncio.Event()

    async def resume() -> dict:
        resume_started.set()
        await release_resume.wait()
        return {"state": "resumed"}

    async def commit() -> dict:
        commit_started.set()
        return {"state": "committed"}

    resume_task = asyncio.create_task(
        fence.run_operation(
            "ckpt-1",
            "resume",
            allowed_phases=frozenset({CheckpointPhase.PREPARED}),
            phase_during=None,
            phase_after=CheckpointPhase.IDLE,
            run=resume,
            retire_outcome="resumed",
        )
    )
    await resume_started.wait()
    commit_task = asyncio.create_task(
        fence.run_operation(
            "ckpt-1",
            "commit",
            allowed_phases=frozenset({CheckpointPhase.PREPARED}),
            phase_during=CheckpointPhase.COMMITTING,
            phase_after=CheckpointPhase.COMMITTED_PAUSED,
            run=commit,
        )
    )
    await asyncio.sleep(0)
    assert not commit_started.is_set()

    release_resume.set()
    assert await resume_task == {"state": "resumed"}
    with pytest.raises(StaleCheckpointError):
        await commit_task
    assert not commit_started.is_set()


@pytest.mark.asyncio
async def test_fence_serializes_commit_before_resume() -> None:
    fence = ControlFence()

    async def prepare() -> dict:
        return {"state": "prepared"}

    await fence.run_operation("ckpt-1", "pause", run=prepare, **_prepare_kwargs())
    commit_started = asyncio.Event()
    release_commit = asyncio.Event()
    resume_started = asyncio.Event()
    operation_order: list[str] = []

    async def commit() -> dict:
        operation_order.append("commit")
        commit_started.set()
        await release_commit.wait()
        return {"state": "committed"}

    async def resume() -> dict:
        operation_order.append("resume")
        resume_started.set()
        assert fence.phase == CheckpointPhase.COMMITTED_PAUSED
        return {"state": "resumed"}

    commit_task = asyncio.create_task(
        fence.run_operation(
            "ckpt-1",
            "commit",
            allowed_phases=frozenset({CheckpointPhase.PREPARED}),
            phase_during=CheckpointPhase.COMMITTING,
            phase_after=CheckpointPhase.COMMITTED_PAUSED,
            run=commit,
        )
    )
    await commit_started.wait()
    resume_task = asyncio.create_task(
        fence.run_operation(
            "ckpt-1",
            "resume",
            allowed_phases=frozenset({CheckpointPhase.PREPARED, CheckpointPhase.COMMITTED_PAUSED}),
            phase_during=None,
            phase_after=CheckpointPhase.IDLE,
            run=resume,
            retire_outcome="resumed",
        )
    )
    await asyncio.sleep(0)
    assert not resume_started.is_set()

    release_commit.set()
    assert await commit_task == {"state": "committed"}
    assert await resume_task == {"state": "resumed"}
    assert operation_order == ["commit", "resume"]
    assert fence.phase == CheckpointPhase.IDLE


@pytest.mark.asyncio
async def test_fence_phase_preserving_operation_keeps_live_phase_update() -> None:
    fence = ControlFence()

    async def prepare() -> dict:
        return {"state": "preparing"}

    await fence.run_operation(
        "ckpt-1",
        "pause",
        run=prepare,
        **_prepare_kwargs(phase_after=CheckpointPhase.PREPARING),
    )
    operation_started = asyncio.Event()
    release_operation = asyncio.Event()

    async def preserve_phase() -> dict:
        operation_started.set()
        await release_operation.wait()
        return {}

    operation = asyncio.create_task(
        fence.run_operation(
            "ckpt-1",
            "abort-inflight",
            allowed_phases=frozenset({CheckpointPhase.PREPARING, CheckpointPhase.PREPARED}),
            phase_during=None,
            phase_after=None,
            run=preserve_phase,
        )
    )
    await operation_started.wait()
    fence.mark_prepared("ckpt-1")
    release_operation.set()
    await operation
    assert fence.phase == CheckpointPhase.PREPARED


@pytest.mark.asyncio
async def test_fence_rejects_conflicting_checkpoint() -> None:
    fence = ControlFence()

    async def run() -> dict:
        return {}

    await fence.run_operation("ckpt-1", "pause", run=run, **_prepare_kwargs())
    with pytest.raises(CheckpointConflictError):
        await fence.run_operation("ckpt-2", "pause", run=run, **_prepare_kwargs())


@pytest.mark.asyncio
async def test_fence_rejects_conflicting_checkpoint_while_operation_runs() -> None:
    fence = ControlFence()
    started = asyncio.Event()
    release = asyncio.Event()

    async def run() -> dict:
        started.set()
        await release.wait()
        return {}

    active = asyncio.create_task(fence.run_operation("ckpt-1", "pause", run=run, **_prepare_kwargs()))
    await started.wait()
    with pytest.raises(CheckpointConflictError):
        await fence.run_operation("ckpt-2", "pause", run=run, **_prepare_kwargs())
    release.set()
    await active


@pytest.mark.asyncio
async def test_fence_rejects_invalid_phase() -> None:
    fence = ControlFence()

    async def run() -> dict:
        return {}

    # Commit before prepare: the fence is still idle, commit requires PREPARED.
    with pytest.raises(InvalidPhaseError):
        await fence.run_operation(
            "ckpt-1",
            "commit",
            allowed_phases=frozenset({CheckpointPhase.PREPARED}),
            phase_during=CheckpointPhase.COMMITTING,
            phase_after=CheckpointPhase.COMMITTED_PAUSED,
            run=run,
        )


@pytest.mark.asyncio
async def test_fence_retires_checkpoint_and_rejects_stale_coordinator() -> None:
    fence = ControlFence()

    async def run() -> dict:
        return {"state": "done"}

    await fence.run_operation("ckpt-1", "pause", run=run, **_prepare_kwargs())
    await fence.run_operation(
        "ckpt-1",
        "resume",
        allowed_phases=frozenset({CheckpointPhase.PREPARED}),
        phase_during=CheckpointPhase.PREPARED,
        phase_after=CheckpointPhase.IDLE,
        run=run,
        retire_outcome="resumed",
    )
    assert fence.phase == CheckpointPhase.IDLE
    assert fence.active_checkpoint_id is None

    # The retired id is stale forever, but its recorded results still replay
    # so a coordinator retrying its final call gets the same answer.
    replay = await fence.run_operation("ckpt-1", "resume", run=run, **_prepare_kwargs())
    assert replay == {"state": "done"}
    with pytest.raises(StaleCheckpointError):
        await fence.run_operation("ckpt-1", "pause", run=run, **_prepare_kwargs())
    with pytest.raises(StaleCheckpointError):
        await fence.run_operation("ckpt-1", "commit", run=run, **_prepare_kwargs())

    # A new checkpoint can start after retirement.
    assert await fence.run_operation("ckpt-2", "pause", run=run, **_prepare_kwargs()) == {"state": "done"}


@pytest.mark.asyncio
async def test_fence_failure_restores_entry_phase_and_allows_retry() -> None:
    fence = ControlFence()
    attempts = 0

    async def failing() -> dict:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("drain blew up")

    with pytest.raises(RuntimeError):
        await fence.run_operation("ckpt-1", "pause", run=failing, **_prepare_kwargs())
    assert fence.phase == CheckpointPhase.IDLE
    assert fence.active_checkpoint_id is None

    async def run() -> dict:
        return {"state": "prepared"}

    # The failure was not recorded: a retry runs the operation again.
    assert await fence.run_operation("ckpt-1", "pause", run=run, **_prepare_kwargs()) == {"state": "prepared"}
    assert attempts == 1


@pytest.mark.asyncio
async def test_fence_sets_deadline_during_operation() -> None:
    fence = ControlFence()

    async def run() -> dict:
        assert fence.deadline is not None
        assert fence.deadline.deadline_ts == 12345.0
        return {}

    await fence.run_operation("ckpt-1", "pause", run=run, deadline=Deadline(deadline_ts=12345.0), **_prepare_kwargs())
    assert fence.snapshot()["deadline_ts"] == 12345.0


# --- capabilities route on the server bases ---


def _server_client() -> ServerClient:
    return ServerClient(
        head_server_config=BaseServerConfig(host="head.test", port=80),
        global_config_dict=OmegaConf.create({}),
    )


class _StatelessResourcesConfig(BaseResourcesServerConfig):
    CHECKPOINT_RECOVERY_MODE = "stateless"


class _StatelessResources(SimpleResourcesServer):
    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        return BaseVerifyResponse(**body.model_dump(), reward=0.0)


class _RestartOnlyResources(SimpleResourcesServer):
    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        return BaseVerifyResponse(**body.model_dump(), reward=0.0)


def _resources_config(
    num_workers=None,
    config_cls=BaseResourcesServerConfig,
) -> BaseResourcesServerConfig:
    return config_cls(
        host="resources.test",
        port=80,
        entrypoint="app.py",
        name="resources",
        num_workers=num_workers,
    )


def test_stateless_resources_server_capabilities() -> None:
    server = _StatelessResources(
        config=_resources_config(config_cls=_StatelessResourcesConfig),
        server_client=_server_client(),
    )
    client = TestClient(server.setup_webserver())
    body = client.get(f"{CONTROL_URL_PREFIX}/capabilities").json()
    assert body["component"] == "resources_servers"
    assert body["name"] == "resources"
    assert body["schema_version"] == CONTROL_SCHEMA_VERSION
    assert body["checkpoint_mode"] == "stateless"
    assert body["concurrency_contract"] == "stateless"
    assert body["admission_states"] == ["accepting"]
    assert body["multi_process"] == {"mode": "single_worker", "num_workers": 1}
    assert body["phase"] == "idle"
    assert body["active_checkpoint_id"] is None


def test_resources_server_defaults_to_restart_only() -> None:
    server = _RestartOnlyResources(config=_resources_config(), server_client=_server_client())
    body = TestClient(server.setup_webserver()).get(f"{CONTROL_URL_PREFIX}/capabilities").json()
    assert body["checkpoint_mode"] == "restart_only"
    assert body["concurrency_contract"] == "stateless"


def test_agent_server_capabilities(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CHECKPOINT_CONTROL_TOKEN_ENV, "checkpoint-secret")

    class _Agent(SimpleResponsesAPIAgent):
        async def responses(self, body):
            raise NotImplementedError

        async def run(self, body):
            raise NotImplementedError

    agent = _Agent(
        config=BaseResponsesAPIAgentConfig(host="agent.test", port=80, entrypoint="app.py", name="agent"),
        server_client=_server_client(),
    )
    body = TestClient(agent.setup_webserver()).get(f"{CONTROL_URL_PREFIX}/capabilities").json()
    assert body["component"] == "responses_api_agents"
    assert body["name"] == "agent"
    assert body["checkpoint_mode"] == "export_restore"
    assert body["concurrency_contract"] == "serialized_per_session"
    assert body["features"] == ["completed_result_acknowledgement"]
    assert agent._checkpoint_participant is not None

    class _WhiteboxAgent(_Agent):
        checkpoint_continuation_supported = True
        checkpoint_resource_dependencies_supported = True

    whitebox = _WhiteboxAgent(
        config=BaseResponsesAPIAgentConfig(host="agent.test", port=80, entrypoint="app.py", name="whitebox"),
        server_client=_server_client(),
    )
    whitebox_client = TestClient(whitebox.setup_webserver())
    whitebox_body = whitebox_client.get(f"{CONTROL_URL_PREFIX}/capabilities").json()
    assert whitebox_body["checkpoint_mode"] == "export_restore"
    assert whitebox_body["concurrency_contract"] == "serialized_per_session"
    assert whitebox_body["features"] == [
        "completed_result_acknowledgement",
        "agent_continuation_index_v1",
        "discard_restored_continuation_v1",
        "agent_resource_dependency_index_v1",
    ]


def test_capabilities_route_reflects_live_fence_phase() -> None:
    server = _StatelessResources(config=_resources_config(), server_client=_server_client())
    client = TestClient(server.setup_webserver())
    server.checkpoint_fence().phase = CheckpointPhase.PREPARED
    server.checkpoint_fence().active_checkpoint_id = "ckpt-9"
    body = client.get(f"{CONTROL_URL_PREFIX}/capabilities").json()
    assert body["phase"] == "prepared"
    assert body["active_checkpoint_id"] == "ckpt-9"


def test_control_plane_installable_on_custom_app() -> None:
    # Servers that build their own FastAPI app (e.g. GymnasiumServer) call
    # setup_control_plane themselves, mirroring setup_session_state_routes.
    server = _StatelessResources(config=_resources_config(), server_client=_server_client())
    app = FastAPI()
    server.setup_control_plane(app)
    body = TestClient(app).get(f"{CONTROL_URL_PREFIX}/capabilities").json()
    assert body["component"] == "resources_servers"
