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
"""Cross-layer regression coverage for one partial-rollout checkpoint cut."""

import asyncio
import hashlib
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import orjson
import pytest
from fastapi import FastAPI, Request
from omegaconf import OmegaConf

from nemo_gym._checkpoint import (
    AGENT_CHECKPOINT_URL_PREFIX,
    GATED_MODEL_ROUTE_SUFFIXES,
    MODEL_ADMISSION_URL_PREFIX,
    MODEL_CHECKPOINT_URL_PREFIX,
    RESOURCES_CHECKPOINT_URL_PREFIX,
    AdmissionLimiter,
    AdmissionMiddleware,
    AgentAdmissionClosedError,
    AgentBoundaryRecord,
    AgentCheckpointParticipant,
    ControlCapabilities,
    ControlFence,
    MultiProcessCapability,
    ResourcesCheckpointParticipant,
    ResourceSnapshot,
    StaleAttemptError,
    install_agent_checkpoint,
    install_control_plane,
    install_model_admission,
    install_model_checkpoint,
    install_resources_checkpoint,
)
from nemo_gym.config_types import BaseServerConfig
from nemo_gym.rollout_correlation import (
    ATTEMPT_INDEX_HEADER,
    PARENT_MODEL_CALL_ID_HEADER,
    ROLLOUT_ID_HEADER,
    SOURCE_CAPTURE_KEY_HEADER,
    RolloutContextMiddleware,
    capture_key_for,
    checkpoint_parent_context,
    rollout_context,
)
from nemo_gym.server_utils import ServerClient
from nemo_gym.token_id_capture.lineage import FileLineageStore


AUTH_TOKEN = "checkpoint-token"
AUTH_HEADERS = {"authorization": f"Bearer {AUTH_TOKEN}"}
CHECKPOINT_ID = "checkpoint-1"
RESTORE_ID = "restore-1"
ROLLOUT_ID = "rollout-a"
STRAGGLER_ID = "rollout-straggler"
PARENT_CALL_ID = "call-parent"


def _capabilities(component: str, name: str) -> ControlCapabilities:
    return ControlCapabilities(
        component=component,
        name=name,
        multi_process=MultiProcessCapability(mode="single_worker", num_workers=1),
    )


def _agent_app(participant: AgentCheckpointParticipant, name: str) -> FastAPI:
    app = FastAPI()
    fence = ControlFence()
    install_control_plane(app, capabilities=_capabilities("responses_api_agents", name), fence=fence)
    install_agent_checkpoint(app, participant=participant, fence=fence, auth_token=AUTH_TOKEN)
    return app


def _model_app(ledger_root: Path, request_headers: list[dict[str, str]]) -> tuple[FastAPI, AdmissionLimiter]:
    app = FastAPI()
    fence = ControlFence()
    limiter = AdmissionLimiter()
    ledger = FileLineageStore(ledger_root)
    capabilities = _capabilities("responses_api_models", "policy")
    capabilities.instance_role = "policy"
    install_control_plane(app, capabilities=capabilities, fence=fence)
    install_model_admission(
        app,
        limiter=limiter,
        fence=fence,
        instance_role="policy",
        auth_token=AUTH_TOKEN,
    )
    install_model_checkpoint(
        app,
        fence=fence,
        limiter=limiter,
        ledger_provider=lambda: ledger,
        file_ledger_root_provider=lambda: ledger.checkpoint_root,
        instance_role="policy",
        server_name="policy",
        auth_token=AUTH_TOKEN,
    )

    @app.post("/v1/responses")
    async def responses(request: Request) -> dict:
        request_headers.append(dict(request.headers))
        call_index = len(request_headers)
        return {
            "id": f"response-{call_index}",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": f"attempt-1-call-{call_index}"}],
                }
            ],
        }

    app.add_middleware(
        AdmissionMiddleware,
        limiter=limiter,
        gated_suffixes=GATED_MODEL_ROUTE_SUFFIXES,
    )
    return app, limiter


def _resources_app(
    participant: ResourcesCheckpointParticipant,
    state: dict[tuple[str, int], dict],
) -> FastAPI:
    app = FastAPI()
    fence = ControlFence()
    install_control_plane(
        app,
        capabilities=_capabilities("resources_servers", "resources"),
        fence=fence,
    )

    @app.post("/seed_session")
    async def seed_session() -> dict:
        from nemo_gym.rollout_correlation import current_execution_identity

        identity = current_execution_identity()
        state[identity] = {"value": state.get(identity, {}).get("value", 0) + 1}
        return state[identity]

    @app.post("/state")
    async def get_state() -> dict:
        from nemo_gym.rollout_correlation import current_execution_identity

        return state[current_execution_identity()]

    def route_kind(path: str, method: str):
        if method != "POST":
            return None
        if path == "/seed_session":
            return "start"
        if path == "/state":
            return "read"
        return None

    install_resources_checkpoint(
        app,
        participant=participant,
        fence=fence,
        auth_token=AUTH_TOKEN,
        server_name="resources",
        route_kind=route_kind,
    )
    app.add_middleware(RolloutContextMiddleware)
    return app


def _resources_participant(
    state: dict[tuple[str, int], dict],
    *,
    restore_expected: bool = False,
) -> ResourcesCheckpointParticipant:
    async def export(rollout_id: str, attempt_index: int) -> dict:
        return dict(state[(rollout_id, attempt_index)])

    async def restore(snapshots: list[ResourceSnapshot]) -> None:
        replacement = {(snapshot.rollout_id, snapshot.attempt_index): dict(snapshot.state) for snapshot in snapshots}
        state.clear()
        state.update(replacement)

    async def retire(rollout_id: str, attempt_index: int) -> None:
        state.pop((rollout_id, attempt_index), None)

    return ResourcesCheckpointParticipant(
        export_state=export,
        restore_states=restore,
        retire_state=retire,
        restore_expected=restore_expected,
    )


class _Response:
    def __init__(self, response: httpx.Response) -> None:
        self.status = response.status_code
        self.ok = response.is_success
        self.cookies = response.cookies
        self._content = response.content

    async def read(self) -> bytes:
        return self._content


async def _post(client: httpx.AsyncClient, path: str, body: dict) -> httpx.Response:
    return await client.post(path, json=body, headers=AUTH_HEADERS)


async def _record_lineage(
    root: Path,
    rollout_id: str,
    call_id: str,
    *,
    request_items: list[dict],
    output_items: list[dict],
    parent_call_id: str | None = None,
) -> None:
    prev_len = 0 if parent_call_id is None else 3
    await FileLineageStore(root).record(
        rollout_id,
        call_id,
        request_items=request_items,
        response_items=output_items,
        cumulative_token_ids=[],
        digest="1" * 64,
        parent_call_id=parent_call_id,
        staging_key=f"stage/{rollout_id}/{call_id}",
        weight_version=7,
        prev_len=prev_len,
        delta_len=3,
        cum_len=prev_len + 3,
        staging_digest="2" * 64,
        extras_digest="3" * 64,
        mode="text" if parent_call_id is None else "token_in",
        logical_request_id=f"request-{rollout_id}",
        admitted_at=1.0,
        staging_chain=[f"stage/{rollout_id}/{call_id}"],
        chain_hash="4" * 64,
        cumulative_hash="5" * 64,
        response_id=f"response-{rollout_id}",
        output_fingerprint="6" * 64,
        continuation_fingerprint="7" * 64,
        fingerprint_version=1,
    )


@pytest.mark.asyncio
async def test_complete_partial_rollout_checkpoint_cycle(tmp_path, monkeypatch) -> None:
    checkpoint_dir = tmp_path / "controller-checkpoint"
    source_ledger_root = tmp_path / "source-ledger"
    source_model_headers: list[dict[str, str]] = []
    source_model_app, source_limiter = _model_app(source_ledger_root, source_model_headers)

    boundary_items = [
        {"type": "function_call", "call_id": "tool-1", "name": "increment", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "tool-1", "output": "1"},
    ]
    await _record_lineage(
        source_ledger_root,
        ROLLOUT_ID,
        PARENT_CALL_ID,
        request_items=[],
        output_items=boundary_items,
    )
    await _record_lineage(
        source_ledger_root,
        STRAGGLER_ID,
        "call-straggler",
        request_items=[],
        output_items=boundary_items,
    )
    for rollout_id in (ROLLOUT_ID, STRAGGLER_ID):
        source_limiter.release(source_limiter.admit(rollout_id=rollout_id, attempt_index=0))

    source_agent = AgentCheckpointParticipant("whitebox-agent")
    source_agent_app = _agent_app(source_agent, "whitebox-agent")
    source_execution = await source_agent.begin(ROLLOUT_ID, 0, task=asyncio.current_task())
    await source_agent.begin(STRAGGLER_ID, 0, task=None)

    source_resource_state: dict[tuple[str, int], dict] = {}
    source_resources = _resources_participant(source_resource_state)
    source_resources_app = _resources_app(source_resources, source_resource_state)

    # Completed results without framework acknowledgement and executions parked
    # without a boundary both make agent publication impossible.
    completed_blocker = AgentCheckpointParticipant("completed-blocker")
    completed = await completed_blocker.begin("completed", 0, task=asyncio.current_task())
    await completed_blocker.finish(completed, outcome="completed", result={"reward": 1.0})
    parked_blocker = AgentCheckpointParticipant("parked-blocker")
    parked_execution = await parked_blocker.begin("parked", 0, task=asyncio.current_task())
    parked_task = asyncio.create_task(parked_blocker.park(parked_execution))
    while parked_blocker.status()["parked_without_boundary"] != 1:
        await asyncio.sleep(0)

    async with (
        httpx.AsyncClient(transport=httpx.ASGITransport(app=source_model_app), base_url="http://model") as model,
        httpx.AsyncClient(transport=httpx.ASGITransport(app=source_agent_app), base_url="http://agent") as agent,
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=source_resources_app),
            base_url="http://resources",
        ) as resources,
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=_agent_app(completed_blocker, "completed-blocker")),
            base_url="http://completed",
        ) as completed_client,
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=_agent_app(parked_blocker, "parked-blocker")),
            base_url="http://parked",
        ) as parked_client,
    ):
        prepare_body = {"checkpoint_id": CHECKPOINT_ID, "deadline_ts": time.time() + 5}
        for blocker_client in (completed_client, parked_client):
            blocked = await _post(
                blocker_client,
                f"{AGENT_CHECKPOINT_URL_PREFIX}/prepare",
                prepare_body,
            )
            assert blocked.status_code == 409
            assert blocked.json()["error"]["code"] == "agent_prepare_incomplete"

        identity_headers = {ROLLOUT_ID_HEADER: ROLLOUT_ID, ATTEMPT_INDEX_HEADER: "0"}
        straggler_headers = {ROLLOUT_ID_HEADER: STRAGGLER_ID, ATTEMPT_INDEX_HEADER: "0"}
        assert (await resources.post("/seed_session", headers=identity_headers)).json() == {"value": 1}
        assert (await resources.post("/seed_session", headers=straggler_headers)).json() == {"value": 1}
        assert source_resources.revision_for(ROLLOUT_ID, 0) == 1

        pause = await _post(model, f"{MODEL_ADMISSION_URL_PREFIX}/pause", prepare_body)
        assert pause.json()["state"] == "paused"
        aborted = await _post(
            model,
            f"{MODEL_ADMISSION_URL_PREFIX}/abort_inflight",
            {**prepare_body, "rollout_id": STRAGGLER_ID, "attempt_index": 0},
        )
        assert aborted.status_code == 200

        agent_prepare_task = asyncio.create_task(_post(agent, f"{AGENT_CHECKPOINT_URL_PREFIX}/prepare", prepare_body))
        await asyncio.sleep(0)
        boundary_task = asyncio.create_task(
            source_agent.commit_boundary(
                source_execution,
                AgentBoundaryRecord(
                    rollout_id=ROLLOUT_ID,
                    attempt_index=0,
                    boundary_index=1,
                    output_items=boundary_items,
                    last_committed_model_call_id=PARENT_CALL_ID,
                    resource_state_revisions={"resources": 1},
                    agent_state={"step": 1},
                ),
            )
        )
        retired_agent = await _post(
            agent,
            f"{AGENT_CHECKPOINT_URL_PREFIX}/retire",
            {**prepare_body, "rollout_id": STRAGGLER_ID, "attempt_index": 0},
        )
        assert retired_agent.json()["retired"] is True
        agent_prepare = await agent_prepare_task
        assert agent_prepare.json()["ready_to_commit"] is True
        assert not boundary_task.done()

        resources_prepare = await _post(
            resources,
            f"{RESOURCES_CHECKPOINT_URL_PREFIX}/prepare",
            prepare_body,
        )
        assert resources_prepare.json()["sessions"] == 2
        retired_resources = await _post(
            resources,
            f"{RESOURCES_CHECKPOINT_URL_PREFIX}/retire",
            {**prepare_body, "rollout_id": STRAGGLER_ID, "attempt_index": 0},
        )
        assert retired_resources.json()["retired"] is True

        commit_body = {**prepare_body, "checkpoint_dir": str(checkpoint_dir)}
        model_commit = await _post(model, f"{MODEL_CHECKPOINT_URL_PREFIX}/commit", commit_body)
        agent_commit = await _post(agent, f"{AGENT_CHECKPOINT_URL_PREFIX}/commit", commit_body)
        resources_commit = await _post(resources, f"{RESOURCES_CHECKPOINT_URL_PREFIX}/commit", commit_body)
        assert model_commit.json()["excluded_tombstoned"] == 1
        assert agent_commit.json()["records"] == 1
        assert resources_commit.json()["sessions"] == 1

    agent_namespace = (
        checkpoint_dir / "agent" / f"instance-{hashlib.sha256(b'whitebox-agent').hexdigest()}" / "manifest.json"
    )
    assert (checkpoint_dir / "model-ledger" / "policy" / "manifest.json").exists()
    assert agent_namespace.exists()
    assert (checkpoint_dir / "resources" / "resources" / "manifest.json").exists()
    assert len(list(checkpoint_dir.rglob("manifest.json"))) == 3

    restored_model_headers: list[dict[str, str]] = []
    restored_model_app, restored_limiter = _model_app(tmp_path / "restored-ledger", restored_model_headers)
    restored_agent = AgentCheckpointParticipant("whitebox-agent")
    restored_agent_app = _agent_app(restored_agent, "whitebox-agent")
    restored_resource_state: dict[tuple[str, int], dict] = {}
    restored_resources = _resources_participant(restored_resource_state, restore_expected=True)
    restored_resources_app = _resources_app(restored_resources, restored_resource_state)

    async with (
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=restored_model_app), base_url="http://policy.test"
        ) as model,
        httpx.AsyncClient(transport=httpx.ASGITransport(app=restored_agent_app), base_url="http://agent") as agent,
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=restored_resources_app),
            base_url="http://resources",
        ) as resources,
    ):
        restore_body = {
            "checkpoint_id": RESTORE_ID,
            "deadline_ts": time.time() + 5,
            "checkpoint_dir": str(checkpoint_dir),
        }
        assert (await _post(model, f"{MODEL_CHECKPOINT_URL_PREFIX}/restore", restore_body)).status_code == 200
        assert (await _post(agent, f"{AGENT_CHECKPOINT_URL_PREFIX}/restore", restore_body)).status_code == 200
        assert (await _post(resources, f"{RESOURCES_CHECKPOINT_URL_PREFIX}/restore", restore_body)).status_code == 200

        early_model = await model.post("/v1/responses", json={"input": []})
        assert early_model.status_code == 409
        with pytest.raises(AgentAdmissionClosedError):
            await restored_agent.begin(ROLLOUT_ID, 1, task=None)
        early_resource = await resources.post(
            "/state",
            headers={ROLLOUT_ID_HEADER: ROLLOUT_ID, ATTEMPT_INDEX_HEADER: "1"},
        )
        assert early_resource.status_code == 409

        resume_body = {"checkpoint_id": RESTORE_ID, "deadline_ts": time.time() + 5}
        assert (await _post(model, f"{MODEL_ADMISSION_URL_PREFIX}/resume", resume_body)).status_code == 200
        assert (await _post(agent, f"{AGENT_CHECKPOINT_URL_PREFIX}/resume", resume_body)).status_code == 200
        assert (await _post(resources, f"{RESOURCES_CHECKPOINT_URL_PREFIX}/resume", resume_body)).status_code == 200

        replacement = await restored_agent.begin(ROLLOUT_ID, 1, task=asyncio.current_task())
        continuation = restored_agent.continuation(replacement)
        assert continuation is not None
        assert continuation.resource_state_revisions == {"resources": 1}
        assert continuation.agent_state == {"step": 1}
        assert restored_resources.revision_for(ROLLOUT_ID, 1) == 1
        state_response = await resources.post(
            "/state",
            headers={ROLLOUT_ID_HEADER: ROLLOUT_ID, ATTEMPT_INDEX_HEADER: "1"},
        )
        assert state_response.json() == {"value": 1}

        config = OmegaConf.create({"policy": {"responses_api_models": {"model": {"host": "policy.test", "port": 80}}}})
        server_client = ServerClient(
            head_server_config=BaseServerConfig(host="head.test", port=80),
            global_config_dict=config,
        )

        async def dispatch(method: str, url: str, **kwargs) -> _Response:
            parsed = urlsplit(url)
            response = await model.request(
                method,
                parsed.path,
                json=kwargs.get("json"),
                headers=kwargs.get("headers"),
            )
            return _Response(response)

        import nemo_gym.server_utils

        monkeypatch.setattr(nemo_gym.server_utils, "request", dispatch)
        with rollout_context(
            capture_key_for(ROLLOUT_ID, 1),
            attempt_index=1,
            logical_rollout_id=ROLLOUT_ID,
        ):
            with checkpoint_parent_context(
                capture_key_for(continuation.rollout_id, continuation.attempt_index),
                continuation.last_committed_model_call_id,
            ):
                first = await server_client.post("policy", "/v1/responses", json={"input": boundary_items})
                first_payload = orjson.loads(await first.read())
                await _record_lineage(
                    tmp_path / "restored-ledger",
                    capture_key_for(ROLLOUT_ID, 1),
                    "call-attempt-1",
                    request_items=boundary_items,
                    output_items=first_payload["output"],
                    parent_call_id=PARENT_CALL_ID,
                )
                second_input = boundary_items + first_payload["output"]
                second = await server_client.post("policy", "/v1/responses", json={"input": second_input})
        assert first_payload["id"] == "response-1"
        assert orjson.loads(await second.read())["id"] == "response-2"

    assert restored_model_headers[0][SOURCE_CAPTURE_KEY_HEADER] == capture_key_for(ROLLOUT_ID, 0)
    assert restored_model_headers[0][PARENT_MODEL_CALL_ID_HEADER] == PARENT_CALL_ID
    assert restored_model_headers[0][ROLLOUT_ID_HEADER] == ROLLOUT_ID
    assert restored_model_headers[0][ATTEMPT_INDEX_HEADER] == "1"
    assert SOURCE_CAPTURE_KEY_HEADER not in restored_model_headers[1]
    assert PARENT_MODEL_CALL_ID_HEADER not in restored_model_headers[1]

    next_parent = await FileLineageStore(tmp_path / "restored-ledger").resolve(
        capture_key_for(ROLLOUT_ID, 1),
        second_input,
    )
    assert next_parent.match is not None
    assert next_parent.match.model_call_id == "call-attempt-1"
    assert next_parent.match.model_call_id != PARENT_CALL_ID

    source_row = orjson.loads(
        (tmp_path / "restored-ledger" / f"{ROLLOUT_ID}.lineage.jsonl").read_bytes().splitlines()[0]
    )
    assert {
        "parent_call_id": source_row["parent_call_id"],
        "staging_key": source_row["staging_key"],
        "weight_version": source_row["weight_version"],
        "prev_len": source_row["prev_len"],
        "delta_len": source_row["delta_len"],
        "cum_len": source_row["cum_len"],
        "staging_digest": source_row["staging_digest"],
        "extras_digest": source_row["extras_digest"],
        "mode": source_row["mode"],
        "staging_chain": source_row["staging_chain"],
        "chain_hash": source_row["chain_hash"],
        "cumulative_hash": source_row["cumulative_hash"],
    } == {
        "parent_call_id": None,
        "staging_key": "stage/rollout-a/call-parent",
        "weight_version": 7,
        "prev_len": 0,
        "delta_len": 3,
        "cum_len": 3,
        "staging_digest": "2" * 64,
        "extras_digest": "3" * 64,
        "mode": "text",
        "staging_chain": ["stage/rollout-a/call-parent"],
        "chain_hash": "4" * 64,
        "cumulative_hash": "5" * 64,
    }
    assert not (tmp_path / "restored-ledger" / f"{STRAGGLER_ID}.lineage.jsonl").exists()
    assert restored_resources.revision_for(STRAGGLER_ID, 1) is None
    assert restored_agent.resolve(STRAGGLER_ID, 1) is None
    with pytest.raises(StaleAttemptError):
        restored_limiter.admit(rollout_id=STRAGGLER_ID, attempt_index=0)

    await restored_agent.finish(replacement, outcome="completed", result={"reward": 1.0})
    await source_agent.resume()
    await boundary_task
    parked_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await parked_task
