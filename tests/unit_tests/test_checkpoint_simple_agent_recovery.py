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
"""A real SimpleAgent turn resumes without replaying its resource mutation."""

import asyncio
import time
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import urlsplit

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from omegaconf import OmegaConf

import nemo_gym.server_utils
from nemo_gym._checkpoint import (
    AGENT_CHECKPOINT_URL_PREFIX,
    CHECKPOINT_CONTROL_TOKEN_ENV,
    GATED_MODEL_ROUTE_SUFFIXES,
    MODEL_ADMISSION_URL_PREFIX,
    MODEL_CHECKPOINT_URL_PREFIX,
    RESOURCES_CHECKPOINT_URL_PREFIX,
    AdmissionLimiter,
    AdmissionMiddleware,
    ControlCapabilities,
    ControlFence,
    MultiProcessCapability,
    install_control_plane,
    install_model_admission,
    install_model_checkpoint,
)
from nemo_gym.config_types import BaseServerConfig, ModelServerRef, ResourcesServerRef
from nemo_gym.rollout_correlation import (
    ATTEMPT_INDEX_HEADER,
    MODEL_CALL_ID_HEADER,
    PARENT_MODEL_CALL_ID_HEADER,
    ROLLOUT_ID_HEADER,
    SOURCE_CAPTURE_KEY_HEADER,
    RolloutContextMiddleware,
    current_attempt_index,
    current_logical_rollout_id,
)
from nemo_gym.server_utils import ServerClient
from nemo_gym.token_id_capture.lineage import FileLineageStore
from nemo_gym.token_id_capture.staging.records import CallRecord, CaptureLedgerCommit, RolloutManifest
from resources_servers.example_session_state_mgmt.app import (
    IncrementCounterRequest,
    IncrementCounterResponse,
    StatefulCounterResourcesServer,
    StatefulCounterResourcesServerConfig,
)
from responses_api_agents.simple_agent.app import SimpleAgent, SimpleAgentConfig


AUTH_TOKEN = "checkpoint-token"
AUTH_HEADERS = {"authorization": f"Bearer {AUTH_TOKEN}"}
ROLLOUT_ID = "partial-rollout"
COMPLETED_ROLLOUT_ID = "completed-rollout"
CHECKPOINT_ID = "checkpoint-1"
RESTORE_ID = "restore-1"


class _AsyncBody:
    def __init__(self, content: bytes) -> None:
        self._content = content

    async def read(self) -> bytes:
        return self._content


class _Response:
    def __init__(self, response: httpx.Response) -> None:
        self._response = response
        self.status = response.status_code
        self.ok = response.is_success
        self.cookies = response.cookies
        self.headers = response.headers
        self.content = _AsyncBody(response.content)
        self._content = response.content

    async def read(self) -> bytes:
        return self._content

    def raise_for_status(self) -> None:
        self._response.raise_for_status()


class _CountingResourcesServer(StatefulCounterResourcesServer):
    incremented: ClassVar[list[tuple[str, int]]] = []

    async def increment_counter(self, request: Request, body: IncrementCounterRequest) -> IncrementCounterResponse:
        identity = (current_logical_rollout_id(), current_attempt_index())
        assert identity[0] is not None and identity[1] is not None
        self.incremented.append((identity[0], identity[1]))
        return await super().increment_counter(request, body)


def _global_config() -> Any:
    return OmegaConf.create(
        {
            "observability_enabled": True,
            "policy": {"responses_api_models": {"policy": {"host": "policy.test", "port": 80}}},
            "resources": {"resources_servers": {"resources": {"host": "resources.test", "port": 80}}},
            "agent": {"responses_api_agents": {"agent": {"host": "agent.test", "port": 80}}},
        }
    )


def _server_client() -> ServerClient:
    return ServerClient(
        head_server_config=BaseServerConfig(host="head.test", port=80),
        global_config_dict=_global_config(),
    )


def _agent() -> SimpleAgent:
    return SimpleAgent(
        config=SimpleAgentConfig(
            host="agent.test",
            port=80,
            entrypoint="app.py",
            name="agent",
            model_server=ModelServerRef(type="responses_api_models", name="policy"),
            resources_server=ResourcesServerRef(type="resources_servers", name="resources"),
        ),
        server_client=_server_client(),
    )


def _resources(*, restore_expected: bool = False) -> _CountingResourcesServer:
    return _CountingResourcesServer(
        config=StatefulCounterResourcesServerConfig(
            host="resources.test",
            port=80,
            entrypoint="app.py",
            name="resources",
            checkpoint_restore_expected=restore_expected,
        ),
        server_client=_server_client(),
    )


def _model_response(*, response_id: str, output: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": response_id,
        "created_at": 1.0,
        "model": "deterministic-policy",
        "object": "response",
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


def _model_app(
    ledger_root: Path,
    *,
    first_call_returned: asyncio.Event | None = None,
) -> tuple[FastAPI, list[dict[str, Any]]]:
    app = FastAPI()
    fence = ControlFence()
    limiter = AdmissionLimiter()
    requests: list[dict[str, Any]] = []
    ledger = FileLineageStore(ledger_root)
    capabilities = ControlCapabilities(
        component="responses_api_models",
        name="policy",
        instance_role="policy",
        multi_process=MultiProcessCapability(mode="single_worker", num_workers=1),
    )
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
    async def responses(request: Request) -> JSONResponse:
        body = await request.json()
        rollout_id = current_logical_rollout_id()
        attempt_index = current_attempt_index()
        assert rollout_id is not None and attempt_index is not None
        input_items = body["input"]
        has_tool_result = any(item.get("type") == "function_call_output" for item in input_items)
        if rollout_id == COMPLETED_ROLLOUT_ID or has_tool_result:
            output = [
                {
                    "id": f"message-{rollout_id}-a{attempt_index}",
                    "content": [{"annotations": [], "text": "done", "type": "output_text"}],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ]
        else:
            output = [
                {
                    "id": "function-call-1",
                    "call_id": "counter-call-1",
                    "name": "increment_counter",
                    "arguments": '{"count":2}',
                    "type": "function_call",
                    "status": "completed",
                }
            ]
        call_id = f"{rollout_id}-a{attempt_index}-call-{len(requests) + 1}"
        requests.append({"body": body, "headers": dict(request.headers), "call_id": call_id})
        capture_key = f"{rollout_id}{'' if attempt_index == 0 else f'-a{attempt_index}'}"
        staging_key = f"stage/{rollout_id}/a{attempt_index}/{call_id}"
        parent_call_id = request.headers.get(PARENT_MODEL_CALL_ID_HEADER)
        parent_match = None
        if parent_call_id is not None:
            source_capture_key = request.headers[SOURCE_CAPTURE_KEY_HEADER]
            parent_match = (await ledger.resolve_explicit(source_capture_key, parent_call_id, input_items)).match
            assert parent_match is not None
        prev_len = parent_match.prev_len if parent_match is not None else 0
        staging_chain = (*parent_match.staging_chain, staging_key) if parent_match is not None else (staging_key,)
        await ledger.record(
            CaptureLedgerCommit(
                rollout_id=capture_key,
                record=CallRecord(
                    model_call_id=call_id,
                    parent_call_id=parent_call_id,
                    staging_key=staging_key,
                    weight_version=7,
                    prev_len=prev_len,
                    delta_len=1,
                    cum_len=prev_len + 1,
                    digest="2" * 64,
                    extras_digest="3" * 64,
                    mode="text" if parent_call_id is None else "token_in",
                    admitted_at=1.0,
                    chain_hash="4" * 64,
                    cumulative_hash="5" * 64,
                    response_id=f"response-{call_id}",
                    output_fingerprint="6" * 64,
                    continuation_fingerprint="7" * 64,
                    fingerprint_version=1,
                ),
                staging_chain=staging_chain,
                parent_manifest=(parent_match.parent_manifest if parent_match is not None else ()),
                request_items=input_items,
                response_items=output,
            )
        )
        if first_call_returned is not None and rollout_id == ROLLOUT_ID and not has_tool_result:
            first_call_returned.set()
        return JSONResponse(
            _model_response(response_id=f"response-{call_id}", output=output),
            headers={MODEL_CALL_ID_HEADER: call_id},
        )

    app.add_middleware(
        AdmissionMiddleware,
        limiter=limiter,
        gated_suffixes=GATED_MODEL_ROUTE_SUFFIXES,
    )
    app.add_middleware(RolloutContextMiddleware)
    return app, requests


async def _post_control(client: httpx.AsyncClient, path: str, body: dict[str, Any]) -> httpx.Response:
    return await client.post(path, json=body, headers=AUTH_HEADERS)


async def _wait_until(predicate, *, timeout: float = 2.0) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0)


def _run_body(rollout_id: str, attempt_index: int, *, initial_count: int, expected_count: int) -> dict[str, Any]:
    return {
        "_ng_rollout_id": rollout_id,
        "_ng_attempt_index": attempt_index,
        "initial_count": initial_count,
        "expected_count": expected_count,
        "responses_create_params": {
            "input": [{"role": "user", "content": "increment the counter by two, then finish"}],
            "tools": [
                {
                    "name": "increment_counter",
                    "parameters": {
                        "type": "object",
                        "properties": {"count": {"type": "integer"}},
                        "required": ["count"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                    "type": "function",
                }
            ],
        },
    }


@pytest.mark.asyncio
async def test_simple_agent_restores_next_turn_without_repeating_resource_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(CHECKPOINT_CONTROL_TOKEN_ENV, AUTH_TOKEN)
    _CountingResourcesServer.incremented = []
    checkpoint_dir = tmp_path / "checkpoint"
    first_call_returned = asyncio.Event()

    source_agent = _agent()
    source_resources = _resources()
    source_model_app, source_model_requests = _model_app(
        tmp_path / "source-lineage",
        first_call_returned=first_call_returned,
    )
    source_apps = {
        "agent.test": source_agent.setup_webserver(),
        "resources.test": source_resources.setup_webserver(),
        "policy.test": source_model_app,
    }
    source_clients = {
        host: httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=f"http://{host}")
        for host, app in source_apps.items()
    }

    async def source_dispatch(method: str, url: str, **kwargs: Any) -> _Response:
        parsed = urlsplit(url)
        payload = kwargs.get("json")
        if hasattr(payload, "model_dump"):
            payload = payload.model_dump(mode="json", by_alias=True, exclude_none=True)
        response = await source_clients[parsed.hostname].request(
            method,
            parsed.path,
            json=payload,
            cookies=kwargs.get("cookies"),
            headers=kwargs.get("headers"),
        )
        return _Response(response)

    monkeypatch.setattr(nemo_gym.server_utils, "request", source_dispatch)
    try:
        completed = await source_clients["agent.test"].post(
            "/run",
            json=_run_body(COMPLETED_ROLLOUT_ID, 0, initial_count=3, expected_count=3),
        )
        assert completed.status_code == 200
        assert completed.json()["reward"] == 1.0
        completion_receipt = next(
            item["completion_receipt"]
            for item in source_agent.checkpoint_participant().status()["completed_unacknowledged_attempts"]
            if item["rollout_id"] == COMPLETED_ROLLOUT_ID and item["attempt_index"] == 0
        )
        acknowledgement = await _post_control(
            source_clients["agent.test"],
            f"{AGENT_CHECKPOINT_URL_PREFIX}/acknowledge-completed",
            {
                "schema_version": 1,
                "executions": [completion_receipt],
            },
        )
        assert acknowledgement.status_code == 200

        partial_run = asyncio.create_task(
            source_clients["agent.test"].post(
                "/run",
                json=_run_body(ROLLOUT_ID, 0, initial_count=3, expected_count=5),
            )
        )
        await first_call_returned.wait()
        prepare_body = {"checkpoint_id": CHECKPOINT_ID, "deadline_ts": time.time() + 5}
        paused = await _post_control(
            source_clients["policy.test"],
            f"{MODEL_ADMISSION_URL_PREFIX}/pause",
            prepare_body,
        )
        assert paused.status_code == 200
        await _wait_until(lambda: source_agent.checkpoint_participant().status()["parked_with_boundary"] == 1)

        agent_prepare = await _post_control(
            source_clients["agent.test"],
            f"{AGENT_CHECKPOINT_URL_PREFIX}/prepare",
            prepare_body,
        )
        resources_prepare = await _post_control(
            source_clients["resources.test"],
            f"{RESOURCES_CHECKPOINT_URL_PREFIX}/prepare",
            prepare_body,
        )
        assert agent_prepare.json()["ready_to_commit"] is True
        assert resources_prepare.json()["sessions"] == 1
        # Session creation is revision 1 and the counter mutation is revision 2.
        assert source_resources.checkpoint_participant().revision_for(ROLLOUT_ID, 0) == 2

        commit_body = {**prepare_body, "checkpoint_dir": str(checkpoint_dir)}
        agent_commit = await _post_control(
            source_clients["agent.test"],
            f"{AGENT_CHECKPOINT_URL_PREFIX}/commit",
            commit_body,
        )
        assert agent_commit.status_code == 200, agent_commit.text
        model_commit = await _post_control(
            source_clients["policy.test"],
            f"{MODEL_CHECKPOINT_URL_PREFIX}/commit",
            {
                **commit_body,
                "continuation_indexes": [agent_commit.json()["continuation_index"]],
            },
        )
        assert model_commit.status_code == 200, model_commit.text
        resources_commit = await _post_control(
            source_clients["resources.test"],
            f"{RESOURCES_CHECKPOINT_URL_PREFIX}/commit",
            commit_body,
        )
        assert resources_commit.status_code == 200, resources_commit.text

        partial_run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await partial_run
    finally:
        for client in source_clients.values():
            await client.aclose()

    restored_agent = _agent()
    restored_resources = _resources(restore_expected=True)
    restored_model_app, restored_model_requests = _model_app(tmp_path / "restored-lineage")
    restored_apps = {
        "agent.test": restored_agent.setup_webserver(),
        "resources.test": restored_resources.setup_webserver(),
        "policy.test": restored_model_app,
    }
    restored_clients = {
        host: httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=f"http://{host}")
        for host, app in restored_apps.items()
    }

    async def restored_dispatch(method: str, url: str, **kwargs: Any) -> _Response:
        parsed = urlsplit(url)
        payload = kwargs.get("json")
        if hasattr(payload, "model_dump"):
            payload = payload.model_dump(mode="json", by_alias=True, exclude_none=True)
        response = await restored_clients[parsed.hostname].request(
            method,
            parsed.path,
            json=payload,
            cookies=kwargs.get("cookies"),
            headers=kwargs.get("headers"),
        )
        return _Response(response)

    monkeypatch.setattr(nemo_gym.server_utils, "request", restored_dispatch)
    try:
        restore_body = {
            "checkpoint_id": RESTORE_ID,
            "deadline_ts": time.time() + 5,
            "checkpoint_dir": str(checkpoint_dir),
        }
        for client, path in (
            (restored_clients["policy.test"], f"{MODEL_CHECKPOINT_URL_PREFIX}/restore"),
            (restored_clients["agent.test"], f"{AGENT_CHECKPOINT_URL_PREFIX}/restore"),
            (restored_clients["resources.test"], f"{RESOURCES_CHECKPOINT_URL_PREFIX}/restore"),
        ):
            restored = await _post_control(client, path, restore_body)
            assert restored.status_code == 200, restored.text

        resume_body = {"checkpoint_id": RESTORE_ID, "deadline_ts": time.time() + 5}
        for client, path in (
            (restored_clients["policy.test"], f"{MODEL_ADMISSION_URL_PREFIX}/resume"),
            (restored_clients["agent.test"], f"{AGENT_CHECKPOINT_URL_PREFIX}/resume"),
            (restored_clients["resources.test"], f"{RESOURCES_CHECKPOINT_URL_PREFIX}/resume"),
        ):
            resumed = await _post_control(client, path, resume_body)
            assert resumed.status_code == 200, resumed.text

        result = await restored_clients["agent.test"].post(
            "/run",
            json=_run_body(ROLLOUT_ID, 1, initial_count=3, expected_count=5),
        )
        assert result.status_code == 200, result.text
        assert result.json()["reward"] == 1.0
        restored_manifest = RolloutManifest.model_validate(
            await FileLineageStore(tmp_path / "restored-lineage").manifest(f"{ROLLOUT_ID}-a1")
        )
        assert [record.capture_key for record in restored_manifest.records] == [ROLLOUT_ID, f"{ROLLOUT_ID}-a1"]
        assert [record.model_call_id for record in restored_manifest.records] == [
            source_model_requests[-1]["call_id"],
            restored_model_requests[0]["call_id"],
        ]
    finally:
        for client in restored_clients.values():
            await client.aclose()

    assert _CountingResourcesServer.incremented == [(ROLLOUT_ID, 0)]
    assert len(source_model_requests) == 2
    assert len(restored_model_requests) == 1
    source_partial_request = source_model_requests[-1]
    restored_request = restored_model_requests[0]
    restored_input = restored_request["body"]["input"]
    assert [item["type"] for item in restored_input[-2:]] == ["function_call", "function_call_output"]
    assert restored_request["headers"][ROLLOUT_ID_HEADER] == ROLLOUT_ID
    assert restored_request["headers"][ATTEMPT_INDEX_HEADER] == "1"
    assert restored_request["headers"][SOURCE_CAPTURE_KEY_HEADER] == ROLLOUT_ID
    assert restored_request["headers"][PARENT_MODEL_CALL_ID_HEADER] == source_partial_request["call_id"]
    assert restored_agent.checkpoint_participant().resolve(COMPLETED_ROLLOUT_ID, 1) is None
