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
"""A sandbox failure before the agent ran keeps no response, so it retries within the cap; a reset /run is resent."""

import asyncio
import json
import socket
import struct

import httpx
import pytest
from aiohttp import ClientSession
from omegaconf import OmegaConf

import nemo_gym.client.http_client as transport
import nemo_gym.rollouts.rollout_collection as collection
from nemo_gym.artifacts.rollout_rows import NG_NO_GENERATION_METADATA
from nemo_gym.client.server_client import ServerClient
from nemo_gym.config.config_types import BaseServerConfig
from nemo_gym.rollouts.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper
from nemo_gym.rollouts.rollout_recovery import RunManifest, has_recorded_grade, has_saved_generation
from nemo_gym.rollouts.rollout_store import RolloutStore
from tests.helpers.single_agent_turn import _environment_server, _request


# The agent's own response object when its sandbox was never created: no output items and no status.
_EMPTY = {"id": "resp-empty", "object": "response", "output": [], "error": None, "status": None, "model": "policy"}


def _sandbox_credentials_rejected(**extra):
    return {
        "_ng_failure_class": "infrastructure_error",
        "failure_kind": None,
        "failure_reason": "SandboxApiException",
        "evaluation_completed": False,
        "mask_sample": False,
        **extra,
    }


def test_the_collector_keeps_no_response_object_for_an_agent_that_never_ran():
    diagnostics = collection._failure_diagnostics(_sandbox_credentials_rejected(response=_EMPTY))
    assert "_ng_failure_response" not in diagnostics
    assert diagnostics["failure_reason"] == "SandboxApiException"
    # A response the producer finished stays evidence, as before.
    finished = _EMPTY | {"status": "completed"}
    assert (
        collection._failure_diagnostics(_sandbox_credentials_rejected(response=finished))["_ng_failure_response"]
        == finished
    )


def test_an_empty_response_object_is_not_a_kept_answer():
    assert not has_saved_generation({"_ng_failure_response": _EMPTY})
    assert has_saved_generation({"_ng_failure_response": _EMPTY | {"status": "completed"}})


# A model's legitimate blank answer: a completed response holding one empty assistant message.
_BLANK_ANSWER = {
    "id": "resp-blank",
    "object": "response",
    "output": [
        {
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "id": "msg_0",
            "content": [{"type": "output_text", "text": "", "annotations": []}],
        }
    ],
    "error": None,
    "status": "completed",
    "model": "policy",
}
# The same shape built by an agent's failure path when no generation happened, carrying the explicit marker.
_NO_GENERATION = _BLANK_ANSWER | {"id": "agent-failure", "metadata": dict(NG_NO_GENERATION_METADATA)}


@pytest.mark.parametrize("key", ["response", "_ng_failure_response"])
def test_a_marked_no_generation_placeholder_is_not_a_kept_answer_but_a_blank_answer_is(key):
    assert not has_saved_generation({key: _NO_GENERATION})
    assert has_saved_generation({key: _BLANK_ANSWER})


def test_an_unmarked_placeholder_saved_before_the_marker_stays_a_kept_answer():
    """Without the marker an old placeholder has the shape of a blank model answer, so it is kept, never rerun."""
    legacy = _BLANK_ANSWER | {"id": "remote_agent_failure", "model": "remote_agent", "created_at": 0.0}
    assert has_saved_generation({"_ng_failure_response": legacy})


def test_a_completed_evaluation_is_a_recorded_grade():
    assert has_recorded_grade({"evaluation_completed": True})
    assert not has_recorded_grade({"evaluation_completed": False})
    assert not has_recorded_grade({"reward": 0.0})  # collection never keeps a placeholder score


@pytest.mark.parametrize("graded,pending", [(False, [1]), (True, [])], ids=["no-grade-retried", "graded-kept"])
def test_the_store_retries_a_sandbox_failure_saved_with_an_empty_response(tmp_path, graded, pending):
    """No kept answer and no recorded grade: a fresh retry. A recorded grade is a graded result: it is kept."""
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    rows = [{"_ng_task_index": 0, "_ng_rollout_index": 0}]
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(rows[0]) + "\n")

    def prepare():
        return rows, RunManifest.create(source, rows, {}, {}).model_copy(update={"max_rollout_attempts": 2})

    with RolloutStore.start_or_resume(artifacts / "rollouts.jsonl", prepare, resume=False) as store:
        row = dict(rows[0])
        store.allocate_attempt(row)
        record = {
            "schema_version": 1,
            "episode_id": {"rollout_id": "0-0", "attempt": 0},
            "run_id": store.manifest.run_id,
            "source": "environment",
            "delivery": "delivered",
            "failure": {
                "failure_reason": "SandboxApiException",
                "terminal": False,
                "failure_kind": "infrastructure_error",
            },
            "http_status": None,
            "exception_type": "SandboxApiException",
        }
        # Rows saved before this change still carry the empty response object.
        saved = _sandbox_credentials_rejected(_ng_failure_record=record, _ng_failure_response=_EMPTY)
        store.record_outcome(row | saved | {"evaluation_completed": graded})
        assert [row["_ng_attempt_index"] for row in store.pending()] == pending


@pytest.mark.parametrize(
    "response,pending",
    [(_NO_GENERATION, [1]), (_BLANK_ANSWER, [])],
    ids=["no-generation-retried", "blank-answer-kept"],
)
def test_the_store_retries_an_infrastructure_failure_whose_agent_never_generated(tmp_path, response, pending):
    """A retryable infrastructure failure retries past a no-generation placeholder; a real blank answer is kept."""
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    rows = [{"_ng_task_index": 0, "_ng_rollout_index": 0}]
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(rows[0]) + "\n")

    def prepare():
        return rows, RunManifest.create(source, rows, {}, {}).model_copy(update={"max_rollout_attempts": 2})

    with RolloutStore.start_or_resume(artifacts / "rollouts.jsonl", prepare, resume=False) as store:
        row = dict(rows[0])
        store.allocate_attempt(row)
        record = {
            "schema_version": 1,
            "episode_id": {"rollout_id": "0-0", "attempt": 0},
            "run_id": store.manifest.run_id,
            "source": "environment",
            "delivery": "delivered",
            "failure": {
                "failure_reason": "SandboxApiException",
                "terminal": False,
                "failure_kind": "infrastructure_error",
            },
            "http_status": None,
            "exception_type": "SandboxApiException",
        }
        saved = _sandbox_credentials_rejected(_ng_failure_record=record, _ng_failure_response=response)
        store.record_outcome(row | saved)
        assert [row["_ng_attempt_index"] for row in store.pending()] == pending


async def test_a_run_with_routing_off_survives_a_connection_reset_before_any_reply(tmp_path, monkeypatch):
    environment, dependencies = _environment_server()
    dependencies.responses.extend(list(dependencies.responses))
    requests = []
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=environment.setup_webserver()), base_url="http://environment"
    ) as endpoint:

        async def handle(reader, writer):
            headers = await reader.readuntil(b"\r\n\r\n")
            length = int(
                next(
                    line.split(b":", 1)[1]
                    for line in headers.split(b"\r\n")
                    if line.lower().startswith(b"content-length:")
                )
            )
            body = json.loads(await reader.readexactly(length))
            requests.append(body)
            if len(requests) == 1:
                # A reset before any reply byte, as a closed keep-alive connection gives the client.
                sock = writer.get_extra_info("socket")
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
                writer.transport.abort()
                return
            response = await endpoint.post("/run", json=body)
            payload = response.content
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                + str(len(payload)).encode()
                + b"\r\nConnection: close\r\n\r\n"
                + payload
            )
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        tcp = await asyncio.start_server(handle, "127.0.0.1", 0)
        global_config = OmegaConf.create(
            {
                "environment": {
                    "environment_servers": {
                        "single_agent_turn": {"host": "127.0.0.1", "port": tcp.sockets[0].getsockname()[1]}
                    }
                }
            }
        )
        client = ServerClient(
            head_server_config=BaseServerConfig(host="head", port=1), global_config_dict=global_config
        )
        monkeypatch.setattr(collection, "setup_server_client_utils", lambda *args, **kwargs: client)
        source = tmp_path / "input.jsonl"
        source.write_text(_request().task.model_dump_json() + "\n")
        (tmp_path / "artifacts").mkdir()
        output = tmp_path / "artifacts" / "rollouts.jsonl"
        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(source),
            output_jsonl_fpath=str(output),
            environment_server_routes={"test:source": "environment"},
            route_failures_to_sidecar=False,
            require_complete=False,
            disable_aggregation=True,
            disable_health_check=True,
        )
        async with tcp, ClientSession() as session:
            monkeypatch.setattr(transport, "get_global_aiohttp_client", lambda: session)
            await asyncio.wait_for(RolloutCollectionHelper().run_from_config(config), timeout=10)

    assert len(requests) == 2
    store = RolloutStore.read(output)
    assert store.coverage()["measured"] == 1 and store.coverage()["attempts"] == 1
    assert output.with_name("rollouts_failures.jsonl").read_text() == ""
