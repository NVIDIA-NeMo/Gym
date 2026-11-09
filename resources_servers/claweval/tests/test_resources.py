# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise real HTTP/session/MCP wiring without allocating native sandboxes."""

from copy import deepcopy
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.client import ServerClient
from resources_servers.claweval.app import ClawEvalResourcesConfig, ClawEvalResourcesServer
from resources_servers.claweval.process import TaskProcess


@pytest.fixture
def native(monkeypatch):
    operations = []

    async def start(self, python, output):
        self.task = None

    async def call(self, operation, payload, *, timeout):
        operations.append((operation, deepcopy(payload)))
        if operation == "seed":
            self.task = payload["task_data"]["task_id"]
            return {
                "input": self.task,
                "instructions": "Native prompt",
                "max_user_rounds": 0,
                "timeout_seconds": 5,
                "max_turns": 3,
                "tools": [
                    {
                        "name": self.task,
                        "description": "Native task tool",
                        "input_schema": {
                            "type": "object",
                            "properties": {"value": {"type": "string"}},
                            "required": ["value"],
                        },
                    }
                ],
            }
        if operation == "tool":
            if payload["name"] != self.task:
                raise ValueError("tool is unavailable in this task")
            return {
                "content": [{"type": "text", "text": self.task + ":" + payload["arguments"]["value"]}],
                "isError": False,
            }
        if operation == "verify":
            return {"reward": 0.715, "reward_components": {"completion": 0.8}}
        raise AssertionError(operation)

    async def close(self):
        operations.append(("close", {"task": self.task}))

    monkeypatch.setattr(TaskProcess, "start", start)
    monkeypatch.setattr(TaskProcess, "call", call)
    monkeypatch.setattr(TaskProcess, "close", close)
    return operations


def server():
    config = ClawEvalResourcesConfig(
        name="claweval_resources",
        host="127.0.0.1",
        port=8000,
        entrypoint="app.py",
        claweval_root="/native/source",
        claweval_revision="a" * 40,
        claweval_python="python",
        claweval_config="config.yaml",
        sandbox_image="native.sqsh",
        sandbox_dependencies="/native/deps",
    )
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {}
    return ClawEvalResourcesServer(config=config, server_client=client)


def seed(client, identifier="first", task="T001"):
    body = {
        "resources_session_id": identifier,
        "episode_id": {"rollout_id": identifier},
        "task_id": {"taskset": "claweval", "task_id": task},
        "task_data": {"task_id": task, "task_sha256": "a" * 64, "prompt": "Task prompt", "split": "general"},
    }
    response = client.post("/seed_session", json=body)
    assert response.status_code == 200, response.text
    return response.json(), body


def rpc(client, headers, method, params=None):
    return client.post(
        "/mcp/",
        json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}},
        headers={**headers, "Accept": "application/json, text/event-stream"},
    )


def test_mcp_lists_exact_task_schema_and_scopes_tokens_to_live_tasks(native):
    resources = server()
    with TestClient(resources.setup_webserver()) as client:
        first, first_body = seed(client)
        second, _ = seed(client, "second", "T002")
        for data, name in [(first, "T001"), (second, "T002")]:
            headers = data["resources_tools"]["headers"]
            listed = rpc(client, headers, "tools/list")
            assert listed.status_code == 200, listed.text
            assert [tool["name"] for tool in listed.json()["result"]["tools"]] == [name]
            result = rpc(client, headers, "tools/call", {"name": name, "arguments": {"value": "actual"}})
            assert result.json()["result"]["content"][0]["text"] == name + ":actual"
        assert rpc(client, {"Authorization": "Bearer wrong"}, "tools/list").status_code == 401
        close = client.post(
            "/close_session", json={"resources_session_id": "first", "episode_id": first_body["episode_id"]}
        )
        assert close.status_code == 200, close.text
        assert rpc(client, first["resources_tools"]["headers"], "tools/list").status_code == 401
        assert rpc(client, second["resources_tools"]["headers"], "tools/list").status_code == 200


def test_verify_retries_replay_score_and_never_grade_or_stop_twice(native):
    resources = server()
    with TestClient(resources.setup_webserver()) as client:
        _, body = seed(client)
        verify = {
            "episode_id": body["episode_id"],
            "task_id": body["task_id"],
            "verification_input": {"conversation": [], "model_turns": 1},
        }
        first = client.post("/verify", json=verify)
        second = client.post("/verify", json=verify)
        assert first.status_code == second.status_code == 200
        assert first.json() == second.json()
    assert [op for op, _ in native].count("verify") == 1
    assert [op for op, _ in native].count("close") == 1


def test_closed_session_cannot_be_reseeded_and_other_episode_cannot_close_it(native):
    resources = server()
    with TestClient(resources.setup_webserver()) as client:
        _, body = seed(client)
        wrong = client.post(
            "/close_session", json={"resources_session_id": "first", "episode_id": {"rollout_id": "intruder"}}
        )
        assert wrong.status_code == 409
        assert resources.has_session_state("first")
        closed = client.post(
            "/close_session", json={"resources_session_id": "first", "episode_id": body["episode_id"]}
        )
        assert closed.status_code == 200
        assert client.post("/seed_session", json=body).status_code == 409


def test_shutdown_attempts_every_session_and_retains_failed_cleanup(native, monkeypatch):
    resources = server()

    async def close(self):
        native.append(("shutdown", {"task": self.task}))
        if self.task == "T001":
            raise RuntimeError("native teardown failed")

    with pytest.raises(BaseExceptionGroup, match="ClawEval resource cleanup failed"):
        with TestClient(resources.setup_webserver()) as client:
            seed(client)
            seed(client, "second", "T002")
            monkeypatch.setattr(TaskProcess, "close", close)
    assert {payload["task"] for op, payload in native if op == "shutdown"} == {"T001", "T002"}
    assert resources.has_session_state("first")
    assert not resources.has_session_state("second")


def test_image_results_preserve_media_over_mcp_and_direct_http(native, monkeypatch):
    original = TaskProcess.call
    image = {"type": "image", "data": "cGl4ZWxz", "mimeType": "image/png"}

    async def call(self, operation, payload, *, timeout):
        result = await original(self, operation, payload, timeout=timeout)
        if operation == "tool":
            result["content"].append(image)
        return result

    monkeypatch.setattr(TaskProcess, "call", call)
    with TestClient(server().setup_webserver()) as client:
        data, _ = seed(client)
        response = rpc(
            client,
            data["resources_tools"]["headers"],
            "tools/call",
            {"name": "T001", "arguments": {"value": "actual"}},
        )
        assert response.json()["result"]["content"][1] == image
        response = client.post("/T001", json={"value": "actual"})
        assert response.status_code == 200
        assert response.json() == {
            "output": "T001:actual",
            "tool_output_images": [{"image_url": "data:image/png;base64,cGl4ZWxz"}],
        }
