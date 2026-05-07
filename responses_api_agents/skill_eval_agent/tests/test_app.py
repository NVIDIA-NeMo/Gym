# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Unit tests for skill_eval_agent.

Server interactions go through a FakeServerClient that records every request
so tests can assert: seed_session → model loop → /verify → /close sequencing,
SKILL.md prepending, tool-call forwarding, tools schema injection.
"""

import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.server_utils import ServerClient
from responses_api_agents.skill_eval_agent.app import (
    SkillEvalAgent,
    SkillEvalAgentConfig,
    _tool_call_log_entry,
)


def _message_response(text: str = "done") -> dict:
    return {
        "id": "resp_1",
        "created_at": 1.0,
        "model": "dummy",
        "object": "response",
        "output": [
            {
                "id": "msg_1",
                "content": [{"annotations": [], "text": text, "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


def _tool_call_response(name: str, arguments: str, call_id: str = "call_1") -> dict:
    return {
        "id": "resp_tc",
        "created_at": 1.0,
        "model": "dummy",
        "object": "response",
        "output": [
            {
                "id": "fc_1",
                "type": "function_call",
                "name": name,
                "arguments": arguments,
                "call_id": call_id,
                "status": "completed",
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


def _verify_response(reward: float, grades: list[dict] | None = None) -> dict:
    return {
        "responses_create_params": {"input": []},
        "response": _message_response(),
        "reward": reward,
        "grades": grades or [],
        "judge_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        "parse_error": None,
    }


@dataclass
class Call:
    server_name: str
    url_path: str
    json: Any
    cookies: Any = None


class FakeResponse:
    def __init__(self, payload: dict | str, ok: bool = True, status: int = 200):
        self._body = json.dumps(payload) if isinstance(payload, dict) else payload
        self.ok = ok
        self.status = status
        self.cookies = MagicMock()
        content = MagicMock()
        content.read = self.read
        self.content = content

    async def read(self) -> bytes:
        return self._body.encode()


@dataclass
class FakeServerClient:
    """Records calls and returns queued responses keyed by (server_name, url_path)."""

    responses: dict[tuple[str, str], list[FakeResponse]] = field(default_factory=dict)
    calls: list[Call] = field(default_factory=list)

    def queue(self, server_name: str, url_path: str, *resps: FakeResponse) -> None:
        self.responses.setdefault((server_name, url_path), []).extend(resps)

    async def post(
        self, *, server_name: str, url_path: str, json: Any = None, cookies: Any = None, **_
    ) -> FakeResponse:
        self.calls.append(Call(server_name=server_name, url_path=url_path, json=json, cookies=cookies))
        queue = self.responses.get((server_name, url_path))
        if not queue:
            raise AssertionError(f"No fake response queued for {server_name}{url_path}")
        return queue.pop(0) if len(queue) > 1 else queue[0]


def _make_agent(fake: FakeServerClient) -> SkillEvalAgent:
    config = SkillEvalAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="skill_eval_agent",
        workspace_server=ResourcesServerRef(type="resources_servers", name="skill_workspace"),
        judge_server=ResourcesServerRef(type="resources_servers", name="skill_judge"),
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        max_steps=4,
    )
    # Bypass Pydantic's strict ServerClient type check by passing a spec'd mock with our fake as delegate.
    client_mock = MagicMock(spec=ServerClient)
    client_mock.post = fake.post
    return SkillEvalAgent(config=config, server_client=client_mock)


def _run_body(
    *,
    with_skill: bool,
    skill_md: str = "skill instructions",
    assertions: list[str] | None = None,
    files: list[str] | None = None,
) -> dict:
    return {
        "responses_create_params": {
            "input": [{"role": "user", "content": "solve the task"}],
        },
        "verifier_metadata": {
            "skill_path": "/tmp/skill",
            "scenario_id": 1,
            "files": files or [],
            "with_skill": with_skill,
            "skill_md": skill_md,
            "assertions": assertions or ["assertion A"],
        },
    }


class TestRunEndToEnd:
    def test_with_skill_prepends_skill_md_and_calls_seed_verify_close(self) -> None:
        fake = FakeServerClient()
        fake.queue("skill_workspace", "/seed_session", FakeResponse({"env_id": "E1"}))
        fake.queue("policy_model", "/v1/responses", FakeResponse(_message_response("final")))
        fake.queue("skill_judge", "/verify", FakeResponse(_verify_response(reward=1.0)))
        fake.queue("skill_workspace", "/close", FakeResponse({"message": "ok", "success": True}))

        agent = _make_agent(fake)
        client = TestClient(agent.setup_webserver())
        r = client.post("/run", json=_run_body(with_skill=True))
        assert r.status_code == 200
        assert r.json()["reward"] == 1.0

        paths = [(c.server_name, c.url_path) for c in fake.calls]
        assert paths == [
            ("skill_workspace", "/seed_session"),
            ("policy_model", "/v1/responses"),
            ("skill_judge", "/verify"),
            ("skill_workspace", "/close"),
        ]

        seed_body = fake.calls[0].json
        assert seed_body["skill_path"] == "/tmp/skill"
        assert seed_body["scenario_id"] == 1

        # SKILL.md must be prepended as a system message when with_skill=True.
        model_body = fake.calls[1].json
        # model_copy(update=...) yields dict via model_dump; but server_client.post gets the pydantic model.
        input_msgs = model_body.input if hasattr(model_body, "input") else model_body["input"]

        def _role(msg):
            return msg.role if hasattr(msg, "role") else msg["role"]

        def _content(msg):
            return msg.content if hasattr(msg, "content") else msg["content"]

        assert _role(input_msgs[0]) == "system"
        assert "skill instructions" in _content(input_msgs[0])
        # Tools must be injected when none were supplied.
        tools = model_body.tools if hasattr(model_body, "tools") else model_body["tools"]
        tool_names = {(t.get("name") if isinstance(t, dict) else t["name"]) for t in tools}
        assert tool_names == {"run_bash", "read_file"}

    def test_without_skill_omits_skill_system_message(self) -> None:
        fake = FakeServerClient()
        fake.queue("skill_workspace", "/seed_session", FakeResponse({"env_id": "E2"}))
        fake.queue("policy_model", "/v1/responses", FakeResponse(_message_response()))
        fake.queue("skill_judge", "/verify", FakeResponse(_verify_response(reward=0.5)))
        fake.queue("skill_workspace", "/close", FakeResponse({"message": "ok", "success": True}))

        agent = _make_agent(fake)
        client = TestClient(agent.setup_webserver())
        r = client.post("/run", json=_run_body(with_skill=False))
        assert r.status_code == 200

        model_body = fake.calls[1].json
        input_msgs = model_body.input if hasattr(model_body, "input") else model_body["input"]

        def _role(msg):
            return msg.role if hasattr(msg, "role") else msg["role"]

        assert all(_role(m) != "system" for m in input_msgs)

    def test_tool_call_is_dispatched_to_workspace_and_forwarded_to_judge(self) -> None:
        fake = FakeServerClient()
        fake.queue("skill_workspace", "/seed_session", FakeResponse({"env_id": "E3"}))
        fake.queue(
            "policy_model",
            "/v1/responses",
            FakeResponse(_tool_call_response("run_bash", json.dumps({"cmd": "ls"}))),
            FakeResponse(_message_response("done")),
        )
        fake.queue(
            "skill_workspace",
            "/run_bash",
            FakeResponse(
                {"stdout": "file.txt\n", "stderr": "", "exit_code": 0, "truncated": False, "timed_out": False}
            ),
        )
        fake.queue("skill_judge", "/verify", FakeResponse(_verify_response(reward=1.0)))
        fake.queue("skill_workspace", "/close", FakeResponse({"message": "ok", "success": True}))

        agent = _make_agent(fake)
        client = TestClient(agent.setup_webserver())
        r = client.post("/run", json=_run_body(with_skill=True))
        assert r.status_code == 200

        bash_call = next(c for c in fake.calls if c.url_path == "/run_bash")
        assert bash_call.json == {"env_id": "E3", "cmd": "ls"}

        verify_call = next(c for c in fake.calls if c.url_path == "/verify")
        tool_calls = verify_call.json["verifier_metadata"]["tool_calls"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "run_bash"
        assert tool_calls[0]["exit_code"] == 0
        assert "file.txt" in tool_calls[0]["stdout_snippet"]

    def test_workspace_error_recorded_and_close_still_runs(self) -> None:
        fake = FakeServerClient()
        fake.queue("skill_workspace", "/seed_session", FakeResponse({"env_id": "E4"}))
        fake.queue(
            "policy_model",
            "/v1/responses",
            FakeResponse(_tool_call_response("read_file", json.dumps({"path": "escapes/../etc"}))),
            FakeResponse(_message_response("fallback")),
        )
        fake.queue(
            "skill_workspace",
            "/read_file",
            FakeResponse({"detail": "Path escapes the workspace"}, ok=False, status=400),
        )
        fake.queue("skill_judge", "/verify", FakeResponse(_verify_response(reward=0.0)))
        fake.queue("skill_workspace", "/close", FakeResponse({"message": "ok", "success": True}))

        agent = _make_agent(fake)
        client = TestClient(agent.setup_webserver())
        r = client.post("/run", json=_run_body(with_skill=False))
        assert r.status_code == 200

        verify_call = next(c for c in fake.calls if c.url_path == "/verify")
        tool_calls = verify_call.json["verifier_metadata"]["tool_calls"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "read_file"
        assert "Path escapes" in tool_calls[0]["stderr_snippet"]

        # Close must still run even though a tool call errored mid-loop.
        assert any(c.url_path == "/close" for c in fake.calls)

    def test_close_runs_even_if_judge_raises(self) -> None:
        fake = FakeServerClient()
        fake.queue("skill_workspace", "/seed_session", FakeResponse({"env_id": "E5"}))
        fake.queue("policy_model", "/v1/responses", FakeResponse(_message_response()))
        fake.queue(
            "skill_judge",
            "/verify",
            FakeResponse({"detail": "judge boom"}, ok=False, status=500),
        )
        fake.queue("skill_workspace", "/close", FakeResponse({"message": "ok", "success": True}))

        agent = _make_agent(fake)
        client = TestClient(agent.setup_webserver())
        with pytest.raises(Exception):
            client.post("/run", json=_run_body(with_skill=False))

        assert any(c.url_path == "/close" for c in fake.calls)

    def test_missing_skill_metadata_raises(self) -> None:
        fake = FakeServerClient()
        agent = _make_agent(fake)
        client = TestClient(agent.setup_webserver())
        body = {"responses_create_params": {"input": [{"role": "user", "content": "x"}]}, "verifier_metadata": {}}
        with pytest.raises(Exception):
            client.post("/run", json=body)


class TestResponsesProxy:
    def test_responses_is_pure_model_proxy(self) -> None:
        fake = FakeServerClient()
        fake.queue("policy_model", "/v1/responses", FakeResponse(_message_response("hi")))

        agent = _make_agent(fake)
        client = TestClient(agent.setup_webserver())
        r = client.post("/v1/responses", json={"input": [{"role": "user", "content": "hello"}]})
        assert r.status_code == 200
        assert r.json()["output"][0]["content"][0]["text"] == "hi"

        assert [(c.server_name, c.url_path) for c in fake.calls] == [("policy_model", "/v1/responses")]


class TestToolCallLogEntry:
    def test_run_bash_payload_produces_stdout_and_exit_code(self) -> None:
        entry = _tool_call_log_entry(
            "run_bash",
            '{"cmd": "ls"}',
            {"stdout": "hello", "stderr": "", "exit_code": 0, "truncated": False},
        )
        assert entry["name"] == "run_bash"
        assert entry["exit_code"] == 0
        assert entry["stdout_snippet"] == "hello"

    def test_read_file_payload_uses_content_as_stdout(self) -> None:
        entry = _tool_call_log_entry("read_file", '{"path": "f"}', {"content": "file-body", "truncated": False})
        assert entry["name"] == "read_file"
        assert entry["stdout_snippet"] == "file-body"
        assert "exit_code" not in entry
