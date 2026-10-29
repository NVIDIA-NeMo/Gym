# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The BIRD-Interact Environment Server against the real Resources Server, with a scripted agent."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from aiohttp import ClientResponseError, RequestInfo
from fastapi.testclient import TestClient
from multidict import CIMultiDict, CIMultiDictProxy
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, ValidationError
from yarl import URL

from environment_servers.bird_interact.app import (
    BirdInteractEnvironmentServer,
    BirdInteractEnvironmentServerConfig,
    BirdInteractRequest,
    extract_sql,
)
from nemo_gym.base_responses_api_agent import AgentSeedSessionRequest
from nemo_gym.config_types import AgentServerRef, ModelServerRef, ResourcesServerRef
from nemo_gym.episode_types import EpisodeId, MaterializedTask, TaskId
from nemo_gym.mcp_auto_exposure import maybe_auto_expose
from nemo_gym.server_utils import BaseServerConfig, ServerClient
from nemo_gym.tool_access import MCPToolAccess
from resources_servers.bird_interact.app import BirdInteractResourcesServer, BirdInteractResourcesServerConfig
from resources_servers.bird_interact.task_data import BirdInteractTask


DATA = Path(__file__).parents[2].parent / "resources_servers/bird_interact/data"
TASKS = {row["instance_id"]: row for row in map(json.loads, (DATA / "example.jsonl").read_text().splitlines())}
# The Resources Server's session sandboxes run on the host, in each test's own workspace.
LOCAL_SANDBOX: dict[str, dict[str, str]] = {"local": {}}


@pytest.fixture(autouse=True)
def sandbox_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(LOCAL_SANDBOX["local"], "workspace_root", str(tmp_path / "sandboxes"))


def _message(text: str) -> dict:
    return {
        "id": f"response-{abs(hash(text))}",
        "created_at": 0,
        "model": "model",
        "object": "response",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "id": "message",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "tools": [],
    }


class _Cookie:
    def __init__(self, value: str) -> None:
        self.value = value


class _Content:
    def __init__(self, body: bytes) -> None:
        self.body = body

    async def read(self) -> bytes:
        return self.body


class _Reply:
    """An aiohttp-shaped response."""

    def __init__(self, status: int, body: bytes, cookies: dict[str, str]) -> None:
        self.status, self.ok, self.body = status, status < 400, body
        self.content = _Content(body)
        self.cookies = {name: _Cookie(value) for name, value in cookies.items()}
        self.headers: dict[str, str] = {}

    async def read(self) -> bytes:
        return self.body

    def raise_for_status(self) -> None:
        if not self.ok:
            url = URL("http://test")
            info = RequestInfo(url, "POST", CIMultiDictProxy(CIMultiDict()), url)
            raise ClientResponseError(request_info=info, history=(), status=self.status, message="error")


def _from_httpx(response: httpx.Response) -> _Reply:
    return _Reply(response.status_code, response.content, dict(response.cookies))


AgentTurn = Callable[[dict, TestClient], str | None]


class _Client(ServerClient):
    """Routes the Resources Server's calls into its app and the agent's into a script."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    resources: TestClient
    agent_turns: list[AgentTurn]
    agent_calls: list[tuple[str, dict]] = []

    async def post(self, server_name: str, url_path: str, **kwargs: Any) -> _Reply:
        body = kwargs.get("json")
        if isinstance(body, BaseModel):
            body = json.loads(json.dumps(body.model_dump(mode="json", exclude_unset=True)))
        if server_name == "bird_interact":
            return _from_httpx(self.resources.post(url_path, json=body))
        self.agent_calls.append((url_path, body))
        if url_path == "/v1/agent_sessions":
            reply = {"agent_session_id": body["agent_session_id"]}
            if "continuation" in body:
                reply["capabilities"] = {"mode": "resume_harness_session", "observations": ["trajectory", "timing"]}
            return _Reply(200, json.dumps(reply).encode(), {"agent": "cookie"})
        if url_path == "/v1/agent_sessions/close":
            return _Reply(200, json.dumps({"agent_session_id": body["agent_session_id"]}).encode(), {})
        text = self.agent_turns.pop(0)(body, self.resources)
        return _Reply(200, json.dumps(_message(text or "Done.")).encode(), {})

    def _resolve_base_url(self, server_name: str) -> str:
        return f"http://{server_name}:8000"


def _says(text: str) -> AgentTurn:
    return lambda body, resources: text


def _sql(sql: str) -> AgentTurn:
    return _says(f"Here is my query:\n```sqlite\n{sql}\n```")


def _servers(
    mode: str,
    agent_turns: list[AgentTurn],
    *,
    max_turns: int,
    user_model_replies: list[str] = (),
    transports: tuple[str, ...] = ("direct_http",),
    expose_tools_over_mcp: bool = False,
):
    resources_client = MagicMock(spec=ServerClient)
    resources_client.global_config_dict = {}
    resources_client.post = AsyncMock(
        side_effect=[_Reply(200, json.dumps(_message(text)).encode(), {}) for text in user_model_replies]
    )
    resources = BirdInteractResourcesServer(
        config=BirdInteractResourcesServerConfig(
            name="bird_interact",
            host="resources",
            port=8000,
            entrypoint="app.py",
            database_dir=str(DATA / "example_databases"),
            user_model=ModelServerRef(type="responses_api_models", name="user_model"),
            sandbox_provider=LOCAL_SANDBOX,
            expose_tools_over_mcp=expose_tools_over_mcp,
        ),
        server_client=resources_client,
    )
    resources_app = resources.setup_webserver()
    maybe_auto_expose(resources, resources_app)
    client = _Client(
        head_server_config=BaseServerConfig(host="head", port=1),
        global_config_dict=OmegaConf.create(
            {"agent": {"responses_api_agents": {"simple_agent": {"host": "agent", "port": 1, "entrypoint": "app.py"}}}}
        ),
        resources=TestClient(resources_app),
        agent_turns=agent_turns,
        agent_calls=[],
    )
    environment = BirdInteractEnvironmentServer(
        config=BirdInteractEnvironmentServerConfig(
            name="environment",
            host="environment",
            port=8001,
            entrypoint="app.py",
            agent_server=AgentServerRef(type="responses_api_agents", name="agent"),
            resources_server=ResourcesServerRef(type="resources_servers", name="bird_interact"),
            mode=mode,
            resources_tool_transports=list(transports),
            max_turns=max_turns,
            turn_timeout_seconds=30,
            cleanup_timeout_seconds=5,
        ),
        server_client=client,
    )
    return environment, client


def _request(task: str, **overrides: Any) -> BirdInteractRequest:
    return BirdInteractRequest(
        episode_id=EpisodeId(rollout_id=f"rollout-{task}"),
        task=MaterializedTask(
            task_id=TaskId(taskset="bird_interact:example", task_id=task),
            task_input=BirdInteractTask.model_validate(TASKS[task] | overrides),
        ),
    )


def _activation_inputs(client: _Client) -> list[list[dict]]:
    return [body["input"] for path, body in client.agent_calls if path.endswith("/v1/responses")]


class TestConversationalMode:
    async def test_a_question_a_failed_attempt_and_a_debugged_answer(self) -> None:
        environment, client = _servers(
            "c_interact",
            [
                _says("What do you mean by bestsellers?"),
                _sql("SELECT name FROM authors"),
                _sql(TASKS["bookshop_1"]["sol_sql"][0]),
            ],
            max_turns=10,
            user_model_replies=['labeled("bestsellers")', "Books that sold at least 10 copies in total."],
        )

        response = await environment.run_request(_request("bookshop_1"))

        assert response.failure is None
        result = response.result
        assert (result.reward, result.stop_reason, len(result.turns)) == (0.5, "solved", 3)
        assert result.phases[0].attempts == 2
        assert result.questions_asked == 1
        opening, answer, debug = _activation_inputs(client)
        assert [message["role"] for message in opening] == ["system", "user"]
        assert "at most 5 in total" in opening[0]["content"]
        assert "Bestseller" not in opening[1]["content"]  # The masked knowledge entry stays hidden.
        assert answer == [{"role": "user", "content": "Books that sold at least 10 copies in total."}]
        assert "one more attempt" in debug[0]["content"]
        seed = AgentSeedSessionRequest.model_validate(client.agent_calls[0][1])
        assert seed.continuation is not None and seed.tool_accesses == []

    async def test_the_follow_up_question_is_the_next_user_turn(self) -> None:
        follow_up = TASKS["bookshop_3"]["follow_up"]
        environment, client = _servers(
            "c_interact",
            [_sql(TASKS["bookshop_3"]["sol_sql"][0]), _sql(follow_up["sol_sql"][0])],
            max_turns=10,
        )

        response = await environment.run_request(_request("bookshop_3"))

        assert response.result.reward == pytest.approx(1.0)
        assert follow_up["query"] in _activation_inputs(client)[1][0]["content"]
        assert [phase.phase for phase in response.result.phases] == [1, 2]

    async def test_a_question_after_the_budget_ends_the_episode_without_a_submission(self) -> None:
        questions = [_says(f"Question {index}?") for index in range(6)]
        environment, _ = _servers("c_interact", questions, max_turns=10, user_model_replies=["unanswerable()"] * 5)

        response = await environment.run_request(_request("bookshop_1"))

        assert (response.result.stop_reason, response.result.reward, len(response.result.turns)) == (
            "no_sql_submitted",
            0.0,
            6,
        )

    async def test_a_question_in_the_debug_turn_ends_the_episode(self) -> None:
        environment, _ = _servers(
            "c_interact", [_sql("SELECT name FROM authors"), _says("What should change?")], max_turns=10
        )

        response = await environment.run_request(_request("bookshop_1"))

        result = response.result
        assert (result.stop_reason, result.reward, result.questions_asked) == ("no_sql_submitted", 0.0, 0)
        assert result.phases[0].attempts == 1

    async def test_a_task_without_ground_truth_fails_at_seed_for_good(self) -> None:
        environment, client = _servers("c_interact", [], max_turns=10)

        response = await environment.run_request(_request("bookshop_1", sol_sql=[]))

        assert (response.failure.stage, response.failure.terminal) == ("seed", True)
        assert "has no sol_sql" in response.failure.failure_reason
        assert client.agent_calls == []


def _mcp(resources: TestClient, access: MCPToolAccess, method: str, params: dict) -> dict:
    headers = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
    body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    reply = resources.post(
        URL(str(access.connection.url)).path, headers=headers | access.connection.headers, json=body
    )
    return reply.json()["result"]


class TestAgenticMode:
    async def test_the_agent_works_through_tools_in_one_turn(self) -> None:
        def agent_uses_tools(body: dict, resources: TestClient) -> str:
            resources.post("/get_all_knowledge_definitions")
            resources.post("/submit", json={"sql": TASKS["bookshop_1"]["sol_sql"][0]})
            return "Submitted."

        environment, client = _servers("a_interact", [agent_uses_tools], max_turns=1)

        response = await environment.run_request(_request("bookshop_1"))

        assert (response.result.reward, response.result.stop_reason) == (0.7, "agent_finished")
        assert response.result.coins_spent == 4.0
        seed = AgentSeedSessionRequest.model_validate(client.agent_calls[0][1])
        assert seed.continuation is None
        assert str(seed.tool_accesses[0].base_url) == "http://bird_interact:8000/"
        opening = next(body for path, body in client.agent_calls if path.endswith("/v1/responses"))
        assert len(opening["tools"]) == 9

    async def test_over_mcp_the_agent_lists_the_tools_from_the_resources_server(self) -> None:
        listed: list[str] = []

        def agent_uses_mcp(body: dict, resources: TestClient) -> str:
            (access,) = AgentSeedSessionRequest.model_validate(client.agent_calls[0][1]).tool_accesses
            listed.extend(tool["name"] for tool in _mcp(resources, access, "tools/list", {})["tools"])
            submit = {"name": "submit", "arguments": {"sql": TASKS["bookshop_1"]["sol_sql"][0]}}
            _mcp(resources, access, "tools/call", submit)
            return "Submitted."

        environment, client = _servers(
            "a_interact", [agent_uses_mcp], max_turns=1, transports=("mcp",), expose_tools_over_mcp=True
        )

        with client.resources:
            response = await environment.run_request(_request("bookshop_1"))

        assert (response.result.reward, response.result.coins_spent) == (0.7, 3.0)
        (access,) = AgentSeedSessionRequest.model_validate(client.agent_calls[0][1]).tool_accesses
        assert (access.kind, str(access.connection.url)) == ("mcp", "http://bird_interact:8000/mcp")
        assert len(listed) == 9
        # The tools come from the MCP server; Codex refuses a request that also declares them.
        opening = next(body for path, body in client.agent_calls if path.endswith("/v1/responses"))
        assert opening["tools"] == []

    async def test_mcp_without_exposure_on_the_resources_server_fails_at_seed_for_good(self) -> None:
        environment, client = _servers("a_interact", [], max_turns=1, transports=("mcp",))

        response = await environment.run_request(_request("bookshop_1"))

        assert (response.failure.stage, response.failure.terminal) == ("seed", True)
        assert "expose_tools_over_mcp" in response.failure.failure_reason
        assert client.agent_calls == []

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [({"max_turns": 2}, "max_turns to 1"), ({"resources_tool_transports": []}, "resources_tool_transports")],
    )
    def test_a_interact_runs_in_one_turn_with_a_tool_transport(self, overrides: dict, message: str) -> None:
        with pytest.raises(ValidationError, match=message):
            BirdInteractEnvironmentServerConfig(
                **{
                    "name": "environment",
                    "host": "environment",
                    "port": 8001,
                    "entrypoint": "app.py",
                    "agent_server": AgentServerRef(type="responses_api_agents", name="agent"),
                    "resources_server": ResourcesServerRef(type="resources_servers", name="bird_interact"),
                    "mode": "a_interact",
                    "max_turns": 1,
                    "turn_timeout_seconds": 30,
                    "cleanup_timeout_seconds": 5,
                }
                | overrides
            )


@pytest.mark.parametrize(
    ("reply", "sql"),
    [
        ("```sqlite\nSELECT 1\n```", "SELECT 1"),
        ("first\n```sql\nSELECT 1\n```\nthen\n```SQL\nSELECT 2;\n```", "SELECT 2;"),
        ("Which year do you mean?", None),
        ("```python\nprint(1)\n```", None),
    ],
)
def test_only_a_fenced_sql_block_is_a_submission(reply: str, sql: str | None) -> None:
    assert extract_sql(reply) == sql
