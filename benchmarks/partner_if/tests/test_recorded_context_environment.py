# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real HTTP episodes use the stock Gymnasium agent and Resources-owned frozen tools.

Only semantic judgments and model replies are scripted. These tests exercise stopping and evidence binding;
they do not calibrate the semantic judge.
"""

import copy
import hashlib
import json
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from unittest.mock import patch
from uuid import uuid4

import aiohttp
import httpx
import pytest
from aiohttp import web
from omegaconf import OmegaConf
from pydantic import PrivateAttr

from benchmarks.partner_if.tests.fixture_data import BUNDLE, PROBE, SOURCE, TASK
from environment_servers.legacy_agent.app import LegacyAgentEnvironmentServer, LegacyAgentEnvironmentServerConfig
from environment_servers.partner_if.app import PartnerIFEnvironmentServer, PartnerIFEnvironmentServerConfig
from nemo_gym.client import ServerClient, get_response_json, raise_boundary_failure
from nemo_gym.config import BaseServerConfig
from resources_servers.gymnasium import GymnasiumServer
from resources_servers.partner_if.engine import aggregate
from resources_servers.partner_if.grading_contract import RawJudgment
from resources_servers.partner_if.grading_service import GradeResponse
from resources_servers.partner_if.recorded_context.assets import ReplayConfig, actor_identity
from resources_servers.partner_if.recorded_context_app import (
    RecordedContextResourcesServer,
    RecordedContextResourcesServerConfig,
)
from responses_api_agents.gymnasium_agent.app import GymnasiumAgent, GymnasiumAgentConfig


RESULT_KEY = "partner_if_recorded_context_result"
RESOURCE_NAME = "partner_if_recorded_context"


def call(name: str, arguments: dict, identity: str) -> dict:
    return {
        "type": "function_call",
        "id": "fc-" + identity,
        "call_id": identity,
        "name": name,
        "arguments": json.dumps(arguments),
        "status": "completed",
    }


def read_skill() -> dict:
    return call("Read", {"file_path": "/opt/agent/skills/registry-access/SKILL.md"}, "read")


def probe() -> dict:
    return call("Bash", {"command": f"python3 -I {PROBE} --uri '{BUNDLE['sibling']['input_uri']}'"}, "probe")


def prescription() -> dict:
    return call(
        "Write",
        {"file_path": "/workspace/logs/answer.txt", "content": "Publish a new artifact or choose another version."},
        "draft",
    )


def response(*calls: dict, content: str | None = None, **extras) -> dict:
    output = list(calls)
    if content is not None:
        output.insert(
            0,
            {
                "id": "message-" + uuid4().hex,
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": content, "annotations": []}],
            },
        )
    return {
        "id": "response-" + uuid4().hex,
        "object": "response",
        "created_at": 0,
        "model": "test",
        "status": "completed",
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "output": output,
        "usage": {
            "input_tokens": 100,
            "input_tokens_details": {"cached_tokens": None},
            "output_tokens": 20,
            "output_tokens_details": {"reasoning_tokens": None},
            "total_tokens": 120,
        },
    } | extras


def chat_response(*calls: dict, content: str | None = None) -> dict:
    return {
        "id": "chat-" + uuid4().hex,
        "model": "test",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [
                        {
                            "id": item["call_id"],
                            "type": "function",
                            "function": {"name": item["name"], "arguments": item["arguments"]},
                        }
                        for item in calls
                    ],
                },
                "finish_reason": "tool_calls" if calls else "stop",
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
    }


class LocalClient(ServerClient):
    _urls: dict[str, str] = PrivateAttr(default_factory=dict)

    def _resolve_base_url(self, server_name: str) -> str:
        return self._urls[server_name]


@dataclass
class RecordedContextPair:
    policy_responses: list[dict | Exception]
    fixture_passed: bool = True
    max_parent_turns: int = 6
    max_policy_calls: int = 8
    repeat_step: bool = False
    trace: list[dict] = field(default_factory=list)
    policy_requests: list[dict] = field(default_factory=list)
    grade_calls: list[dict] = field(default_factory=list)
    client: LocalClient | None = None
    agent: GymnasiumAgent | None = None
    resources: RecordedContextResourcesServer | None = None
    environment: LegacyAgentEnvironmentServer | None = None
    _asgi_clients: dict = field(default_factory=dict)

    async def _dispatch(self, service: str, request: web.Request) -> web.Response:
        payload = await request.json() if request.can_read_body else {}
        event = {"server": service, "path": request.path, "json": copy.deepcopy(payload)}
        self.trace.append(event)
        if service == "policy_model":
            self.policy_requests.append({"path": request.path, "json": copy.deepcopy(payload)})
            assert self.policy_responses, "Unexpected duplicate or post-decision actor request"
            raw = self.policy_responses.pop(0)
            if isinstance(raw, Exception):
                return web.json_response({"error": {"message": "Declared synthetic provider failure"}}, status=503)
            return web.json_response(copy.deepcopy(raw))
        downstream = self._asgi_clients[service]
        downstream.cookies.clear()
        forwarded = await downstream.request(
            request.method,
            request.path_qs,
            content=await request.read(),
            headers={key: value for key, value in request.headers.items() if key.lower() != "host"},
        )
        event["status"] = forwarded.status_code
        if forwarded.status_code == 200:
            event["reply"] = forwarded.json()
        if service == RESOURCE_NAME and request.path.endswith("/step") and self.repeat_step:
            self.repeat_step = False
            repeated = await downstream.request(
                request.method,
                request.path_qs,
                content=await request.read(),
                headers={key: value for key, value in request.headers.items() if key.lower() != "host"},
            )
            assert repeated.status_code == 200, repeated.text
            assert repeated.json() == forwarded.json()
            event["identical_retry_confirmed"] = True
        return web.Response(
            status=forwarded.status_code,
            body=forwarded.content,
            headers=[
                (key, value)
                for key, value in forwarded.headers.multi_items()
                if key.lower() not in {"content-length", "transfer-encoding", "content-encoding"}
            ],
        )

    async def _fixture_grade(self, contract, evidence, **kwargs):
        self.grade_calls.append({"task_id": contract.task_id, "evidence": evidence})
        return GradeResponse(
            grading_result=aggregate(
                contract,
                [
                    RawJudgment(
                        id=item.id,
                        passed=self.fixture_passed,
                        reason="Declared synthetic verdict for stopping tests",
                        evidence="This fixture does not calibrate the semantic judge",
                    )
                    for item in contract.checks
                ],
            )
        )

    @asynccontextmanager
    async def running(self) -> AsyncIterator["RecordedContextPair"]:
        async with AsyncExitStack() as stack:
            transport = await stack.enter_async_context(aiohttp.ClientSession())
            with (
                patch("nemo_gym.client.http_client.get_global_aiohttp_client", return_value=transport),
                patch("nemo_gym.client.http_client.get_global_aiohttp_control_client", return_value=transport),
                patch("resources_servers.partner_if.app.grade", side_effect=self._fixture_grade),
            ):
                urls = {}
                for service in ("agent", RESOURCE_NAME, "policy_model", "environment"):
                    app = web.Application()

                    async def dispatch(request, selected=service):
                        return await self._dispatch(selected, request)

                    app.router.add_route("*", "/{path:.*}", dispatch)
                    runner = web.AppRunner(app, access_log=None, handler_cancellation=True)
                    await runner.setup()
                    stack.push_async_callback(runner.cleanup)
                    site = web.TCPSite(runner, "127.0.0.1", 0)
                    await site.start()
                    urls[service] = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
                reference = {"type": "resources_servers", "name": RESOURCE_NAME}
                self.client = LocalClient(
                    head_server_config=BaseServerConfig(host="127.0.0.1", port=1),
                    global_config_dict=OmegaConf.create(
                        {
                            "observability_enabled": False,
                            "policy_model": {
                                "responses_api_models": {
                                    "vllm_model": {
                                        "entrypoint": "app.py",
                                        "host": "127.0.0.1",
                                        "port": int(urls["policy_model"].rsplit(":", 1)[1]),
                                    }
                                }
                            },
                            "agent": {
                                "responses_api_agents": {
                                    "gymnasium_agent": {"entrypoint": "app.py", "resources_server": reference}
                                }
                            },
                            RESOURCE_NAME: {
                                "resources_servers": {"partner_if": {"entrypoint": "recorded_context_app.py"}}
                            },
                            "environment": {
                                "environment_servers": {
                                    "legacy_agent": {
                                        "entrypoint": "app.py",
                                        "agent_server": {"type": "responses_api_agents", "name": "agent"},
                                    }
                                }
                            },
                        }
                    ),
                )
                self.client._urls = urls
                self.agent = GymnasiumAgent(
                    config=GymnasiumAgentConfig(
                        name="agent",
                        host="127.0.0.1",
                        port=1,
                        entrypoint="app.py",
                        resources_server=reference,
                        model_server={"type": "responses_api_models", "name": "policy_model"},
                        max_steps=self.max_parent_turns,
                        model_server_transport_max_attempts=1,
                    ),
                    server_client=self.client,
                )
                self.resources = RecordedContextResourcesServer(
                    config=RecordedContextResourcesServerConfig(
                        name=RESOURCE_NAME,
                        host="127.0.0.1",
                        port=1,
                        entrypoint="recorded_context_app.py",
                        policy_model_server={"type": "responses_api_models", "name": "policy_model"},
                        policy_model="test",
                        judge_model_server={"type": "responses_api_models", "name": "judge"},
                        accept_verification_input=True,
                        max_parent_turns=self.max_parent_turns,
                        replay=ReplayConfig(
                            package_dir=SOURCE / "assets",
                            coverage_profile="expanded-v4",
                            max_policy_calls=self.max_policy_calls,
                            max_delegate_turns=6,
                        ),
                    ),
                    server_client=self.client,
                )
                self.environment = LegacyAgentEnvironmentServer(
                    config=LegacyAgentEnvironmentServerConfig(
                        name="environment",
                        host="127.0.0.1",
                        port=1,
                        entrypoint="app.py",
                        default_episode_timeout_seconds=None,
                        agent_server={"type": "responses_api_agents", "name": "agent"},
                    ),
                    server_client=self.client,
                )
                apps = {
                    "agent": self.agent.setup_webserver(),
                    RESOURCE_NAME: self.resources.setup_webserver(),
                    "environment": self.environment.setup_webserver(),
                }
                for name, app in apps.items():
                    self._asgi_clients[name] = await stack.enter_async_context(
                        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=urls[name])
                    )
                yield self

    async def run(self, row: dict | None = None) -> tuple[dict, dict]:
        original = copy.deepcopy(TASK if row is None else row)
        original.update(_ng_task_index=27, _ng_rollout_index=3, _ng_attempt_index=2)
        result = await self.client.post(server_name="environment", url_path="/run", json=original)
        await raise_boundary_failure(result)
        return await get_response_json(result), original


def requests(pair: RecordedContextPair, service: str, suffix: str) -> list[dict]:
    return [event for event in pair.trace if event["server"] == service and event["path"].endswith(suffix)]


def canonical_result(reply: dict) -> dict:
    return reply["info"][RESULT_KEY]


def assert_stock_episode(pair: RecordedContextPair, reply: dict, original: dict) -> dict:
    assert type(pair.agent) is GymnasiumAgent
    assert type(pair.environment) is LegacyAgentEnvironmentServer
    assert isinstance(pair.resources, GymnasiumServer)
    assert len(requests(pair, "environment", "/run")) == len(requests(pair, "agent", "/run")) == 1
    resets = requests(pair, RESOURCE_NAME, "/reset")
    steps = requests(pair, RESOURCE_NAME, "/step")
    assert len(resets) == 1 and resets[0]["reply"]["observation"] is None
    assert len(requests(pair, RESOURCE_NAME, "/close")) == 1
    assert not any(
        event["path"].endswith(("/verify", "/replay/prefix-review", "/v1/agent_sessions")) for event in pair.trace
    )
    assert actor_identity(pair.policy_requests[0]["json"]) == actor_identity(original["responses_create_params"])
    result = canonical_result(reply)
    assert result == steps[-1]["reply"]["info"][RESULT_KEY]
    bound = {key: value for key, value in result.items() if key != "recorded_context_result_sha256"}
    assert (
        result["recorded_context_result_sha256"]
        == hashlib.sha256(
            json.dumps(bound, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
        ).hexdigest()
    )
    assert result["verifier_metadata"] == original["verifier_metadata"]
    assert actor_identity(result["responses_create_params"]) == actor_identity(original["responses_create_params"])
    assert all(event["json"].get("_ng_step_request_id") for event in steps)
    assert not pair.resources.session_state
    return result


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["pass", "fail", "same_batch", "turn_cap", "safe_final", "capture_gap"])
async def test_stock_agent_stops_at_the_resources_decision_and_preserves_the_frozen_input(mode):
    queue = {
        "pass": [response(read_skill()), response(probe())],
        "fail": [response(prescription())],
        "same_batch": [response(probe(), prescription())],
        "turn_cap": [response(read_skill())],
        "safe_final": [response(content="I am investigating.")],
        "capture_gap": [response(call("Read", {"file_path": "/missing"}, "missing"))],
    }[mode]
    pair = RecordedContextPair(
        policy_responses=queue,
        fixture_passed=mode not in {"fail", "same_batch"},
        max_parent_turns=1 if mode == "turn_cap" else 6,
    )
    frozen = copy.deepcopy(TASK)
    async with pair.running():
        reply, original = await pair.run()
    assert TASK == frozen
    result = assert_stock_episode(pair, reply, original)
    masked = mode in {"turn_cap", "safe_final", "capture_gap"}
    assert result["mask_sample"] is masked and reply["mask_sample"] is masked
    assert result["evaluation_completed"] is not masked
    assert result["reward"] == (1.0 if mode == "pass" else 0.0)
    assert reply["reward"] == result["reward"]
    assert len(pair.policy_requests) == (2 if mode == "pass" else 1)
    assert not pair.policy_responses
    if mode == "pass":
        pending = requests(pair, RESOURCE_NAME, "/step")[0]["reply"]
        assert pending["terminated"] is False and pending["truncated"] is False
        assert pending["reward"] == 0 and pending["mask_sample"] is False
        next_input = pair.policy_requests[1]["json"]["input"]
        assert (
            next_input[: len(frozen["responses_create_params"]["input"])] == pair.policy_requests[0]["json"]["input"]
        )
        assert any(item.get("type") == "function_call_output" and item["call_id"] == "read" for item in next_input)
        assert result["response"]["usage"]["total_tokens"] == 240
    calls = {item["call_id"] for item in result["response"]["output"] if item["type"] == "function_call"}
    outputs = {item["call_id"] for item in result["response"]["output"] if item["type"] == "function_call_output"}
    assert calls == outputs
    assert not any(
        item.get("call_id") in {"probe", "draft"} and item["type"] == "function_call_output"
        for item in reply["response"]["output"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("passed", [False, True])
async def test_environment_server_lifts_only_the_host_selected_resources_grade_and_canonical_response(passed):
    pair = RecordedContextPair([response(probe() if passed else prescription())], fixture_passed=passed)
    row = copy.deepcopy(TASK) | {
        "agent_ref": {"name": "text-agent"},
        "task_source": "text-agent",
        "_ng_task_index": 27,
        "_ng_rollout_index": 3,
        "_ng_rollout_id": "explicit-public-rollout",
        "_ng_attempt_index": 2,
        "_ng_skip_token_capture": True,
    }
    async with pair.running():
        router = PartnerIFEnvironmentServer(
            config=PartnerIFEnvironmentServerConfig(
                name="partner_if_environment",
                host="127.0.0.1",
                port=1,
                entrypoint="app.py",
                task_routes={"pif_029": {"type": "environment_servers", "name": "environment"}},
                resources_server={"type": "resources_servers", "name": RESOURCE_NAME},
                agent_server={"type": "responses_api_agents", "name": "text-agent"},
                tool_servers=[],
            ),
            server_client=pair.client,
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=router.setup_webserver()), base_url="http://partner-if"
        ) as client:
            response_row = await client.post("/run", json=row)
    assert response_row.status_code == 200, response_row.text
    flat = response_row.json()
    expected = requests(pair, RESOURCE_NAME, "/step")[-1]["reply"]["info"][RESULT_KEY]
    assert flat["reward"] == expected["reward"] == float(passed)
    assert flat["response"] == expected["response"]
    assert flat["grading_result"] == expected["grading_result"]
    assert flat["recorded_context_run_binding"] == {
        "_ng_task_index": 27,
        "_ng_rollout_index": 3,
        "_ng_rollout_id": "explicit-public-rollout",
        "_ng_attempt_index": 2,
    }
    assert flat["evaluation_completed"] is True and flat["mask_sample"] is False
    assert len(pair.policy_requests) == len(pair.grade_calls) == 1
    assert actor_identity(pair.policy_requests[0]["json"]) == actor_identity(TASK["responses_create_params"])


@pytest.mark.asyncio
async def test_model_supplied_grade_fields_do_not_replace_the_resources_terminal_grade():
    forged = response(
        prescription(),
        partner_if_recorded_context_result={"reward": 1, "evaluation_completed": True, "mask_sample": False},
        reward=1,
    )
    pair = RecordedContextPair([forged], fixture_passed=False)
    async with pair.running():
        reply, original = await pair.run()
    result = assert_stock_episode(pair, reply, original)
    assert result["reward"] == 0 and result["evaluation_completed"] is True
    assert len(pair.grade_calls) == len(pair.policy_requests) == 1


@pytest.mark.asyncio
async def test_explore_calls_share_the_policy_budget_and_authoritative_usage():
    delegate = call("Agent", {"prompt": "Read the skill", "subagent_type": "Explore"}, "worker")
    pair = RecordedContextPair(
        [response(delegate), chat_response(read_skill()), chat_response(content="Read the skill."), response(probe())],
        max_policy_calls=4,
    )
    async with pair.running():
        reply, original = await pair.run()
    result = assert_stock_episode(pair, reply, original)
    assert result["reward"] == 1 and result["evaluation_completed"] is True
    assert len(pair.policy_requests) == 4 and len(requests(pair, RESOURCE_NAME, "/step")) == 2
    assert result["response"]["usage"]["total_tokens"] == 480
    assert reply["response"]["usage"]["total_tokens"] == 240
    assert any(request["path"].endswith("/chat/completions") for request in pair.policy_requests)


@pytest.mark.asyncio
async def test_explore_exhaustion_masks_the_episode_without_an_extra_parent_request():
    delegate = call("Agent", {"prompt": "Read the skill", "subagent_type": "Explore"}, "worker")
    pair = RecordedContextPair([response(delegate), chat_response(read_skill())], max_policy_calls=2)
    async with pair.running():
        reply, original = await pair.run()
    result = assert_stock_episode(pair, reply, original)
    assert result["mask_sample"] is True and result["evaluation_completed"] is False
    assert len(pair.policy_requests) == 2 and len(requests(pair, RESOURCE_NAME, "/step")) == 1
    assert not pair.policy_responses


@pytest.mark.asyncio
async def test_identical_step_retry_reuses_the_terminal_grade_and_tool_execution():
    pair = RecordedContextPair([response(probe())], repeat_step=True)
    async with pair.running():
        reply, original = await pair.run()
    result = assert_stock_episode(pair, reply, original)
    assert result["reward"] == 1 and result["evaluation_completed"] is True
    assert requests(pair, RESOURCE_NAME, "/step")[0]["identical_retry_confirmed"] is True
    assert len(pair.policy_requests) == len(pair.grade_calls) == 1


@pytest.mark.asyncio
async def test_failed_explore_dispatch_keeps_total_usage_unknown_and_is_not_retried():
    delegate = call("Agent", {"prompt": "Read the skill", "subagent_type": "Explore"}, "worker")
    pair = RecordedContextPair([response(delegate), TimeoutError("Declared synthetic worker failure")])
    async with pair.running():
        reply, original = await pair.run()
    result = assert_stock_episode(pair, reply, original)
    assert result["mask_sample"] is True and result["evaluation_completed"] is False
    assert result["response"]["usage"] is None
    assert len(pair.policy_requests) == 2
    assert len(requests(pair, RESOURCE_NAME, "/step")) == 1
