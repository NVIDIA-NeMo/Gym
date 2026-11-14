# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise real replay/verifier/environment boundaries; scripted grades are not semantic calibration."""

import copy
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

import aiohttp
import httpx
from aiohttp import web
from omegaconf import OmegaConf
from pydantic import PrivateAttr

from benchmarks.partner_if.tests.test_replay import SOURCE, TASK
from environment_servers.single_agent_turn.app import (
    SingleAgentTurnEnvironmentServer,
    SingleAgentTurnEnvironmentServerConfig,
)
from nemo_gym.client import ServerClient, get_response_json, raise_boundary_failure
from nemo_gym.config import BaseServerConfig
from nemo_gym.data import materialize_task
from nemo_gym.server_runtime import SingleAgentTurnRequest, SingleAgentTurnResponse
from resources_servers.partner_if.app import PartnerIFResourcesServer, PartnerIFResourcesServerConfig
from resources_servers.partner_if.engine import aggregate
from resources_servers.partner_if.grading_contract import RawJudgment
from resources_servers.partner_if.grading_service import GradeResponse
from responses_api_agents.ngc_recorded_context.native_agent import NativeReplayAgent, NativeReplayAgentConfig
from responses_api_agents.ngc_recorded_context.replay import ReplayConfig


class LocalClient(ServerClient):
    """Retain the normal Gym HTTP transport while resolving only this test's local services."""

    _urls: dict[str, str] = PrivateAttr(default_factory=dict)

    def _resolve_base_url(self, server_name: str) -> str:
        return self._urls[server_name]


@dataclass
class GenericReplayPair:
    """A test-only router shared with the opt-in external real-model smoke runner."""

    directory: Path
    model: str = "test"
    max_activations: int = 6
    policy_responses: list[dict | Exception] | None = None
    fixture_passed: bool = True
    credentials: dict[str, str] | None = None
    trace: list[dict] = field(default_factory=list)
    policy_requests: list[dict] = field(default_factory=list)
    client: LocalClient | None = None
    agent: NativeReplayAgent | None = None
    resources: PartnerIFResourcesServer | None = None
    environment: SingleAgentTurnEnvironmentServer | None = None
    _apps: dict = field(default_factory=dict)
    _asgi_clients: dict = field(default_factory=dict)

    async def _dispatch(self, service: str, request: web.Request) -> web.Response:
        payload = await request.json() if request.can_read_body else {}
        event = {"server": service, "path": request.path, "json": copy.deepcopy(payload)}
        self.trace.append(event)
        if service == "partner_if" and request.path.endswith("/verify"):
            assert self.agent is not None
            assert self.agent._session_records
            assert all(
                record.state is None and record.close_response.cleanup_confirmed
                for record in self.agent._session_records.values()
            ), "The verifier must see a closed, finalized agent capture"
        if service == "policy_model":
            self.policy_requests.append(copy.deepcopy(payload))
            if self.policy_responses is not None:
                assert self.policy_responses, "Unexpected duplicate or post-decision model request"
                raw = self.policy_responses.pop(0)
                if isinstance(raw, Exception):
                    return web.json_response({"error": {"message": "scripted provider failure"}}, status=503)
                raw = copy.deepcopy(raw)
                raw["model"] = self.model
                raw["usage"] = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
                return web.json_response(raw)
        # The proxy must not invent cookie state beyond Gym's explicit session headers.
        self._asgi_clients[service].cookies.clear()
        forwarded = await self._asgi_clients[service].request(
            request.method,
            request.path_qs,
            content=await request.read(),
            headers={key: value for key, value in request.headers.items() if key.lower() != "host"},
        )
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
        """Only the semantic verdict is scripted; capture validation and ordering decisions remain real."""
        return GradeResponse(
            grading_result=aggregate(
                contract,
                [
                    RawJudgment(
                        id=item.id,
                        passed=self.fixture_passed,
                        reason="Declared synthetic fixture verdict",
                        evidence="This unit fixture does not calibrate the semantic judge",
                    )
                    for item in contract.checks
                ],
            )
        )

    @asynccontextmanager
    async def running(self):
        async with AsyncExitStack() as stack:
            transport = await stack.enter_async_context(aiohttp.ClientSession())
            with (
                patch("nemo_gym.client.http_client.get_global_aiohttp_client", return_value=transport),
                patch("nemo_gym.client.http_client.get_global_aiohttp_control_client", return_value=transport),
            ):
                urls = {}
                services = ["agent", "partner_if", "policy_model", "environment"]
                if self.credentials is not None:
                    services.append("judge")
                for service in services:
                    app = web.Application()

                    async def dispatch(request, selected=service):
                        return await self._dispatch(selected, request)

                    app.router.add_route("*", "/{path:.*}", dispatch)
                    runner = web.AppRunner(app, access_log=None)
                    await runner.setup()
                    stack.push_async_callback(runner.cleanup)
                    site = web.TCPSite(runner, "127.0.0.1", 0)
                    await site.start()
                    port = site._server.sockets[0].getsockname()[1]
                    urls[service] = f"http://127.0.0.1:{port}"
                config = {
                    "observability_enabled": self.credentials is not None,
                    "model_call_capture_dir": str((self.directory / "model_calls").resolve()),
                    "mlflow_tracking_required": False,
                    "mlflow_tracking_uri": None,
                    "mlflow_tracking_token": None,
                    "agent": {
                        "responses_api_agents": {
                            "ngc_recorded_context": {"entrypoint": "native_agent.py", "tool_servers": []}
                        }
                    },
                    "partner_if": {"resources_servers": {"partner_if": {"entrypoint": "app.py"}}},
                    "environment": {
                        "environment_servers": {
                            "single_agent_turn": {
                                "entrypoint": "app.py",
                                "agent_server": {"type": "responses_api_agents", "name": "agent"},
                                "resources_server": {"type": "resources_servers", "name": "partner_if"},
                            }
                        }
                    },
                }
                for name in ("policy_model", "judge"):
                    if name in urls:
                        config[name] = {
                            "responses_api_models": {
                                "vllm_model": {
                                    "entrypoint": "app.py",
                                    "host": "127.0.0.1",
                                    "port": int(urls[name].rsplit(":", 1)[1]),
                                }
                            }
                        }
                self.client = LocalClient(
                    head_server_config=BaseServerConfig(host="127.0.0.1", port=1),
                    global_config_dict=OmegaConf.create(config),
                )
                self.client._urls = urls
                self.agent = NativeReplayAgent(
                    config=NativeReplayAgentConfig(
                        name="agent",
                        host="127.0.0.1",
                        port=1,
                        entrypoint="native_agent.py",
                        model=self.model,
                        model_server={"type": "responses_api_models", "name": "policy_model"},
                        replay=ReplayConfig(
                            package_dir=SOURCE / "assets",
                            artifacts_dir=self.directory / "captures",
                            coverage_profile="expanded-v4",
                            max_new_turns=6,
                            max_policy_calls=8,
                            gym_reasoning_decode=True,
                        ),
                    ),
                    server_client=self.client,
                )
                self.resources = PartnerIFResourcesServer(
                    config=PartnerIFResourcesServerConfig(
                        name="partner_if",
                        host="127.0.0.1",
                        port=1,
                        entrypoint="app.py",
                        judge_model_server={"type": "responses_api_models", "name": "judge"},
                        accept_verification_input=True,
                        ordering_artifacts_dir=self.directory / "captures",
                        judge_max_concurrency=1,
                        judge_responses_create_params={"input": [], "max_output_tokens": 32768, "temperature": 0.0},
                    ),
                    server_client=self.client,
                )
                self.environment = SingleAgentTurnEnvironmentServer(
                    config=SingleAgentTurnEnvironmentServerConfig(
                        name="generic_replay_environment",
                        host="127.0.0.1",
                        port=1,
                        entrypoint="app.py",
                        agent_server={"type": "responses_api_agents", "name": "agent"},
                        resources_server={"type": "resources_servers", "name": "partner_if"},
                        resources_tool_transports=[],
                        tool_servers=[],
                        forward_agent_evidence=True,
                        verifier_continuation={"max_activations": self.max_activations},
                        default_episode_timeout_seconds=900,
                        cleanup_timeout_seconds=5,
                    ),
                    server_client=self.client,
                )
                self._apps = {
                    "agent": self.agent.setup_webserver(),
                    "partner_if": self.resources.setup_webserver(),
                    "environment": self.environment.setup_webserver(),
                }
                if self.credentials is not None:
                    from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig

                    for name, model, key in [
                        ("policy_model", self.model, self.credentials["policy_api_key"]),
                        ("judge", self.credentials["judge_model_name"], self.credentials["judge_api_key"]),
                    ]:
                        model_server = VLLMModel(
                            config=VLLMModelConfig(
                                name=name,
                                host="127.0.0.1",
                                port=1,
                                entrypoint="app.py",
                                model=model,
                                base_url=self.credentials[
                                    "policy_base_url" if name == "policy_model" else "judge_base_url"
                                ],
                                api_key=key,
                                return_token_id_information=False,
                                uses_reasoning_parser=True,
                                uses_interleaved_reasoning=True,
                                max_num_tries=1,
                                max_http_attempts=1,
                                request_timeout_seconds=180,
                                sampling_overrides={"temperature": 0.7, "top_p": 0.95, "max_tokens": 8192}
                                if name == "policy_model"
                                else None,
                            ),
                            server_client=self.client,
                        )
                        self._apps[name] = model_server.setup_webserver()
                for name, app in self._apps.items():
                    self._asgi_clients[name] = await stack.enter_async_context(
                        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=urls[name])
                    )
                if self.policy_responses is not None:
                    with patch("resources_servers.partner_if.app.grade", side_effect=self._fixture_grade):
                        yield self
                else:
                    yield self

    async def run(self, *, task_name="pif_029", positional_identity=False):
        task = copy.deepcopy(TASK)
        task["task_name"] = task_name
        materialized = (
            materialize_task(task, taskset="test", task_index=27)
            if positional_identity
            else {"task_id": {"taskset": "test", "task_id": "pif_029"}, "task_input": task}
        )
        request = SingleAgentTurnRequest.model_validate(
            {"episode_id": {"rollout_id": "generic-replay", "attempt": 1}, "task": materialized}
        )
        result = await self.client.post(server_name="environment", url_path="/run", json=request)
        await raise_boundary_failure(result)
        reply = SingleAgentTurnResponse.model_validate(await get_response_json(result))
        return reply.model_dump(mode="json"), request
