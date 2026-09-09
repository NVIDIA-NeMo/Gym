# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared host for running an unchanged harness process in a task sandbox."""

import asyncio
import json
import logging
import shlex
import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import HTTPException, Request
from pydantic import ConfigDict, PrivateAttr

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import Body, SimpleResponsesAPIAgent
from nemo_gym.global_config import get_first_server_config_dict
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import AgentObservationBundle, SandboxObservation
from nemo_gym.sandbox import AsyncSandbox, SandboxSpec, create_provider
from nemo_gym.sandbox.agent_dependencies import install_agent_dependencies
from nemo_gym.sandbox.config import resolve_provider_config
from nemo_gym.sandbox.workspace import SandboxWorkspace
from nemo_gym.server_utils import get_response_json, raise_for_status


LOG = logging.getLogger(__name__)


class RuntimeRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class RuntimeVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class SandboxedAgentHost(SimpleResponsesAPIAgent):
    """Own seed/execute/verify/cleanup while the harness owns only /responses.

    Constructed by SimpleResponsesAPIAgent.create_server(), never by a harness.
    Dependencies are provisioned automatically unless explicitly disabled.
    """

    harness_class: type
    _sem: asyncio.Semaphore = PrivateAttr()

    def model_post_init(self, context: Any) -> None:
        self._sem = asyncio.Semaphore(self.config.runtime.concurrency)

    def _harness_import(self) -> str:
        module = self.harness_class.__module__
        if module == "__main__":
            block = self.server_client.global_config_dict[self.config.name]["responses_api_agents"]
            implementation = next(iter(block))
            module = (
                f"responses_api_agents.{implementation}.{self.config.entrypoint.removesuffix('.py').replace('/', '.')}"
            )
        return f"{module}:{self.harness_class.__name__}"

    def _worker_payload(self, body: RuntimeRunRequest, cookies: dict) -> dict:
        runtime = self.config.runtime
        config = self.config.model_dump(mode="json") | runtime.agent_config
        config["runtime"] = {"type": "local"}
        config["token_id_capture"] = self._token_id_capture_enabled()
        # Only route information crosses into the agent; never the verifier config,
        # task answers, provider credentials, or the complete global configuration.
        global_config = {
            key: self.server_client.global_config_dict.get(key, False) for key in ("observability_enabled",)
        }
        global_config["token_id_capture"] = {
            "enabled": self._token_id_capture_enabled(),
            "all_agents": False,
        }

        def collect_refs(value):
            if isinstance(value, dict):
                if value.get("type") in {"responses_api_models", "resources_servers", "responses_api_agents"}:
                    name = value["name"]
                    source = get_first_server_config_dict(self.server_client.global_config_dict, name)
                    route = {"host": source["host"], "port": source["port"]}
                    if name in runtime.server_urls:
                        route["runtime_base_url"] = runtime.server_urls[name].rstrip("/")
                    global_config[name] = {value["type"]: {"route": route}}
                else:
                    for child in value.values():
                        collect_refs(child)
            elif isinstance(value, list):
                for child in value:
                    collect_refs(child)

        collect_refs(config)
        return {
            "harness": self._harness_import(),
            "config": config,
            "global_config": global_config,
            "body": body.responses_create_params.model_dump(mode="json"),
            "path": self.url_path_for_run("/v1/responses", body),
            "cookies": cookies,
        }

    async def _execute(self, sandbox: AsyncSandbox, body: RuntimeRunRequest, cookies: dict) -> dict:
        runtime = self.config.runtime
        for local_path, remote_path in runtime.uploads.items():
            await sandbox.upload(local_path, remote_path)
        if runtime.setup_command:
            setup = await sandbox.exec(runtime.setup_command, env=runtime.env, timeout_s=runtime.setup_timeout_s)
            if setup.error_type or setup.return_code != 0:
                raise RuntimeError(f"Agent runtime setup failed (exit={setup.return_code}, error={setup.error_type})")
        remote_dir = f"/tmp/nemo-gym-agent-{uuid4().hex}"
        created = await sandbox.exec(f"mkdir -m 700 {shlex.quote(remote_dir)}", timeout_s=runtime.setup_timeout_s)
        if created.error_type or created.return_code != 0:
            raise RuntimeError("Failed to create agent worker directory")
        with tempfile.TemporaryDirectory(prefix="nemo-gym-agent-") as temp:
            python, worker_env = runtime.python, runtime.env
            if runtime.dependencies.enabled:
                python, worker_env = await install_agent_dependencies(
                    sandbox, runtime, self._harness_import(), remote_dir, Path(temp)
                )
            payload_path = Path(temp) / "request.json"
            payload_path.write_text(json.dumps(self._worker_payload(body, cookies)))
            await sandbox.upload(payload_path, f"{remote_dir}/request.json")
            await sandbox.upload(Path(__file__).with_name("agent_runtime_worker.py"), f"{remote_dir}/worker.py")
            result = await sandbox.exec(
                shlex.join(
                    [
                        python,
                        f"{remote_dir}/worker.py",
                        f"{remote_dir}/request.json",
                        f"{remote_dir}/response.json",
                    ]
                ),
                timeout_s=runtime.timeout_s,
                env=worker_env,
            )
            if result.error_type or result.return_code != 0:
                raise RuntimeError(
                    f"Agent worker failed (exit={result.return_code}, error={result.error_type}): {(result.stderr or '')[-2000:]}"
                )
            output_path = Path(temp) / "response.json"
            await sandbox.download(f"{remote_dir}/response.json", output_path)
            return json.loads(output_path.read_text())

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        raise HTTPException(status_code=400, detail="Sandbox placement requires /run to establish a task workspace")

    async def run(self, request: Request, body: RuntimeRunRequest) -> RuntimeVerifyResponse:
        async with self._sem:
            resources = getattr(self.config, "resources_server", None)
            if resources is None:
                raise ValueError("Sandbox placement requires a resources_server reference")
            runtime = self.config.runtime
            cookies = dict(request.cookies)
            sandbox = None
            provider = None
            environment_workspace = False
            try:
                seed = await self.server_client.post(
                    server_name=resources.name,
                    url_path="/seed_session",
                    json=body.model_dump(),
                    cookies=cookies,
                )
                await raise_for_status(seed)
                cookies |= dict(seed.cookies)
                seeded = await get_response_json(seed)
                environment_workspace = bool(seeded.get("workspace"))
                if runtime.sandbox_source == "environment":
                    if not environment_workspace:
                        raise ValueError("runtime.sandbox_source=environment requires a workspace from /seed_session")
                    workspace = SandboxWorkspace.model_validate(seeded["workspace"])
                    provider = create_provider(
                        resolve_provider_config(workspace.provider, self.server_client.global_config_dict)
                    )
                    sandbox = await AsyncSandbox.connect(workspace.descriptor, provider=provider)
                else:
                    if environment_workspace:
                        raise ValueError(
                            "Environment supplied a task workspace; use runtime.sandbox_source=environment"
                        )
                    provider = create_provider(
                        resolve_provider_config(runtime.provider, self.server_client.global_config_dict)
                    )
                    sandbox = await AsyncSandbox(provider).start(SandboxSpec(**runtime.spec))
                output = await self._execute(sandbox, body, cookies)
                cookies |= output.get("cookies", {})
                raw_response = output["response"]
                observations = raw_response.pop("_ng_agent_observations", None)
                response = NeMoGymResponse.model_validate(raw_response)
                if self.config.skip_verification:
                    verified = body.model_dump() | {
                        "response": response,
                        "reward": self.config.skip_verification_reward,
                    }
                else:
                    verification = await self.server_client.post(
                        server_name=resources.name,
                        url_path="/verify",
                        json=body.model_dump() | {"response": response.model_dump(mode="json")},
                        cookies=cookies,
                    )
                    await raise_for_status(verification)
                    verified = await get_response_json(verification)
                if observations is not None:
                    bundle = AgentObservationBundle.model_validate(observations)
                    # Host execution is sandboxed even when the inner harness uses subprocesses.
                    bundle.gaps = [gap for gap in bundle.gaps if gap.code != "no_sandbox_runtime"]
                    bundle.records.append(
                        SandboxObservation(role="agent", provider=provider.name, outcome="completed")
                    )
                    verifier_observation = verified.pop("verifier_sandbox_observation", None)
                    if verifier_observation is not None:
                        bundle.records.append(SandboxObservation.model_validate(verifier_observation))
                    verified["ng_agent_observations"] = bundle
                return RuntimeVerifyResponse.model_validate(verified)
            finally:
                if environment_workspace:
                    try:
                        cleanup = await self.server_client.post(
                            server_name=resources.name,
                            url_path="/cleanup_session",
                            json={},
                            cookies=cookies,
                        )
                        await raise_for_status(cleanup)
                    except Exception:
                        LOG.exception("Failed to clean up environment workspace")
                elif sandbox is not None:
                    try:
                        await sandbox.stop()
                    except Exception:
                        LOG.exception("Failed to stop agent runtime")
                if provider is not None and (environment_workspace or sandbox is None):
                    try:
                        await provider.aclose()
                    except Exception:
                        LOG.exception("Failed to close agent runtime provider")
