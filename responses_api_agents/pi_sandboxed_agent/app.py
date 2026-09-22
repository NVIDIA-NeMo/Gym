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

"""Run the native Pi CLI in a provider-managed sandbox; parse and grade on the host."""

import json
import logging
from asyncio import Semaphore
from contextvars import ContextVar
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from shlex import join, quote
from tempfile import TemporaryDirectory
from time import time
from typing import Any, Literal
from uuid import uuid4

from fastapi import Request
from pydantic import Field

from nemo_gym.base_responses_api_agent import Body, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import AgentInvocation, SandboxObservation, ToolCallObservation
from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec, create_provider
from nemo_gym.sandbox.agent_tools import restricted_network_policy, sandbox_server_url, seed_mcp_servers
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.sandbox.utils import cpu_cap_env
from nemo_gym.server_utils import get_response_json, is_nemo_gym_fastapi_entrypoint, raise_for_status
from responses_api_agents.pi_agent.app import (
    PiAgent,
    PiAgentConfig,
    PiAgentRunRequest,
    PiAgentVerifyResponse,
    parse_pi_events,
)


LOG = logging.getLogger(__name__)
_RUN: ContextVar[dict[str, Any] | None] = ContextVar("pi_sandboxed_run", default=None)


class PiSandboxedAgentConfig(PiAgentConfig):
    """Pi CLI settings plus provider allocation and host-side tool services."""

    model_server: ModelServerRef
    sandbox_provider: str
    sandbox_config: dict[str, Any]
    network_access: Literal["inherit", "model_only", "model_and_tools"] = "inherit"
    tool_servers: list[ResourcesServerRef] = Field(default_factory=list)
    execution_failure_reward_zero: bool = False
    artifacts_dir: str = "results/pi_sandboxed"


class PiSandboxedAgentVerifyResponse(PiAgentVerifyResponse):
    pi_failed: bool = False
    pi_exit_code: int | None = None
    pi_error_type: str | None = None
    pi_results_dir: str = ""


class PiSandboxedAgent(PiAgent):
    config: PiSandboxedAgentConfig

    def model_post_init(self, __context: Any) -> None:
        # Pi is preinstalled in the sandbox image, never installed on the Gym host.
        SimpleResponsesAPIAgent.model_post_init(self, __context)
        self.sem = Semaphore(self.config.concurrency)

    def _resolve_model_base_url(self, rollout_id: str | None = None) -> str:
        context = _RUN.get()
        if context is None:
            raise RuntimeError("Pi sandbox model routing requires a seeded /run request")
        return (
            self.base_url_for_run(sandbox_server_url(self.config.model_server.name), context["body"]).rstrip("/")
            + "/v1"
        )

    async def _start_sandbox(self, seed: dict) -> AsyncSandbox:
        global_config = get_global_config_dict()
        provider = create_provider(resolve_provider_config(self.config.sandbox_provider, global_config))
        descriptor = seed.get("sandbox_descriptor")
        if not descriptor and seed.get("sandbox_handle"):
            descriptor = {"sandbox_id": seed["sandbox_handle"]}
        if descriptor:
            if self.config.network_access != "inherit":
                raise ValueError("Cannot verify network policy on a resource-owned sandbox")
            return await AsyncSandbox.connect(descriptor, provider=provider)
        options = deepcopy(self.config.sandbox_config)
        resources = SandboxResources.from_mapping(options.pop("resources", {}))
        env = cpu_cap_env(resources.cpu) if options.pop("derive_cpu_env", True) else {}
        env.update(options.pop("env", {}))
        metadata = resolve_provider_metadata(self.config.sandbox_provider, global_config)
        metadata = metadata | options.pop("metadata", {}) | {"nemo_gym_agent": self.config.name}
        if self.config.network_access != "inherit":
            urls = [sandbox_server_url(self.config.model_server.name)]
            if self.config.network_access == "model_and_tools":
                if not self.config.tool_servers:
                    raise ValueError("model_and_tools requires tool_servers")
                urls.extend(sandbox_server_url(s.name) for s in self.config.tool_servers)
            options.setdefault("provider_options", {})["network_policy"] = restricted_network_policy(
                provider.name, urls
            )
        sandbox = AsyncSandbox(provider)
        await sandbox.start(SandboxSpec(resources=resources, env=env, metadata=metadata, **options))
        return sandbox

    async def _run_pi(
        self,
        instruction: str,
        system_prompt: str | None,
        *,
        rollout_id: str | None = None,
        collect_observations: bool = True,
    ) -> tuple[list[Any], dict[str, int], str, list[tuple[float, dict[str, Any]]]]:
        context = _RUN.get()
        if context is None:
            raise RuntimeError("Pi sandbox execution requires a seeded /run request")
        sandbox = context["sandbox"]
        remote = "/tmp/nemo-gym-pi-" + uuid4().hex
        agent_home = remote + "/home/.pi/agent"
        files = {
            agent_home + "/models.json": json.dumps(self._build_models_config(rollout_id)),
            agent_home + "/settings.json": json.dumps({"compaction": {"enabled": self.config.auto_compaction}}),
        }
        files[remote + "/capture.py"] = Path(__file__).with_name("capture.py").read_text()
        env = dict(self.config.env) | {"HOME": remote + "/home", "PI_SKIP_VERSION_CHECK": "1", "PI_TELEMETRY": "0"}
        cmd = [*self.config.command_parts, "--print", "--mode", "json", "--no-session"]
        extensions = []
        if self.config.bash_timeout is not None:
            env["NEMO_GYM_PI_BASH_TIMEOUT"] = str(self.config.bash_timeout)
            extensions.append("bash-timeout.mjs")
        if self.config.output_token_policy == "remaining_context":
            extensions.append("remaining-context.mjs")
        mcp = {name: server.model_dump() for name, server in self.config.mcp_servers.items()} | context["mcp"]
        if mcp:
            files[remote + "/mcp.json"] = json.dumps(mcp)
            env["NEMO_GYM_PI_MCP_CONFIG"] = remote + "/mcp.json"
            extensions.append("gym_mcp.mjs")
        for extension in extensions:
            files[remote + "/" + extension] = (Path(__file__).parent.parent / "pi_agent" / extension).read_text()
            cmd.extend(["--extension", remote + "/" + extension])
        cmd += ["--provider", "nemo", "--model", self.config.model]
        if self.config.thinking:
            cmd += ["--thinking", self.config.thinking]
        if system_prompt:
            cmd += ["--append-system-prompt", system_prompt]
        cmd += self.config.extra_args + [instruction]
        with TemporaryDirectory() as directory:
            for i, (target, contents) in enumerate(files.items()):
                local = Path(directory) / str(i)
                local.write_text(contents)
                local.chmod(0o600)
                await sandbox.upload(local, target)
        # Configuration failures are infrastructure errors, not zero-reward generations.
        setup = await sandbox.exec(command=f"chmod -R go-rwx {quote(remote)}", timeout_s=self.config.timeout)
        if setup.return_code != 0 or setup.error_type:
            raise RuntimeError("Unable to protect Pi session configuration")
        if self.config.pi_version:
            version = await sandbox.exec(
                command=join([*self.config.command_parts, "--version"]), timeout_s=self.config.timeout
            )
            if version.return_code != 0 or (version.stdout or "").strip() != self.config.pi_version:
                raise RuntimeError("Preinstalled Pi version does not match pi_version")
        # Persist the native stream in the sandbox too: provider stdout may be truncated.
        stdout_path, stderr_path = remote + "/stdout.jsonl", remote + "/stderr.log"
        result = None
        error_type = None
        try:
            result = await sandbox.exec(
                command=f"{join(['python3', remote + '/capture.py', remote + '/events.jsonl', *cmd])} > {quote(stdout_path)} 2> {quote(stderr_path)}",
                env=env,
                timeout_s=self.config.timeout,
            )
        except Exception as exc:
            error_type = type(exc).__name__
            LOG.exception("Pi sandbox execution failed")
        root = Path(self.config.artifacts_dir) / sha256((rollout_id or uuid4().hex).encode()).hexdigest()
        root.mkdir(parents=True, exist_ok=True)
        # Export failures propagate; retrying such a request must not become a scored zero.
        await sandbox.download(stdout_path, root / "stdout.jsonl")
        await sandbox.download(stderr_path, root / "stderr.log")
        await sandbox.download(remote + "/events.jsonl", root / "events.jsonl")
        stdout = (root / "stdout.jsonl").read_text(errors="replace")
        error_type = error_type or getattr(result, "error_type", None)
        return_code = getattr(result, "return_code", None)
        events = []
        lines = (root / "events.jsonl").read_text().split("\n")
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                # A killed capture process may leave its last write incomplete. Keep the
                # raw artifact, but only tolerate an unterminated tail on failed execution.
                if index != len(lines) - 1 or (not error_type and return_code == 0):
                    raise
                LOG.warning("Ignoring incomplete trailing Pi event after failed execution: %s", root / "events.jsonl")
        if error_type or return_code != 0:
            events.append(
                (
                    time(),
                    {
                        "type": "_ng_process_exit",
                        "return_code": return_code or 1,
                        "timed_out": "timeout" in (error_type or "").lower(),
                    },
                )
            )
        context["execution"] = {
            "pi_failed": bool(error_type or return_code != 0),
            "pi_exit_code": return_code,
            "pi_error_type": error_type,
            "pi_results_dir": str(root),
        }
        output, usage = parse_pi_events(stdout)
        return output, usage, self.config.model, events

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        raise ValueError("PiSandboxedAgent requires /run to seed tools and allocate its sandbox")

    async def run(self, request: Request, body: PiAgentRunRequest) -> PiSandboxedAgentVerifyResponse:
        async with self.sem:
            seeded = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/seed_session",
                json=body.model_dump(),
                cookies=request.cookies,
            )
            await raise_for_status(seeded)
            cookies = request.cookies | seeded.cookies
            seed = await get_response_json(seeded)
            mcp = await seed_mcp_servers(
                self.server_client, self.config.tool_servers, body, cookies, timeout_s=self.config.timeout
            )
            sandbox = await self._start_sandbox(seed)
            rollout_id = self.rollout_id_from_run(body) or uuid4().hex
            context = {"sandbox": sandbox, "body": body, "mcp": mcp}
            token = _RUN.set(context)
            try:
                episode = await self._create_episode(body.responses_create_params, rollout_id=rollout_id)
                observations = episode.observations
                observations.gaps = [g for g in observations.gaps if g.code != "no_sandbox_runtime"]
                execution = context["execution"]
                failed = execution["pi_failed"] or any(
                    isinstance(r, AgentInvocation) and r.status in {"failed", "incomplete"}
                    for r in observations.records
                )
                execution["pi_failed"] = failed
                handle = sandbox._handle
                observations.records.append(
                    SandboxObservation(
                        role="agent",
                        sandbox_id=handle.sandbox_id,
                        provider=handle.provider_name,
                        outcome="failed" if failed else "completed",
                        exit_code=execution["pi_exit_code"],
                        error_type=execution["pi_error_type"],
                    )
                )
                for record in observations.records:
                    if isinstance(record, ToolCallObservation):
                        record.sandbox_id = handle.sandbox_id
                metadata = execution | {"ng_agent_observations": observations.model_dump(mode="json")}
                payload = body.model_dump(mode="json") | {"response": episode.response.model_dump(mode="json")}
                root = Path(execution["pi_results_dir"])
                (root / "generation.json").write_text(json.dumps(payload | metadata))
                if failed and self.config.execution_failure_reward_zero:
                    result = payload | {"reward": 0.0}
                else:
                    verified = await self.server_client.post(
                        server_name=self.config.resources_server.name,
                        url_path="/verify",
                        json=payload,
                        cookies=cookies,
                    )
                    await raise_for_status(verified)
                    result = await get_response_json(verified)
                return PiSandboxedAgentVerifyResponse.model_validate(
                    result
                    | metadata
                    | {
                        "turns_used": sum(getattr(i, "type", None) == "message" for i in episode.response.output),
                        "finished_naturally": not failed,
                    }
                )
            finally:
                _RUN.reset(token)
                # Connected resource-owned sandboxes remain alive through verification, just like OpenCode.
                await sandbox.stop()


if __name__ == "__main__":
    PiSandboxedAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = PiSandboxedAgent.run_webserver()
