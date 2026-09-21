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
import ipaddress
import json
import logging
import re
import shlex
import socket
import subprocess
import tempfile
from asyncio import Semaphore
from contextvars import ContextVar
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, Mapping
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

from fastapi import Request
from omegaconf import ListConfig
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import NEMO_GYM_MCP_METADATA_KEY, BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.global_config import get_first_server_config_dict
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.rollout_observability import AgentObservationBundle, SandboxObservation
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.sandbox.providers.base import ConnectableProvider, SandboxSpec
from nemo_gym.sandbox.providers.registry import create_provider
from nemo_gym.server_utils import get_response_json, is_nemo_gym_fastapi_entrypoint, raise_for_status


LOG = logging.getLogger(__name__)
_RUN_CONTEXT: ContextVar[dict[str, Any] | None] = ContextVar("harness_agent_run_context", default=None)


_AGENTS = {
    "claude_code": ("responses_api_agents.claude_code_agent.app", "ClaudeCodeAgent", "ClaudeCodeAgentConfig"),
    "cline": ("responses_api_agents.cline_agent.app", "ClineAgent", "ClineAgentConfig"),
    "codex": ("responses_api_agents.codex_agent.app", "CodexAgent", "CodexAgentConfig"),
    "deepagents": ("responses_api_agents.nemo_fabric_agent.app", "NeMoFabricAgent", "NeMoFabricAgentConfig"),
    "hermes": ("responses_api_agents.hermes_agent.app", "HermesAgent", "HermesAgentConfig"),
    "kilocode": ("responses_api_agents.kilocode_agent.app", "KiloCodeAgent", "KiloCodeAgentConfig"),
    "mini_swe": ("responses_api_agents.nemo_fabric_agent.app", "NeMoFabricAgent", "NeMoFabricAgentConfig"),
    "openclaw": ("responses_api_agents.openclaw_agent.app", "OpenClawAgent", "OpenClawAgentConfig"),
    "opencode": ("responses_api_agents.opencode_agent.app", "OpenCodeAgent", "OpenCodeAgentConfig"),
    "pi": ("responses_api_agents.pi_agent.app", "PiAgent", "PiAgentConfig"),
    "prime": ("responses_api_agents.prime_agent.app", "PrimeAgent", "PrimeAgentConfig"),
    "terminus_2": ("responses_api_agents.terminus_2_agent.app", "Terminus2Agent", "Terminus2AgentConfig"),
}

_FABRIC_ADAPTERS = {
    "deepagents": "nvidia.fabric.langchain.deepagents",
    "mini_swe": "nvidia.fabric.mini-swe-agent",
}

_WORKDIR_KEYS = {
    "cline": "repo_dir",
    "codex": "cwd",
    "deepagents": "cwd",
    "kilocode": "repo_dir",
    "mini_swe": "cwd",
    "opencode": "repo_dir",
    "terminus_2": "workspace_root",
}


def resolve_agent(name: str) -> tuple[str, str, str]:
    try:
        return _AGENTS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown agent: {name}") from exc


async def stage_and_run_eval(
    provider,
    handle,
    eval_files: Mapping[str, str],
    eval_command: str,
    reward_file: str,
    timeout_s: int,
) -> float:
    """Stage eval files into the sandbox, run them and get the reward."""
    if handle.provider_name == "local":
        paths = {str(parent) for path in [*eval_files, reward_file] for parent in [Path(path), *Path(path).parents]}
        for path in sorted(paths - {"/", "."}, key=len, reverse=True):
            eval_command = eval_command.replace(path, path.lstrip("/"))
        eval_files = {path.lstrip("/"): content for path, content in eval_files.items()}
        reward_file = reward_file.lstrip("/")
    with tempfile.TemporaryDirectory() as td:
        for target, content in eval_files.items():
            local = Path(td) / uuid4().hex
            local.write_text(content)
            await provider.exec(handle, f"mkdir -p {Path(target).parent}", timeout_s=30)
            await provider.upload_file(handle, local, target)
        result = await provider.exec(handle, eval_command, timeout_s=timeout_s)
        if result.return_code != 0:
            raise RuntimeError(
                f"eval command failed ({result.return_code}): {(result.stderr or result.stdout or '')[:300]}"
            )
        local = Path(td) / "reward"
        await provider.download_file(handle, reward_file, local)
        reward = local.read_text().strip()
        if not reward:
            raise RuntimeError(f"reward file {reward_file} is empty")
        return float(reward)


class HarnessAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    concurrency: int = 64
    agent: str
    agent_kwargs: dict[str, Any] = Field(default_factory=dict)

    sandbox_provider: str | dict[str, Any]
    sandbox_model_base_url: str | None = None
    sandbox_image: str = "python:3.12-slim"
    sandbox_spec: dict[str, Any] = Field(default_factory=dict)
    setup_commands: list[str] = Field(default_factory=list)
    sandbox_python: str = "python3"
    gym_source: str = "auto"
    network_access: Literal["inherit", "model_only", "model_and_tools"] = "inherit"
    tool_servers: list[ResourcesServerRef] = Field(default_factory=list)
    artifacts_dir: str | None = None
    execution_failure_reward_zero: bool = False

    eval_timeout: int = 1800
    rollout_timeout: int = 2400


class HarnessAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class HarnessAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    harness_failed: bool = False
    ng_agent_observations: AgentObservationBundle | None = None


class HarnessAgent(SimpleResponsesAPIAgent):
    config: HarnessAgentConfig
    sem: Semaphore = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        if self.config.tool_servers and self.config.agent not in {"opencode", "pi"}:
            raise ValueError("Authenticated remote MCP tools currently require the OpenCode or Pi adapter")
        if self.config.network_access == "model_only" and self.config.tool_servers:
            raise ValueError("Remote tools require network_access=model_and_tools or inherit")
        if self.config.network_access == "model_and_tools" and not self.config.tool_servers:
            raise ValueError("network_access=model_and_tools requires tool_servers")
        self.sem = Semaphore(self.config.concurrency)
        global_config = self.server_client.global_config_dict
        provider_config = resolve_provider_config(
            self.config.sandbox_provider,
            global_config,
        )
        self._sandbox_metadata = resolve_provider_metadata(
            self.config.sandbox_provider,
            global_config,
        )
        self._provider = create_provider(provider_config)
        self._gym_tar = None
        self._gym_source_url = None
        if self.config.gym_source == "auto":
            self._gym_tar = self._build_gym_tar()
        elif "://" in self.config.gym_source:
            self._gym_source_url = self.config.gym_source
        else:
            self._gym_tar = Path(self.config.gym_source)

    def _build_gym_tar(self) -> Path:
        root = Path(__file__).resolve().parent.parent.parent
        agent_module, _, _ = resolve_agent(self.config.agent)
        agent_pkg = "/".join(agent_module.split(".")[:-1])
        tar_path = Path(tempfile.gettempdir()) / f"gym_src_{uuid4().hex}.tar.gz"
        subprocess.run(
            [
                "tar",
                "czf",
                str(tar_path),
                "--exclude=__pycache__",
                "--exclude=*.pyc",
                "--exclude=.*",
                "--exclude=node_modules",
                "--exclude=data",
                "--exclude=tests",
                "--exclude=outputs",
                "--exclude=workspaces",
                "-C",
                str(root),
                "nemo_gym",
                agent_pkg,
            ],
            check=True,
        )
        size_mb = tar_path.stat().st_size / 1e6
        if size_mb > 50:
            LOG.warning("gym source tar is %.0fMB, sandbox uploads will be slow", size_mb)
        return tar_path

    def _sandbox_model_url(self, request: Request) -> str:
        base = self.config.sandbox_model_base_url
        if not base:
            cfg = get_first_server_config_dict(self.server_client.global_config_dict, self.config.model_server.name)
            # Use the Gym model server so backend credentials and request
            # adapters stay outside the sandbox. Explicit overrides are allowed.
            base = self.server_client._build_server_base_url(cfg)
        if isinstance(base, (list, ListConfig)):
            base = base[0]
        base = re.sub(r"/v1/?$", "", str(base))
        if not self.config.sandbox_model_base_url:
            base = self._rewrite_loopback_url(base)
        prefix = (_RUN_CONTEXT.get() or {}).get("url_prefix") or self.url_path_for_request("", request)
        return f"{base.rstrip('/')}{prefix}"

    @staticmethod
    def _rewrite_loopback_url(base: str) -> str:
        """Advertise the host address to sandboxes for local Gym services."""
        parsed = urlsplit(base if "://" in base else f"http://{base}")
        if parsed.hostname not in {"127.0.0.1", "localhost", "0.0.0.0", "::1", "::"}:
            return base
        try:
            host = socket.gethostbyname(socket.gethostname())
        except OSError:
            return base
        host = f"[{host}]" if ":" in host else host
        netloc = f"{host}:{parsed.port}" if parsed.port else host
        return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))

    def _runner(self) -> tuple[str, dict, str]:
        script = (Path(__file__).parent / "agent_runner.py").read_text()
        agent_module, agent_class, agent_config_class = resolve_agent(self.config.agent)
        runner_config = {
            "agent_module": agent_module,
            "agent_class": agent_class,
            "agent_config_class": agent_config_class,
        }
        return script, runner_config, f"{self.config.sandbox_python} runner.py"

    @staticmethod
    def _box_path(handle, path: str) -> str:
        return path.lstrip("/") if handle.provider_name == "local" else path

    async def _upload_files(self, handle, files: Mapping[str, str]) -> None:
        with tempfile.TemporaryDirectory() as td:
            for i, (target, content) in enumerate(files.items()):
                local = Path(td) / str(i)
                local.write_text(content)
                await self._provider.upload_file(handle, local, self._box_path(handle, target))

    async def run(self, request: Request, body: HarnessAgentRunRequest) -> HarnessAgentVerifyResponse:
        async with self.sem:
            cookies = request.cookies

            seed_resp = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/seed_session",
                json=body.model_dump(),
                cookies=cookies,
            )
            await raise_for_status(seed_resp)
            cookies = cookies | seed_resp.cookies
            seed = await get_response_json(seed_resp)
            descriptor = seed.get("sandbox_descriptor")
            if not descriptor and isinstance(seed.get("sandbox_handle"), str):
                descriptor = {"sandbox_id": seed["sandbox_handle"]}
            run_context = {
                "sandbox_descriptor": descriptor,
                "url_prefix": self.url_path_for_run("", body),
                "rollout_id": self.rollout_id_from_run(body) or uuid4().hex,
            }
            token = _RUN_CONTEXT.set(run_context)
            try:
                if self.config.tool_servers:
                    run_context["mcp"] = await self._seed_tool_servers(body, cookies)
                agent_resp = await self.responses(request, body.responses_create_params)
                agent_resp_json = agent_resp.model_dump(mode="json")
                observations = run_context.get("observations")
                failed = run_context.get("harness_failed", False)
                if failed and self.config.execution_failure_reward_zero:
                    await self._close_run_sandbox(run_context)
                    return HarnessAgentVerifyResponse.model_validate(
                        body.model_dump()
                        | {
                            "response": agent_resp_json,
                            "reward": 0.0,
                            "harness_failed": True,
                            "ng_agent_observations": observations,
                        }
                    )
                verify_resp = await self.server_client.post(
                    server_name=self.config.resources_server.name,
                    url_path="/verify",
                    json=body.model_dump() | {"response": agent_resp_json},
                    cookies=cookies,
                )
                await raise_for_status(verify_resp)
                return HarnessAgentVerifyResponse.model_validate(
                    await get_response_json(verify_resp)
                    | {"harness_failed": failed, "ng_agent_observations": observations}
                )
            except BaseException:
                await self._close_run_sandbox(run_context)
                raise
            finally:
                _RUN_CONTEXT.reset(token)

    def _tool_server_url(self, server: ResourcesServerRef) -> str:
        cfg = get_first_server_config_dict(self.server_client.global_config_dict, server.name)
        return self._rewrite_loopback_url(str(self.server_client._build_server_base_url(cfg))).rstrip("/")

    async def _seed_tool_servers(self, body: HarnessAgentRunRequest, cookies: Mapping[str, str]) -> dict[str, Any]:
        entries = {}
        for server in self.config.tool_servers:
            seeded = await self.server_client.post(
                server_name=server.name, url_path="/seed_session", json=body.model_dump(), cookies=cookies
            )
            await raise_for_status(seeded)
            metadata = (await get_response_json(seeded)).get(NEMO_GYM_MCP_METADATA_KEY)
            if not isinstance(metadata, dict) or not metadata.get("headers"):
                raise ValueError(f"Tool server {server.name} must expose authenticated MCP tools")
            entries[server.name] = {
                "type": "remote",
                "url": self._tool_server_url(server) + "/" + metadata.get("url_path", "/mcp").lstrip("/"),
                "headers": metadata["headers"],
                "enabled": True,
                "timeout": self.config.rollout_timeout * 1000,
            }
        return entries

    async def _grade_in_box(self, handle, grade_spec: dict) -> float:
        return await stage_and_run_eval(
            self._provider,
            handle,
            eval_files=grade_spec.get("eval_files") or {},
            eval_command=grade_spec.get("eval_command") or "bash /tests/test.sh",
            reward_file=grade_spec.get("eval_reward_file") or "/logs/verifier/reward.txt",
            timeout_s=self.config.eval_timeout,
        )

    async def _provision_box(self, image: str, files: dict[str, str], model_url: str):
        descriptor = (_RUN_CONTEXT.get() or {}).get("sandbox_descriptor")
        if descriptor:
            if self.config.network_access != "inherit":
                raise ValueError("Cannot verify network policy on a resource-owned sandbox")
            if not isinstance(self._provider, ConnectableProvider):
                raise TypeError("sandbox provider cannot connect to a resource-owned sandbox")
            handle = await self._provider.connect(descriptor)
        else:
            spec_config = deepcopy(self.config.sandbox_spec)
            metadata = self._sandbox_metadata | dict(spec_config.pop("metadata", {}))
            if self.config.network_access != "inherit":
                if self._provider.name != "opensandbox":
                    raise ValueError("Restricted network access requires the OpenSandbox network-policy provider")
                urls = [model_url]
                if self.config.network_access == "model_and_tools":
                    urls.extend(self._tool_server_url(server) for server in self.config.tool_servers)
                targets = set()
                for url in urls:
                    target = urlsplit(url).hostname
                    if not target or target.lower() == "localhost":
                        raise ValueError("Model/tool endpoint must advertise a sandbox-reachable host")
                    try:
                        address = ipaddress.ip_address(target)
                    except ValueError:
                        address = None
                    if address and (address.is_loopback or address.is_unspecified):
                        raise ValueError("Model/tool endpoint must advertise a sandbox-reachable host")
                    targets.add(target)
                spec_config.setdefault("provider_options", {})["network_policy"] = {
                    "defaultAction": "deny",
                    "egress": [{"action": "allow", "target": target} for target in sorted(targets)],
                }
            handle = await self._provider.create(SandboxSpec(image=image, metadata=metadata, **spec_config))
        try:
            work_dir = self._box_path(handle, "/work")
            await self._provider.exec(handle, f"mkdir -p {shlex.quote(work_dir)}", timeout_s=60)
            # Provider.create() does not stage files (AsyncSandbox.start() does).
            await self._upload_files(handle, self.config.sandbox_spec.get("files", {}))
            await self._upload_files(handle, files)
            if self._gym_tar is not None or self._gym_source_url is not None:
                archive = self._box_path(handle, "/work/gym_src.tar.gz")
                gym_mount = self._box_path(handle, "/work/gym_mount")
                if self._gym_tar is not None:
                    await self._provider.upload_file(handle, self._gym_tar, archive)
                else:
                    r = await self._provider.exec(
                        handle,
                        f"curl -fsSL -o {shlex.quote(archive)} {shlex.quote(self._gym_source_url)}",
                        timeout_s=600,
                    )
                    if r.return_code != 0:
                        raise RuntimeError(f"gym source fetch failed: {(r.stderr or '')[:300]}")
                r = await self._provider.exec(
                    handle,
                    f"mkdir -p {shlex.quote(gym_mount)} && tar xzf {shlex.quote(archive)} -C {shlex.quote(gym_mount)}",
                    timeout_s=300,
                )
                if r.return_code != 0:
                    raise RuntimeError(f"gym source extraction failed: {(r.stderr or '')[:300]}")
            for cmd in self.config.setup_commands:
                r = await self._provider.exec(handle, cmd, timeout_s=900)
                if r.return_code != 0:
                    raise RuntimeError(f"setup failed ({r.return_code}): {cmd} | {(r.stderr or '')[:300]}")

            pm = re.match(r"https?://([^:/]+):(\d+)", model_url)
            if pm:
                net = await self._provider.exec(
                    handle,
                    f"bash -c 'echo > /dev/tcp/{pm.group(1)}/{pm.group(2)}'",
                    timeout_s=5,
                )
                if net.return_code != 0:
                    raise RuntimeError(f"model endpoint {pm.group(1)}:{pm.group(2)} unreachable from sandbox")
            return handle
        except BaseException:
            if not descriptor:
                await self._close_box(handle)
            raise

    async def _close_box(self, handle) -> None:
        try:
            await self._provider.close(handle)
        except Exception:
            LOG.warning("sandbox close failed", exc_info=True)

    async def _close_run_sandbox(self, run_context: dict[str, Any]) -> None:
        descriptor = run_context.get("sandbox_descriptor")
        if not descriptor:
            return
        if not isinstance(self._provider, ConnectableProvider):
            LOG.warning("sandbox provider cannot clean up a resource-owned sandbox")
            return
        try:
            handle = await self._provider.connect(descriptor)
        except Exception:
            LOG.warning("sandbox cleanup connect failed", exc_info=True)
            return
        await self._close_box(handle)

    async def _download_json(self, handle, path: str) -> Any:
        with tempfile.TemporaryDirectory() as td:
            local = Path(td) / "out"
            await self._provider.download_file(handle, path, local)
            return json.loads(local.read_text())

    def _write_generation(self, rollout_id: str, record: dict[str, Any], *, failure: bool = False) -> Path | None:
        if not self.config.artifacts_dir:
            return None
        # IDs can come from dataset metadata; never interpret them as paths.
        destination = Path(self.config.artifacts_dir) / sha256(rollout_id.encode()).hexdigest()
        destination.mkdir(parents=True, exist_ok=True)
        filename = "failure.json" if failure else "generation.json"
        (destination / filename).write_text(json.dumps({"rollout_id": rollout_id, **record}))
        return destination

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        meta = getattr(body, "metadata", None) or {}
        image = meta.get("docker_image") or self.config.sandbox_image

        runner_script, runner_config, runner_cmd = self._runner()
        context = _RUN_CONTEXT.get()
        if context is None:
            context = {}
        path_params = getattr(request, "path_params", None)
        request_rollout_id = path_params.get("rollout_id") if isinstance(path_params, Mapping) else None
        runner_config["rollout_id"] = context.get("rollout_id") or request_rollout_id or uuid4().hex
        agent_body = body.model_copy(deep=True)
        if getattr(agent_body, "metadata", None):
            agent_body.metadata = {k: v for k, v in agent_body.metadata.items() if k != "sandbox_eval"}
        agent_config = deepcopy(self.config.agent_kwargs)
        if mcp := context.get("mcp"):
            if self.config.agent == "pi":
                agent_config.setdefault("mcp_servers", {}).update(mcp)
            else:
                agent_config.setdefault("opencode_config", {}).setdefault("mcp", {}).update(mcp)
        if adapter_id := _FABRIC_ADAPTERS.get(self.config.agent):
            agent_config.setdefault("adapter_id", adapter_id)
        agent_config.setdefault("resources_server", self.config.resources_server.model_dump(mode="json"))
        agent_config.setdefault("model_server", self.config.model_server.model_dump(mode="json"))
        descriptor = (_RUN_CONTEXT.get() or {}).get("sandbox_descriptor")
        workdir = meta.get("workdir") or (descriptor or {}).get("workdir")
        if workdir:
            runner_config["cwd"] = workdir
            if workdir_key := _WORKDIR_KEYS.get(self.config.agent):
                agent_config[workdir_key] = workdir
        model_url = self._sandbox_model_url(request)
        files = {
            "/work/model_url.txt": model_url,
            "/work/request.json": agent_body.model_dump_json(),
            "/work/agent_config.json": json.dumps(agent_config),
            "/work/runner_config.json": json.dumps(runner_config),
            "/work/runner.py": runner_script,
        }
        handle = await self._provision_box(image, files, model_url)
        try:
            work_dir = self._box_path(handle, "/work")
            if descriptor and not workdir:
                pwd = await self._provider.exec(handle, "pwd", timeout_s=30)
                workdir = (pwd.stdout or "").strip() if pwd.return_code == 0 else ""
                if workdir:
                    runner_config["cwd"] = workdir
                    if workdir_key := _WORKDIR_KEYS.get(self.config.agent):
                        agent_config[workdir_key] = workdir
                    await self._upload_files(
                        handle,
                        {
                            "/work/agent_config.json": json.dumps(agent_config),
                            "/work/runner_config.json": json.dumps(runner_config),
                        },
                    )
            r = await self._provider.exec(
                handle,
                f"{runner_cmd} > runner.out 2>&1",
                cwd=work_dir,
                timeout_s=self.config.rollout_timeout,
            )
            logs = await self._provider.exec(handle, "tail -c 6000 runner.out", cwd=work_dir, timeout_s=30)
            if r.return_code != 0 or "RUNNER_DONE" not in (logs.stdout or ""):
                raise RuntimeError(
                    f"runner failed ({r.return_code}): {(logs.stdout or logs.stderr or r.stderr or '')[-6000:]}"
                )

            resp = NeMoGymResponse.model_validate(
                await self._download_json(handle, self._box_path(handle, "/work/response.json"))
            )
            raw_observations = getattr(resp, "_ng_agent_observations", None)
            response_json = resp.model_dump(mode="json")
            response_json.pop("_ng_agent_observations", None)
            resp = NeMoGymResponse.model_validate(response_json)
            observations = AgentObservationBundle.model_validate(raw_observations) if raw_observations else None
            failed = False
            if observations is not None:
                failed = any(
                    getattr(record, "parent_invocation_id", None) is None
                    and getattr(record, "status", None) in {"failed", "incomplete", "cancelled"}
                    for record in observations.records
                    if getattr(record, "kind", None) == "agent_invocation"
                )
                observations.gaps = [gap for gap in observations.gaps if gap.code != "no_sandbox_runtime"]
                observations.records.append(
                    SandboxObservation(
                        role="agent",
                        provider=handle.provider_name,
                        sandbox_id=handle.sandbox_id,
                        outcome="failed" if failed else "completed",
                        exit_code=r.return_code,
                    )
                )
            context["observations"] = observations
            context["harness_failed"] = failed
            destination = self._write_generation(
                runner_config["rollout_id"],
                {
                    "response": resp.model_dump(mode="json"),
                    "harness_failed": failed,
                    "observations": observations.model_dump(mode="json") if observations else None,
                },
            )
            if destination:
                try:
                    await self._provider.download_file(
                        handle, self._box_path(handle, "/work/runner.out"), destination / "runner.log"
                    )
                except Exception:
                    # The validated response and receipt are already saved. A missing
                    # diagnostic log must not turn a completed rollout into a failure.
                    LOG.warning("Could not download runner log for sandbox %s", handle.sandbox_id, exc_info=True)

            grade_raw = meta.get("sandbox_eval")
            grade_spec = json.loads(grade_raw) if isinstance(grade_raw, str) else grade_raw
            if grade_spec:
                reward = await self._grade_in_box(handle, grade_spec)
                resp.metadata = (resp.metadata or {}) | {"sandbox_reward": str(reward)}
            return resp
        except BaseException as exc:
            # Keep partial generation receipts if export or grading fails.
            if self.config.artifacts_dir:
                self._write_generation(
                    runner_config["rollout_id"],
                    {
                        "error_type": type(exc).__name__,
                        "sandbox_id": handle.sandbox_id,
                    },
                    failure=True,
                )
            raise
        finally:
            if not descriptor:
                await self._close_box(handle)


if __name__ == "__main__":
    HarnessAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = HarnessAgent.run_webserver()
