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
"""Runs any Gym environment inside a sandbox.

Two modes:
  agent_only_runner: import another agent's responses() in the sandbox and use an external
    resources server for scoring. No gym servers in the sandbox.
  gym_runner: start Nemo Gym servers in the sandbox and run the task e2e inside.
    Wraps environments without a clean responses/verify split.
"""

import asyncio
import base64
import copy
import json
import logging
import posixpath
import re
import shlex
import socket
import subprocess
import tarfile
import tempfile
from asyncio import Semaphore
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Mapping, Optional
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import httpx
from fastapi import Request
from pydantic import BaseModel, ConfigDict, Field, model_validator

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import (
    RUN_TOKEN_HEADER,
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.sandbox.providers.base import SandboxSpec
from nemo_gym.sandbox.providers.registry import create_provider
from nemo_gym.server_utils import get_response_json, raise_for_status


LOG = logging.getLogger(__name__)
SANDBOX_SETUP_METADATA_KEY = "_nemo_gym_sandbox_setup"
SANDBOX_ARTIFACTS_METADATA_KEY = "nemo_gym_sandbox_artifacts"


class SandboxWorkspaceSetup(BaseModel):
    workspace_path: Optional[str] = None
    workdir: str = "/workspace"
    artifact_paths: list[str] = Field(default_factory=list)
    max_artifact_bytes: int = 10 * 1024 * 1024

    @model_validator(mode="after")
    def validate_paths(self) -> "SandboxWorkspaceSetup":
        if not self.workdir.startswith("/"):
            raise ValueError("sandbox workdir must be absolute")
        if self.max_artifact_bytes < 1:
            raise ValueError("max_artifact_bytes must be positive")
        for artifact_path in self.artifact_paths:
            path = PurePosixPath(artifact_path)
            if path.is_absolute() or ".." in path.parts or not path.parts:
                raise ValueError(f"sandbox artifact path must be relative: {artifact_path!r}")
        return self


async def stage_and_run_eval(
    provider,
    handle,
    eval_files: Mapping[str, str],
    eval_command: str,
    reward_file: str,
    timeout_s: int,
) -> float:
    """Stage eval files into the sandbox, run them and get the reward."""
    with tempfile.TemporaryDirectory() as td:
        for target, content in eval_files.items():
            local = Path(td) / uuid4().hex
            local.write_text(content)
            await provider.exec(handle, f"mkdir -p {Path(target).parent}", timeout_s=30)
            await provider.upload_file(handle, local, target)
        await provider.exec(handle, eval_command, timeout_s=timeout_s)
        local = Path(td) / "reward"
        await provider.download_file(handle, reward_file, local)
        return float(local.read_text().strip() or 0.0)


def archive_workspace(workspace: Path, destination: Path) -> None:
    """Archive a seeded workspace without changing its files."""
    if not workspace.is_dir():
        raise FileNotFoundError(f"sandbox workspace does not exist: {workspace}")
    with tarfile.open(destination, "w:gz") as archive:
        for child in sorted(workspace.iterdir(), key=lambda item: item.name):
            archive.add(child, arcname=child.name, recursive=True)


class SandboxAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: Optional[ResourcesServerRef] = None
    model_server: Optional[ModelServerRef] = None
    concurrency: int = 64
    mode: Literal["agent_only_runner", "gym_runner"] = "agent_only_runner"

    agent_module: Optional[str] = None
    agent_class: Optional[str] = None
    agent_config_class: Optional[str] = None
    agent_config: dict[str, Any] = Field(default_factory=dict)

    nested_config_paths: list[str] = Field(default_factory=list)
    nested_overrides: list[str] = Field(default_factory=list)
    nested_agent_name: Optional[str] = None
    nested_agent_port: int = 11001

    sandbox_provider: dict[str, Any]
    sandbox_image: str = "python:3.12-slim"
    sandbox_spec: dict[str, Any] = Field(default_factory=dict)
    model_transport: Literal["direct", "endpoint_bridge"] = "direct"
    model_bridge_port: int = 18080
    setup_commands: list[str] = Field(default_factory=list)
    sandbox_python: str = "python3"
    # "auto" tars the local repo, or a path to a prebuilt tar.gz, or a URL fetched in the sandbox
    gym_source: str = "auto"

    eval_timeout: int = 1800
    rollout_timeout: int = 2400
    empty_trajectory_retries: int = 0

    @model_validator(mode="after")
    def validate_model_transport(self) -> "SandboxAgentConfig":
        if not 1 <= self.model_bridge_port <= 65535:
            raise ValueError("model_bridge_port must be between 1 and 65535")
        if self.model_transport == "endpoint_bridge" and self.mode != "agent_only_runner":
            raise ValueError("endpoint_bridge is only supported in agent_only_runner mode")
        if self.empty_trajectory_retries < 0:
            raise ValueError("empty_trajectory_retries must be non-negative")
        return self


class SandboxAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class SandboxAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class SandboxAgent(SimpleResponsesAPIAgent):
    config: SandboxAgentConfig
    sem: Semaphore = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)
        self._provider = create_provider(self.config.sandbox_provider)
        self._gym_tar = None
        self._gym_source_url = None
        if self.config.mode == "agent_only_runner":
            if self.config.gym_source == "auto":
                self._gym_tar = self._build_gym_tar()
            elif "://" in self.config.gym_source:
                self._gym_source_url = self.config.gym_source
            else:
                self._gym_tar = Path(self.config.gym_source)

    def _build_gym_tar(self) -> Path:
        root = Path(__file__).resolve().parent.parent.parent
        agent_pkg = "/".join((self.config.agent_module or "responses_api_agents").split(".")[:-1])
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
        """Run-scoped model target for the configured transport, without a /v1 suffix."""
        base = self.harness_base_url(request)
        if base is None:
            raise RuntimeError("sandbox_agent requires a model_server")
        base = re.sub(r"/v1/?$", "", base)
        if self.config.model_transport == "endpoint_bridge":
            # The forwarder is colocated with Gym, so loopback is the correct target.
            return base
        parsed = urlsplit(base if "://" in base else f"http://{base}")
        host = parsed.hostname or ""
        try:
            if host in ("127.0.0.1", "localhost", "0.0.0.0"):
                # loopback binds are unreachable from a sandbox
                host = socket.gethostbyname(socket.gethostname())
            else:
                host = socket.gethostbyname(host)
        except OSError:
            pass
        netloc = f"{host}:{parsed.port}" if parsed.port else host
        return urlunsplit((parsed.scheme or "http", netloc, parsed.path, parsed.query, parsed.fragment))

## NOTE: hacky thing for preclusters like prenyx should not usually be needed 
    def _runner_model_url(self, target_model_url: str) -> str:
        if self.config.model_transport == "endpoint_bridge":
            return f"http://127.0.0.1:{self.config.model_bridge_port}"
        return target_model_url

    @staticmethod
    async def _forward_model_request(
        pending: dict[str, Any],
        target_model_url: str,
        model_client: httpx.AsyncClient,
    ) -> dict[str, Any]:
        request_path = str(pending["path"])
        if not request_path.startswith("/") or request_path.startswith("/bridge/"):
            raise RuntimeError(f"invalid model bridge request path: {request_path!r}")
        excluded_request_headers = {"host", "content-length", "connection", "transfer-encoding"}
        request_headers = {
            str(key): str(value)
            for key, value in (pending.get("headers") or {}).items()
            if str(key).lower() not in excluded_request_headers
        }
        response = await model_client.request(
            str(pending.get("method") or "POST"),
            f"{target_model_url.rstrip('/')}{request_path}",
            headers=request_headers,
            content=base64.b64decode(pending.get("body_b64") or ""),
        )
        excluded_response_headers = {
            "connection",
            "content-encoding",
            "content-length",
            "transfer-encoding",
        }
        response_headers = {
            str(key): str(value)
            for key, value in response.headers.items()
            if str(key).lower() not in excluded_response_headers
        }
        return {
            "status": response.status_code,
            "headers": response_headers,
            "body_b64": base64.b64encode(response.content).decode(),
        }

    async def _start_model_bridge(self, handle: Any) -> Any:
        port = self.config.model_bridge_port
        result = await self._provider.exec(
            handle,
            f"BBH_MODEL_BRIDGE_PORT={port} nohup {self.config.sandbox_python} "
            "/work/model_bridge_server.py >/tmp/model_bridge.log 2>&1 </dev/null & "
            "for i in $(seq 1 30); do "
            f'{self.config.sandbox_python} -c "import urllib.request; '
            f"urllib.request.urlopen('http://127.0.0.1:{port}/health', timeout=1)\" "
            "&& exit 0; sleep 1; done; exit 1",
            timeout_s=60,
        )
        if result.return_code != 0:
            raise RuntimeError(
                f"sandbox model bridge did not start: {result.stderr or result.stdout or 'unknown error'}"
            )
        get_endpoint = getattr(handle.raw, "get_endpoint", None)
        if get_endpoint is None:
            raise RuntimeError("endpoint_bridge requires a sandbox provider with public endpoint support")
        return await get_endpoint(port)

    def _model_bridge_endpoint_url(self, endpoint: Any) -> str:
        value = getattr(endpoint, "url", None) or getattr(endpoint, "endpoint", None)
        if not value:
            raise RuntimeError("sandbox public endpoint is missing its URL")
        value = str(value).strip().rstrip("/")
        if "://" not in value:
            provider_config = self.config.sandbox_provider.get("opensandbox") or {}
            connection_config = provider_config.get("connection") or {}
            protocol = str(connection_config.get("protocol") or "http").lower()
            if protocol not in {"http", "https"}:
                raise RuntimeError(f"unsupported sandbox endpoint protocol: {protocol!r}")
            value = f"{protocol}://{value.lstrip('/')}"
        return value

    async def _forward_model_requests(self, endpoint: Any, target_model_url: str) -> None:
        bridge_headers = getattr(endpoint, "headers", None) or {}
        endpoint_url = self._model_bridge_endpoint_url(endpoint)
        bridge_timeout = httpx.Timeout(30.0)
        model_timeout = httpx.Timeout(float(self.config.rollout_timeout))
        async with (
            httpx.AsyncClient(headers=bridge_headers, timeout=bridge_timeout) as bridge_client,
            httpx.AsyncClient(timeout=model_timeout) as model_client,
        ):
            while True:
                pending_response = await bridge_client.get(f"{endpoint_url}/bridge/next")
                if pending_response.status_code == 204:
                    continue
                pending_response.raise_for_status()
                pending = pending_response.json()
                request_id = str(pending["request_id"])
                try:
                    reply = await self._forward_model_request(pending, target_model_url, model_client)
                except Exception as exc:
                    LOG.exception("model bridge request failed")
                    reply = {
                        "status": 502,
                        "headers": {"content-type": "application/json"},
                        "body_b64": base64.b64encode(
                            json.dumps({"error": {"message": f"model bridge failed: {type(exc).__name__}"}}).encode()
                        ).decode(),
                    }
                posted = await bridge_client.post(f"{endpoint_url}/bridge/reply/{request_id}", json=reply)
                posted.raise_for_status()

    async def _exec_runner(self, handle: Any, runner_cmd: str, target_model_url: str) -> Any:
        if self.config.model_transport == "direct":
            return await self._provider.exec(handle, runner_cmd, timeout_s=self.config.rollout_timeout)

        endpoint = await self._start_model_bridge(handle)
        forward_task = asyncio.create_task(self._forward_model_requests(endpoint, target_model_url))
        runner_task = asyncio.create_task(
            self._provider.exec(handle, runner_cmd, timeout_s=self.config.rollout_timeout)
        )
        try:
            done, _ = await asyncio.wait({runner_task, forward_task}, return_when=asyncio.FIRST_COMPLETED)
            if forward_task in done:
                exception = forward_task.exception()
                if exception is not None:
                    raise exception
                raise RuntimeError("model bridge forwarding stopped before the sandbox agent completed")
            return await runner_task
        finally:
            if not runner_task.done():
                runner_task.cancel()
            forward_task.cancel()
            await asyncio.gather(runner_task, forward_task, return_exceptions=True)

    @staticmethod
    def _log_runner_result(result: Any) -> None:
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        if stderr:
            LOG.info("sandbox runner stderr tail: %s", stderr[-6000:])
        if result.return_code != 0 or "RUNNER_DONE" not in stdout:
            LOG.warning(
                "runner incomplete (exit %s): %s",
                result.return_code,
                (stderr or stdout)[-6000:],
            )

    def _runner(self) -> tuple[str, dict, str]:
        if self.config.mode == "agent_only_runner":
            script = (Path(__file__).parent / "agent_runner.py").read_text()
            runner_config = {
                "agent_module": self.config.agent_module,
                "agent_class": self.config.agent_class,
                "agent_config_class": self.config.agent_config_class,
            }
        else:
            script = (Path(__file__).parent / "gym_runner.py").read_text()
            runner_config = {
                "config_paths": list(self.config.nested_config_paths),
                "overrides": list(self.config.nested_overrides),
                "agent_name": self.config.nested_agent_name,
                "agent_port": self.config.nested_agent_port,
                "timeout": self.config.rollout_timeout,
            }
        return script, runner_config, f"{self.config.sandbox_python} /work/runner.py"

    @staticmethod
    def _agent_config_for_runner(config: Mapping[str, Any]) -> dict[str, Any]:
        """Return JSON-ready agent config with OpenCode numeric limits normalized."""
        normalized = copy.deepcopy(dict(config))
        providers = (normalized.get("opencode_config") or {}).get("provider") or {}
        for provider in providers.values():
            if not isinstance(provider, dict):
                continue
            for model in (provider.get("models") or {}).values():
                if not isinstance(model, dict):
                    continue
                limits = model.get("limit") or {}
                for key in ("context", "output"):
                    value = limits.get(key)
                    if isinstance(value, str) and value.isdigit():
                        limits[key] = int(value)
        return normalized

    @staticmethod
    def _body_from_seed(body: SandboxAgentRunRequest, seed: dict[str, Any]) -> SandboxAgentRunRequest:
        params_raw = seed.get("responses_create_params")
        params = (
            NeMoGymResponseCreateParamsNonStreaming.model_validate(params_raw)
            if params_raw is not None
            else body.responses_create_params.model_copy(deep=True)
        )
        setup_raw = seed.get("sandbox_setup")
        if setup_raw is not None:
            setup = SandboxWorkspaceSetup.model_validate(setup_raw)
            metadata = dict(params.metadata or {})
            metadata[SANDBOX_SETUP_METADATA_KEY] = setup.model_dump_json()
            params = params.model_copy(update={"metadata": metadata})
        return body.model_copy(update={"responses_create_params": params})

    @staticmethod
    def _verify_seed_context(seed: dict[str, Any]) -> dict[str, Any]:
        context = dict(seed.get("verify_context") or {})
        if seed.get("env_id") is not None:
            context.setdefault("env_id", seed["env_id"])
        return context

    async def run(self, request: Request, body: SandboxAgentRunRequest) -> BaseVerifyResponse:
        if self.config.mode == "gym_runner":
            async with self.sem:
                return await self._run_nested(request, body)
        async with self.sem:
            cookies = request.cookies

            seed_resp = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/seed_session",
                json=body.model_dump(),
                cookies=cookies,
            )
            await raise_for_status(seed_resp)
            cookies = seed_resp.cookies
            seed_json = await get_response_json(seed_resp)
            seeded_body = self._body_from_seed(body, seed_json)

            max_attempts = 1 + self.config.empty_trajectory_retries if self.config.model_server else 1
            for attempt in range(1, max_attempts + 1):
                run_token = uuid4().hex if self.config.model_server else None
                run_header = {"headers": {RUN_TOKEN_HEADER: run_token}} if run_token else {}

                agent_resp = await self.server_client.post(
                    server_name=self.config.name,
                    url_path="/v1/responses",
                    json=seeded_body.responses_create_params,
                    cookies=cookies,
                    **run_header,
                )
                await raise_for_status(agent_resp)
                cookies = agent_resp.cookies
                agent_resp_json = await get_response_json(agent_resp)

                gym_resp = NeMoGymResponse.model_validate(agent_resp_json)
                if not run_token:
                    break
                constructed = await self.get_monotonic_trajectory(self.config.model_server.name, run_token)
                if constructed:
                    gym_resp.output = constructed
                    agent_resp_json = gym_resp.model_dump()
                    break
                if attempt < max_attempts:
                    LOG.warning(
                        "no constructed trajectory for run %s (env %s); retrying sandbox agent (%d/%d)",
                        run_token,
                        seed_json.get("env_id"),
                        attempt,
                        max_attempts,
                    )
                    continue
                LOG.warning("no constructed trajectory for run %s, falling back to harness output", run_token)
                agent_resp_json = gym_resp.model_dump()

            verify_resp = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=seeded_body.model_dump()
                | {
                    "response": agent_resp_json,
                    "seed_session": self._verify_seed_context(seed_json),
                },
                cookies=cookies,
            )
            await raise_for_status(verify_resp)
            return SandboxAgentVerifyResponse.model_validate(await get_response_json(verify_resp))

    async def _grade_in_box(self, handle, grade_spec: dict) -> float:
        try:
            return await stage_and_run_eval(
                self._provider,
                handle,
                eval_files=grade_spec.get("eval_files") or {},
                eval_command=grade_spec.get("eval_command") or "bash /tests/test.sh",
                reward_file=grade_spec.get("eval_reward_file") or "/logs/verifier/reward.txt",
                timeout_s=self.config.eval_timeout,
            )
        except Exception:
            LOG.warning("in-box grading failed, reward=0", exc_info=True)
            return 0.0

    async def _provision_box(
        self,
        image: str,
        files: dict[str, str],
        model_url: str,
        workspace_setup: Optional[SandboxWorkspaceSetup] = None,
    ):
        spec = SandboxSpec(image=image, **dict(self.config.sandbox_spec))
        handle = await self._provider.create(spec)
        try:
            await self._provider.exec(handle, "mkdir -p /work", timeout_s=60)
            with tempfile.TemporaryDirectory() as td:
                for i, (target, content) in enumerate(files.items()):
                    local = Path(td) / str(i)
                    local.write_text(content)
                    await self._provider.upload_file(handle, local, target)
            if self._gym_tar is not None or self._gym_source_url is not None:
                if self._gym_tar is not None:
                    await self._provider.upload_file(handle, self._gym_tar, "/work/gym_src.tar.gz")
                else:
                    r = await self._provider.exec(
                        handle,
                        f"curl -fsSL -o /work/gym_src.tar.gz {shlex.quote(self._gym_source_url)}",
                        timeout_s=600,
                    )
                    if r.return_code != 0:
                        raise RuntimeError(f"gym source fetch failed: {(r.stderr or '')[:300]}")
                r = await self._provider.exec(
                    handle, "mkdir -p /gym_mount && tar xzf /work/gym_src.tar.gz -C /gym_mount", timeout_s=300
                )
                if r.return_code != 0:
                    raise RuntimeError(f"gym source extraction failed: {(r.stderr or '')[:300]}")
            if workspace_setup is not None and workspace_setup.workspace_path:
                with tempfile.TemporaryDirectory() as td:
                    archive_path = Path(td) / "workspace.tar.gz"
                    archive_workspace(Path(workspace_setup.workspace_path), archive_path)
                    await self._provider.upload_file(handle, archive_path, "/work/workspace.tar.gz")
                command = (
                    f"mkdir -p {shlex.quote(workspace_setup.workdir)} && "
                    f"tar xzf /work/workspace.tar.gz -C {shlex.quote(workspace_setup.workdir)}"
                )
                r = await self._provider.exec(handle, command, timeout_s=300)
                if r.return_code != 0:
                    raise RuntimeError(f"workspace extraction failed: {(r.stderr or r.stdout or '')[:300]}")
            for cmd in self.config.setup_commands:
                r = await self._provider.exec(handle, cmd, timeout_s=900)
                if r.return_code != 0:
                    raise RuntimeError(
                        f"sandbox setup failed ({r.return_code}): {cmd}: {(r.stderr or r.stdout or '')[:300]}"
                    )

            pm = re.match(r"https?://([^:/]+):(\d+)", model_url)
            if pm and self.config.model_transport == "direct":
                net = await self._provider.exec(
                    handle,
                    f"timeout 5 bash -c 'echo > /dev/tcp/{pm.group(1)}/{pm.group(2)}' && echo NET_OK || echo NET_FAIL",
                    timeout_s=30,
                )
                if "NET_FAIL" in (net.stdout or ""):
                    raise RuntimeError(f"model endpoint {pm.group(1)}:{pm.group(2)} unreachable from sandbox")
            return handle
        except Exception:
            await self._close_box(handle)
            raise

    async def _close_box(self, handle) -> None:
        try:
            await self._provider.close(handle)
        except Exception:
            LOG.warning("sandbox close failed", exc_info=True)

    async def _download_json(self, handle, path: str) -> Any:
        with tempfile.TemporaryDirectory() as td:
            local = Path(td) / "out"
            await self._provider.download_file(handle, path, local)
            return json.loads(local.read_text().strip().splitlines()[0])

    async def _download_artifacts(
        self,
        handle,
        setup: SandboxWorkspaceSetup,
    ) -> dict[str, str]:
        artifacts = {}
        with tempfile.TemporaryDirectory() as td:
            for index, relative_path in enumerate(setup.artifact_paths):
                local = Path(td) / str(index)
                remote = posixpath.join(setup.workdir, relative_path)
                try:
                    await self._provider.download_file(handle, remote, local)
                except Exception:
                    LOG.info("sandbox artifact not found: %s", relative_path)
                    continue
                if local.stat().st_size > setup.max_artifact_bytes:
                    LOG.warning(
                        "sandbox artifact %s exceeds %d bytes; skipping",
                        relative_path,
                        setup.max_artifact_bytes,
                    )
                    continue
                artifacts[relative_path] = local.read_text(encoding="utf-8")
        return artifacts

    async def _run_nested(self, request: Request, body: SandboxAgentRunRequest) -> BaseVerifyResponse:
        row = body.model_dump()
        row.pop("agent_ref", None)
        meta = (row.get("responses_create_params") or {}).get("metadata") or {}
        image = meta.get("docker_image") or self.config.sandbox_image
        runner_script, runner_config, runner_cmd = self._runner()
        target_model_url = self._sandbox_model_url(request)
        model_url = self._runner_model_url(target_model_url)
        files = {
            "/work/model_url.txt": model_url,
            "/work/input.jsonl": json.dumps(row) + "\n",
            "/work/runner_config.json": json.dumps(runner_config),
            "/work/runner.py": runner_script,
        }
        handle = await self._provision_box(image, files, model_url)
        try:
            await self._provider.exec(
                handle, f"nohup {runner_cmd} > /work/runner.out 2>&1 & echo started", timeout_s=60
            )
            deadline = self.config.rollout_timeout
            waited = 0
            while waited < deadline:
                await asyncio.sleep(20)
                waited += 20
                r = await self._provider.exec(handle, "test -f /work/done && echo DONE", timeout_s=30)
                if "DONE" in (r.stdout or ""):
                    break
            r = await self._provider.exec(handle, "tail -c 6000 /work/runner.out", timeout_s=30)
            if "RUNNER_DONE" not in (r.stdout or ""):
                LOG.warning("runner incomplete: %s", (r.stdout or "")[-6000:])
            rows = await self._download_json(handle, "/work/rollouts.jsonl")
            return BaseVerifyResponse.model_validate(rows)
        finally:
            await self._close_box(handle)

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        meta = getattr(body, "metadata", None) or {}
        image = meta.get("docker_image") or self.config.sandbox_image
        setup_raw = meta.get(SANDBOX_SETUP_METADATA_KEY)
        workspace_setup = (
            SandboxWorkspaceSetup.model_validate_json(setup_raw)
            if isinstance(setup_raw, str)
            else SandboxWorkspaceSetup.model_validate(setup_raw)
            if setup_raw is not None
            else None
        )

        runner_script, runner_config, runner_cmd = self._runner()
        agent_body = body.model_copy(deep=True)
        if getattr(agent_body, "metadata", None):
            agent_body.metadata = {
                key: value
                for key, value in agent_body.metadata.items()
                if key not in {"sandbox_eval", SANDBOX_SETUP_METADATA_KEY}
            }
        agent_config = self._agent_config_for_runner(self.config.agent_config)
        agent_workdir = meta.get("workdir") or (workspace_setup.workdir if workspace_setup else None)
        if agent_workdir:
            agent_config = agent_config | {"repo_dir": agent_workdir}
        target_model_url = self._sandbox_model_url(request)
        model_url = self._runner_model_url(target_model_url)
        files = {
            "/work/model_url.txt": model_url,
            "/work/request.json": agent_body.model_dump_json(),
            "/work/agent_config.json": json.dumps(agent_config),
            "/work/runner_config.json": json.dumps(runner_config),
            "/work/runner.py": runner_script,
        }
        if self.config.model_transport == "endpoint_bridge":
            files["/work/model_bridge_server.py"] = (
                Path(__file__).with_name("model_bridge_server.py").read_text(encoding="utf-8")
            )
        handle = await self._provision_box(image, files, model_url, workspace_setup)
        try:
            r = await self._exec_runner(handle, runner_cmd, target_model_url)
            self._log_runner_result(r)

            resp = NeMoGymResponse.model_validate(await self._download_json(handle, "/work/response.json"))

            if workspace_setup and workspace_setup.artifact_paths:
                artifacts = await self._download_artifacts(handle, workspace_setup)
                resp.metadata = (resp.metadata or {}) | {SANDBOX_ARTIFACTS_METADATA_KEY: json.dumps(artifacts)}

            if meta.get("patch_workdir"):
                wd = meta["patch_workdir"]
                r = await self._provider.exec(
                    handle, f"git -C {wd} add -A . && git -C {wd} diff --cached", timeout_s=120
                )
                resp.metadata = (resp.metadata or {}) | {"model_patch": r.stdout or ""}

            grade_raw = meta.get("sandbox_eval")
            grade_spec = json.loads(grade_raw) if isinstance(grade_raw, str) else grade_raw
            if grade_spec:
                reward = await self._grade_in_box(handle, grade_spec)
                resp.metadata = (resp.metadata or {}) | {"sandbox_reward": str(reward)}
            return resp
        finally:
            await self._close_box(handle)


if __name__ == "__main__":
    SandboxAgent.run_webserver()
