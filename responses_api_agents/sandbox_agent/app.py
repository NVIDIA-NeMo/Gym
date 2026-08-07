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
import json
import logging
import re
import shlex
import socket
import subprocess
import tempfile
from asyncio import Semaphore
from pathlib import Path
from typing import Any, Literal, Mapping, Optional
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

from fastapi import Request
from omegaconf import ListConfig
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
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
from nemo_gym.sandbox.providers.base import SandboxSpec
from nemo_gym.sandbox.providers.registry import create_provider
from nemo_gym.server_utils import get_response_json, raise_for_status


LOG = logging.getLogger(__name__)


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
    setup_commands: list[str] = Field(default_factory=list)
    sandbox_python: str = "python3"
    # "auto" tars the local repo, or a path to a prebuilt tar.gz, or a URL fetched in the sandbox
    gym_source: str = "auto"

    eval_timeout: int = 1800
    rollout_timeout: int = 2400


class SandboxAgentRunRequest(BaseRunRequest):
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
        """Model endpoint URL, host rewritten to an IP reachable from sandboxes, no /v1 suffix."""
        cfg = get_first_server_config_dict(self.server_client.global_config_dict, self.config.model_server.name)
        base = cfg.get("base_url") or self.server_client._build_server_base_url(cfg)
        if isinstance(base, (list, ListConfig)):
            base = base[0]
        base = re.sub(r"/v1/?$", "", str(base))
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

            agent_resp = await self.server_client.post(
                server_name=self.config.name,
                url_path="/v1/responses",
                json=body.responses_create_params,
                cookies=cookies,
            )
            await raise_for_status(agent_resp)
            cookies = agent_resp.cookies
            agent_resp_json = await get_response_json(agent_resp)

            verify_resp = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=body.model_dump() | {"response": agent_resp_json},
                cookies=cookies,
            )
            await raise_for_status(verify_resp)
            return BaseVerifyResponse.model_validate(await get_response_json(verify_resp))

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

    async def _provision_box(self, image: str, files: dict[str, str], model_url: str):
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
            for cmd in self.config.setup_commands:
                r = await self._provider.exec(handle, cmd, timeout_s=900)
                if r.return_code != 0:
                    LOG.warning("setup failed (%d): %s | %s", r.return_code, cmd, (r.stderr or "")[:300])

            pm = re.match(r"https?://([^:/]+):(\d+)", model_url)
            if pm:
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

    async def _run_nested(self, request: Request, body: SandboxAgentRunRequest) -> BaseVerifyResponse:
        row = body.model_dump()
        row.pop("agent_ref", None)
        meta = (row.get("responses_create_params") or {}).get("metadata") or {}
        image = meta.get("docker_image") or self.config.sandbox_image
        runner_script, runner_config, runner_cmd = self._runner()
        model_url = self._sandbox_model_url(request)
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

        runner_script, runner_config, runner_cmd = self._runner()
        agent_body = body.model_copy(deep=True)
        if getattr(agent_body, "metadata", None):
            agent_body.metadata = {k: v for k, v in agent_body.metadata.items() if k != "sandbox_eval"}
        agent_config = dict(self.config.agent_config)
        if meta.get("workdir"):
            agent_config = agent_config | {"repo_dir": meta["workdir"]}
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
            r = await self._provider.exec(handle, runner_cmd, timeout_s=self.config.rollout_timeout)
            if "RUNNER_DONE" not in (r.stdout or ""):
                LOG.warning("runner incomplete: %s", (r.stderr or r.stdout or "")[-6000:])

            resp = NeMoGymResponse.model_validate(await self._download_json(handle, "/work/response.json"))

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
