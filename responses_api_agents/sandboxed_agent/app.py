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

import json
import logging
import re
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
    """Stage eval files into the live sandbox, run the eval command, and read back the reward file."""
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


DELEGATE_RUNNER = """
import asyncio, json, sys
sys.path.insert(0, "/gym_mount")
import nemo_gym
assert nemo_gym.__file__.startswith("/gym_mount"), f"wrong nemo_gym: {{nemo_gym.__file__}}"
from unittest.mock import MagicMock
from omegaconf import OmegaConf
from {delegate_module} import {delegate_class}, {delegate_config_class}
from nemo_gym.config_types import BaseServerConfig
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient

body = json.load(open("/work/request.json"))
model_url = open("/work/model_url.txt").read().strip()
# any __SANDBOX_MODEL_URL__ in the delegate config resolves to the sandbox-reachable model URL
cfg_raw = open("/work/delegate_config.json").read().replace("__SANDBOX_MODEL_URL__", model_url)
cfg_dict = json.loads(cfg_raw)

cfg = {delegate_config_class}(host="", port=0, entrypoint="", name="delegate", **cfg_dict)
sc = ServerClient(head_server_config=BaseServerConfig(host="127.0.0.1", port=0), global_config_dict=OmegaConf.create({{}}))
agent = {delegate_class}(config=cfg, server_client=sc)

params = NeMoGymResponseCreateParamsNonStreaming.model_validate(body)
resp = asyncio.run(agent.responses(MagicMock(), params))
open("/work/response.json", "w").write(resp.model_dump_json())
print("RUNNER_DONE")
"""

NESTED_GYM_RUNNER = """
set -e
export NEMO_GYM_POLICY_BASE_URL="$(cat /work/model_url.txt)"
cd /gym
nohup ng_run "+config_paths=[{config_paths}]" > /work/gym.log 2>&1 &
for i in $(seq 1 150); do
  curl -s -m 2 http://127.0.0.1:{agent_port}/health > /dev/null && break
  sleep 2
done
curl -s -m {timeout} -X POST http://127.0.0.1:{agent_port}/run \\
  -H "content-type: application/json" -d @/work/request.json > /work/response.json
echo RUNNER_DONE
"""


class SandboxedAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: Optional[ModelServerRef] = None
    concurrency: int = 64
    mode: Literal["agent_only_runner", "gym_runner"] = "agent_only_runner"

    delegate_module: Optional[str] = None
    delegate_class: Optional[str] = None
    delegate_config_class: Optional[str] = None
    delegate_config: dict[str, Any] = Field(default_factory=dict)

    nested_config_paths: list[str] = Field(default_factory=list)
    nested_agent_port: int = 11001

    sandbox_provider: dict[str, Any]
    sandbox_image: str = "python:3.12-slim"
    image_from_metadata_key: Optional[str] = None
    workspace_from_metadata_key: Optional[str] = None
    sandbox_spec: dict[str, Any] = Field(default_factory=dict)
    setup_commands: list[str] = Field(default_factory=list)
    sandbox_python: str = "python3"

    grade_in_box: bool = False
    patch_workdir: Optional[str] = None # TODO a bit too swe specific right now
    grade_metadata_key: str = "sandbox_eval"
    eval_timeout: int = 1800
    rollout_timeout: int = 2400


class SandboxedAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class SandboxedAgent(SimpleResponsesAPIAgent):
    config: SandboxedAgentConfig
    sem: Semaphore = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)
        self._provider = create_provider(self.config.sandbox_provider)
        self._gym_tar = self._build_gym_tar() if self.config.mode == "agent_only_runner" else None

    def _build_gym_tar(self) -> Path:
        root = Path(__file__).resolve().parent.parent.parent
        delegate_pkg = "/".join((self.config.delegate_module or "responses_api_agents").split(".")[:-1])
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
                delegate_pkg,
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

    def _runner(self) -> tuple[str, str, str]:
        if self.config.mode == "agent_only_runner":
            script = DELEGATE_RUNNER.format(
                delegate_module=self.config.delegate_module,
                delegate_class=self.config.delegate_class,
                delegate_config_class=self.config.delegate_config_class,
            )
            return "/work/runner.py", script, f"{self.config.sandbox_python} /work/runner.py"
        script = NESTED_GYM_RUNNER.format(
            config_paths=",".join(self.config.nested_config_paths),
            agent_port=self.config.nested_agent_port,
            timeout=self.config.rollout_timeout,
        )
        return "/work/runner.sh", script, "bash /work/runner.sh"

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        return await self._run_in_sandbox(request, body)

    async def run(self, request: Request, body: SandboxedAgentRunRequest) -> BaseVerifyResponse:
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

    async def _run_in_sandbox(self, request, body) -> NeMoGymResponse:
        image = self.config.sandbox_image
        meta = getattr(body, "metadata", None) or {}
        if self.config.image_from_metadata_key and meta.get(self.config.image_from_metadata_key):
            image = meta[self.config.image_from_metadata_key]

        runner_path, runner_script, runner_cmd = self._runner()
        delegate_body = body.model_copy(deep=True)
        if getattr(delegate_body, "metadata", None):
            delegate_body.metadata = {
                k: v for k, v in delegate_body.metadata.items() if k != self.config.grade_metadata_key
            }
        delegate_config = dict(self.config.delegate_config)
        workdir = (
            meta.get(self.config.workspace_from_metadata_key) if self.config.workspace_from_metadata_key else None
        )
        if workdir:
            delegate_config = delegate_config | {"repo_dir": workdir}
        model_url = self._sandbox_model_url(request)
        files = {
            "/work/model_url.txt": model_url,
            "/work/request.json": delegate_body.model_dump_json(),
            "/work/delegate_config.json": json.dumps(delegate_config),
            runner_path: runner_script,
        }
        spec = SandboxSpec(image=image, **dict(self.config.sandbox_spec))
        handle = await self._provider.create(spec)
        try:
            await self._provider.exec(handle, "mkdir -p /work", timeout_s=60)
            with tempfile.TemporaryDirectory() as td:
                for i, (target, content) in enumerate(files.items()):
                    local = Path(td) / str(i)
                    local.write_text(content)
                    await self._provider.upload_file(handle, local, target)
            if self._gym_tar is not None:
                await self._provider.upload_file(handle, self._gym_tar, "/work/gym_src.tar.gz")
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
            r = await self._provider.exec(handle, runner_cmd, timeout_s=self.config.rollout_timeout)
            if "RUNNER_DONE" not in (r.stdout or ""):
                LOG.warning("runner incomplete: %s", (r.stderr or r.stdout or "")[-6000:])

            with tempfile.TemporaryDirectory() as td:
                local = Path(td) / "response.json"
                await self._provider.download_file(handle, "/work/response.json", local)
                resp = NeMoGymResponse.model_validate(json.loads(local.read_text()))

            if self.config.patch_workdir:
                wd = self.config.patch_workdir
                r = await self._provider.exec(
                    handle, f"git -C {wd} add -A . && git -C {wd} diff --cached", timeout_s=120
                )
                resp.metadata = (resp.metadata or {}) | {"model_patch": r.stdout or ""}

            grade_raw = meta.get(self.config.grade_metadata_key) if self.config.grade_in_box else None
            grade_spec = json.loads(grade_raw) if isinstance(grade_raw, str) else grade_raw
            if grade_spec:
                reward = await self._grade_in_box(handle, grade_spec)
                resp.metadata = (resp.metadata or {}) | {"sandbox_reward": str(reward)}
            return resp
        finally:
            if handle is not None:
                try:
                    await self._provider.close(handle)
                except Exception:
                    LOG.warning("sandbox close failed", exc_info=True)


if __name__ == "__main__":
    SandboxedAgent.run_webserver()
