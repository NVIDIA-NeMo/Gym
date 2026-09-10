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
import json
import logging
import re
import shlex
import socket
import subprocess
import tempfile
from asyncio import Semaphore
from pathlib import Path
from typing import Any, Mapping, Optional
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

from fastapi import Request
from omegaconf import ListConfig
from pydantic import ConfigDict, Field

from nemo_gym.agents.config import AgentHarnessConfig
from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.global_config import get_first_server_config_dict, get_global_config_dict
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.rollout_observability import AgentObservationBundle
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.sandbox.providers.base import SandboxSpec
from nemo_gym.sandbox.providers.registry import create_provider
from nemo_gym.server_utils import get_response_json, raise_for_status


LOG = logging.getLogger(__name__)
_INTERNAL_OBSERVATIONS_KEY = "_ng_agent_observations"


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
    model_server: Optional[ModelServerRef] = None
    concurrency: int = 64
    agent: str
    agent_kwargs: AgentHarnessConfig

    sandbox_provider: str | dict[str, Any]
    sandbox_image: str = "python:3.12-slim"
    sandbox_spec: dict[str, Any] = Field(default_factory=dict)
    setup_commands: list[str] = Field(default_factory=list)
    sandbox_python: str = "python3"
    # "auto" tars the local repo, or a path to a prebuilt tar.gz, or a URL fetched in the sandbox
    gym_source: str = "auto"

    eval_timeout: int = 1800
    rollout_timeout: int = 2400


class HarnessAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class HarnessAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    turns_used: int = 0
    finished_naturally: bool = False
    ng_agent_observations: Optional[AgentObservationBundle] = None


class HarnessAgent(SimpleResponsesAPIAgent):
    config: HarnessAgentConfig
    sem: Semaphore = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)
        named_configs = get_global_config_dict() if isinstance(self.config.sandbox_provider, str) else None
        self._provider = create_provider(resolve_provider_config(self.config.sandbox_provider, named_configs))
        self._sandbox_default_metadata = resolve_provider_metadata(self.config.sandbox_provider, named_configs)
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
            ],
            check=True,
        )
        size_mb = tar_path.stat().st_size / 1e6
        if size_mb > 50:
            LOG.warning("gym source tar is %.0fMB, sandbox uploads will be slow", size_mb)
        return tar_path

    def _sandbox_model_url(self, request: Request) -> str:
        """Model endpoint URL, host rewritten to an IP reachable from sandboxes, no /v1 suffix."""
        if self.config.model_server is None:
            return (self.config.agent_kwargs.model.base_url or "").removesuffix("/v1")
        cfg = get_first_server_config_dict(self.server_client.global_config_dict, self.config.model_server.name)
        base = cfg.get("base_url") or self.server_client._build_server_base_url(cfg)
        if isinstance(base, (list, ListConfig)):
            base = base[0]
        base = re.sub(r"/v1/?$", "", str(base))
        parsed = urlsplit(base if "://" in base else f"http://{base}")
        host = parsed.hostname or ""
        if host in ("127.0.0.1", "localhost", "0.0.0.0") and self._provider.name != "local":
            try:
                # loopback binds are unreachable from a sandbox
                host = socket.gethostbyname(socket.gethostname())
            except OSError:
                pass
        netloc = f"{host}:{parsed.port}" if parsed.port else host
        return urlunsplit((parsed.scheme or "http", netloc, parsed.path, parsed.query, parsed.fragment))

    def _runner(self) -> tuple[str, dict, str]:
        script = (Path(__file__).parent / "agent_runner.py").read_text()
        runner_config = {
            "agent": self.config.agent,
            "model_ref": self.config.model_server.model_dump(mode="json") if self.config.model_server else None,
        }
        return script, runner_config, f"{self.config.sandbox_python} runner.py"

    @staticmethod
    def _box_path(handle, path: str) -> str:
        return path.lstrip("/") if handle.provider_name == "local" else path

    async def run(self, request: Request, body: HarnessAgentRunRequest) -> BaseVerifyResponse:
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
            observations = agent_resp_json.pop(_INTERNAL_OBSERVATIONS_KEY, None)

            verify_resp = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=body.model_dump() | {"response": agent_resp_json},
                cookies=cookies,
            )
            await raise_for_status(verify_resp)
            result = await get_response_json(verify_resp)
            gym_response = NeMoGymResponse.model_validate(agent_resp_json)
            turns = sum(
                getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant"
                for item in gym_response.output
            )
            last = gym_response.output[-1] if gym_response.output else None
            return HarnessAgentVerifyResponse.model_validate(
                result
                | {
                    "turns_used": turns,
                    "finished_naturally": getattr(last, "type", None) == "message"
                    and getattr(last, "role", None) == "assistant",
                    "ng_agent_observations": observations,
                }
            )

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
        sandbox_spec = dict(self.config.sandbox_spec)
        sandbox_spec["metadata"] = self._sandbox_default_metadata | dict(sandbox_spec.get("metadata", {}))
        spec = SandboxSpec(image=image, **sandbox_spec)
        handle = await self._provider.create(spec)
        try:
            work_dir = self._box_path(handle, "/work")
            await self._provider.exec(handle, f"mkdir -p {shlex.quote(work_dir)}", timeout_s=60)
            with tempfile.TemporaryDirectory() as td:
                for i, (target, content) in enumerate(files.items()):
                    local = Path(td) / str(i)
                    local.write_text(content)
                    await self._provider.upload_file(handle, local, self._box_path(handle, target))
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
            lines = local.read_text().splitlines()
            if len(lines) != 1:
                raise RuntimeError(f"expected one JSON row in {path}, got {len(lines)}")
            return json.loads(lines[0])

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
        agent_kwargs = self.config.agent_kwargs.model_dump(mode="json")
        if meta.get("workdir"):
            agent_kwargs = agent_kwargs | {"workspace": meta["workdir"]}
        model_url = self._sandbox_model_url(request)
        files = {
            "/work/model_url.txt": model_url,
            "/work/request.json": agent_body.model_dump_json(),
            "/work/agent_kwargs.json": json.dumps(agent_kwargs),
            "/work/runner_config.json": json.dumps(runner_config),
            "/work/runner.py": runner_script,
        }
        handle = await self._provision_box(image, files, model_url)
        try:
            work_dir = self._box_path(handle, "/work")
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

            response_json = await self._download_json(handle, self._box_path(handle, "/work/response.json"))
            if not getattr(request, "path_params", {}).get("rollout_id"):
                response_json.pop(_INTERNAL_OBSERVATIONS_KEY, None)
            resp = NeMoGymResponse.model_validate(response_json)

            grade_raw = meta.get("sandbox_eval")
            grade_spec = json.loads(grade_raw) if isinstance(grade_raw, str) else grade_raw
            if grade_spec:
                reward = await self._grade_in_box(handle, grade_spec)
                resp.metadata = (resp.metadata or {}) | {"sandbox_reward": str(reward)}
            return resp
        finally:
            await self._close_box(handle)


if __name__ == "__main__":
    HarnessAgent.run_webserver()
