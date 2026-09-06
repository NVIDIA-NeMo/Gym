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

"""GDPVal agent that runs a pluggable harness inside the GDPVal container."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from time import monotonic
from typing import Any, Dict, Optional

from fastapi import Request
from pydantic import ConfigDict, Field, field_validator

from nemo_gym import PARENT_DIR
from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, Body, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import AsyncSandbox, SandboxSpec, resolve_provider_config, resolve_provider_metadata
from nemo_gym.server_utils import get_response_json, raise_for_status
from responses_api_agents.stirrup_agent.tasks.gdpval import _download_reference_files, _parse_json_str


_RUNNER_SOURCE_PATH = Path(__file__).with_name("sandbox_entrypoint.py")
_PROMPT_PATH = Path(__file__).parent / "prompts" / "gdpval_user_prompt.txt"

# Only what is never a deliverable. Source files are the deliverable for the software tasks,
# so .py stays harvestable and .ipynb is not scratch.
_SCRATCH_SUFFIXES = frozenset({".pyc", ".pyo", ".log"})
# Directories a build leaves behind. One task shipped 1497 node_modules files, which flattened
# into the target and exceeded the judge's context limit.
_EXCLUDED_DIRS = frozenset({"node_modules", ".git", "__pycache__", ".venv", "venv", ".cache"})
_MAX_DELIVERABLES = 100
_DELIVERABLE_SUFFIXES = frozenset(
    {
        ".docx",
        ".doc",
        ".odt",
        ".rtf",
        ".xlsx",
        ".xls",
        ".ods",
        ".csv",
        ".pptx",
        ".ppt",
        ".odp",
        ".pdf",
        ".md",
        ".txt",
        ".html",
        ".json",
        ".xml",
        ".png",
        ".jpg",
        ".jpeg",
        ".svg",
        ".gif",
        ".mp3",
        ".mp4",
        ".wav",
        ".zip",
        ".eml",
        ".ipynb",
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".css",
        ".sol",
        ".circom",
        ".yaml",
        ".yml",
        ".sql",
        ".sh",
        ".step",
        ".stp",
        ".psd",
        ".tex",
    }
)


def agent_key(agent_server_module: str) -> str:
    """responses_api_agents.hermes_agent.app maps to hermes_agent, the deps-script key."""
    parts = agent_server_module.split(".")
    return parts[-2] if len(parts) >= 2 else agent_server_module


def deps_recipe_key(*paths: Path) -> str:
    blob = b"".join(p.read_bytes() for p in paths if p.exists()) or b"no-script"
    return hashlib.sha256(blob).hexdigest()


def deps_build_env(deps_dir: Path, portable_python_sh: Path) -> dict[str, str]:
    build_dir = deps_dir.parent / f".{deps_dir.name}-build"
    cache_dir, temp_dir, home_dir = build_dir / "cache", build_dir / "tmp", build_dir / "home"
    for path in (cache_dir, temp_dir, home_dir):
        path.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "DEPS_DIR": str(deps_dir),
            "NEMO_GYM_ROOT": str(PARENT_DIR),
            "PORTABLE_PYTHON_SH": str(portable_python_sh),
            "HOME": str(home_dir),
            "PYTHONPATH": "",
            "PIP_CACHE_DIR": str(cache_dir / "pip"),
            "UV_CACHE_DIR": str(cache_dir / "uv"),
            "XDG_CACHE_HOME": str(cache_dir),
            "TMPDIR": str(temp_dir),
        }
    )
    return env


def build_user_prompt(task_prompt: str, reference_paths: list[str], output_dir: str) -> str:
    listing = "\n".join(f"- {p}" for p in sorted(reference_paths)) or "None"
    return _PROMPT_PATH.read_text(encoding="utf-8").format(
        task=task_prompt, reference_files=listing, output_dir=output_dir
    )


def is_deliverable(name: str) -> bool:
    suffix = Path(name).suffix.lower()
    return suffix in _DELIVERABLE_SUFFIXES and suffix not in _SCRATCH_SUFFIXES


class GDPValAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: Optional[ModelServerRef] = None

    concurrency: int = 4
    timeout: int = 5400
    system_prompt: Optional[str] = None

    agent_server_module: str = "responses_api_agents.hermes_agent.app"
    agent_server_class: str = "HermesAgent"
    agent_config_class: str = "HermesAgentConfig"
    agent_kwargs: Dict[str, Any] = Field(default_factory=dict)

    image: str
    sandbox_provider: str | Dict[str, Any] = Field(default_factory=lambda: {"apptainer": {}})
    sandbox_spec: Dict[str, Any] = Field(default_factory=dict)
    container_workdir: str = "/workspace"
    container_deps_dir: str = "/agent_deps_mount"

    # The resources server resolves this from its own working directory, which differs from ours.
    persist_deliverables_dir: str

    @field_validator("persist_deliverables_dir")
    @classmethod
    def _require_absolute(cls, value: str) -> str:
        if not Path(value).is_absolute():
            raise ValueError("persist_deliverables_dir must be an absolute path")
        return value


class GDPValAgentRunRequest(BaseRunRequest):
    # The GDPVal row fields and the framework's rollout index arrive as extra keys and must
    # reach /verify; without them the judge receives no rubric.
    model_config = ConfigDict(extra="allow")


class GDPValAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class GDPValAgent(SimpleResponsesAPIAgent):
    config: GDPValAgentConfig
    model_config = ConfigDict(arbitrary_types_allowed=True)

    sem: Any = None
    _setup_lock: Any = None
    _deps_dir: Any = None
    _image: Any = None
    _sandbox_provider: Any = None
    _sandbox_metadata: Any = None

    def model_post_init(self, __context: Any) -> None:
        self.sem = asyncio.Semaphore(self.config.concurrency)
        self._setup_lock = asyncio.Lock()
        self._sandbox_provider = resolve_provider_config(self.config.sandbox_provider)
        self._sandbox_metadata = resolve_provider_metadata(self._sandbox_provider)
        # At startup rather than on the first rollout, so a pre-warm that only starts the
        # servers still leaves the prefix in place.
        self._deps_dir = self._provision_deps()

    def _provision_deps(self) -> Path:
        key = agent_key(self.config.agent_server_module)
        script = PARENT_DIR / "responses_api_agents" / key / "scripts" / f"{key}_deps.sh"
        if not script.exists():
            raise RuntimeError(f"no setup script for {key!r} at {script}")
        portable_python_sh = Path(__file__).parent / "setup_scripts" / "_portable_python.sh"
        # Named the way the evaluation pipeline derives it, so a pinned harness version can
        # drop the sentinel to force a rebuild.
        deps_dir = Path(__file__).parent / "deps" / f"gdpval_{key}_deps"
        sentinel = deps_dir / ".installed"
        recipe = deps_recipe_key(script, portable_python_sh)
        if sentinel.exists() and sentinel.read_text().strip() == recipe:
            return deps_dir
        deps_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["bash", str(script)], env=deps_build_env(deps_dir, portable_python_sh), check=True)
        sentinel.write_text(recipe)
        return deps_dir

    def _resolve_image(self) -> str:
        image = self.config.image.strip()
        if not image:
            raise ValueError("no container image configured; set GDPVAL_CONTAINER_PATH")
        is_apptainer = "apptainer" in self._sandbox_provider
        if image.endswith(".sif") or image.startswith(("/", ".")):
            if not is_apptainer:
                raise ValueError("local or .sif images require the Apptainer sandbox provider")
            return image
        return image.removeprefix("docker://")

    def _model_url(self, rollout_id: Optional[str]) -> str:
        if not self.config.model_server:
            return ""
        # Rollout-prefixed, so the model server can attribute the harness's calls to this rollout.
        # The bare base URL reaches the same server but files the calls under no rollout at all.
        return self.resolve_model_base_url(self.config.model_server.name, rollout_id)

    def _paths(self) -> tuple[str, str, str, str]:
        wd = self.config.container_workdir.rstrip("/")
        return wd, f"{wd}/input", f"{wd}/output", f"{wd}/.nv"

    def _build_spec(
        self, body: GDPValAgentRunRequest, instruction: str, image: str, deps_dir: Path,
        rollout_id: Optional[str],
    ) -> SandboxSpec:
        wd, _, _, traj = self._paths()
        extra = dict(self.config.sandbox_spec)
        provider_options = dict(extra.pop("provider_options", {}) or {})
        binds = list(provider_options.pop("binds", []) or [])
        binds.append(f"{deps_dir}:{self.config.container_deps_dir}:ro")
        provider_options["binds"] = binds
        metadata = dict(self._sandbox_metadata)
        metadata.update(extra.pop("metadata", {}) or {})
        return SandboxSpec(
            image=image,
            workdir=wd,
            env={
                "GDPVAL_MODEL_URL": self._model_url(rollout_id),
                "GDPVAL_ROLLOUT_ID": rollout_id or "",
                "GDPVAL_MODEL_NAME": body.responses_create_params.model or "model",
                "GDPVAL_AGENT_KWARGS": json.dumps(self.config.agent_kwargs),
                "GDPVAL_AGENT_HOME": f"{wd}/.home",
                "GDPVAL_SYSTEM_PROMPT": self.config.system_prompt or "",
                "GDPVAL_TRAJ_DIR": traj,
                "GDPVAL_WORKDIR": wd,
                "GDPVAL_AGENT_MODULE": self.config.agent_server_module,
                "GDPVAL_AGENT_CLASS": self.config.agent_server_class,
                "GDPVAL_AGENT_CFG_CLASS": self.config.agent_config_class,
                "GDPVAL_AGENT_DEPS_DIR": self.config.container_deps_dir,
            },
            files={
                f"{traj}/instruction.txt": instruction,
                f"{traj}/agent_runner.py": _RUNNER_SOURCE_PATH.read_text(encoding="utf-8"),
            },
            metadata=metadata,
            provider_options=provider_options,
            **extra,
        )

    async def _list_deliverables(self, box: AsyncSandbox) -> list[str]:
        wd, input_dir, output_dir, traj = self._paths()
        listing = await box.exec(f"find {shlex.quote(output_dir)} -type f 2>/dev/null", timeout_s=60)
        found = [line.strip() for line in (listing.stdout or "").splitlines() if line.strip()]
        if found:
            return found
        # The harness has no tool for declaring deliverables, so a model that ignored the
        # output directory still leaves its work somewhere under the workspace.
        excluded = " ".join(f"-not -path {shlex.quote(f'{d}/*')}" for d in (traj, input_dir, f"{wd}/.home"))
        sweep = await box.exec(
            f"find {shlex.quote(wd)} -maxdepth 2 -type f "
            f"-newer {shlex.quote(f'{traj}/started')} {excluded} 2>/dev/null",
            timeout_s=60,
        )
        return [line.strip() for line in (sweep.stdout or "").splitlines() if line.strip()]

    def _deliverables_dir(self, body: GDPValAgentRunRequest) -> Path:
        extra = body.model_extra or {}
        task_id = extra.get("task_id") or "unknown"
        repeat = extra.get("_ng_rollout_index")
        repeat_name = f"repeat_{repeat}" if repeat is not None else "repeat_0"
        return Path(self.config.persist_deliverables_dir) / f"task_{task_id}" / repeat_name

    async def _collect(self, box: AsyncSandbox, target: Path) -> int:
        shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True, exist_ok=True)
        collected = 0
        for remote in await self._list_deliverables(box):
            name = Path(remote).name
            if not is_deliverable(name):
                continue
            if _EXCLUDED_DIRS.intersection(Path(remote).parts):
                continue
            if collected >= _MAX_DELIVERABLES:
                print(f"[gdpval_agent] stopping at {_MAX_DELIVERABLES} deliverables", flush=True)
                break
            # The scorer lists this directory non-recursively, so everything lands flat.
            dest = target / name
            if dest.exists():
                dest = target / f"{Path(remote).parent.name}_{name}"
            try:
                await box.download(remote, dest)
                collected += 1
            except Exception as e:
                print(f"[gdpval_agent] could not download {remote}: {e}", flush=True)
        return collected

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        raise NotImplementedError("the harness runs inside the sandbox and serves its own responses endpoint")

    async def run(self, request: Request, body: GDPValAgentRunRequest) -> GDPValAgentVerifyResponse:
        rollout_id = self.rollout_id_from_run(body)
        extra = body.model_extra or {}
        task_prompt = extra.get("prompt") or ""
        reference_files = _parse_json_str(extra.get("reference_files") or [], [])
        reference_urls = _parse_json_str(extra.get("reference_file_urls") or [], [])
        wd, input_dir, output_dir, traj = self._paths()

        async with self.sem:
            async with self._setup_lock:
                if self._image is None:
                    self._image = await asyncio.to_thread(self._resolve_image)

            with tempfile.TemporaryDirectory(prefix="gdpval_agent_") as scratch:
                staged = Path(scratch) / "input"
                staged.mkdir(parents=True, exist_ok=True)
                downloaded = []
                if reference_files and reference_urls:
                    downloaded = await asyncio.to_thread(
                        _download_reference_files, reference_files, reference_urls, staged
                    )
                    if len(downloaded) != len(reference_files):
                        print(
                            f"[gdpval_agent] staged {len(downloaded)}/{len(reference_files)} reference files",
                            flush=True,
                        )

                local_refs = sorted(p for p in staged.rglob("*") if p.is_file())
                container_refs = [f"{input_dir}/{p.relative_to(staged).as_posix()}" for p in local_refs]
                instruction = build_user_prompt(task_prompt, container_refs, output_dir)
                spec = self._build_spec(body, instruction, self._image, self._deps_dir, rollout_id)

                async with AsyncSandbox(self._sandbox_provider, spec) as box:
                    await box.start()
                    for local, remote in zip(local_refs, container_refs):
                        await box.upload(local, remote)
                    await box.exec(
                        f"mkdir -p {shlex.quote(output_dir)} && touch {shlex.quote(traj)}/started", timeout_s=60
                    )

                    started = monotonic()
                    result = await box.exec(
                        f"{shlex.quote(self.config.container_deps_dir)}/bin/python {traj}/agent_runner.py",
                        cwd=wd,
                        timeout_s=self.config.timeout,
                    )
                    agent_run_time = monotonic() - started
                    if result.return_code != 0:
                        details = (result.stderr or result.stdout or "")[-2000:]
                        print(f"[gdpval_agent] harness exited {result.return_code}: {details}", flush=True)

                    response = await self._load_response(box, Path(scratch) / "response.json")
                    target = self._deliverables_dir(body)
                    collected = await self._collect(box, target)

            print(f"[gdpval_agent] collected {collected} deliverable(s) into {target}", flush=True)
            payload = body.model_dump() | {
                "response": response.model_dump(mode="json"),
                "deliverables_dir": str(target),
                "agent_run_time": round(agent_run_time, 3),
            }
            verify_resp = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=payload,
                cookies=request.cookies,
            )
            await raise_for_status(verify_resp)
            return await get_response_json(verify_resp)

    async def _load_response(self, box: AsyncSandbox, local: Path) -> NeMoGymResponse:
        _, _, _, traj = self._paths()
        try:
            await box.download(f"{traj}/response.json", local)
            return NeMoGymResponse.model_validate_json(local.read_text())
        except Exception as e:
            print(f"[gdpval_agent] could not load harness response: {e}", flush=True)
            return NeMoGymResponse(
                id="gdpval-agent-error",
                created_at=0.0,
                model="error",
                object="response",
                output=[],
                tools=[],
                parallel_tool_calls=False,
                tool_choice="auto",
            )


if __name__ == "__main__":
    GDPValAgent.run_webserver()
