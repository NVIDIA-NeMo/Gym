# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""harness-agnostic cvdp agentic agent on the provider-neutral sandbox api.

boots any gym responses() agent (chosen by agent_server_module/class/config_class) inside an
eda sim sandbox, harvests the rtl it edits, and grades it via /verify. built on AsyncSandbox
plus a single-key sandbox_provider config, so the backend (apptainer, opensandbox) and the
harness are both config, not code.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from subprocess import Popen
from time import time
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import Request
from pydantic import ConfigDict

from nemo_gym import PARENT_DIR
from nemo_gym.base_resources_server import BaseRunRequest
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, Body, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.global_config import get_first_server_config_dict
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import AsyncSandbox, SandboxSpec
from nemo_gym.server_utils import get_response_json, raise_for_status
from responses_api_agents.cvdp_agent.sandbox_runner import agent_key, deps_recipe_key, harvest, load_runner_source


_DEFAULT_HARVEST_GLOBS = ["rtl/**/*.sv", "rtl/**/*.v", "rtl/**/*.vhd", "verif/**/*.sv", "verif/**/*.v"]


class CvdpGenericAgentRunRequest(BaseRunRequest):
    # extra="allow" so verifier_metadata (and any other task fields) survive parsing and are
    # carried through to /verify. BaseRunRequest drops unknown fields, which would 422 /verify.
    model_config = ConfigDict(extra="allow")


def _is_harness_path(rel: str) -> bool:
    """True for paths that belong to the hidden grading harness and must never be
    seeded into the agent workspace (the test scripts in ``src/`` and the compose
    file)."""
    norm = rel.replace("\\", "/").strip("/")
    return (
        norm == "src"
        or norm.startswith("src/")
        or norm == "docker-compose.yml"
        or norm.endswith("/docker-compose.yml")
    )


def _safe_rel(rel: str) -> bool:
    """reject absolute paths or ones that escape the workspace via ..."""
    if rel.startswith("/"):
        return False
    parts = Path(rel).parts
    return ".." not in parts


class CvdpGenericAgentConfig(BaseResponsesAPIAgentConfig):
    """config for the harness-agnostic, provider-neutral sandboxed cvdp agent."""

    # Optional: when a harness brings its own model endpoint (e.g. claude_code via
    # anthropic_base_url/api_key in agent_kwargs), leave this unset and don't start a Gym
    # model server. When set, its URL is passed into the sandbox as NV_MODEL_URL.
    model_server: Optional[ModelServerRef] = None
    resources_server: ResourcesServerRef
    concurrency: int = 8
    system_prompt: Optional[str] = None
    timeout: int = 1800

    # which gym agent to boot in the sandbox (the whole any-harness surface)
    agent_server_module: str = "responses_api_agents.claude_code_agent.app"
    agent_server_class: str = "ClaudeCodeAgent"
    agent_config_class: str = "ClaudeCodeAgentConfig"
    agent_kwargs: Dict[str, Any] = {}

    # sandbox wiring (provider-neutral). sandbox_provider is a single-key provider config,
    # e.g. {"apptainer": {...}} or {"opensandbox": {...}}. sandbox_spec carries extra spec
    # fields (provider_options, ttl_s, ...) merged onto the per-task spec.
    # may be a bare docker ref (e.g. "nvidia/cvdp-sim:v1.0.0"), an explicit .sif path, or a
    # docker:// / oras:// uri. A bare docker ref is resolved to a cached .sif on the host (same
    # convention as the cvdp verifier's harness), so one image value works for agent + verifier.
    image: str = "nvidia/cvdp-sim:v1.0.0"
    sif_cache_dir: str = ""  # defaults to ~/.cache/nemo-gym/sif (matches the cvdp harness cache)
    sandbox_provider: Dict[str, Any] = {"apptainer": {}}
    sandbox_spec: Dict[str, Any] = {}
    container_workdir: str = "/code"
    harvest_globs: list[str] = _DEFAULT_HARVEST_GLOBS
    # how nemo_gym + the agent deps prefix reach the sandbox: "bind" (zero-copy, apptainer/local)
    # or "baked" (already in the image, e.g. opensandbox)
    deps_provision: str = "bind"


class CvdpGenericAgent(SimpleResponsesAPIAgent):
    """runs any gym agent inside an eda sim sandbox for rtl tasks, provider-neutral."""

    config: CvdpGenericAgentConfig
    model_config = ConfigDict(arbitrary_types_allowed=True)

    sem: Any = None
    _deps_dir: Any = None
    _image: Any = None
    _setup_lock: Any = None

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self.sem = asyncio.Semaphore(self.config.concurrency)
        self._deps_dir = None
        self._image = None
        # serialize one-time host setup (deps prefix + image pull) so concurrent
        # requests don't race into the same deps dir / sif cache on first run.
        self._setup_lock = asyncio.Lock()

    def _provision_deps(self) -> Path:
        """install the configured agent's deps prefix once, mounted at /agent_deps_mount."""
        key = agent_key(self.config.agent_server_module)
        scripts_dir = Path(__file__).parent / "setup_scripts"
        deps_dir = Path(__file__).parent / "deps" / key
        script = scripts_dir / f"{key}_deps.sh"
        sentinel = deps_dir / ".installed"
        # fingerprint the per-harness script AND the shared helper it sources, so editing
        # either one invalidates cached prefixes and forces a rebuild.
        recipe = deps_recipe_key(script, scripts_dir / "_portable_python.sh")
        if sentinel.exists() and sentinel.read_text().strip() == recipe:
            return deps_dir
        if not script.exists():
            raise RuntimeError(f"no setup script for {key!r} at {script}")
        deps_dir.mkdir(parents=True, exist_ok=True)
        proc = Popen(f"DEPS_DIR={deps_dir} NEMO_GYM_ROOT={PARENT_DIR} bash {script}", shell=True)
        assert proc.wait() == 0, f"agent deps setup failed ({script})"
        sentinel.write_text(recipe)
        return deps_dir

    def _resolve_image(self) -> str:
        """Map config.image to something the provider can start directly.

        An explicit .sif path or a fully-qualified uri (docker://, oras://, ...) is used as-is.
        A bare docker ref is converted to a cached .sif under sif_cache_dir, pulling it on first
        use. This mirrors the cvdp verifier's harness cache (same safe-name scheme), so the same
        ``nvidia/cvdp-sim:v1.0.0`` value resolves to the identical .sif on both sides and never
        triggers a docker.io pull at apptainer ``instance start`` time.
        """
        img = self.config.image
        if img.endswith(".sif") or img.startswith(("/", ".")) or "://" in img:
            return img
        cache = self.config.sif_cache_dir or os.path.join(Path.home(), ".cache", "nemo-gym", "sif")
        os.makedirs(cache, exist_ok=True)
        sif_path = os.path.join(cache, img.replace("/", "_").replace(":", "_") + ".sif")
        if os.path.exists(sif_path):
            return sif_path
        tmp = sif_path + ".pulling"
        proc = subprocess.run(
            ["apptainer", "pull", "--force", tmp, f"docker://{img}"],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise RuntimeError(f"apptainer pull failed for {img} (exit {proc.returncode}): {proc.stderr}")
        os.rename(tmp, sif_path)
        return sif_path

    def _model_url(self) -> str:
        # no Gym model server -> empty URL; the runner then leaves the harness on its own
        # endpoint (e.g. claude_code's anthropic_base_url from agent_kwargs).
        if not self.config.model_server:
            return ""
        cfg = get_first_server_config_dict(self.server_client.global_config_dict, self.config.model_server.name)
        return self.server_client._build_server_base_url(cfg)

    def _seed_files(
        self, workdir: str, context_files: Dict[str, str], harness_files: Optional[dict]
    ) -> Dict[str, str]:
        """context files to upload into the workspace, skipping harness-like or unsafe paths."""
        forbidden = set(harness_files or {})
        out: Dict[str, str] = {}
        for rel, content in (context_files or {}).items():
            if content is None or rel in forbidden or _is_harness_path(rel) or not _safe_rel(rel):
                continue
            out[f"{workdir.rstrip('/')}/{rel}"] = content
        return out

    def _build_spec(
        self, body: BaseRunRequest, instruction: str, deps_dir: Path, files: Dict[str, str], image: str
    ) -> SandboxSpec:
        wd = self.config.container_workdir
        extra = dict(self.config.sandbox_spec)
        binds = list((extra.pop("provider_options", {}) or {}).get("binds", []))
        if self.config.deps_provision == "bind":
            binds += [f"{PARENT_DIR}:/nemo_gym_mount:ro", f"{deps_dir}:/agent_deps_mount:ro"]
        provider_options = {**(self.config.sandbox_spec.get("provider_options", {})), "binds": binds}
        # runner and instruction live under the workdir mount, not a separate /trajectories_mount,
        # because the provider-neutral spec.files upload only delivers paths under the mount point
        traj = self._traj_dir()
        return SandboxSpec(
            image=image,
            workdir=wd,
            env={
                "NV_MODEL_URL": self._model_url(),
                "NV_MODEL_NAME": body.responses_create_params.model or "model",
                "NV_AGENT_KWARGS": json.dumps(self.config.agent_kwargs),
                "NV_SYSTEM_PROMPT": self.config.system_prompt or "",
                "NV_TRAJ_DIR": traj,
                "NV_AGENT_MODULE": self.config.agent_server_module,
                "NV_AGENT_CLASS": self.config.agent_server_class,
                "NV_AGENT_CFG_CLASS": self.config.agent_config_class,
            },
            files={
                f"{traj}/instruction.txt": instruction,
                f"{traj}/agent_runner.py": load_runner_source(),
                **files,
            },
            provider_options=provider_options,
            **{k: v for k, v in extra.items() if k != "provider_options"},
        )

    def _traj_dir(self) -> str:
        """runner, instruction, and response location, kept under the workdir mount so spec.files lands."""
        return f"{self.config.container_workdir.rstrip('/')}/.nv"

    async def _remote_harvest(self, box: AsyncSandbox, workdir: str, globs: list[str], seeded: Dict[str, str]) -> dict:
        """list and download files matching globs from the sandbox, then filter via harvest()."""
        dirs = sorted({g.split("/")[0] for g in globs if "/" in g}) or ["."]
        listing = await box.exec(
            f"cd {shlex.quote(workdir)} && find {' '.join(shlex.quote(d) for d in dirs)} -type f 2>/dev/null"
        )
        rels = [line.strip().lstrip("./") for line in (listing.stdout or "").splitlines() if line.strip()]
        mirror = Path(tempfile.mkdtemp(prefix="cvdp_harvest_"))
        for rel in rels:
            if not _safe_rel(rel):
                continue
            dest = mirror / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                await box.download(f"{workdir.rstrip('/')}/{rel}", dest)
            except Exception:
                pass
        return harvest(mirror, globs, seeded=seeded)

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        # driven via /run (seed, run, harvest, verify), kept concrete for the base class
        raise NotImplementedError("CvdpGenericAgent is driven via /run, not /responses")

    async def run(self, request: Request, body: CvdpGenericAgentRunRequest):
        meta = (body.model_extra or {}).get("verifier_metadata") or {}
        context_files = meta.get("context_files") or {}
        target_files = meta.get("target_files") or []
        wd = self.config.container_workdir

        inp = body.responses_create_params.input
        instruction = (
            inp
            if isinstance(inp, str)
            else "\n\n".join(
                m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "") for m in (inp or [])
            )
        )

        async with self.sem:
            # one-time host setup: only the first task provisions/pulls, others wait
            # then reuse the memoized results (avoids concurrent-write races).
            async with self._setup_lock:
                if self._deps_dir is None:
                    self._deps_dir = await asyncio.to_thread(self._provision_deps)
                if self._image is None:
                    self._image = await asyncio.to_thread(self._resolve_image)
            deps_dir = self._deps_dir
            image = self._image
            seeded = self._seed_files(wd, context_files, meta.get("harness_files"))
            spec = self._build_spec(body, instruction, deps_dir, seeded, image)

            async with AsyncSandbox(self.config.sandbox_provider, spec) as box:
                await box.start()
                traj = self._traj_dir()
                await box.exec(
                    f"/agent_deps_mount/bin/python {traj}/agent_runner.py",
                    cwd=wd,
                    timeout_s=self.config.timeout,
                )

                # the runner wrote the trajectory under the workdir mount
                resp_local = Path(tempfile.mkdtemp(prefix="cvdp_resp_")) / "response.json"
                try:
                    await box.download(f"{traj}/response.json", resp_local)
                except Exception:
                    pass
                response = (
                    NeMoGymResponse.model_validate_json(resp_local.read_text())
                    if resp_local.exists()
                    else NeMoGymResponse(
                        id=f"resp_{uuid4().hex}",
                        created_at=int(time()),
                        model=body.responses_create_params.model or "model",
                        object="response",
                        output=[],
                        parallel_tool_calls=False,
                        tool_choice="auto",
                        tools=[],
                    )
                )

                rtl_files = await self._remote_harvest(box, wd, self.config.harvest_globs, context_files)
                # always include declared targets that exist, even outside the harvested dirs
                for tf in target_files:
                    if tf in rtl_files or not _safe_rel(tf):
                        continue
                    dest = Path(tempfile.mkdtemp(prefix="cvdp_tf_")) / "f"
                    try:
                        await box.download(f"{wd.rstrip('/')}/{tf}", dest)
                        rtl_files[tf] = dest.read_text(encoding="utf-8")
                    except Exception:
                        pass

            payload = body.model_dump() | {"response": response.model_dump()}
            if rtl_files:
                payload["rtl_files"] = rtl_files
            verify_resp = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=payload,
                cookies=request.cookies,
            )
            await raise_for_status(verify_resp)
            return await get_response_json(verify_resp)


if __name__ == "__main__":
    CvdpGenericAgent.run_webserver()
