# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""CVDP agent — one class, two flavors selected by ``config.simple_agent``.

- **simple** (``simple_agent: true``): no harness, no sandbox. The model itself is asked to
  produce the RTL/code (optionally with resources-server tool calls); ``/verify`` grades
  whatever the model emitted. Fast; used by the non-agentic CVDP dataset.
- **agentic** (``simple_agent: false``): a coding harness (Claude, Hermes, ...) is booted
  inside an EDA-sim sandbox so it can edit files on disk and self-test with the in-container
  EDA tools; the produced HDL files are harvested and graded through the same ``/verify``.
  The harness is config-selected (``agent_server_module``/``class``/``config_class``), built
  on the provider-neutral sandbox API so the backend is config, not code.
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from subprocess import Popen
from time import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import Request, Response
from pydantic import ConfigDict, ValidationError

from nemo_gym import PARENT_DIR
from nemo_gym.base_resources_server import (
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.global_config import get_first_server_config_dict
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
)
from nemo_gym.sandbox import AsyncSandbox, SandboxSpec
from nemo_gym.server_utils import get_response_json, raise_for_status


_DEFAULT_HARVEST_GLOBS = ["rtl/**/*.sv", "rtl/**/*.v", "rtl/**/*.vhd", "verif/**/*.sv", "verif/**/*.v"]

# guest entrypoint injected into the sandbox and run as ``python agent_runner.py``.
_RUNNER_SOURCE_PATH = Path(__file__).with_name("sandbox_entrypoint.py")


# ----------------------------
# Agentic-flavor host helpers
# ----------------------------


def agent_key(agent_server_module: str) -> str:
    """responses_api_agents.hermes_agent.app maps to hermes_agent, the deps-script key."""
    parts = agent_server_module.split(".")
    return parts[-2] if len(parts) >= 2 else agent_server_module


def load_runner_source() -> str:
    """return the guest entrypoint script (``sandbox_entrypoint.py``) verbatim.

    it is a plain module rather than a templated string, so it is diffable and syntax-checked
    with the rest of the package. it reads the agent module/class from ``NV_AGENT_*`` env vars
    set by the caller, so no per-agent rendering is needed.
    """
    return _RUNNER_SOURCE_PATH.read_text(encoding="utf-8")


def deps_recipe_key(*paths: Path) -> str:
    """stable hash of the deps-install inputs so a prefix is reused until its recipe changes."""
    blob = b"".join(p.read_bytes() for p in paths if p.exists()) or b"no-script"
    return hashlib.sha256(blob).hexdigest()


def harvest(workdir: Path, globs: list[str], *, seeded: dict[str, str] | None = None) -> dict[str, str]:
    """collect files the agent produced under workdir that match any glob.

    returns {relative_posix_path: text_content}. files identical to a seeded input are skipped
    so unchanged context files are not reported as produced. unreadable or binary files are
    skipped. point it at e.g. ["rtl/**/*.sv", "rtl/**/*.v"].
    """
    workdir = Path(workdir)
    seeded = seeded or {}
    produced: dict[str, str] = {}
    for pattern in globs:
        for fpath in sorted(workdir.glob(pattern)):
            if not fpath.is_file():
                continue
            rel = fpath.relative_to(workdir).as_posix()
            if rel in produced:
                continue
            try:
                content = fpath.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if seeded.get(rel) == content:
                continue  # unchanged context file
            produced[rel] = content
    return produced


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


# ----------------------------
# Config + request/response schemas
# ----------------------------


class CVDPAgentConfig(BaseResponsesAPIAgentConfig):
    """Config for the CVDP agent. ``simple_agent`` picks the flavor; the fields below it are
    grouped by the flavor that reads them (the other flavor ignores them)."""

    # flavor selector: simple (model emits code directly) vs agentic (harness in a sandbox).
    simple_agent: bool = False

    resources_server: ResourcesServerRef
    # required for the simple flavor; optional for agentic (a harness may bring its own model
    # endpoint, e.g. claude_code via anthropic_base_url/api_key in agent_kwargs). When set for
    # agentic, its URL is passed into the sandbox as NV_MODEL_URL.
    model_server: Optional[ModelServerRef] = None

    # --- simple flavor ---
    max_steps: Optional[int] = None
    llm_parse_retries: int = 3  # Retry model+verify on parse failure or model error (mirrors CVDP LLM_RETRY_COUNT)

    # --- agentic flavor ---
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
    # image may be a bare docker ref (e.g. "nvidia/cvdp-sim:v1.0.0"), an explicit .sif path, or
    # a docker:// / oras:// uri. A bare docker ref is resolved to a cached .sif on the host (same
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


class CVDPAgentRunRequest(BaseRunRequest):
    # extra="allow" so verifier_metadata (and any other task fields) survive parsing and are
    # carried through to /verify. BaseRunRequest drops unknown fields, which would 422 /verify.
    model_config = ConfigDict(extra="allow")


class CVDPAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class CVDPAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class CVDPAgent(SimpleResponsesAPIAgent):
    """CVDP agent for both flavors; ``run()`` dispatches on ``config.simple_agent``."""

    config: CVDPAgentConfig
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # agentic-flavor runtime state (unused in the simple flavor)
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

    async def run(self, request: Request, body: CVDPAgentRunRequest):
        if self.config.simple_agent:
            return await self._run_simple(request, body)
        return await self._run_agentic(request, body)

    # ----------------------------
    # Simple flavor
    # ----------------------------

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        # the simple flavor drives the model directly; the agentic flavor never calls this
        # (its run() boots a harness whose own responses() does the work).
        if not self.config.simple_agent:
            raise NotImplementedError("agentic CVDPAgent is driven via /run, not /responses")
        if self.config.model_server is None:
            raise RuntimeError("simple_agent mode requires model_server to be configured")

        body = body.model_copy(deep=True)

        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        new_outputs = []
        usage = None
        step = 0
        model_server_cookies = None  # update the cookies on every model response
        resources_server_cookies = request.cookies  # update the cookies on every resources server response

        while True:
            step += 1
            new_body = body.model_copy(update={"input": body.input + new_outputs})

            model_response = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path=self.url_path_for_request("/v1/responses", request),
                json=new_body,
                cookies=model_server_cookies,
            )
            # We raise for status here since we expect model calls to always work.
            await raise_for_status(model_response)
            model_response_json = await get_response_json(model_response)
            model_server_cookies = model_response.cookies
            try:
                model_response = NeMoGymResponse.model_validate(model_response_json)
            except ValidationError as e:
                raise RuntimeError(
                    f"Received an invalid response from model server: {json.dumps(model_response_json)}"
                ) from e

            output = model_response.output
            new_outputs.extend(output)

            if not usage:
                usage = model_response.usage
                model_response.usage = None

            if usage and model_response.usage:
                usage.input_tokens += model_response.usage.input_tokens
                usage.output_tokens += model_response.usage.output_tokens
                usage.total_tokens += model_response.usage.total_tokens

                # TODO support more advanced token details
                usage.input_tokens_details.cached_tokens = 0
                usage.output_tokens_details.reasoning_tokens = 0

            if model_response.incomplete_details and model_response.incomplete_details.reason == "max_output_tokens":
                break

            all_fn_calls: List[NeMoGymResponseFunctionToolCall] = [o for o in output if o.type == "function_call"]
            all_output_messages: List[NeMoGymResponseOutputMessage] = [
                o for o in output if o.type == "message" and o.role == "assistant"
            ]
            if not all_fn_calls and all_output_messages:
                break

            for output_function_call in all_fn_calls:
                api_response = await self.server_client.post(
                    server_name=self.config.resources_server.name,
                    url_path=f"/{output_function_call.name}",
                    json=json.loads(output_function_call.arguments),
                    cookies=resources_server_cookies,
                )
                # We don't raise for status here since it's a valid return for the API to error e.g. if the model outputs an invalid call or something.
                resources_server_cookies = api_response.cookies

                tool_response = NeMoGymFunctionCallOutput(
                    type="function_call_output",
                    call_id=output_function_call.call_id,
                    output=(await api_response.content.read()).decode(),
                )
                new_outputs.append(tool_response)

            # Check if max steps is not None and if we have exhausted it.
            if self.config.max_steps and step >= self.config.max_steps:
                break

        # Propogate any extra cookies necessary for downstream verification
        for k, v in (*resources_server_cookies.items(), *model_server_cookies.items()):
            response.set_cookie(k, v)

        model_response.output = new_outputs
        model_response.usage = usage
        return model_response

    async def _run_simple(self, request: Request, body: CVDPAgentRunRequest) -> CVDPAgentVerifyResponse:
        cookies = request.cookies

        seed_session_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(seed_session_response)
        cookies = seed_session_response.cookies

        # Retry loop — mirrors CVDP's LLM_RETRY_COUNT in dataset_processor.py.
        # Re-calls the model and re-verifies on:
        #   1. Parse failure (resource server returns parse_failed=True)
        #   2. Model call exception (vllm/network error)
        task_id = (
            (body.verifier_metadata or {}).get("task_id")
            if isinstance(body.verifier_metadata, dict)
            else getattr(body.verifier_metadata, "task_id", None)
        )
        retries_left = self.config.llm_parse_retries
        while True:
            try:
                response = await self.server_client.post(
                    server_name=self.config.name,
                    url_path=self.url_path_for_run("/v1/responses", body),
                    json=body.responses_create_params,
                    cookies=cookies,
                )
                await raise_for_status(response)
                cookies = response.cookies

                verify_request = CVDPAgentVerifyRequest.model_validate(
                    body.model_dump() | {"response": await get_response_json(response)}
                )

                verify_response = await self.server_client.post(
                    server_name=self.config.resources_server.name,
                    url_path="/verify",
                    json=verify_request.model_dump(),
                    cookies=cookies,
                )
                await raise_for_status(verify_response)
                result = CVDPAgentVerifyResponse.model_validate(await get_response_json(verify_response))

                # Check for parse failure — resource server signals this when the
                # model produced output but RTL/code extraction failed.
                if getattr(result, "parse_failed", False) and retries_left > 0:
                    retries_left -= 1
                    print(f"[RETRY] parse_failed for task_id={task_id}, retries_left={retries_left}")
                    continue

                return result

            except Exception as e:
                if retries_left > 0:
                    retries_left -= 1
                    print(f"[RETRY] exception for task_id={task_id}, retries_left={retries_left}, error={e}")
                    continue
                raise

    # ----------------------------
    # Agentic flavor
    # ----------------------------

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
        # Hold an exclusive file lock so concurrent rollouts (even across processes, since
        # the agent and verifier share this cache dir) don't pull the same image at once.
        lock_path = sif_path + ".lock"
        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
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

    async def _remote_harvest(
        self,
        box: AsyncSandbox,
        workdir: str,
        globs: list[str],
        seeded: Dict[str, str],
        mirror: Path,
    ) -> dict:
        """list and download files matching globs from the sandbox, then filter via harvest()."""
        dirs = sorted({g.split("/")[0] for g in globs if "/" in g}) or ["."]
        listing = await box.exec(
            f"cd {shlex.quote(workdir)} && find {' '.join(shlex.quote(d) for d in dirs)} -type f 2>/dev/null"
        )
        rels = [line.strip().lstrip("./") for line in (listing.stdout or "").splitlines() if line.strip()]
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

    async def _run_agentic(self, request: Request, body: CVDPAgentRunRequest):
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

                with tempfile.TemporaryDirectory(prefix="cvdp_agent_run_") as scratch:
                    scratch_path = Path(scratch)
                    resp_local = scratch_path / "response.json"
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

                    rtl_files = await self._remote_harvest(
                        box,
                        wd,
                        self.config.harvest_globs,
                        context_files,
                        scratch_path / "harvest",
                    )
                    # always include declared targets that exist, even outside the harvested dirs
                    for tf in target_files:
                        if tf in rtl_files or not _safe_rel(tf):
                            continue
                        dest = scratch_path / "targets" / tf
                        dest.parent.mkdir(parents=True, exist_ok=True)
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
    CVDPAgent.run_webserver()
