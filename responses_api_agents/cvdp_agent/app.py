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

"""CVDP simple and sandboxed-harness agent."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shlex
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import Request, Response
from pydantic import ConfigDict, Field, ValidationError

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
from nemo_gym.sandbox import (
    AsyncSandbox,
    SandboxSpec,
    resolve_provider_config,
    resolve_provider_metadata,
)
from nemo_gym.server_utils import get_response_json, raise_for_status


_DEFAULT_HARVEST_GLOBS = ["rtl/**/*.sv", "rtl/**/*.v", "rtl/**/*.vhd", "verif/**/*.sv", "verif/**/*.v"]

_RUNNER_SOURCE_PATH = Path(__file__).with_name("sandbox_entrypoint.py")


def agent_key(agent_server_module: str) -> str:
    """responses_api_agents.hermes_agent.app maps to hermes_agent, the deps-script key."""
    parts = agent_server_module.split(".")
    return parts[-2] if len(parts) >= 2 else agent_server_module


def load_runner_source() -> str:
    """Return the guest entrypoint source."""
    return _RUNNER_SOURCE_PATH.read_text(encoding="utf-8")


def deps_recipe_key(*paths: Path) -> str:
    """stable hash of the deps-install inputs so a prefix is reused until its recipe changes."""
    blob = b"".join(p.read_bytes() for p in paths if p.exists()) or b"no-script"
    return hashlib.sha256(blob).hexdigest()


def deps_build_env(deps_dir: Path) -> dict[str, str]:
    """Give dependency installers private state outside the archived runtime."""
    build_dir = deps_dir.parent / f".{deps_dir.name}-build"
    cache_dir = build_dir / "cache"
    temp_dir = build_dir / "tmp"
    home_dir = build_dir / "home"
    for path in (cache_dir, temp_dir, home_dir):
        path.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "DEPS_DIR": str(deps_dir),
            "NEMO_GYM_ROOT": str(PARENT_DIR),
            "HOME": str(home_dir),
            "PYTHONPATH": "",
            "PIP_CACHE_DIR": str(cache_dir / "pip"),
            "NPM_CONFIG_CACHE": str(cache_dir / "npm"),
            "UV_CACHE_DIR": str(cache_dir / "uv"),
            "XDG_CACHE_HOME": str(cache_dir),
            "TMPDIR": str(temp_dir),
        }
    )
    return env


def harvest(workdir: Path, globs: list[str], *, seeded: dict[str, str] | None = None) -> dict[str, str]:
    """Collect changed text files under workdir that match any glob."""
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


class CVDPAgentConfig(BaseResponsesAPIAgentConfig):
    """Configuration for simple or agentic CVDP execution."""

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

    agent_server_module: str = "responses_api_agents.claude_code_agent.app"
    agent_server_class: str = "ClaudeCodeAgent"
    agent_config_class: str = "ClaudeCodeAgentConfig"
    agent_kwargs: Dict[str, Any] = Field(default_factory=dict)

    image: str = "nvidia/cvdp-sim:v1.0.0"
    sandbox_provider: str | Dict[str, Any] = Field(default_factory=lambda: {"apptainer": {}})
    sandbox_spec: Dict[str, Any] = Field(default_factory=dict)
    container_workdir: str = "/code"
    harvest_globs: list[str] = _DEFAULT_HARVEST_GLOBS
    deps_provision: Literal["archive", "baked"] = "archive"


class CVDPAgentRunRequest(BaseRunRequest):
    # extra="allow" so verifier_metadata (and any other task fields) survive parsing and are
    # carried through to /verify. BaseRunRequest drops unknown fields, which would 422 /verify.
    model_config = ConfigDict(extra="allow")


class CVDPAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class CVDPAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class CVDPAgent(SimpleResponsesAPIAgent):
    """CVDP agent for simple and agentic execution."""

    config: CVDPAgentConfig
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # agentic-flavor runtime state (unused in the simple flavor)
    sem: Any = None
    _deps_dir: Any = None
    _deps_archive: Any = None
    _image: Any = None
    _sandbox_provider: Any = None
    _sandbox_metadata: Any = None
    _setup_lock: Any = None

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self.sem = asyncio.Semaphore(self.config.concurrency)
        self._deps_dir = None
        self._deps_archive = None
        self._image = None
        self._sandbox_provider = resolve_provider_config(
            self.config.sandbox_provider,
            getattr(self.server_client, "global_config_dict", None),
        )
        self._sandbox_metadata = resolve_provider_metadata(
            self.config.sandbox_provider,
            getattr(self.server_client, "global_config_dict", None),
        )
        self._setup_lock = asyncio.Lock()

    async def run(self, request: Request, body: CVDPAgentRunRequest):
        if self.config.simple_agent:
            return await self._run_simple(request, body)
        return await self._run_agentic(request, body)

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

    def _provision_deps(self) -> Path:
        """Install the configured agent's portable dependency prefix once."""
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
        subprocess.run(["bash", str(script)], env=deps_build_env(deps_dir), check=True)
        sentinel.write_text(recipe)
        return deps_dir

    def _resolve_image(self) -> str:
        """Validate and normalize the configured image reference."""
        image = self.config.image.strip()
        is_apptainer = "apptainer" in self._sandbox_provider
        if image.endswith(".sif") or image.startswith(("/", ".")):
            if not is_apptainer:
                raise ValueError("local or .sif images require the Apptainer sandbox provider")
            return image
        if "://" in image and not image.startswith("docker://"):
            if not is_apptainer:
                raise ValueError(f"image URI {image!r} is not portable. Use a bare OCI image reference")
            return image
        return image.removeprefix("docker://")

    def _archive_deps(self, deps_dir: Path) -> Path:
        """Package the portable harness runtime once."""
        sentinel = deps_dir / ".installed"
        recipe = sentinel.read_text().strip()
        archive = deps_dir.parent / f".{deps_dir.name}-{recipe[:16]}.tar.gz"
        if archive.exists() and archive.stat().st_mtime >= sentinel.stat().st_mtime:
            return archive
        temporary = archive.with_suffix(".tmp")
        with tarfile.open(temporary, "w:gz", compresslevel=1) as tar:
            for child in sorted(deps_dir.iterdir()):
                tar.add(child, arcname=child.name)
        temporary.replace(archive)
        return archive

    def _model_url(self) -> str:
        if not self.config.model_server:
            return ""
        cfg = get_first_server_config_dict(self.server_client.global_config_dict, self.config.model_server.name)
        return self.server_client._build_server_base_url(cfg)

    def _seed_files(
        self, workdir: str, context_files: Dict[str, str], harness_files: Optional[dict]
    ) -> Dict[str, str]:
        """Select safe context files for the workspace."""
        forbidden = set(harness_files or {})
        out: Dict[str, str] = {}
        for rel, content in (context_files or {}).items():
            if content is None or rel in forbidden or _is_harness_path(rel) or not _safe_rel(rel):
                continue
            out[f"{workdir.rstrip('/')}/{rel}"] = content
        return out

    def _build_spec(self, body: BaseRunRequest, instruction: str, files: Dict[str, str], image: str) -> SandboxSpec:
        wd = self.config.container_workdir
        extra = dict(self.config.sandbox_spec)
        provider_options = dict(extra.pop("provider_options", {}) or {})
        metadata = dict(self._sandbox_metadata)
        metadata.update(extra.pop("metadata", {}) or {})
        traj = self._traj_dir()
        return SandboxSpec(
            image=image,
            workdir=wd,
            env={
                "NV_MODEL_URL": self._model_url(),
                "NV_MODEL_NAME": body.responses_create_params.model or "model",
                "NV_AGENT_KWARGS": json.dumps(self.config.agent_kwargs),
                "NV_AGENT_HOME": f"{wd.rstrip('/')}/.home",
                "NV_SYSTEM_PROMPT": self.config.system_prompt or "",
                "NV_TRAJ_DIR": traj,
                "NV_AGENT_MODULE": self.config.agent_server_module,
                "NV_AGENT_CLASS": self.config.agent_server_class,
                "NV_AGENT_CFG_CLASS": self.config.agent_config_class,
                "NV_AGENT_DEPS_DIR": self._agent_deps_dir(),
            },
            files={
                f"{traj}/instruction.txt": instruction,
                f"{traj}/agent_runner.py": load_runner_source(),
                **files,
            },
            metadata=metadata,
            provider_options=provider_options,
            **extra,
        )

    def _traj_dir(self) -> str:
        """Return the trajectory directory."""
        return f"{self.config.container_workdir.rstrip('/')}/.nv"

    def _agent_deps_dir(self) -> str:
        if self.config.deps_provision == "archive":
            return f"{self.config.container_workdir.rstrip('/')}/.agent_deps"
        return "/agent_deps_mount"

    def _failed_agentic_response(
        self,
        body: CVDPAgentRunRequest,
        error: str,
        *,
        return_code: int | None = None,
    ) -> CVDPAgentVerifyResponse:
        """Return a zero-reward rollout for a per-task harness failure."""
        meta = (body.model_extra or {}).get("verifier_metadata") or {}
        categories = meta.get("categories") or []
        response = NeMoGymResponse(
            id="cvdp-agent-error",
            created_at=0.0,
            model=body.responses_create_params.model or "error",
            object="response",
            output=[],
            tools=[],
            parallel_tool_calls=False,
            tool_choice="auto",
        )
        print(f"[CVDP AGENT ERROR] task_id={meta.get('task_id')}: {error}", flush=True)
        return CVDPAgentVerifyResponse(
            **body.model_dump(),
            response=response,
            reward=0.0,
            task_id=meta.get("task_id"),
            category=categories[0] if categories else None,
            difficulty=meta.get("difficulty"),
            container_exit_code=return_code,
            container_stderr=error,
        )

    async def _remote_harvest(
        self,
        box: AsyncSandbox,
        workdir: str,
        globs: list[str],
        seeded: Dict[str, str],
        mirror: Path,
    ) -> dict:
        """Download files matching the harvest globs."""
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

    async def _run_agentic(self, request: Request, body: CVDPAgentRunRequest) -> CVDPAgentVerifyResponse:
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
            async with self._setup_lock:
                if self.config.deps_provision == "archive" and self._deps_dir is None:
                    self._deps_dir = await asyncio.to_thread(self._provision_deps)
                if self.config.deps_provision == "archive" and self._deps_archive is None:
                    self._deps_archive = await asyncio.to_thread(self._archive_deps, self._deps_dir)
                if self._image is None:
                    self._image = await asyncio.to_thread(self._resolve_image)
            deps_archive = self._deps_archive
            image = self._image
            seeded = self._seed_files(wd, context_files, meta.get("harness_files"))
            spec = self._build_spec(body, instruction, seeded, image)

            async with AsyncSandbox(self._sandbox_provider, spec) as box:
                await box.start()
                traj = self._traj_dir()
                agent_deps_dir = self._agent_deps_dir()
                if self.config.deps_provision == "archive":
                    deps_archive_path = f"{traj}/agent-deps.tar.gz"
                    await box.exec(f"mkdir -p {shlex.quote(agent_deps_dir)}", timeout_s=30)
                    await box.upload(deps_archive, deps_archive_path)
                    unpack = await box.exec(
                        f"tar -xzf {shlex.quote(deps_archive_path)} -C {shlex.quote(agent_deps_dir)}",
                        timeout_s=600,
                    )
                    if unpack.return_code != 0:
                        raise RuntimeError(f"agent runtime extraction failed: {(unpack.stderr or '')[-500:]}")
                elif self.config.deps_provision == "baked":
                    runtime = await box.exec(f"test -x {shlex.quote(agent_deps_dir)}/bin/python", timeout_s=30)
                    if runtime.return_code != 0:
                        raise RuntimeError(f"sandbox image does not contain {agent_deps_dir}/bin/python")
                agent_result = await box.exec(
                    f"{shlex.quote(agent_deps_dir)}/bin/python {traj}/agent_runner.py",
                    cwd=wd,
                    timeout_s=self.config.timeout,
                )
                if agent_result.return_code != 0:
                    details = (agent_result.stderr or agent_result.stdout or "")[-2000:]
                    return self._failed_agentic_response(
                        body,
                        f"sandbox harness exited with code {agent_result.return_code}: {details}",
                        return_code=agent_result.return_code,
                    )

                with tempfile.TemporaryDirectory(prefix="cvdp_agent_run_") as scratch:
                    scratch_path = Path(scratch)
                    resp_local = scratch_path / "response.json"
                    try:
                        await box.download(f"{traj}/response.json", resp_local)
                        response = NeMoGymResponse.model_validate_json(resp_local.read_text())
                    except Exception as e:
                        return self._failed_agentic_response(body, f"could not load harness response.json: {e}")

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
