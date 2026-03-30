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

import asyncio
import hashlib
import os
import shlex
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from cvdp_lib.model_helpers import ModelHelpers
from pydantic import BaseModel

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


_helpers = ModelHelpers()


# ----------------------------
# Config
# ----------------------------


class CVDPResourcesServerConfig(BaseResourcesServerConfig):
    oss_sim_image: str = "ghcr.io/hdl/sim/osvb"
    oss_pnr_image: str = "ghcr.io/hdl/impl/pnr"
    eda_sim_image: str = ""  # Set to a commercial EDA image (e.g. Cadence Xcelium)
    container_timeout: int = 600
    num_processes: int = 4  # Max concurrent Apptainer jobs
    sif_cache_dir: str = ""  # Defaults to ~/.cache/nemo-gym/sif


# ----------------------------
# Schemas
# ----------------------------


class CVDPVerifierMetadata(BaseModel):
    task_id: str
    categories: List[str] = []
    difficulty: str = ""
    target_files: List[str]
    harness_files: Dict[str, Optional[str]]


class CVDPRunRequest(BaseRunRequest):
    pass


class CVDPVerifyRequest(CVDPRunRequest, BaseVerifyRequest):
    verifier_metadata: Dict[str, Any]


class CVDPVerifyResponse(BaseVerifyResponse):
    task_id: Optional[str] = None
    category: Optional[str] = None  # e.g. "cid003" — for CVDP report
    difficulty: Optional[str] = None  # e.g. "easy" — for CVDP report
    extracted_rtl: Optional[Dict[str, str]] = None
    container_exit_code: Optional[int] = None
    container_stderr: Optional[str] = None
    container_services: Optional[List[Dict]] = None  # per-service results: [{"service", "exit_code", "stderr"}]
    execution_time: Optional[float] = None  # total harness wall time in seconds
    parse_failed: bool = False  # True when model produced output but RTL extraction failed


# ----------------------------
# Code extraction helpers
# ----------------------------


def _parse_model_response(res: str, target_files: List[str]) -> Optional[Dict[str, str]]:
    """
    Parse model output using ModelHelpers.parse_model_response().
    Returns {filename: code} or None on failure.
    """
    if not target_files:
        return None

    no_schema = len(target_files) == 1

    # Match CVDP's openai_llm.py: strip response before parsing
    res = res.strip()

    # Match CVDP's openai_llm.py: fix JSON formatting for multi-file responses
    if not no_schema and res.startswith("{") and res.endswith("}"):
        res = _helpers.fix_json_formatting(res)

    output, success = _helpers.parse_model_response(res, files=target_files, no_schema=no_schema)

    if not success:
        return None

    if no_schema:  # schema is one first or multiple
        code = output.get("direct_text") or output.get("response")
        return {target_files[0]: code} if code else None

    result: Dict[str, str] = {}
    if "code" in output and isinstance(output["code"], list):
        for item in output["code"]:
            if isinstance(item, dict):
                result.update(item)

    return result if result else None


# ----------------------------
# Apptainer harness helpers
# ----------------------------


def _apply_substitutions(content: str, config: CVDPResourcesServerConfig) -> str:
    """
    Replace image placeholders in harness file content — mirrors repository.apply_template_substitution() but with Apptainer syntax.
    """
    substitutions = {
        "__VERIF_EDA_IMAGE__": config.eda_sim_image,
        "__OSS_SIM_IMAGE__": config.oss_sim_image,
        "__OSS_PNR_IMAGE__": config.oss_pnr_image,
    }
    for placeholder, value in substitutions.items():
        if value and placeholder in content:
            content = content.replace(placeholder, value)
    return content


def _resolve_image_for_service(
    compose_data: dict,
    service_name: str,
    harness_files: Dict[str, Optional[str]],
    config: CVDPResourcesServerConfig,
) -> Tuple[str, List[str]]:
    """
    Resolve the container image for a service that uses ``build:`` instead of
    ``image:`` in its docker-compose definition.

    Docker Compose handles ``build:`` natively by reading a Dockerfile and
    building an image on the fly.  Apptainer cannot do this directly, so we
    parse the Dockerfile to extract the base image (FROM) and any RUN / ADD
    commands, then replay them via ``apptainer build`` with a def file.

    Returns (base_image, post_commands) where *post_commands* are shell
    commands for the ``%post`` section of an Apptainer definition file.
    If the service already has ``image:``, returns (image, []).
    """
    svc = (compose_data.get("services") or {}).get(service_name, {})
    image = svc.get("image", "")
    if image:
        return image, []

    # Determine Dockerfile path from build: config
    build_cfg = svc.get("build", {})
    if isinstance(build_cfg, str):
        dockerfile_path = os.path.join(build_cfg, "Dockerfile")
    elif isinstance(build_cfg, dict):
        dockerfile_path = build_cfg.get("dockerfile", "Dockerfile")
    else:
        return "", []

    # Look for the Dockerfile in harness_files (try multiple path variants)
    dockerfile_content = None
    candidates = [
        dockerfile_path,
        f"src/{dockerfile_path}",
        dockerfile_path.replace("src/", ""),
    ]
    for candidate in candidates:
        for hf_path, hf_content in harness_files.items():
            if hf_content and (hf_path == candidate or hf_path.endswith(os.path.basename(candidate))):
                dockerfile_content = _apply_substitutions(hf_content, config)
                break
        if dockerfile_content:
            break

    if not dockerfile_content:
        return "", []

    # Parse Dockerfile: extract FROM base image and RUN/ADD commands
    base_image = ""
    post_commands: List[str] = []
    for line in dockerfile_content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.upper().startswith("FROM "):
            parts = line.split()
            base_image = parts[1] if len(parts) > 1 else ""
            if " AS " in base_image.upper():
                base_image = base_image.split()[0]
        elif line.upper().startswith("RUN "):
            post_commands.append(line[4:].strip())
        elif line.upper().startswith("ADD ") and "http" in line.lower():
            # Convert ADD <url> <dest> to wget/curl
            parts = line.split()
            if len(parts) >= 3:
                url, dest = parts[1], parts[2]
                post_commands.append(f"wget -q -O {dest} {url} || curl -sL -o {dest} {url}")

    return base_image, post_commands


def _parse_compose_service(compose_content: str, service_name: str) -> Dict[str, Any]:
    """
    Extract image, command, entrypoint, volumes, working_dir, and environment
    from a docker-compose service definition.  The compose YAML is only used as
    metadata — Apptainer handles the actual execution.
    """
    data = yaml.safe_load(compose_content) or {}
    service = (data.get("services") or {}).get(service_name, {})
    return {
        "image": service.get("image", ""),
        "command": service.get("command", ""),
        "entrypoint": service.get("entrypoint"),
        "volumes": service.get("volumes", []),
        "working_dir": service.get("working_dir", "/code/rundir"),
        "environment": service.get("environment", {}),
    }


def _build_bind_args(workdir: str, compose_volumes: List[str]) -> List[str]:
    """
    Build --bind arguments for Apptainer from:
    1. The standard /code/* workspace mounts
    2. Non-/code volumes from the docker-compose service definition
    """
    bind_args: List[str] = []

    # Standard /code/* mounts
    for vol in ["docs", "rundir", "rtl", "verif", "src"]:
        bind_args += ["--bind", f"{workdir}/{vol}:/code/{vol}"]

    # Compose-defined volumes (skip /code mounts — handled above)
    for vol_str in compose_volumes:
        parts = vol_str.split(":")
        host_path = parts[0]
        container_path = parts[1] if len(parts) > 1 else host_path
        opts = parts[2] if len(parts) > 2 else ""

        if "/code" in container_path:
            continue

        # Resolve relative paths against workdir
        if host_path.startswith("./") or host_path.startswith("../") or not os.path.isabs(host_path):
            host_path = os.path.normpath(os.path.join(workdir, host_path))

        bind_spec = f"{host_path}:{container_path}"
        if opts:
            bind_spec += f":{opts}"
        bind_args += ["--bind", bind_spec]

    return bind_args


def _load_dot_env(workdir: str) -> Dict[str, str]:
    """
    Parse the src/.env file (KEY=value lines) from the workspace.
    Docker Compose auto-loads env_file directives; Apptainer does not,
    so we read them ourselves and pass them via --env.
    """
    env_path = os.path.join(workdir, "src", ".env")
    env_vars: Dict[str, str] = {}
    if not os.path.isfile(env_path):
        return env_vars
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, val = line.partition("=")
                env_vars[key.strip()] = val.strip()
    return env_vars


def _build_env_args(environment: Any, dot_env: Optional[Dict[str, str]] = None) -> List[str]:
    """Build --env arguments for Apptainer from a compose environment field
    and any variables loaded from the workspace src/.env file."""
    env_args: List[str] = []
    # Load dot_env first so compose environment can override
    if dot_env:
        for key, val in dot_env.items():
            env_args += ["--env", f"{key}={val}"]
    if isinstance(environment, dict):
        for key, val in environment.items():
            env_args += ["--env", f"{key}={val}"]
    elif isinstance(environment, list):
        for item in environment:
            env_args += ["--env", str(item)]
    return env_args


def _build_command(entrypoint: Any, command: Any) -> List[str]:
    """Build the command list from compose entrypoint + command fields."""
    cmd_parts: List[str] = []

    if entrypoint:
        if isinstance(entrypoint, str):
            cmd_parts = shlex.split(entrypoint)
        else:
            cmd_parts = list(entrypoint)

    if command:
        if isinstance(command, str):
            cmd_parts += shlex.split(command)
        else:
            cmd_parts += list(command)

    return cmd_parts


# ----------------------------
# Server
# ----------------------------


class CVDPResourcesServer(SimpleResourcesServer):
    config: CVDPResourcesServerConfig

    def model_post_init(self, context: Any) -> None:
        self._semaphore = asyncio.Semaphore(value=self.config.num_processes)
        self._sif_locks: Dict[str, asyncio.Lock] = {}
        self._sif_lock_guard = asyncio.Lock()
        cache = self.config.sif_cache_dir
        if not cache:
            cache = os.path.join(Path.home(), ".cache", "nemo-gym", "sif")
        self._sif_cache_dir = cache
        os.makedirs(self._sif_cache_dir, exist_ok=True)

    async def verify(self, body: CVDPVerifyRequest) -> CVDPVerifyResponse:
        # print(f"[DEBUG] verify entry from model")
        meta = CVDPVerifierMetadata.model_validate(body.verifier_metadata)

        # categories is [category_id, difficulty], e.g. ["cid003", "medium"]
        category, difficulty = meta.categories[0], meta.categories[1]

        model_out = body.response.output_text

        # Fallback: if output_text is empty, try extracting RTL from reasoning content.
        # Mirrors cvdp's logic: if message.content is None, fall back to message.reasoning_content.
        if not model_out or not model_out.strip():
            for item in body.response.output:
                if getattr(item, "type", None) == "reasoning":
                    reasoning_texts = [s.text for s in (item.summary or []) if s.text]
                    if reasoning_texts:
                        model_out = "\n".join(reasoning_texts)
                        # print(f"[verify] output_text empty — falling back to reasoning content (len={len(model_out)})")
                        break

        has_model_output = bool(model_out and model_out.strip())
        if not has_model_output:
            return CVDPVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                task_id=meta.task_id,
                category=category,
                difficulty=difficulty,
                extracted_rtl=None,
                container_exit_code=None,
                container_stderr=None,
                container_services=None,
                execution_time=0.0,
                parse_failed=False,
            )

        rtl_files = _parse_model_response(model_out, meta.target_files)

        # If model produced output but parsing failed, signal parse_failed so the
        # agent can retry with a fresh model completion — mirrors CVDP's
        # LLM_RETRY_COUNT loop in dataset_processor.py.
        if rtl_files is None:
            return CVDPVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                task_id=meta.task_id,
                category=category,
                difficulty=difficulty,
                extracted_rtl=None,
                container_exit_code=None,
                container_stderr="parse_failed: could not extract RTL from model output",
                container_services=[],
                execution_time=0.0,
                parse_failed=True,
            )

        async with self._semaphore:
            t0 = time.time()
            exit_code, stderr, service_results = await self._run_harness(
                rtl_files=rtl_files or {},
                harness_files=meta.harness_files,
                task_id=meta.task_id,
            )
            execution_time = time.time() - t0

        return CVDPVerifyResponse(
            **body.model_dump(),
            reward=1.0 if exit_code == 0 else 0.0,
            task_id=meta.task_id,
            category=category,
            difficulty=difficulty,
            extracted_rtl=rtl_files,
            container_exit_code=exit_code,
            container_stderr=stderr,
            container_services=service_results,
            execution_time=execution_time,
        )

    async def _run_harness(
        self,
        rtl_files: Dict[str, str],
        harness_files: Dict[str, Optional[str]],
        task_id: str,
    ) -> Tuple[int, str, List[Dict]]:
        """
        Write harness + RTL to a temp workspace and run verification via Apptainer.

        Mirrors repository.py prepare() + obj_harness():
          Workspace layout:
            workdir/
              docker-compose.yml   (parsed for service metadata, not executed directly)
              src/                 (test scripts and .env from harness_files)
              rtl/                 (model-generated RTL, bound as /code/rtl)
              verif/               (empty, bound as /code/verif)
              docs/                (empty, bound as /code/docs)
              rundir/              (execution output, bound as /code/rundir)
        """
        with tempfile.TemporaryDirectory(prefix=f"cvdp_{task_id}_") as workdir:
            workdir_path = Path(workdir)

            # Create all mount dirs — mirrors repository.create_folders()
            for d in ["rtl", "verif", "docs", "src", "rundir"]:
                (workdir_path / d).mkdir()

            # Write harness files — mirrors repository.restore_files()
            compose_content: Optional[str] = None
            for filepath, content in harness_files.items():
                if content is None:
                    continue
                content = _apply_substitutions(content, self.config)
                if filepath.endswith("docker-compose.yml"):
                    compose_content = content
                dest = workdir_path / filepath
                dest.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with open(str(dest), "w+", encoding="utf-8") as f:
                        f.write(content)
                except Exception:
                    print(f"Failed to write file: {filepath}")

            if compose_content is None:
                return 1, "No docker-compose.yml found in harness_files", []

            # Write model-generated RTL files
            for filepath, code in rtl_files.items():
                rel = Path(filepath)
                if rel.parts[0] == "rtl":
                    dest = (workdir_path / "rtl").joinpath(*rel.parts[1:])
                else:
                    dest = workdir_path / "rtl" / rel.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with open(str(dest), "w+", encoding="utf-8") as f:
                        f.write(code)
                except Exception:
                    print(f"Failed to write file: {filepath}")

                # print(f"DEBUG: wrote model generated RTL files: {filepath}")

            # Run each service — mirrors repository.obj_harness()
            compose_data = yaml.safe_load(compose_content)
            services = list((compose_data.get("services") or {}).keys())

            service_results: List[Dict] = []
            for service in services:
                exit_code, output = await self._run_service(workdir, service, task_id, compose_content, harness_files)
                service_results.append({"service": service, "exit_code": exit_code, "stderr": output})

            final_exit_code = 0 if all(r["exit_code"] == 0 for r in service_results) else 1
            combined_stderr = "\n".join(f"[{r['service']}] {r['stderr']}" for r in service_results if r["stderr"])
            return final_exit_code, combined_stderr, service_results

    async def _run_service(
        self,
        workdir: str,
        service: str,
        task_id: str,
        compose_content: str,
        harness_files: Optional[Dict[str, Optional[str]]] = None,
    ) -> Tuple[int, str]:
        """
        Run a single service from the compose definition using Apptainer. — mirrors repository.log_docker().

        Pulls the Docker image as a SIF (cached), then executes with
        --bind mounts equivalent to the original Docker volume mappings.
        Apptainer uses host networking by default, so no network setup is needed.

        """
        path = os.path.abspath(workdir)
        svc = _parse_compose_service(compose_content, service)

        # Resolve image — handles both image: and build: services.
        # Docker Compose builds from Dockerfiles automatically; for Apptainer
        # we parse the Dockerfile and build a SIF with the equivalent commands.
        image = svc["image"]
        post_commands: List[str] = []
        if not image and harness_files:
            compose_data = yaml.safe_load(compose_content)
            image, post_commands = _resolve_image_for_service(compose_data, service, harness_files, self.config)
        if not image:
            return 1, f"No image defined for service '{service}'"

        try:
            if post_commands:
                sif_path = await self._ensure_built_sif(image, post_commands)
            else:
                sif_path = await self._ensure_sif(image)
        except RuntimeError as exc:
            return 1, str(exc)

        bind_args = _build_bind_args(path, svc["volumes"])
        dot_env = _load_dot_env(path)
        env_args = _build_env_args(svc["environment"], dot_env)
        cmd_parts = _build_command(svc["entrypoint"], svc["command"])

        # Fix working_dir paths that don't exist under Apptainer's bind mounts.
        # Some compose files use /src/rundir/ which exists in Docker (via volume
        # mount) but not in Apptainer (which only binds to /code/*).
        working_dir = svc["working_dir"] or "/code/rundir"
        if "/code/" not in working_dir:
            working_dir = "/code/rundir"

        if cmd_parts:
            cmd = [
                "apptainer",
                "exec",
                "--writable-tmpfs",
                "--home",
                "/code/rundir",
                *bind_args,
                *env_args,
                "--pwd",
                working_dir,
                sif_path,
                *cmd_parts,
            ]
        else:
            # No explicit command — use the container's default runscript
            cmd = [
                "apptainer",
                "run",
                "--writable-tmpfs",
                "--home",
                "/code/rundir",
                *bind_args,
                *env_args,
                "--pwd",
                working_dir,
                sif_path,
            ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=workdir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        exit_code = -1
        stdout_bytes = b""
        stderr_bytes = b""
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.config.container_timeout,
            )
            exit_code = proc.returncode
        except asyncio.TimeoutError:
            proc.kill()
            stdout_bytes, stderr_bytes = await proc.communicate()
            exit_code = -1
            stderr_bytes = f"apptainer exec timed out after {self.config.container_timeout}s".encode()

        combined = (stderr_bytes + stdout_bytes).decode("utf-8", errors="replace")
        return exit_code, combined

    async def _ensure_built_sif(self, base_image: str, post_commands: List[str]) -> str:
        """
        Build a SIF that extends a base image with extra commands from a Dockerfile.

        This replicates what ``docker compose build`` does: take a base image,
        run additional commands (pip install, etc.), and produce a new image.
        For Apptainer we generate a definition file and run ``apptainer build``.
        Results are cached by a hash of the commands.
        """
        if not post_commands:
            return await self._ensure_sif(base_image)

        cmd_hash = hashlib.md5("\n".join(post_commands).encode()).hexdigest()[:12]
        safe_name = base_image.replace("/", "_").replace(":", "_") + f"__built_{cmd_hash}.sif"
        sif_path = os.path.join(self._sif_cache_dir, safe_name)

        if os.path.exists(sif_path):
            return sif_path

        # Reuse the per-image locking pattern
        async with self._sif_lock_guard:
            if safe_name not in self._sif_locks:
                self._sif_locks[safe_name] = asyncio.Lock()
            lock = self._sif_locks[safe_name]

        async with lock:
            if os.path.exists(sif_path):
                return sif_path

            base_sif = await self._ensure_sif(base_image)

            post_section = "\n    ".join(post_commands)
            def_content = f"Bootstrap: localimage\nFrom: {base_sif}\n\n%post\n    {post_section}\n"
            tmp_def = sif_path + ".def"
            tmp_sif = sif_path + ".building"
            with open(tmp_def, "w") as f:
                f.write(def_content)

            proc = await asyncio.create_subprocess_exec(
                "apptainer",
                "build",
                "--force",
                tmp_sif,
                tmp_def,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            os.unlink(tmp_def)
            if proc.returncode != 0:
                if os.path.exists(tmp_sif):
                    os.unlink(tmp_sif)
                raise RuntimeError(f"apptainer build failed: {stderr.decode(errors='replace')}")
            os.rename(tmp_sif, sif_path)
            return sif_path

    async def _ensure_sif(self, image: str) -> str:
        """
        Return the path to a cached SIF file for the given Docker image,
        pulling it from the registry if not already cached.
        Mirrors the cleanup() trap in repository.log_docker()'s generated shell script.
        """
        safe_name = image.replace("/", "_").replace(":", "_") + ".sif"
        sif_path = os.path.join(self._sif_cache_dir, safe_name)

        if os.path.exists(sif_path):
            return sif_path

        # Per-image lock to avoid concurrent pulls of the same image
        async with self._sif_lock_guard:
            if image not in self._sif_locks:
                self._sif_locks[image] = asyncio.Lock()
            lock = self._sif_locks[image]

        async with lock:
            # Double-check after acquiring lock
            if os.path.exists(sif_path):
                return sif_path

            tmp_path = sif_path + ".pulling"
            proc = await asyncio.create_subprocess_exec(
                "apptainer",
                "pull",
                "--force",
                tmp_path,
                f"docker://{image}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise RuntimeError(
                    f"apptainer pull failed for {image} (exit {proc.returncode}): {stderr.decode(errors='replace')}"
                )
            os.rename(tmp_path, sif_path)
            return sif_path


if __name__ == "__main__":
    CVDPResourcesServer.run_webserver()
