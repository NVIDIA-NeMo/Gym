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

"""Apptainer testbench runner.

This module owns the *mechanism* of CVDP verification: translating a dataset's
docker-compose test harness (the cocotb testbench + compose file each task ships)
into Apptainer calls and executing it in a sandbox. The resources server
(``app.py``) owns the *policy* (the HTTP contract and reward scoring) and
delegates execution to :class:`TestbenchRunner`.

Named for what it runs — CVDP's per-task *test harness* — and deliberately kept
distinct from the "harness" concept on the agent side (a coding agent driven by
``responses_api_agents/cvdp_agent/``).

Layout:
- module-level pure functions: compose -> Apptainer translation (stateless).
- :class:`TestbenchRunner`: stateful executor (SIF cache, per-image locks, the
  lazily-built sandbox provider).
"""

import asyncio
import contextlib
import hashlib
import os
import shlex
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import yaml

from nemo_gym.sandbox import (
    SandboxCreateError,
    SandboxProvider,
    SandboxSpec,
    create_provider,
)


if TYPE_CHECKING:
    from resources_servers.cvdp.app import CVDPResourcesServerConfig


# ----------------------------
# Compose -> Apptainer translation (pure helpers)
# ----------------------------


def _safe_workspace_path(workdir: Path, rel: str) -> Optional[Path]:
    """Resolve ``rel`` under ``workdir``, rejecting absolute paths and traversal.

    The request-provided file maps (``harness_files``, ``context_files``,
    ``rtl_files``) are written into the per-rollout temp workspace before the
    sandbox is started. A hostile or malformed key such as ``/etc/cron.d/x`` or
    ``../../../x`` would otherwise escape the workspace on the host. Returns the
    resolved destination path only if it stays strictly inside ``workdir``,
    otherwise ``None`` (caller skips it). The agent side already filters
    agent-harvested paths, but ``/verify`` can receive ``rtl_files`` directly, so
    the verifier validates here too.
    """
    if not rel or os.path.isabs(rel):
        return None
    base = workdir.resolve()
    dest = (workdir / rel).resolve()
    if dest == base or base not in dest.parents:
        return None
    return dest


def _apply_substitutions(content: str, config: "CVDPResourcesServerConfig") -> str:
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
    config: "CVDPResourcesServerConfig",
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


def _build_binds(workdir: str, compose_volumes: List[str]) -> List[str]:
    """
    Build a list of Apptainer bind specs ("src:dst[:opts]") from:
    1. The standard /code/* workspace mounts
    2. Non-/code volumes from the docker-compose service definition

    This is the provider-facing form (one string per mount), passed through
    ``SandboxSpec.provider_options['binds']``.
    """
    binds: List[str] = []

    # Standard /code/* mounts
    for vol in ["docs", "rundir", "rtl", "verif", "src"]:
        binds.append(f"{workdir}/{vol}:/code/{vol}")

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
        binds.append(bind_spec)

    return binds


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


def _build_env(environment: Any, dot_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Merge workspace src/.env vars with a compose ``environment`` field into a
    plain {key: value} dict. dot_env is applied first so compose values win."""
    env: Dict[str, str] = {}
    if dot_env:
        env.update(dot_env)
    if isinstance(environment, dict):
        for key, val in environment.items():
            env[str(key)] = str(val)
    elif isinstance(environment, list):
        for item in environment:
            text = str(item)
            if "=" in text:
                key, _, val = text.partition("=")
                env[key] = val
    return env


def _build_runtime_tmp_env(container_tmp_path: str) -> Dict[str, str]:
    """
    Force simulator temp and lock files into writable per-rollout container storage.
    """
    return {
        "TMPDIR": container_tmp_path,
        "TMP": container_tmp_path,
        "TEMP": container_tmp_path,
        "TEMPDIR": container_tmp_path,
        "XCELIUM_TMPDIR": container_tmp_path,
        "CDS_LOCK": f"{container_tmp_path}/.cdslock",
        # imc/Java can still hit /tmp unless java.io.tmpdir is forced.
        "JAVA_TOOL_OPTIONS": f"-Djava.io.tmpdir={container_tmp_path}",
    }


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
# Stateful executor
# ----------------------------


class TestbenchRunner:
    """Runs a dataset's docker-compose harness inside Apptainer.

    Owns the SIF cache, per-image pull/build locks, and the lazily-constructed
    sandbox provider. Construct once per server (the apptainer binary is only
    required when a sandbox is actually started, so this can be built on hosts
    without apptainer).
    """

    def __init__(self, config: "CVDPResourcesServerConfig") -> None:
        self.config = config
        self._sif_locks: Dict[str, asyncio.Lock] = {}
        self._sif_lock_guard = asyncio.Lock()
        # Config-selected sandbox provider — built lazily on first use so this
        # can be constructed on hosts without the backend (e.g. apptainer) present.
        self._provider: Optional[SandboxProvider] = None
        self._provider_lock = asyncio.Lock()
        cache = config.sif_cache_dir
        if not cache:
            cache = os.path.join(Path.home(), ".cache", "nemo-gym", "sif")
        self._sif_cache_dir = cache
        os.makedirs(self._sif_cache_dir, exist_ok=True)

    async def run(
        self,
        rtl_files: Dict[str, str],
        harness_files: Dict[str, Optional[str]],
        task_id: str,
        context_files: Optional[Dict[str, str]] = None,
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
        context_files = context_files or {}
        tmp_root = self.config.harness_workspace_dir.strip()
        if tmp_root:
            os.makedirs(tmp_root, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f"cvdp_{task_id}_", dir=tmp_root or None) as workdir:
            workdir_path = Path(workdir)

            # Create all mount dirs — mirrors repository.create_folders()
            for d in ["rtl", "verif", "docs", "src", "rundir"]:
                (workdir_path / d).mkdir()
            # Optional per-rollout temp storage; cleaned when TemporaryDirectory exits.
            if self.config.container_tmp_bind_path:
                (workdir_path / "rundir" / "tmp").mkdir(parents=True, exist_ok=True)

            # Write harness files — mirrors repository.restore_files()
            compose_content: Optional[str] = None
            for filepath, content in harness_files.items():
                if content is None:
                    continue
                content = _apply_substitutions(content, self.config)
                if filepath.endswith("docker-compose.yml"):
                    compose_content = content
                dest = _safe_workspace_path(workdir_path, filepath)
                if dest is None:
                    print(f"Skipping unsafe harness file path: {filepath}")
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with open(str(dest), "w+", encoding="utf-8") as f:
                        f.write(content)
                except Exception:
                    print(f"Failed to write file: {filepath}")

            if compose_content is None:
                return 1, "No docker-compose.yml found in harness_files", []

            # Write companion files from input.context — mirrors
            # repository.restore_files(self.context). Preserves the full
            # target path (e.g. verif/tb_foo.sv -> workdir/verif/tb_foo.sv).
            for filepath, code in context_files.items():
                dest = _safe_workspace_path(workdir_path, filepath)
                if dest is None:
                    print(f"Skipping unsafe context file path: {filepath}")
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with open(str(dest), "w+", encoding="utf-8") as f:
                        f.write(code)
                except Exception:
                    print(f"Failed to write context file: {filepath}")

            # Write model-generated files (overwrites context files for target slots).
            # Preserves the full target path, matching CVDP's restore_files().
            for filepath, code in rtl_files.items():
                dest = _safe_workspace_path(workdir_path, filepath)
                if dest is None:
                    print(f"Skipping unsafe rtl file path: {filepath}")
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with open(str(dest), "w+", encoding="utf-8") as f:
                        f.write(code)
                except Exception:
                    print(f"Failed to write file: {filepath}")

            # Run each service — mirrors repository.obj_harness()
            compose_data = yaml.safe_load(compose_content)
            services = list((compose_data.get("services") or {}).keys())

            service_results: List[Dict] = []
            for service in services:
                exit_code, output = await self._run_service(workdir, service, compose_content, harness_files)
                service_results.append({"service": service, "exit_code": exit_code, "stderr": output})

            final_exit_code = 0 if all(r["exit_code"] == 0 for r in service_results) else 1
            combined_stderr = "\n".join(f"[{r['service']}] {r['stderr']}" for r in service_results if r["stderr"])
            return final_exit_code, combined_stderr, service_results

    def _build_sandbox_provider_config(self) -> Dict[str, Any]:
        """Resolve the single-key sandbox provider config used for grading.

        The backend is config-selected (``config.sandbox_provider``) so the
        verifier is not hard-wired to Apptainer. When the apptainer backend is
        used we fill in the knobs the one-shot CVDP grading path needs, without
        clobbering anything the operator set explicitly:
        - ``--writable-tmpfs`` so EDA tools can write to the container rootfs.
        - readiness probe disabled — we exec the real command immediately and
          surface its failure directly, so an extra probe round-trip is wasted.
        - timeouts pinned to ``container_timeout``; concurrency comfortably above
          the outer ``num_processes`` gate so the provider never bottlenecks.
        """
        provider_cfg: Dict[str, Any] = dict(self.config.sandbox_provider or {"apptainer": {}})
        if "apptainer" not in provider_cfg:
            return provider_cfg
        apptainer = dict(provider_cfg.get("apptainer") or {})
        create = dict(apptainer.get("create") or {})
        create.setdefault("start_timeout_s", self.config.container_timeout)
        create.setdefault("extra_start_args", ["--writable-tmpfs"])
        exec_cfg = dict(apptainer.get("exec") or {})
        exec_cfg.setdefault("default_timeout_s", self.config.container_timeout)
        exec_cfg.setdefault("concurrency", max(32, self.config.num_processes * 4))
        probe = dict(apptainer.get("probe") or {})
        probe.setdefault("command", None)
        apptainer["create"] = create
        apptainer["exec"] = exec_cfg
        apptainer["probe"] = probe
        provider_cfg["apptainer"] = apptainer
        return provider_cfg

    async def _get_provider(self) -> SandboxProvider:
        """Build (once) and return the config-selected sandbox provider.

        Instantiated through the generic provider registry so the grading backend
        is swappable via config; see ``_build_sandbox_provider_config`` for the
        apptainer defaults applied to the one-shot harness run.
        """
        if self._provider is None:
            async with self._provider_lock:
                if self._provider is None:
                    self._provider = create_provider(self._build_sandbox_provider_config())
        return self._provider

    async def _run_service(
        self,
        workdir: str,
        service: str,
        compose_content: str,
        harness_files: Optional[Dict[str, Optional[str]]] = None,
    ) -> Tuple[int, str]:
        """
        Run a single compose service via the Apptainer sandbox provider — mirrors
        repository.log_docker().

        The Docker image is pulled/built into a cached SIF first (the provider
        does not pull or build), then the provider starts an instance with the
        workspace ``/code/*`` mounts (and any compose-defined volumes) bound in,
        execs the service command, and tears the instance down. Apptainer uses
        host networking by default, so no network setup is needed.
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

        # Per-service bind mounts and environment.
        binds = _build_binds(path, svc["volumes"])
        env = _build_env(svc["environment"], _load_dot_env(path))
        if self.config.container_tmp_bind_path:
            binds.append(f"{path}/rundir/tmp:{self.config.container_tmp_bind_path}")
            env.update(_build_runtime_tmp_env(self.config.container_tmp_bind_path))

        # Fix working_dir paths that don't exist under Apptainer's bind mounts.
        # Some compose files use /src/rundir/ which exists in Docker (via volume
        # mount) but not in Apptainer (which only binds to /code/*).
        working_dir = svc["working_dir"] or "/code/rundir"
        if "/code/" not in working_dir:
            working_dir = "/code/rundir"

        cmd_parts = _build_command(svc["entrypoint"], svc["command"])
        # No explicit command -> run the image's default runscript (equivalent to
        # the old ``apptainer run``). HOME is exported in-shell to mirror the old
        # ``--home /code/rundir`` (apptainer refuses HOME via --env).
        inner = shlex.join(cmd_parts) if cmd_parts else "/.singularity.d/runscript"
        command = f"export HOME=/code/rundir; exec {inner}"

        provider = await self._get_provider()
        spec = SandboxSpec(image=sif_path, provider_options={"binds": binds})

        try:
            handle = await provider.create(spec)
        except SandboxCreateError as exc:
            return 1, f"apptainer instance start failed for service '{service}': {exc}"

        try:
            result = await provider.exec(
                handle,
                command,
                cwd=working_dir,
                env=env,
                timeout_s=self.config.container_timeout,
            )
        finally:
            with contextlib.suppress(Exception):
                await provider.close(handle)

        if result.error_type == "timeout":
            return -1, f"apptainer exec timed out after {self.config.container_timeout}s"

        # Mirror the old (stderr + stdout) ordering for combined diagnostics.
        combined = (result.stderr or "") + (result.stdout or "")
        return result.return_code, combined

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
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise RuntimeError(
                    f"apptainer pull failed for {image} (exit {proc.returncode}): {stderr.decode(errors='replace')}"
                )
            os.rename(tmp_path, sif_path)
            return sif_path
