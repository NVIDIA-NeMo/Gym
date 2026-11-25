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

"""Run CVDP Compose harnesses through the common sandbox API."""

import asyncio
import hashlib
import json
import logging
import os
import posixpath
import re
import shlex
import tarfile
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import yaml

from nemo_gym.core import PROVIDER_OOM_KILLED, PROVIDER_UNAVAILABLE, VERIFIER_ERROR, NgFailureEvent, observe_failure
from nemo_gym.sandbox import (
    AsyncSandbox,
    SandboxCreateError,
    SandboxSpec,
    resolve_provider_config,
    resolve_provider_metadata,
)
from nemo_gym.verification import SandboxVerificationError
from resources_servers.cvdp.repository_workspace import (
    RepositoryChanges,
    apply_changes,
    extract_snapshot,
    resolve_archive,
)


if TYPE_CHECKING:
    from resources_servers.cvdp.app import CVDPResourcesServerConfig


logger = logging.getLogger(__name__)


class CVDPInfrastructureError(SandboxVerificationError):
    """Verification did not produce a trustworthy task outcome."""

    def __init__(self, message: str, *, failure_kind: str = VERIFIER_ERROR) -> None:
        super().__init__(failure_kind, message)


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
    """Replace image placeholders in harness content."""
    substitutions = {
        "__VERIF_EDA_IMAGE__": config.eda_sim_image,
        "__OSS_SIM_IMAGE__": config.oss_sim_image,
        "__OSS_PNR_IMAGE__": config.oss_pnr_image,
    }
    for placeholder, value in substitutions.items():
        if value and placeholder in content:
            content = content.replace(placeholder, value)
    return content


def _normalize_build_file(path: str, content: str, config: "CVDPResourcesServerConfig") -> str:
    content = _apply_substitutions(content, config)
    if posixpath.basename(path).startswith("Dockerfile") and config.cocotb_bus_version:
        content = re.sub(
            r"(?m)^(RUN\s+pip3?\s+install\s+)([^\n]*cocotb[-_]bus[^\n]*)$",
            lambda match: (
                match.group(0)
                if "cocotb==" in match.group(2)
                else f"{match.group(1)}cocotb=={config.cocotb_bus_version} {match.group(2)}"
            ),
            content,
        )
    if (
        posixpath.basename(path).startswith("Dockerfile")
        and config.oss_pnr_image.startswith("ghcr.io/hdl/impl/pnr")
        and f"FROM {config.oss_pnr_image}" in content
    ):
        content = content.replace(
            "https://bootstrap.pypa.io/get-pip.py",
            "https://bootstrap.pypa.io/pip/3.9/get-pip.py",
        )
    return content


def _service_build_key(
    compose_data: dict,
    service_name: str,
    harness_files: Dict[str, Optional[str]],
    config: "CVDPResourcesServerConfig",
) -> str:
    """Return the deterministic manifest key for a Compose ``build:`` service."""
    svc = (compose_data.get("services") or {}).get(service_name, {})
    build_cfg = svc.get("build", {})
    substituted_files = {
        path: _normalize_build_file(path, content, config)
        for path, content in sorted(harness_files.items())
        if content is not None
    }
    if isinstance(build_cfg, str):
        dockerfile_path = posixpath.join(build_cfg, "Dockerfile")
    else:
        context = str((build_cfg or {}).get("context") or ".")
        dockerfile_path = posixpath.join(context, str((build_cfg or {}).get("dockerfile") or "Dockerfile"))
    dockerfile_path = posixpath.normpath(dockerfile_path)
    dockerfile = substituted_files.get(posixpath.normpath(dockerfile_path))
    if dockerfile is None:
        matches = [
            content
            for path, content in substituted_files.items()
            if posixpath.basename(path) == posixpath.basename(dockerfile_path)
        ]
        dockerfile = matches[0] if len(matches) == 1 else ""

    # Include local files only when the Dockerfile can consume them.
    uses_local_context = False
    for line in dockerfile.splitlines():
        parts = shlex.split(line, comments=True)
        if parts and parts[0].upper() in {"ADD", "COPY"}:
            sources = [part for part in parts[1:-1] if not part.startswith("--")]
            if any("://" not in source for source in sources):
                uses_local_context = True
                break
    payload_files = substituted_files if uses_local_context else {dockerfile_path: dockerfile}
    payload = json.dumps({"build": build_cfg, "files": payload_files}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _resolve_service_image(
    compose_data: dict,
    service_name: str,
    harness_files: Dict[str, Optional[str]],
    config: "CVDPResourcesServerConfig",
    prepared_images: Dict[str, str],
) -> Tuple[str, Optional[str]]:
    """Resolve a direct image or a prebuilt image from the preparation manifest."""
    svc = (compose_data.get("services") or {}).get(service_name, {})
    image = str(svc.get("image") or "").removeprefix("docker://")
    if image:
        return config.image_overrides.get(image, image), None
    if not svc.get("build"):
        return "", None
    key = _service_build_key(compose_data, service_name, harness_files, config)
    return prepared_images.get(key, ""), key


def _parse_compose_service(compose_content: str, service_name: str) -> Dict[str, Any]:
    """Extract the supported fields from a Compose service."""
    data = yaml.safe_load(compose_content) or {}
    service = (data.get("services") or {}).get(service_name, {})
    return {
        "image": service.get("image", ""),
        "command": service.get("command", ""),
        "entrypoint": service.get("entrypoint"),
        "volumes": service.get("volumes", []),
        "working_dir": service.get("working_dir", "/code/rundir"),
        "environment": service.get("environment", {}),
        "env_file": service.get("env_file"),
    }


def _compose_workspace_links(compose_volumes: List[Any], workspace: str = "/code") -> List[Tuple[str, str]]:
    """Translate Compose bind mounts into in-sandbox workspace symlinks."""
    links: List[Tuple[str, str]] = []
    for volume in compose_volumes:
        if isinstance(volume, str):
            parts = volume.split(":", 2)
            if len(parts) < 2:
                continue
            source, target = parts[0], parts[1]
        elif isinstance(volume, dict):
            source = str(volume.get("source") or volume.get("src") or "")
            target = str(volume.get("target") or volume.get("dst") or "")
        else:
            raise ValueError(f"unsupported Compose volume entry: {volume!r}")

        target = posixpath.normpath(target)
        if not target.startswith("/"):
            raise ValueError(f"Compose volume target must be absolute: {target!r}")
        if target in {"/", workspace}:
            raise ValueError(f"Compose volume target cannot replace the sandbox workspace root: {target!r}")
        if os.path.isabs(source):
            raise ValueError(f"host-absolute Compose volume is not provider-neutral: {source!r}")
        source = posixpath.normpath(source.removeprefix("./"))
        if source in {"", ".", ".."} or source.startswith("../"):
            raise ValueError(f"Compose volume escapes the uploaded workspace: {source!r}")
        uploaded_source = posixpath.join(workspace, source)
        if uploaded_source != target:
            links.append((uploaded_source, target))
    return links


# GNU tar reports a file that grew or vanished while it was archived with exit status 1. The archive is still
# written, but that file (or, for a directory, its new entries) may be incomplete. The only files the next service
# does not read are the simulator's waveform dumps, so only those are tolerated; a directory, RTL or any input
# file is a snapshot failure.
_TAR_RACE_LINE = re.compile(
    r"^tar: (?:\./)?(?:[^:]+/)?sim_build/[^/:]+\.(?:fst|vcd): (?:file changed as we read it|File removed before we read it)$"
)


def _snapshot_only_raced(snapshot: Any) -> bool:
    """Whether a failed workspace snapshot only lost a race with a waveform file that was still being written.

    A simulator can keep writing a waveform (``sim_build/*.fst`` or ``*.vcd``) while the snapshot reads it. GNU tar
    then exits 1 with ``file changed as we read it`` (or ``File removed before we read it``) and still writes the
    archive. The waveform is not an input of the next service, so the archive is usable. The snapshot command merges
    tar's diagnostics into ``stdout`` (``2>&1``), so the check does not depend on how a provider lays out its streams.
    ``stderr`` is then only the provider's own status summary and is accepted only as empty, ``exit status 1`` or
    OpenSandbox's foreground form ``CommandExecError: 1``. Any other message, any other exit code and any other
    diagnostic stays a snapshot failure.
    """
    if snapshot.return_code != 1:
        return False
    stderr = (snapshot.stderr or "").strip()
    if stderr and stderr not in (f"exit status {snapshot.return_code}", f"CommandExecError: {snapshot.return_code}"):
        return False
    lines = [line.strip() for line in (snapshot.stdout or "").splitlines() if line.strip()]
    return bool(lines) and all(_TAR_RACE_LINE.match(line) for line in lines)


def _pack_workspace(workdir: Path, archive: Path) -> None:
    """Create the opaque workspace archive transferred between sandbox services."""
    with tarfile.open(archive, "w:gz") as tar:
        for child in sorted(workdir.iterdir()):
            tar.add(child, arcname=child.name)


def _load_dot_env(workdir: str) -> Dict[str, str]:
    """Load Compose-style variables from ``src/.env``."""
    env_path = os.path.join(workdir, "src", ".env")
    if not os.path.isfile(env_path):
        return {}
    with open(env_path, encoding="utf-8") as f:
        return _parse_dot_env(f.read())


def _parse_dot_env(content: str) -> Dict[str, str]:
    env_vars: Dict[str, str] = {}
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, val = line.partition("=")
            env_vars[key.strip()] = val.strip()
    return env_vars


def _build_env(environment: Any, dot_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Merge service env-file vars with a compose ``environment`` field into a
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
    """Run a dataset's Compose-described harness through a configured sandbox provider."""

    def __init__(
        self,
        config: "CVDPResourcesServerConfig",
        named_sandbox_configs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.config = config
        self._sandbox_provider = resolve_provider_config(config.sandbox_provider, named_sandbox_configs)
        self._sandbox_metadata = resolve_provider_metadata(config.sandbox_provider, named_sandbox_configs)
        self._prepared_images = dict(config.prepared_images)
        if config.prepared_image_manifest:
            manifest_path = Path(config.prepared_image_manifest).expanduser()
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            images = manifest.get("images", manifest)
            if not isinstance(images, dict):
                raise ValueError(f"invalid CVDP prepared image manifest: {manifest_path}")
            self._prepared_images.update({str(key): str(value) for key, value in images.items()})

    def check_prepared_images(self, inputs: Iterable[Path]) -> set[str]:
        """Check dataset image mappings before any model calls or sandbox starts."""
        required: set[str] = set()
        checked_repositories: set[str] = set()
        missing: dict[str, set[str]] = {}
        for path in inputs:
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    meta = json.loads(line).get("verifier_metadata") or {}
                    digest = meta.get("repository_workspace_sha256")
                    if digest and digest not in checked_repositories:
                        resolve_archive(self.config.prepared_repository_dir, digest)
                        checked_repositories.add(digest)
                    files = meta.get("harness_files") or {}
                    compose = yaml.safe_load(_apply_substitutions(files.get("docker-compose.yml") or "", self.config))
                    for name, service in ((compose or {}).get("services") or {}).items():
                        image, key = _resolve_service_image(compose, name, files, self.config, self._prepared_images)
                        if key:
                            required.add(key)
                        if key and not image:
                            missing.setdefault(key, set()).add(name)
                        elif not image or (image.startswith("__") and image.endswith("__")):
                            raise ValueError(
                                f"CVDP service {name!r} has no configured image ({image!r}). "
                                "Configure its image before collecting rollouts."
                            )
                        elif (image.startswith(("/", "./", "../")) or image.endswith(".sif")) and not Path(
                            image
                        ).is_file():
                            raise ValueError(f"CVDP verifier image does not exist: {image}")
        if missing:
            details = "; ".join(
                f"{key} (services: {', '.join(sorted(names))})" for key, names in sorted(missing.items())
            )
            raise ValueError(
                f"CVDP dataset requires {len(missing)} missing prepared image mapping(s): {details}. "
                "Run resources_servers/cvdp/prepare.py for this dataset and configure prepared_image_manifest "
                "before collecting rollouts. Use --check-only to validate preparation without building images."
            )
        return required

    async def run(
        self,
        rtl_files: Dict[str, str],
        harness_files: Dict[str, Optional[str]],
        task_id: str,
        context_files: Optional[Dict[str, str]] = None,
        *,
        repository_workspace_sha256: str = "",
        repository_changes: Optional[RepositoryChanges] = None,
    ) -> Tuple[int, str, List[Dict]]:
        """Run the task harness against the supplied RTL files.

        A nonzero exit code is the harness's verdict on the RTL. When the sandbox or the workspace transfer fails,
        no verdict exists and ``SandboxVerificationError`` is raised instead.
        """
        context_files = context_files or {}
        tmp_root = self.config.harness_workspace_dir.strip()
        if tmp_root:
            os.makedirs(tmp_root, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f"cvdp_{task_id}_", dir=tmp_root or None) as workdir:
            workdir_path = Path(workdir)

            if repository_workspace_sha256:
                try:
                    archive = resolve_archive(self.config.prepared_repository_dir, repository_workspace_sha256)
                    extract_snapshot(archive, workdir_path)
                    if repository_changes is not None:
                        apply_changes(workdir_path, repository_changes)
                except (OSError, ValueError, tarfile.TarError) as error:
                    raise CVDPInfrastructureError(f"Heavy repository restore failed: {error}") from error
            harness_root = workdir_path / ".cvdp_harness" if repository_workspace_sha256 else workdir_path

            for d in ["rtl", "verif", "docs", "src", "rundir"]:
                (harness_root / d).mkdir(parents=True, exist_ok=True)
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
                dest = _safe_workspace_path(harness_root, filepath)
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
                raise CVDPInfrastructureError("No docker-compose.yml found in harness_files")

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
            if not services:
                raise CVDPInfrastructureError("No services found in docker-compose.yml")

            # Preserve shared workspace state across services.
            archive_path = workdir_path.parent / f"{workdir_path.name}.tar.gz"
            _pack_workspace(workdir_path, archive_path)

            service_results: List[Dict] = []
            try:
                for index, service in enumerate(services):
                    service_kwargs = {"repository_workspace": True} if repository_workspace_sha256 else {}
                    exit_code, output = await self._run_service(
                        archive_path,
                        service,
                        compose_content,
                        harness_files,
                        hand_off=index < len(services) - 1,
                        **service_kwargs,
                    )
                    service_results.append({"service": service, "exit_code": exit_code, "stderr": output})
                    if exit_code != 0:
                        break
            finally:
                archive_path.unlink(missing_ok=True)
                archive_path.with_name(f"{archive_path.name}.next").unlink(missing_ok=True)

            final_exit_code = 0 if all(r["exit_code"] == 0 for r in service_results) else 1
            combined_stderr = "\n".join(f"[{r['service']}] {r['stderr']}" for r in service_results if r["stderr"])
            return final_exit_code, combined_stderr, service_results

    def _build_sandbox_provider_config(self) -> Any:
        """Add CVDP defaults to an Apptainer provider config."""
        if not isinstance(self._sandbox_provider, dict):
            return self._sandbox_provider
        provider_cfg: Dict[str, Any] = dict(self._sandbox_provider)
        if "apptainer" not in provider_cfg:
            return provider_cfg
        apptainer = dict(provider_cfg.get("apptainer") or {})
        create = dict(apptainer.get("create") or {})
        create.setdefault("start_timeout_s", self.config.container_timeout)
        create.setdefault("mount_point", self.config.container_transfer_dir)
        create.setdefault(
            "extra_start_args", ["--writable-tmpfs", "--containall", "--no-mount", "hostfs,bind-paths,cwd"]
        )
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

    async def _run_service(
        self,
        workspace_archive: Path,
        service: str,
        compose_content: str,
        harness_files: Optional[Dict[str, Optional[str]]] = None,
        *,
        repository_workspace: bool = False,
        hand_off: bool = True,
    ) -> Tuple[int, str]:
        """Run one Compose service in a sandbox.

        With ``hand_off`` the workspace is archived after the service and replaces ``workspace_archive`` for the next
        service. The last service sets it to ``False``: nothing reads its archive, so no snapshot is taken.
        """
        svc = _parse_compose_service(compose_content, service)
        compose_data = yaml.safe_load(compose_content) or {}
        image, build_key = _resolve_service_image(
            compose_data,
            service,
            harness_files or {},
            self.config,
            self._prepared_images,
        )
        if not image:
            if build_key:
                # The run was not provisioned with this harness image; the RTL was never simulated.
                raise CVDPInfrastructureError(
                    f"No prepared image for service '{service}' (build key {build_key}). "
                    "Run resources_servers/cvdp/prepare.py for this dataset",
                )
            raise CVDPInfrastructureError(f"No image defined for service '{service}'")

        cmd_parts = _build_command(svc["entrypoint"], svc["command"])
        if not cmd_parts:
            raise CVDPInfrastructureError(
                f"Service '{service}' has no explicit command. Image-default commands are not portable"
            )

        workspace = posixpath.normpath(self.config.container_workspace)
        transfer_dir = posixpath.normpath(self.config.container_transfer_dir)
        if not workspace.startswith("/") or workspace == "/":
            raise CVDPInfrastructureError(f"container_workspace must be an absolute non-root path: {workspace!r}")
        if not transfer_dir.startswith("/") or transfer_dir == "/":
            raise CVDPInfrastructureError(
                f"container_transfer_dir must be an absolute non-root path: {transfer_dir!r}"
            )
        if transfer_dir == workspace:
            raise CVDPInfrastructureError("container_transfer_dir must differ from container_workspace")
        try:
            if repository_workspace:
                # Upstream AgenticRepository mounts its project volume at /code,
                # and mounts the protected harness and run directory separately.
                links = [(f"{workspace}/.cvdp_harness/src", "/src"), (f"{workspace}/.cvdp_harness/rundir", "/rundir")]
            else:
                links = _compose_workspace_links(svc["volumes"], workspace)
        except ValueError as exc:
            raise CVDPInfrastructureError(str(exc)) from exc

        dot_env: Dict[str, str] = {}
        env_files = svc["env_file"]
        if env_files is None:
            env_paths = ["src/.env"]
        elif isinstance(env_files, str):
            env_paths = [env_files]
        elif isinstance(env_files, list) and all(isinstance(path, str) for path in env_files):
            env_paths = env_files
        else:
            raise CVDPInfrastructureError(f"Unsupported env_file for service '{service}'")
        files = {posixpath.normpath(path): content for path, content in (harness_files or {}).items()}
        for path in env_paths:
            normalized = posixpath.normpath(path)
            if normalized.startswith("/") or normalized == ".." or normalized.startswith("../"):
                raise CVDPInfrastructureError(f"Unsafe env_file path for service '{service}': {path}")
            content = files.get(normalized)
            if content is None:
                if env_files is not None:
                    raise CVDPInfrastructureError(f"Missing env_file for service '{service}': {path}")
                continue
            dot_env.update(_parse_dot_env(content))
        env = _build_env(svc["environment"], dot_env)
        if self.config.container_tmp_bind_path:
            env.update(_build_runtime_tmp_env(self.config.container_tmp_bind_path))
            links.append((f"{workspace}/rundir/tmp", self.config.container_tmp_bind_path))

        setup_commands = [
            f"mkdir -p {shlex.quote(workspace)}",
            f"tar -xzf {shlex.quote(transfer_dir + '/cvdp-workspace.tar.gz')} -C {shlex.quote(workspace)}",
        ]
        for source, target in links:
            setup_commands.extend(
                [
                    f"mkdir -p {shlex.quote(posixpath.dirname(target))}",
                    f"rm -rf -- {shlex.quote(target)}",
                    f"ln -s {shlex.quote(source)} {shlex.quote(target)}",
                ]
            )

        working_dir = "/rundir" if repository_workspace else svc["working_dir"] or f"{workspace}/rundir"
        if not repository_workspace and working_dir != workspace and not working_dir.startswith(f"{workspace}/"):
            working_dir = f"{workspace}/rundir"
        command = shlex.join(
            [
                "env",
                f"HOME={workspace}/rundir",
                "PYTHONNOUSERSITE=1",
                *cmd_parts,
            ]
        )

        spec_extra = dict(self.config.sandbox_spec)
        if spec_extra.get("ttl_s") is None:
            # Setup, the test command and snapshot each have their own timeout.
            # Leave additional headroom for startup and workspace transfers; the
            # SDK's ten-minute default can expire before a ten-minute test ends.
            spec_extra["ttl_s"] = 3 * self.config.container_timeout + 600
        provider_options = dict(spec_extra.pop("provider_options", {}) or {})
        metadata = dict(self._sandbox_metadata)
        metadata.update(spec_extra.pop("metadata", {}) or {})
        for reserved in ("image", "workdir", "env", "files"):
            spec_extra.pop(reserved, None)
        spec = SandboxSpec(
            image=image,
            workdir=working_dir,
            metadata=metadata,
            provider_options=provider_options,
            **spec_extra,
        )

        outcome = None
        sandbox = None
        try:
            sandbox = AsyncSandbox(self._build_sandbox_provider_config(), spec)
            async with sandbox:
                await sandbox.start()
                input_archive = f"{transfer_dir}/cvdp-workspace.tar.gz"
                output_archive = f"{transfer_dir}/cvdp-workspace-out.tar.gz"
                await sandbox.upload(workspace_archive, input_archive)
                setup = await sandbox.exec(
                    " && ".join(setup_commands),
                    cwd="/",
                    timeout_s=self.config.container_timeout,
                )
                if setup.error_type or setup.return_code != 0:
                    combined = (setup.stderr or "") + (setup.stdout or "")
                    raise CVDPInfrastructureError(f"workspace setup failed for service '{service}': {combined}")

                try:
                    result = await sandbox.exec(
                        command,
                        cwd=working_dir,
                        env=env,
                        timeout_s=self.config.container_timeout,
                    )
                except Exception as exc:
                    message = str(exc)
                    # The provider confirms a guest OOM from structured lifecycle evidence.
                    if isinstance(exc, NgFailureEvent) and exc.failure_kind == PROVIDER_OOM_KILLED:
                        observe_failure(exc)
                        outcome = (137, f"test execution exceeded sandbox memory for service '{service}': {message}")
                        return outcome
                    raise
                if result.error_type == "timeout":
                    combined = (result.stderr or "") + (result.stdout or "")
                    outcome = (-1, f"sandbox exec timed out after {self.config.container_timeout}s\n{combined}")
                    return outcome
                if result.error_type or result.return_code is None:
                    raise CVDPInfrastructureError(
                        f"sandbox exec failed for service '{service}': no exit code ({result.error_type}): "
                        f"{result.stderr or result.stdout or ''}"
                    )

                if result.return_code != 0:
                    outcome = (result.return_code, (result.stderr or "") + (result.stdout or ""))
                    return outcome

                snapshot_note = ""
                if hand_off:
                    # Exclude transfer archives and private runtime files from project output.
                    transfer_relative = posixpath.relpath(transfer_dir, workspace)
                    snapshot_exclude = " --exclude=./.apptainer-workdir"
                    if transfer_relative != ".." and not transfer_relative.startswith("../"):
                        snapshot_exclude += f" --exclude={shlex.quote('./' + transfer_relative)}"
                    snapshot = await sandbox.exec(
                        f"tar -czf {shlex.quote(output_archive)}{snapshot_exclude} -C {shlex.quote(workspace)} . 2>&1",
                        timeout_s=self.config.container_timeout,
                    )
                    if not snapshot.error_type and snapshot.return_code == 0:
                        pass
                    elif not snapshot.error_type and _snapshot_only_raced(snapshot):
                        diagnostics = (snapshot.stdout or "").strip()
                        logger.warning(
                            "CVDP workspace snapshot after service '%s' exited %s but only waveform files changed during "
                            "archiving; handing the archive to the next service. tar diagnostics: %s",
                            service,
                            snapshot.return_code,
                            diagnostics,
                        )
                        snapshot_note = (
                            f"\n[workspace snapshot after service '{service}' exited {snapshot.return_code}: "
                            f"{diagnostics}]"
                        )
                    else:
                        combined = (snapshot.stderr or "") + (snapshot.stdout or "")
                        raise CVDPInfrastructureError(f"workspace snapshot failed for service '{service}': {combined}")
                    next_archive = workspace_archive.with_name(f"{workspace_archive.name}.next")
                    await sandbox.download(output_archive, next_archive)
                    next_archive.replace(workspace_archive)
                outcome = (result.return_code, (result.stderr or "") + (result.stdout or "") + snapshot_note)
        except CVDPInfrastructureError:
            raise
        except SandboxCreateError as exc:
            raise CVDPInfrastructureError(
                f"sandbox create failed for service '{service}': {exc}", failure_kind=PROVIDER_UNAVAILABLE
            ) from exc
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if outcome is not None and sandbox is not None and sandbox.teardown_confirmed is True:
                # Remote teardown succeeded; a client-release error must not rerun
                # a measured result (the harness may be randomized). Keep the
                # cleanup diagnostic in the request failure report.
                event = NgFailureEvent(
                    boundary="sandbox",
                    failure_kind=VERIFIER_ERROR,
                    failure_reason=f"client cleanup failed after confirmed teardown: {exc}",
                    operation="aclose",
                )
                event.__cause__ = exc
                observe_failure(event)
                return outcome
            raise CVDPInfrastructureError(f"sandbox operation failed for service '{service}': {exc}") from exc

        combined = (result.stderr or "") + (result.stdout or "") + snapshot_note
        return result.return_code, combined
