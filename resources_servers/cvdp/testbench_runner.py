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
import os
import posixpath
import shlex
import tarfile
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import yaml

from nemo_gym.sandbox import (
    AsyncSandbox,
    SandboxCreateError,
    SandboxSpec,
    resolve_provider_config,
    resolve_provider_metadata,
)


if TYPE_CHECKING:
    from resources_servers.cvdp.app import CVDPResourcesServerConfig


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
        return image, None
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

    async def run(
        self,
        rtl_files: Dict[str, str],
        harness_files: Dict[str, Optional[str]],
        task_id: str,
        context_files: Optional[Dict[str, str]] = None,
    ) -> Tuple[int, str, List[Dict]]:
        """Run the task harness against the supplied RTL files."""
        context_files = context_files or {}
        tmp_root = self.config.harness_workspace_dir.strip()
        if tmp_root:
            os.makedirs(tmp_root, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f"cvdp_{task_id}_", dir=tmp_root or None) as workdir:
            workdir_path = Path(workdir)

            for d in ["rtl", "verif", "docs", "src", "rundir"]:
                (workdir_path / d).mkdir()
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
            if not services:
                return 1, "No services found in docker-compose.yml", []

            # Preserve shared workspace state across services.
            archive_path = workdir_path.parent / f"{workdir_path.name}.tar.gz"
            _pack_workspace(workdir_path, archive_path)

            service_results: List[Dict] = []
            try:
                for service in services:
                    exit_code, output = await self._run_service(
                        archive_path,
                        service,
                        compose_content,
                        harness_files,
                    )
                    service_results.append({"service": service, "exit_code": exit_code, "stderr": output})
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

    async def _run_service(
        self,
        workspace_archive: Path,
        service: str,
        compose_content: str,
        harness_files: Optional[Dict[str, Optional[str]]] = None,
    ) -> Tuple[int, str]:
        """Run one Compose service in a sandbox."""
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
                return (
                    1,
                    f"No prepared image for service '{service}' (build key {build_key}). "
                    "Run resources_servers/cvdp/prepare.py for this dataset",
                )
            return 1, f"No image defined for service '{service}'"

        cmd_parts = _build_command(svc["entrypoint"], svc["command"])
        if not cmd_parts:
            return 1, f"Service '{service}' has no explicit command. Image-default commands are not portable"

        workspace = posixpath.normpath(self.config.container_workspace)
        transfer_dir = posixpath.normpath(self.config.container_transfer_dir)
        if not workspace.startswith("/") or workspace == "/":
            return 1, f"container_workspace must be an absolute non-root path: {workspace!r}"
        if not transfer_dir.startswith("/") or transfer_dir == "/":
            return 1, f"container_transfer_dir must be an absolute non-root path: {transfer_dir!r}"
        try:
            links = _compose_workspace_links(svc["volumes"], workspace)
        except ValueError as exc:
            return 1, str(exc)

        dot_env: Dict[str, str] = {}
        for path, content in (harness_files or {}).items():
            if content is not None and posixpath.normpath(path) == "src/.env":
                dot_env = _parse_dot_env(content)
                break
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

        working_dir = svc["working_dir"] or f"{workspace}/rundir"
        if working_dir != workspace and not working_dir.startswith(f"{workspace}/"):
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

        try:
            async with AsyncSandbox(self._build_sandbox_provider_config(), spec) as sandbox:
                await sandbox.start()
                input_archive = f"{transfer_dir}/cvdp-workspace.tar.gz"
                output_archive = f"{transfer_dir}/cvdp-workspace-out.tar.gz"
                await sandbox.upload(workspace_archive, input_archive)
                setup = await sandbox.exec(
                    " && ".join(setup_commands),
                    cwd="/",
                    timeout_s=self.config.container_timeout,
                )
                if setup.return_code != 0:
                    combined = (setup.stderr or "") + (setup.stdout or "")
                    return 1, f"workspace setup failed for service '{service}': {combined}"

                result = await sandbox.exec(
                    command,
                    cwd=working_dir,
                    env=env,
                    timeout_s=self.config.container_timeout,
                )

                snapshot = await sandbox.exec(
                    f"tar -czf {shlex.quote(output_archive)} -C {shlex.quote(workspace)} .",
                    timeout_s=self.config.container_timeout,
                )
                if snapshot.return_code == 0:
                    next_archive = workspace_archive.with_name(f"{workspace_archive.name}.next")
                    await sandbox.download(output_archive, next_archive)
                    next_archive.replace(workspace_archive)
                else:
                    combined = (snapshot.stderr or "") + (snapshot.stdout or "")
                    return 1, f"workspace snapshot failed for service '{service}': {combined}"
        except SandboxCreateError as exc:
            return 1, f"sandbox create failed for service '{service}': {exc}"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return 1, f"sandbox operation failed for service '{service}': {exc}"

        if result.error_type == "timeout":
            return -1, f"sandbox exec timed out after {self.config.container_timeout}s"

        combined = (result.stderr or "") + (result.stdout or "")
        return result.return_code, combined
