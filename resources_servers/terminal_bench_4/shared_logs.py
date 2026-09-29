# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Episode-owned EFS logs, with isolated roles and a collected artifact snapshot."""

import asyncio
import shlex
from copy import deepcopy
from pathlib import PurePosixPath
from uuid import uuid4

from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec, resolve_provider_config


class SharedLogs:
    mount = "/tmp/tb4-efs"

    def __init__(self, environment):
        config = environment.config
        self.host_path = config.efs_logs_host_path
        path = PurePosixPath(self.host_path)
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError("EFS logs host path must be absolute without traversal")
        self.relative = "tb4-logs-" + uuid4().hex
        self.owner_token = uuid4().hex
        self.session_id = environment.session_id + "__logs"
        self.archive_name = ".tb4-artifacts-" + uuid4().hex + ".tar.gz"
        self.archive_digest = None
        self.restored_archive = None
        self.closed = False
        self.resources = []
        self.cleanup_errors = []
        self._cleanup_task = None
        self.initialized = False
        provider = deepcopy(environment.provider_config)
        self.pool = environment.pool
        # The helper needs no GPU, but must use the workload's deployment so
        # ownership changes and artifact snapshots reach the same EFS share.
        metadata = dict(environment.build_spec().metadata)
        metadata.update({"tb4-session": self.session_id, "tb4-role": "logs-helper"})
        if self.pool != "default":
            metadata["nemo-gym.nvidia.com/resource-pool"] = self.pool
        self.main = AsyncSandbox(
            resolve_provider_config(provider),
            SandboxSpec(
                image=config.efs_logs_init_image,
                resources=SandboxResources(cpu=1, memory_mib=512),
                ttl_s=config.sandbox_ttl_s,
                ready_timeout_s=config.sandbox_ready_timeout_s,
                entrypoint=["sh", "-c", "while :; do sleep 3600; done"],
                metadata=metadata,
                provider_options={
                    "resource_requests": "limits",
                    "volumes": [{"name": "tb4-efs", "host": {"path": self.host_path}, "mountPath": self.mount}],
                },
            ),
        )

    @property
    def root(self):
        return f"{self.mount}/{self.relative}"

    def volume(self, role):
        if role not in {"agent", "verifier"}:
            raise ValueError("Unknown EFS log role")
        return {
            "name": "tb4-logs",
            "host": {"path": self.host_path},
            "subPath": f"{self.relative}/{role}",
            "mountPath": "/logs",
            "readOnly": False,
        }

    async def python(self, source, *args):
        command = "python3 -c " + shlex.quote(source) + " " + " ".join(shlex.quote(str(a)) for a in args)
        result = await self.main.exec(command, timeout_s=600)
        if result.return_code:
            raise RuntimeError(f"EFS logs operation failed: {result.stderr}")
        return result

    async def start(self):
        await self.main.start()
        # Never adopt an existing episode directory. Roles can only access their
        # own subPath; the shared root is mounted solely in the trusted helper.
        self.initialized = True
        await self.python(
            "from pathlib import Path; import sys\n"
            "root = Path(sys.argv[1]); root.mkdir(mode=0o755)\n"
            "(root / '.owner').write_text(sys.argv[2])\n"
            "for role in ('agent', 'verifier'):\n"
            " p = root / role; p.mkdir(); p.chmod(0o777)\n",
            self.root,
            self.owner_token,
        )

    async def initialize_role(self, environment):
        result = await environment.exec("id -u; id -g", timeout_sec=60, user=environment.role_user)
        try:
            uid, gid = (int(value) for value in result.stdout.split())
            if result.return_code or min(uid, gid) < 0:
                raise ValueError("Invalid workload identity")
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Unable to determine the task log owner") from exc
        # Give the mount to the execution role (or unchanged image default).
        # A world-writable parent would let a dropped verifier child rename protected
        # /logs/verifier and replace its reward file despite chmod 700 there.
        await self.python(
            "import os, sys\n"
            "os.chown(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))\n"
            "os.chmod(sys.argv[1], 0o755)\n",
            f"{self.root}/{environment.log_role}",
            uid,
            gid,
        )

    def collection_archive(self, artifact, artifacts):
        """Optimize only the unambiguous conventional main log artifact.

        Other artifacts retain their ordered host restore, including overlaps,
        same-source recollection, sidecars, and custom destinations.
        """
        if artifact.source != "/logs/artifacts" or artifact.service not in (None, "main"):
            return None
        source, target = PurePosixPath(artifact.source), artifact.host_path
        for other in artifacts:
            if other is artifact:
                continue
            other_source, other_target = PurePosixPath(other.source), other.host_path
            if (
                source == other_source
                or source in other_source.parents
                or other_source in source.parents
                or target == other_target
                or target in other_target.parents
                or other_target in target.parents
            ):
                return None
        return f"/logs/{self.archive_name}"

    def retain_archive(self, digest):
        self.archive_digest = digest

    async def prepare_verifier(self):
        if not self.archive_digest:
            return
        # The exact collected archive survives agent deletion on EFS. Validate
        # its digest and apply the same data filter as download_dir before
        # repacking, retaining numeric ownership/modes from A rather than the
        # helper's UID/GID and umask. This matches metadata-aware host restore.
        result = await self.python(
            "import hashlib, pathlib, sys, tarfile, tempfile\n"
            "root, name, digest = sys.argv[1:]\n"
            "root = pathlib.Path(root); source = root / 'agent' / name\n"
            "if source.is_symlink() or not source.is_file(): sys.exit(0)\n"
            "with source.open('rb') as f:\n"
            " if hashlib.file_digest(f, 'sha256').hexdigest() != digest: sys.exit(0)\n"
            "metadata = {}\n"
            "def data_filter(info, destination):\n"
            " filtered = tarfile.data_filter(info, destination)\n"
            " metadata[pathlib.PurePosixPath(filtered.name).as_posix()] = (info.uid, info.gid, info.mode & 0o777)\n"
            " return filtered\n"
            "def headers(info):\n"
            " info.uid, info.gid, info.mode = metadata[pathlib.PurePosixPath(info.name).as_posix()]\n"
            " info.uname = info.gname = ''\n"
            " return info\n"
            "with tempfile.TemporaryDirectory(dir=root) as tmp:\n"
            " with tarfile.open(source, 'r:gz') as tar: tar.extractall(tmp, filter=data_filter)\n"
            " with tarfile.open(root / 'verifier' / name, 'w:gz') as tar:\n"
            "  tar.add(tmp, arcname='.', filter=headers)\n"
            "print('ready')\n",
            self.root,
            self.archive_name,
            self.archive_digest,
        )
        if result.stdout.strip() == "ready":
            self.restored_archive = f"/logs/{self.archive_name}"

    def resource_identities(self):
        result = [{"efs_host_path": self.host_path, "efs_subpath": self.relative}]
        handle = getattr(self.main, "_handle", None)
        if handle is not None:
            result.append({"service": "logs-helper", "provider": self.pool, "sandbox_id": handle.sandbox_id})
        return result

    async def stop(self, *, remove_data=True):
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._stop(remove_data))
        await asyncio.shield(self._cleanup_task)

    async def _stop(self, remove_data):
        self.resources = self.resource_identities()
        errors = []
        if self.initialized and remove_data:
            try:
                await self.python(
                    "import pathlib, shutil, sys\n"
                    "root = pathlib.Path(sys.argv[1])\n"
                    "assert not root.is_symlink()\n"
                    "if root.exists():\n"
                    " assert (root / '.owner').read_text() == sys.argv[2]\n"
                    " shutil.rmtree(root)\n",
                    self.root,
                    self.owner_token,
                )
            except Exception as exc:
                errors.append(exc)
        try:
            await self.main.stop()
        except Exception as exc:
            errors.append(exc)
        if errors:
            self.cleanup_errors.extend({"error": str(exc), "resources": self.resources} for exc in errors)
            raise ExceptionGroup("EFS logs cleanup failed", errors)
        self.closed = True
