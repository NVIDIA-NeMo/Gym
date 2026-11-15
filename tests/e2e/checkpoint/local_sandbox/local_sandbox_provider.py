# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sandbox providers whose sandboxes are directories on this machine, for the checkpoint e2e harness scenarios.

Registered as ``local_dir`` (with snapshots) and ``local_dir_plain`` (without, as docker or apptainer are).
A sandbox is a directory under ``base_dir``; its commands run in a local shell, so a harness runs under the real process
supervisor and a checkpoint can pause it with real signals. Every process a sandbox runs carries ``NG_LOCAL_SANDBOX`` in
its environment, so stopping the sandbox kills them all, as deleting a container would.

A snapshot copies the sandbox's directory. A fork of it is created at the path the snapshot was taken from, after the
sandbox there is set aside and its processes killed: the fork has the same absolute paths, as a fork of a container
snapshot does. Every process and server reads the same state from ``base_dir``, as they would from a sandbox service.
"""

import asyncio
import contextlib
import json
import os
import shutil
import signal
import stat
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

from nemo_gym.sandbox.providers.base import (
    SandboxExecResult,
    SandboxHandle,
    SandboxSnapshotInfo,
    SandboxSnapshotNotFoundError,
    SandboxSpec,
    SandboxStatus,
)
from nemo_gym.sandbox.providers.registry import register_provider


MARKER = "NG_LOCAL_SANDBOX"


def kill_sandbox_processes(sandbox_id: str) -> int:
    """SIGKILL every process that carries the sandbox's marker, and those they forked meanwhile; return how many."""
    total = 0
    # A process forked while /proc was read is found by the next pass.
    for _ in range(50):
        found = _kill_marked(sandbox_id)
        total += found
        if not found:
            break
        time.sleep(0.02)
    return total


def _kill_marked(sandbox_id: str) -> int:
    found = 0
    needle = f"{MARKER}={sandbox_id}".encode()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            environ = (entry / "environ").read_bytes()
        except OSError:
            continue
        if needle in environ.split(b"\0"):
            try:
                # A killed process stays listed until its parent reaps it: count only the ones still alive.
                if (entry / "stat").read_text().rsplit(")", 1)[1].split()[0] == "Z":
                    continue
                os.kill(int(entry.name), signal.SIGKILL)
            except (OSError, IndexError):
                continue
            found += 1
    return found


def _not_copied(directory: str, names: list[str]) -> list[str]:
    """The entries of ``directory`` a snapshot leaves out: fifos and sockets, which a copy cannot hold, and files
    that went away since the directory was listed."""
    skipped = []
    for name in names:
        try:
            mode = os.lstat(os.path.join(directory, name)).st_mode
        except FileNotFoundError:
            skipped.append(name)
            continue
        if stat.S_ISFIFO(mode) or stat.S_ISSOCK(mode):
            skipped.append(name)
    return skipped


class LocalDirPlainProvider:
    name = "local_dir_plain"

    def __init__(self, base_dir: str) -> None:
        self.base = Path(base_dir)
        for name in ("boxes", "sandboxes", "snapshots"):
            (self.base / name).mkdir(parents=True, exist_ok=True)

    def _box_file(self, sandbox_id: str) -> Path:
        return self.base / "boxes" / f"{sandbox_id}.json"

    def _box(self, sandbox_id: str) -> dict[str, Any]:
        path = self._box_file(sandbox_id)
        if not path.exists():
            raise RuntimeError(f"sandbox {sandbox_id} does not exist")
        return json.loads(path.read_text())

    def _save(self, sandbox_id: str, box: dict[str, Any]) -> None:
        temporary = self._box_file(sandbox_id).with_suffix(".tmp")
        temporary.write_text(json.dumps(box))
        temporary.replace(self._box_file(sandbox_id))

    def _running(self, sandbox_id: str) -> dict[str, Any]:
        box = self._box(sandbox_id)
        if box["state"] != "running":
            raise RuntimeError(f"sandbox {sandbox_id} is {box['state']}")
        return box

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        sandbox_id = f"sb-{uuid.uuid4().hex[:8]}"
        path = self.base / "sandboxes" / sandbox_id
        (path / "work").mkdir(parents=True)
        self._save(sandbox_id, {"id": sandbox_id, "path": str(path), "state": "running", "from_snapshot": None})
        return SandboxHandle(sandbox_id=sandbox_id, provider_name=self.name, raw=None)

    async def exec(
        self,
        handle: SandboxHandle,
        command: str,
        *,
        cwd: Optional[str] = None,
        env: Optional[dict[str, str]] = None,
        timeout_s: Optional[float] = None,
        user: Any = None,
    ) -> SandboxExecResult:
        box = self._running(handle.sandbox_id)
        process = await asyncio.create_subprocess_exec(
            "bash",
            "-c",
            command,
            cwd=cwd or f"{box['path']}/work",
            env={**os.environ, **(env or {}), MARKER: handle.sandbox_id},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_s)
        except TimeoutError:
            # The command may have ended as the timeout fired.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            await process.wait()
            return SandboxExecResult(stdout=None, stderr=None, return_code=-1, error_type="timeout")
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()
            raise
        return SandboxExecResult(
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
            return_code=process.returncode,
        )

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        self._running(handle.sandbox_id)
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, target_path)

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        self._running(handle.sandbox_id)
        shutil.copyfile(source_path, target_path)

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        box = self._box(handle.sandbox_id)
        return SandboxStatus.RUNNING if box["state"] == "running" else SandboxStatus.STOPPED

    async def close(self, handle: SandboxHandle) -> None:
        try:
            box = self._box(handle.sandbox_id)
        except RuntimeError:
            return
        kill_sandbox_processes(handle.sandbox_id)
        if box["state"] == "running":
            box["state"] = "stopped"
            box["stopped_at"] = time.time()
            self._save(handle.sandbox_id, box)

    async def aclose(self) -> None:
        pass

    async def serialize_handle(self, handle: SandboxHandle, *, scope: Optional[str] = None) -> dict[str, Any]:
        return {"sandbox_id": handle.sandbox_id}

    async def connect(self, descriptor: Mapping[str, Any]) -> SandboxHandle:
        sandbox_id = str(descriptor["sandbox_id"])
        self._running(sandbox_id)
        return SandboxHandle(sandbox_id=sandbox_id, provider_name=self.name, raw=None)


class LocalDirProvider(LocalDirPlainProvider):
    name = "local_dir"

    async def create_from_snapshot(self, snapshot_id: str, spec: SandboxSpec) -> SandboxHandle:
        meta_file = self.base / "snapshots" / snapshot_id / "meta.json"
        if not meta_file.exists():
            raise SandboxSnapshotNotFoundError(f"snapshot {snapshot_id} does not exist", snapshot_id=snapshot_id)
        meta = json.loads(meta_file.read_text())
        path = Path(meta["path"])
        # Whatever sandbox is at the path the snapshot was taken from is set aside, with its processes, so the fork
        # keeps its paths: the sandbox the snapshot was taken from or an earlier fork, running or stopped.
        for box_file in (self.base / "boxes").glob("*.json"):
            box = json.loads(box_file.read_text())
            if box["path"] != str(path):
                continue
            kill_sandbox_processes(box["id"])
            aside = path.with_name(f"{path.name}.superseded-{uuid.uuid4().hex[:6]}")
            if path.exists():
                path.rename(aside)
            box["path"] = str(aside)
            self._save(box["id"], box)
        if path.exists():
            # No box names it: a crash fell between a rename and its save.
            path.rename(path.with_name(f"{path.name}.superseded-{uuid.uuid4().hex[:6]}"))
        await asyncio.to_thread(shutil.copytree, self.base / "snapshots" / snapshot_id / "tree", path, symlinks=True)
        sandbox_id = f"sb-{uuid.uuid4().hex[:8]}"
        self._save(sandbox_id, {"id": sandbox_id, "path": str(path), "state": "running", "from_snapshot": snapshot_id})
        return SandboxHandle(sandbox_id=sandbox_id, provider_name=self.name, raw=None)

    async def snapshot(self, handle: SandboxHandle, *, name: Optional[str] = None) -> str:
        box = self._running(handle.sandbox_id)
        snapshot_id = f"snap-{uuid.uuid4().hex[:8]}"
        directory = self.base / "snapshots" / snapshot_id
        try:
            # Sockets and fifos cannot be copied; a container snapshot would not hold them either.
            await asyncio.to_thread(
                shutil.copytree, box["path"], directory / "tree", symlinks=True, ignore=_not_copied
            )
        except BaseException:
            # A snapshot without meta.json is listed by nothing, so nothing would delete it.
            shutil.rmtree(directory, ignore_errors=True)
            raise
        (directory / "meta.json").write_text(
            json.dumps({"name": name or snapshot_id, "path": box["path"], "sandbox_id": handle.sandbox_id})
        )
        return snapshot_id

    async def wait_snapshot_ready(self, snapshot_id: str) -> None:
        if not (self.base / "snapshots" / snapshot_id / "meta.json").exists():
            raise SandboxSnapshotNotFoundError(f"snapshot {snapshot_id} does not exist", snapshot_id=snapshot_id)

    async def delete_snapshot(self, snapshot_id: str) -> None:
        await asyncio.to_thread(shutil.rmtree, self.base / "snapshots" / snapshot_id, ignore_errors=True)

    async def list_snapshots(self, prefix: str) -> list[SandboxSnapshotInfo]:
        listed = []
        for meta_file in (self.base / "snapshots").glob("*/meta.json"):
            try:
                meta = json.loads(meta_file.read_text())
            except FileNotFoundError:  # deleted since it was listed
                continue
            if meta["name"].startswith(prefix):
                listed.append(
                    SandboxSnapshotInfo(
                        snapshot_id=meta_file.parent.name, name=meta["name"], sandbox_id=meta["sandbox_id"]
                    )
                )
        return listed


register_provider(LocalDirProvider.name, LocalDirProvider, override=True)
register_provider(LocalDirPlainProvider.name, LocalDirPlainProvider, override=True)
