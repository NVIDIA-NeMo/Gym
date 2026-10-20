# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task artifact transfer and isolated judge execution."""

import asyncio
import hashlib
import json
import os
import shutil
import signal
import stat
import sys
import zipfile
from pathlib import Path, PurePosixPath
from shlex import quote
from typing import Any

from nemo_gym.sandbox import AsyncSandbox


class ArtifactIO:
    @staticmethod
    async def _upload_tree(
        sandbox: AsyncSandbox, source: Path, remote_root: str
    ) -> None:
        files = [path for path in source.rglob("*") if path.is_file()]
        if any(path.is_symlink() for path in source.rglob("*")):
            raise ValueError("Symlinks are not accepted in staged GDPval inputs")
        parents = sorted(
            {
                str(PurePosixPath(remote_root) / path.relative_to(source).parent)
                for path in files
            }
        )
        if parents:
            result = await sandbox.exec(
                "mkdir -p " + " ".join(quote(p) for p in parents)
            )
            if result.return_code:
                raise RuntimeError(
                    f"Failed to create sandbox input directories: {result.stderr}"
                )
        for path in files:
            remote = str(PurePosixPath(remote_root) / path.relative_to(source))
            await sandbox.upload(path, remote)

    @staticmethod
    def _extract_results(archive: Path, task: Path) -> None:
        roots = {
            "workspace/deliverables": task / "agent" / "deliverables",
            "trace": task / "trace",
        }
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                mode = member.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise ValueError(
                        f"Sandbox output contains a symlink: {member.filename}"
                    )
                name = PurePosixPath(member.filename)
                if name.is_absolute() or ".." in name.parts:
                    raise ValueError(f"Unsafe sandbox output path: {member.filename}")
                destination = None
                for prefix, root in roots.items():
                    prefix_path = PurePosixPath(prefix)
                    try:
                        relative = name.relative_to(prefix_path)
                    except ValueError:
                        continue
                    destination = root.joinpath(*relative.parts)
                    break
                if destination is None or member.is_dir():
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(member) as src, destination.open("wb") as dst:
                    shutil.copyfileobj(src, dst)

    @staticmethod
    def _collect_artifacts(task: Path) -> list[dict[str, Any]]:
        source = task / "agent" / "deliverables"
        paths = list(source.rglob("*"))
        if (
            source.is_symlink()
            or not source.is_dir()
            or any(p.is_symlink() for p in paths)
        ):
            raise ValueError(
                "Deliverables must be a real directory tree without symlinks"
            )
        output = task / "artifacts"
        output.mkdir()
        manifest = []
        for path in sorted(p for p in paths if p.is_file()):
            relative = path.relative_to(source)
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            digest = hashlib.sha256(destination.read_bytes()).hexdigest()
            manifest.append(
                {
                    "path": str(relative),
                    "bytes": destination.stat().st_size,
                    "sha256": digest,
                }
            )
        (task / "artifact-manifest.json").write_text(json.dumps(manifest) + "\n")
        return manifest

    @staticmethod
    async def _run_judge(task: Path, config_path: Path, timeout: int) -> None:
        log_path = task / "judge.log"
        bridge = Path(__file__).with_name("gym_bridge.py")
        with log_path.open("wb") as log:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                str(bridge),
                "judge",
                "--config",
                str(config_path),
                "--task",
                str(task),
                stdout=log,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                await asyncio.wait_for(process.wait(), timeout=timeout + 180)
            except BaseException as error:
                # The judge may have spawned document converters in its process group.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await process.wait()
                if isinstance(error, asyncio.TimeoutError):
                    raise TimeoutError(
                        f"GDPval judge timed out; inspect {log_path}"
                    ) from error
                raise
        if process.returncode:
            raise RuntimeError(f"GDPval judge failed; inspect {log_path}")
