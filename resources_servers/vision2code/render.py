# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One fresh Enroot sandbox per render, using Gym's lifecycle and file transfers."""

import asyncio
import json
import os
import re
import shlex
import time
from dataclasses import asdict
from pathlib import Path
from typing import TypeVar

from PIL import Image
from pydantic import BaseModel, Field, JsonValue

from nemo_gym.sandbox import AsyncSandbox, SandboxSpec


MAX_BYTES = 32 * 1024 * 1024
MAX_PIXELS = 25_000_000
# A fresh PID namespace kills script descendants when its Python init exits,
# including children that would otherwise race validation/download.
ISOLATED = (
    "/usr/bin/env -i PATH=/usr/local/bin:/usr/bin:/bin HOME=/workspace LANG=C.UTF-8 TMPDIR=/workspace/.tmp "
    "MPLBACKEND=Agg MPLCONFIGDIR=/workspace/.matplotlib OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "
)
UNSHARE = "/usr/bin/unshare --user --map-current-user --net --pid --fork --kill-child --mount-proc "


class RenderResult(BaseModel):
    status: str = "infrastructure_failure"
    diagnostics: dict[str, JsonValue] = Field(default_factory=dict)
    timings: dict[str, float] = Field(default_factory=dict)
    width: int | None = None
    height: int | None = None
    dimension_mismatch: bool | None = None


class InfrastructureError(RuntimeError):
    """The attempt did not measure the submitted program."""


def normalize_code(text: str) -> str:
    """Accept raw Python or one complete Python fence; do not repair programs."""
    text = text.strip()
    match = re.fullmatch(r"```(?:python|py)?\s*\n(.*?)\n```", text, flags=re.DOTALL | re.IGNORECASE)
    return (match.group(1) if match else text) + "\n"


def validate_png(path: Path) -> tuple[int, int]:
    """Reject unsafe or invalid downloads before exposing them to the judge."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("Output is not a regular file")
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("Output exceeds 32 MiB")
    with Image.open(path) as im:
        if im.format != "PNG":
            raise ValueError("Output is not PNG")
        width, height = im.size
        if width * height > MAX_PIXELS:
            raise ValueError("Output exceeds 25 million pixels")
        if getattr(im, "n_frames", 1) != 1:
            raise ValueError("Animated output is not supported")
        im.verify()
    with Image.open(path) as im:
        im.load()
    return width, height


T = TypeVar("T")


async def finish_task(task: asyncio.Task[T]) -> T:
    """Finish a lifecycle operation even if its caller is cancelled repeatedly."""
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    result = task.result()
    if cancelled:
        raise asyncio.CancelledError
    return result


async def render(
    *, code: str, width: int, height: int, image: str, provider: dict, artifact_dir: Path, timeout: float
) -> RenderResult:
    """Render and download before stopping, persisting diagnostics on every path."""
    result = RenderResult()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    code_path = artifact_dir / "code.py"
    code_path.write_text(normalize_code(code))
    sandbox = None
    started = time.perf_counter()
    try:
        # Enroot inherits these at provider construction. Cluster defaults often
        # mount $HOME, which must not expose references or credentials.
        profile = Path(__file__).with_name("enroot").resolve()
        environment = {
            "ENROOT_MOUNT_HOME": "no",
            "ENROOT_REMAP_ROOT": "no",
            "ENROOT_ROOTFS_WRITABLE": "no",
            "ENROOT_SYSCONF_PATH": str(profile),
            "ENROOT_CONFIG_PATH": str(profile / "user"),
            "ENROOT_LOGIN_SHELL": "no",
        }
        provider = {
            "enroot": {
                **provider["enroot"],
                "create": {**provider["enroot"].get("create", {}), "rw": False, "bypass_entrypoint": False},
            }
        }
        previous = {key: os.environ.get(key) for key in environment}
        try:
            # The provider snapshots os.environ synchronously. Restore the caller's
            # environment before the first await so concurrent attempts cannot mix it.
            os.environ.update(environment)
            sandbox = AsyncSandbox(provider, SandboxSpec(image=image, workdir="/workspace"))
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        # Enroot.create does not catch CancelledError. Let it finish attaching its
        # handle so the finally block can stop it even during startup cancellation.
        await finish_task(asyncio.create_task(sandbox.start()))
        result.timings["startup_s"] = time.perf_counter() - started
        await sandbox.upload(code_path, "/workspace/code.py")
        await sandbox.upload(Path(__file__).with_name("launcher.py"), "/workspace/launcher.py")
        probe = await sandbox.exec(ISOLATED + UNSHARE + "/bin/true", timeout_s=15)
        result.diagnostics["isolation_probe"] = asdict(probe)
        if probe.error_type or probe.return_code:
            raise InfrastructureError("Network/PID isolation probe failed; no code was executed")
        mount_check = (
            "import json; "
            "mounts=[line.split()[4] for line in open('/proc/self/mountinfo')]; "
            "allowed={'/','/.lock','/workspace','/proc','/dev','/dev/shm','/dev/null',"
            "'/dev/zero','/dev/full','/dev/random','/dev/urandom'}; "
            "unexpected=[p for p in mounts if p not in allowed and not "
            "p.startswith('/proc/')]; "
            "print(json.dumps({'mounts':mounts,'unexpected':unexpected})); "
            "raise SystemExit(bool(unexpected))"
        )
        mounts = await sandbox.exec(
            ISOLATED + UNSHARE + "/usr/local/bin/python -I -c " + shlex.quote(mount_check), timeout_s=15
        )
        result.diagnostics["mount_probe"] = asdict(mounts)
        if mounts.error_type or mounts.return_code:
            raise InfrastructureError("Unexpected host mounts or failed mount inspection; no code was executed")
        command = (
            ISOLATED
            + f"OUTPUT_PATH=/workspace/output.png TARGET_WIDTH={width} TARGET_HEIGHT={height} "
            + UNSHARE
            + "/usr/local/bin/python -I /workspace/launcher.py"
        )
        render_start = time.perf_counter()
        # Finish the bounded exec on cancellation before stopping the container;
        # the provider's subprocess runner does not kill on CancelledError.
        executed = await finish_task(asyncio.create_task(sandbox.exec(command, timeout_s=timeout)))
        result.timings["render_s"] = time.perf_counter() - render_start
        result.diagnostics["execution"] = asdict(executed)
        if executed.error_type == "sandbox":
            raise InfrastructureError(executed.stderr or "Sandbox exec failed")
        if executed.error_type == "timeout":
            result.status = "timeout"
            return result
        if executed.error_type:
            raise InfrastructureError(f"Sandbox error: {executed.error_type}")
        if executed.return_code == 137:
            raise InfrastructureError("Renderer was killed (possible allocation OOM)")
        if executed.return_code:
            result.status = "code_error"
            return result
        # Check in a separate trusted process after the script namespace exits.
        # /workspace is the provider's unique mount: download_file also refuses symlinks.
        check = (
            "import os,stat,json; p='/workspace/output.png'; "
            "s=os.lstat(p) if os.path.lexists(p) else None; "
            "print(json.dumps({'exists':s is not None,'regular':s is not None and stat.S_ISREG(s.st_mode),"
            "'bytes':s.st_size if s else 0}))"
        )
        checked = await sandbox.exec(
            ISOLATED + UNSHARE + "/usr/local/bin/python -I -c " + shlex.quote(check), timeout_s=15
        )
        if checked.error_type or checked.return_code:
            raise InfrastructureError(f"Output inspection failed: {checked.stderr}")
        info = json.loads(checked.stdout)
        result.diagnostics["output"] = info
        if not info["exists"]:
            result.status = "missing_output"
            return result
        if not info["regular"] or info["bytes"] > MAX_BYTES:
            result.status = "invalid_output"
            return result
        path = artifact_dir / "output.png"
        await sandbox.download("/workspace/output.png", path)
        try:
            result.width, result.height = validate_png(path)
        except (ValueError, OSError, Image.DecompressionBombError) as exc:
            path.unlink(missing_ok=True)
            result.status = "invalid_output"
            result.diagnostics["validation_error"] = str(exc)
            return result
        result.dimension_mismatch = (result.width, result.height) != (width, height)
        result.status = "ok"
        return result
    except asyncio.CancelledError:
        result.status = "cancelled"
        raise
    except Exception as exc:
        result.status = "infrastructure_failure"
        result.diagnostics["error"] = f"{type(exc).__name__}: {exc}"
        return result
    finally:
        try:
            if sandbox is not None:
                stop_start = time.perf_counter()
                stop_task = asyncio.create_task(sandbox.stop())
                await finish_task(stop_task)
                result.timings["cleanup_s"] = time.perf_counter() - stop_start
                result.diagnostics["stopped"] = True
        except asyncio.CancelledError:
            result.status = "cancelled"
            result.diagnostics["stopped"] = (
                stop_task.done() and not stop_task.cancelled() and stop_task.exception() is None
            )
            raise
        except Exception as exc:
            result.status = "infrastructure_failure"
            result.diagnostics["cleanup_error"] = f"{type(exc).__name__}: {exc}"
            result.diagnostics["stopped"] = False
        finally:
            result.timings["total_s"] = time.perf_counter() - started
            (artifact_dir / "execution.json").write_text(result.model_dump_json(indent=2))
