# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One fresh Enroot sandbox per render, using Gym's lifecycle and file transfers."""

import ast
import asyncio
import json
import logging
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
LOGGER = logging.getLogger(__name__)
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
    """Remove thinking markup and extract the last complete code fence, or raw Python."""
    try:
        ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        pass
    else:
        return text.strip() + "\n"
    # Consume fences as whole tokens so thinking-tag literals inside Python are
    # never interpreted as response wrappers. Multiline fences close on their
    # own line, allowing inline backticks in the submitted program as well.
    pattern = (
        r"```(?:python|py)?[ \t]*(?:\r?\n(?P<block>.*?)^[ \t]*```[ \t]*|(?P<inline>[^\r\n]*?)```)"
        r"|(?P<other>```[^\r\n]*\r?\n.*?^[ \t]*```[ \t]*)"
        r"|(?P<unfinished>```.*\Z)|(?P<tag></?think>)"
    )
    tokens = list(re.finditer(pattern, text, flags=re.DOTALL | re.MULTILINE | re.IGNORECASE))
    last_close = max(
        (i for i, token in enumerate(tokens) if (token.group("tag") or "").lower() == "</think>"), default=-1
    )
    thinking = False
    parts = []
    code = None
    position = 0
    for i, token in enumerate(tokens):
        if not thinking:
            parts.append(text[position : token.start()])
        tag = (token.group("tag") or "").lower()
        if tag == "<think>":
            # An unmatched opening tag is stripped without discarding the code.
            thinking = i < last_close
        elif tag == "</think>":
            if not thinking:
                # Some endpoints omit the opening tag when returning reasoning.
                parts = []
                code = None
            thinking = False
        elif not thinking:
            body = token.group("block")
            if body is None:
                body = token.group("inline")
            if body is not None:
                code = body  # Keep the final complete program even when invalid.
            parts.append(token.group())
        position = token.end()
    if not thinking:
        parts.append(text[position:])
    if code is None:
        code = "".join(parts)
    return code.strip() + "\n"


def validate_image(path: Path) -> tuple[int, int]:
    """Reject unsafe or invalid downloads before exposing them to the judge."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("Output is not a regular file")
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("Output exceeds 32 MiB")
    with Image.open(path) as im:
        if im.format not in {"PNG", "JPEG", "WEBP", "GIF", "TIFF", "BMP"}:
            raise ValueError(f"Unsupported output image format: {im.format}")
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
        except Exception:
            if not cancelled:
                raise
    if cancelled:
        if not task.cancelled():
            task.exception()  # Retrieve a failure that happened while waiting for cancellation-safe completion.
        raise asyncio.CancelledError
    return task.result()


async def stop_sandbox(sandbox: AsyncSandbox, *, attempts: int, diagnostics: dict[str, JsonValue]) -> bool:
    """Retry teardown on the same handle; the owner retains it when all attempts fail."""
    if attempts < 1:
        raise ValueError("Sandbox cleanup attempts must be positive")
    errors = []
    diagnostics["cleanup_errors"] = errors
    for attempt in range(attempts):
        try:
            await sandbox.stop()
            diagnostics["stopped"] = True
            return True
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            errors.append(error)
            LOGGER.warning("Sandbox teardown attempt %s/%s failed: %s", attempt + 1, attempts, error)
    diagnostics["stopped"] = False
    diagnostics["cleanup_error"] = errors[-1]
    return False


async def render(
    *,
    code: str,
    width: int,
    height: int,
    image: str,
    provider: dict[str, dict],
    artifact_dir: Path,
    timeout: float,
    cleanup_attempts: int = 2,
    pending_cleanup: dict[Path, AsyncSandbox] | None = None,
) -> RenderResult:
    """Render and stop; register failed teardown handles in the owner's pending_cleanup mapping."""
    if cleanup_attempts < 1:
        raise ValueError("Sandbox cleanup attempts must be positive")
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
                "create": {
                    **provider["enroot"].get("create", {}),
                    "mount_point": "/workspace",
                    "rw": False,
                    "bypass_entrypoint": False,
                },
                "probe": {"command": "test -d /workspace", "expected_stdout": None},
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
        # This also probes network/PID namespace creation: a failed unshare never
        # reaches Python. A separate /bin/true probe would repeat the same exec.
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
            raise InfrastructureError("Isolation or mount inspection failed; no code was executed")
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
        if executed.return_code:
            result.status = "code_error"
            return result
        # Check in a separate trusted process after the script namespace exits.
        # /workspace is the provider's unique mount: download_file also refuses symlinks.
        check = (
            "import os,stat,json; os.chmod('/workspace',0o700); p='/workspace/output.png'; "
            "s=os.lstat(p) if os.path.lexists(p) else None; "
            "print(json.dumps({'exists':s is not None,'regular':s is not None and stat.S_ISREG(s.st_mode),"
            "'readable':s is not None and os.access(p,os.R_OK),"
            "'bytes':s.st_size if s else 0}))"
        )
        checked = await sandbox.exec(
            ISOLATED + UNSHARE + "/usr/local/bin/python -I -c " + shlex.quote(check), cwd="/", timeout_s=15
        )
        if checked.error_type or checked.return_code:
            raise InfrastructureError(f"Output inspection failed: {checked.stderr}")
        info = json.loads(checked.stdout)
        result.diagnostics["output"] = info
        if not info["exists"]:
            result.status = "missing_output"
            return result
        if not info["regular"] or not info["readable"] or info["bytes"] > MAX_BYTES:
            result.status = "invalid_output"
            return result
        path = artifact_dir / "output.png"
        await sandbox.download("/workspace/output.png", path)
        try:
            result.width, result.height = validate_image(path)
        except (ValueError, OSError, SyntaxError, Image.DecompressionBombError) as exc:
            # Pillow raises SyntaxError for malformed PNG chunks and checksum failures.
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
                if pending_cleanup is not None:
                    pending_cleanup[artifact_dir] = sandbox
                stop_start = time.perf_counter()
                stop_task = asyncio.create_task(
                    stop_sandbox(sandbox, attempts=cleanup_attempts, diagnostics=result.diagnostics)
                )
                stopped = await finish_task(stop_task)
                result.timings["cleanup_s"] = time.perf_counter() - stop_start
                if stopped:
                    if pending_cleanup is not None:
                        pending_cleanup.pop(artifact_dir, None)
                else:
                    result.status = "infrastructure_failure"
        except asyncio.CancelledError:
            result.status = "cancelled"
            stopped = (
                stop_task.done() and not stop_task.cancelled() and stop_task.exception() is None and stop_task.result()
            )
            result.diagnostics["stopped"] = bool(stopped)
            if stopped and pending_cleanup is not None:
                pending_cleanup.pop(artifact_dir, None)
            raise
        except Exception as exc:
            result.status = "infrastructure_failure"
            result.diagnostics["cleanup_error"] = f"{type(exc).__name__}: {exc}"
            result.diagnostics["stopped"] = False
        finally:
            result.timings["total_s"] = time.perf_counter() - started
            (artifact_dir / "execution.json").write_text(result.model_dump_json(indent=2))
