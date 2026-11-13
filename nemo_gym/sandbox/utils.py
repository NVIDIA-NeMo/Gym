# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sandbox utility helpers."""

import hashlib
import logging
import tempfile
import uuid
from collections.abc import Iterable
from pathlib import Path
from shlex import quote
from typing import Protocol

import anyio

from nemo_gym._moved_paths import moved_names_getattr
from nemo_gym.core.media import MediaFile
from nemo_gym.sandbox.providers.base import SandboxExecResult


class _SandboxFileTransfer(Protocol):
    async def upload(self, local_path: Path | str, remote_path: str) -> None: ...

    async def download(self, remote_path: str, local_path: Path | str) -> None: ...

    async def exec(
        self, command: str, *, cwd: str | None = None, timeout_s: int | float | None = 180
    ) -> SandboxExecResult: ...


LOGGER = logging.getLogger(__name__)


# Parallelism caps for CPU-limited sandboxes, each set by cpu_cap_env() to the
# floored CPU limit (min 1). Tools size worker pools by host core count, not
# the cgroup limit — Python's multiprocessing.cpu_count(), BLAS/OpenMP thread
# pools, Go's runtime, cargo, Node — so on wide hosts with small limits they
# overcommit, CFS-throttle, and OOM. GNU nproc honors OMP_NUM_THREADS, which
# also tames `make -j$(nproc)` builds; PYTHON_CPU_COUNT caps os/multiprocessing
# .cpu_count() itself on Python 3.13+. MAKEFLAGS is deliberately absent:
# injecting `-j` would flip plain `make` runs from serial to parallel, and
# explicit `-j` flags override MAKEFLAGS anyway. Injection is intentionally
# per-call-site (no sandbox-layer default); callers merge explicit env over it.
CPU_CAP_ENV_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "PYTHON_CPU_COUNT",
    "NUMBA_NUM_THREADS",
    "LOKY_MAX_CPU_COUNT",
    "POLARS_MAX_THREADS",
    "PYTEST_XDIST_AUTO_NUM_WORKERS",
    "DJANGO_TEST_PROCESSES",
    "GOMAXPROCS",
    "CARGO_BUILD_JOBS",
    "RAYON_NUM_THREADS",
    "RUST_TEST_THREADS",
    "UV_THREADPOOL_SIZE",
    "CMAKE_BUILD_PARALLEL_LEVEL",
)


async def upload_bytes(sandbox: _SandboxFileTransfer, *, path: str, data: bytes) -> None:
    """Upload bytes unchanged through file transfer without shell interpolation."""
    await _upload(sandbox, path=path, data=data, readable_by_all=False)


async def _upload(sandbox: _SandboxFileTransfer, *, path: str, data: bytes, readable_by_all: bool) -> None:
    with tempfile.TemporaryDirectory(prefix="sandbox-upload-") as directory:
        source = Path(directory) / "payload"
        source.write_bytes(data)
        if readable_by_all:
            # Providers that copy the file keep its mode, which the host's umask may have made private.
            source.chmod(0o644)
        await sandbox.upload(source, path)


async def upload_text(sandbox: _SandboxFileTransfer, *, path: str, text: str) -> None:
    """Upload UTF-8 text through file transfer without shell interpolation."""
    await upload_bytes(sandbox, path=path, data=text.encode("utf-8"))


async def download_bytes(sandbox: _SandboxFileTransfer, *, path: str) -> bytes:
    """Download a file's bytes unchanged through file transfer."""
    with tempfile.TemporaryDirectory(prefix="sandbox-download-") as directory:
        destination = Path(directory) / "payload"
        await sandbox.download(path, destination)
        return destination.read_bytes()


async def read_text(sandbox: _SandboxFileTransfer, *, path: str) -> str:
    """Download UTF-8 text, replacing invalid bytes in partial process output.

    Line endings become ``\\n``, as when reading a file in text mode.
    """
    text = (await download_bytes(sandbox, path=path)).decode("utf-8", errors="replace")
    return text.replace("\r\n", "\n").replace("\r", "\n")


# Prompt media live in one directory per sandbox, named by content, so every rollout of a task gets the same paths.
# Sessions never delete these files, because a session sharing a borrowed sandbox may be reading them. A sandbox that
# isolates its /tmp drops them when it is destroyed; the local provider, and Apptainer without --contain, share the
# host's /tmp, where they stay until the host clears it.
MEDIA_DIRECTORY = "/tmp/nemo-gym-media"
# Bounds removing partial uploads after a failure, so a cancelled caller is not held by an unresponsive sandbox.
_CLEANUP_TIMEOUT_S = 30


def media_path(file: MediaFile) -> str:
    """Return the sandbox path ``stage_media`` writes ``file`` to, which depends only on its content and type."""
    return f"{MEDIA_DIRECTORY}/{hashlib.sha256(file.data).hexdigest()}.{file.extension}"


async def stage_media(sandbox: _SandboxFileTransfer, files: Iterable[MediaFile]) -> list[str]:
    """Write each media file to its ``media_path`` and return the paths, in order.

    Harness CLIs read media from files, so an agent decodes prompt media with ``nemo_gym.core.split_text_and_media``,
    stages it here and passes the paths to the harness.
    The sandbox's default user creates the media directory, so it can rename files there even when uploads run as
    another user. Each file is uploaded under a unique name, readable by every user, and then renamed onto its path in
    the same directory. The rename is atomic, so a session sharing the sandbox never reads a partly written file, and
    replacing an existing file swaps in identical bytes. When staging fails, the partial uploads are removed.

    Raises:
        SandboxUnavailableError: The provider could not run a staging command, for example because it timed out.
        SandboxTransferError: A staging command failed inside the sandbox, for example on a permission error.
    """
    media = list(files)
    paths = [media_path(file) for file in media]
    if not media:
        return paths
    await _run_media_command(sandbox, f"mkdir -p -- {quote(MEDIA_DIRECTORY)}", action="create the media directory")
    partials: dict[str, str] = {}
    try:
        for path, file in zip(paths, media):
            if path not in partials:
                partials[path] = f"{MEDIA_DIRECTORY}/.{uuid.uuid4().hex}.partial"
                await _upload(sandbox, path=partials[path], data=file.data, readable_by_all=True)
        renames = " && ".join(_rename_command(partial, path) for path, partial in partials.items())
        await _run_media_command(sandbox, renames, action="rename the staged media files")
    except BaseException:
        # Shielded because an anyio cancel scope, such as the one that cancels a request on client disconnect, stays
        # cancelled and would cancel the cleanup at once. The deadline keeps an unresponsive sandbox from holding a
        # cancelled caller.
        with anyio.move_on_after(_CLEANUP_TIMEOUT_S, shield=True) as cleanup:
            await _remove_partial_media(sandbox, partials.values())
        if cleanup.cancelled_caught:
            LOGGER.warning("Gave up removing partial media uploads from the sandbox after %s s", _CLEANUP_TIMEOUT_S)
        raise
    return paths


def _rename_command(partial: str, path: str) -> str:
    # mv would move the file into a directory at the target path. GNU mv -T refuses that, but BSD mv, as on macOS,
    # has no -T, so test for the directory first. Only something creating a directory at a content-hash path between
    # the test and the rename could get past it.
    return (
        f"{{ [ ! -d {quote(path)} ] || {{ echo {quote(path + ' is a directory')} >&2; false; }}; }}"
        f" && mv -f -- {quote(partial)} {quote(path)}"
    )


async def _run_media_command(sandbox: _SandboxFileTransfer, command: str, *, action: str) -> None:
    # Imported here so that importing nemo_gym.sandbox, which loads this module eagerly, does not load transfer.
    from nemo_gym.sandbox.transfer import SandboxTransferError, SandboxUnavailableError

    result = await sandbox.exec(command, cwd="/")
    if result.error_type:
        raise SandboxUnavailableError(
            f"The sandbox could not {action}: {result.error_type}: {result.stderr or result.stdout or ''}"
        )
    if result.return_code:
        raise SandboxTransferError(f"Cannot {action} in {MEDIA_DIRECTORY}: {result.stderr or result.stdout or ''}")


async def _remove_partial_media(sandbox: _SandboxFileTransfer, partials: Iterable[str]) -> None:
    """Remove partial uploads after a failure, so a borrowed sandbox does not keep them."""
    command = "rm -f -- " + " ".join(quote(partial) for partial in partials)
    try:
        result = await sandbox.exec(command, cwd="/", timeout_s=_CLEANUP_TIMEOUT_S)
    except Exception:
        LOGGER.warning("Cannot remove partial media uploads from the sandbox", exc_info=True)
        return
    if result.error_type or result.return_code:
        LOGGER.warning("Cannot remove partial media uploads from the sandbox: %s", result.stderr or result.error_type)


def cpu_cap_env(cpu: float | int | None) -> dict[str, str]:
    """CPU_CAP_ENV_VARS derived from a sandbox CPU limit; empty when unset."""
    if cpu is None:
        return {}
    cores = str(max(1, int(cpu)))
    return {name: cores for name in CPU_CAP_ENV_VARS}


def rewrite_image(image: str | None, rewrites: list[dict[str, str]]) -> str | None:
    """Apply ordered image-prefix rewrites used by sandbox configs."""
    if image is None:
        return None
    for rewrite in rewrites:
        from_prefix = rewrite["from"]
        to_prefix = rewrite["to"]
        if image.startswith(from_prefix):
            return to_prefix + image[len(from_prefix) :]
    return image


# Keep the formerly imported class available without eagerly importing the sandbox API.
__getattr__ = moved_names_getattr(__name__)
