# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-rollout artifact fetch + working-tree materialization.

Two `dataset_kind`s:

- `fileset`: artifact is a `.tar.gz` of a repo at a specific SHA. Extract it
  into the scratch dir and use the resulting directory as the working tree.
- `githistory`: artifact is a git bundle. Clone it into the scratch dir and
  `git checkout <head_sha>`.

Artifacts on a local path (or `file://` URL, e.g. a Lustre mirror of the
dataset) are read in place; remote ones are downloaded into the scratch dir.

`SharedWorkingTrees` lets concurrent rollouts of the same task (the GRPO
generations of one prompt) share one materialized tree. Sandboxes mount the
tree read-only, so sharing does not change what a rollout can observe; the
tree is deleted when its last user releases it.
"""

from __future__ import annotations

import asyncio
import shlex
import shutil
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Awaitable, Callable, Dict, Literal, Optional
from urllib.parse import unquote, urlparse

from nemo_gym.client import raise_for_status, request


DatasetKind = Literal["fileset", "githistory"]

DEFAULT_FETCH_TIMEOUT_S = 120


def local_artifact_path(url: str) -> Optional[Path]:
    """The filesystem path of a local (`/path` or `file://`) artifact URL, else None."""
    parsed = urlparse(url)
    if parsed.scheme not in {"", "file"}:
        return None
    src_path = Path(unquote(parsed.path if parsed.scheme == "file" else url))
    if not src_path.exists():
        raise FileNotFoundError(f"local artifact not found: {src_path}")
    return src_path


async def fetch_artifact(
    url: str,
    dest_path: Path,
    timeout_s: int = DEFAULT_FETCH_TIMEOUT_S,
    headers: Optional[Dict[str, str]] = None,
) -> None:
    """Stream-download an artifact via aiohttp into `dest_path`.

    `nemo_gym.client.http_client.request` retries indefinitely on `ClientOSError`
    (designed for transient model-server hiccups during a training run). For
    a broken or unreachable artifact URL the rollout should fail fast with a
    clean exception, not hang forever, so the fetch is wrapped with a hard
    ceiling. The caller's `try/except` in `run()` turns this into
    `reward=0.0` plus an `error` field on the verify response.

    `headers` lets callers attach an Authorization header (e.g.
    ``{"Authorization": "Bearer ${HF_TOKEN}"}``) for fetching from private
    Hugging Face dataset repos. Keys are passed through verbatim to aiohttp.
    """

    parsed = urlparse(url)
    if parsed.scheme in {"", "file"}:
        src_path = Path(unquote(parsed.path if parsed.scheme == "file" else url))
        if not src_path.exists():
            raise FileNotFoundError(f"local artifact not found: {src_path}")
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_path, dest_path)
        return

    async def _do_fetch() -> None:
        kwargs: Dict[str, object] = {}
        if headers:
            kwargs["headers"] = dict(headers)
        response = await request(method="GET", url=url, **kwargs)
        await raise_for_status(response)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with dest_path.open("wb") as f:
            async for chunk in response.content.iter_chunked(64 * 1024):
                f.write(chunk)
        response.release()

    try:
        await asyncio.wait_for(_do_fetch(), timeout=timeout_s)
    except asyncio.TimeoutError as e:
        raise RuntimeError(f"artifact fetch timed out after {timeout_s}s: {url}") from e


async def materialize_fileset(
    artifact_url: str,
    scratch_dir: Path,
    fetch_timeout_s: int = DEFAULT_FETCH_TIMEOUT_S,
    fetch_headers: Optional[Dict[str, str]] = None,
) -> Path:
    """Extract a tarball into `<scratch_dir>/working_tree/` (entering a single wrapping directory, if any)."""
    working_tree = scratch_dir / "working_tree"
    working_tree.mkdir(parents=True, exist_ok=True)

    archive_path = local_artifact_path(artifact_url)
    downloaded = archive_path is None
    if downloaded:
        archive_path = scratch_dir / "source.tar.gz"
        await fetch_artifact(artifact_url, archive_path, timeout_s=fetch_timeout_s, headers=fetch_headers)

    # Extract as-is. VCQA tarballs are `git archive` output with the repo at the archive root; stripping a
    # leading component would drop every top-level file and flatten every top-level directory into the root.
    proc = await asyncio.create_subprocess_exec(
        "tar",
        "-xzf",
        str(archive_path),
        "-C",
        str(working_tree),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        _, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=300)
    except asyncio.TimeoutError:
        proc.kill()
        raise RuntimeError("tar extraction timed out")
    if proc.returncode != 0:
        raise RuntimeError(f"tar extraction failed: {stderr_b.decode(errors='replace')}")

    # An archive that wraps the repo in one directory (`git archive --prefix=<name>/`) is entered instead.
    entries = list(working_tree.iterdir())
    if len(entries) == 1 and entries[0].is_dir() and not entries[0].is_symlink():
        working_tree = entries[0]

    if downloaded:
        try:
            archive_path.unlink()
        except OSError:
            pass

    return working_tree


async def materialize_githistory(
    artifact_url: str,
    head_sha: str,
    scratch_dir: Path,
    fetch_timeout_s: int = DEFAULT_FETCH_TIMEOUT_S,
    fetch_headers: Optional[Dict[str, str]] = None,
) -> Path:
    """Download a git bundle and check out `head_sha` into `working_tree/`."""
    working_tree = scratch_dir / "working_tree"

    bundle_path = local_artifact_path(artifact_url)
    downloaded = bundle_path is None
    if downloaded:
        bundle_path = scratch_dir / "source.bundle"
        await fetch_artifact(artifact_url, bundle_path, timeout_s=fetch_timeout_s, headers=fetch_headers)

    # --no-checkout: the one checkout is the task's head_sha, not the bundle's default branch first.
    cmd = (
        f"git clone --quiet --no-checkout {shlex.quote(str(bundle_path))} {shlex.quote(str(working_tree))} "
        f"&& cd {shlex.quote(str(working_tree))} "
        f"&& git checkout --quiet {shlex.quote(head_sha)}"
    )
    proc = await asyncio.create_subprocess_exec(
        "bash",
        "-lc",
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=600)
    except asyncio.TimeoutError:
        proc.kill()
        raise RuntimeError("git clone/checkout timed out")
    if proc.returncode != 0:
        raise RuntimeError(
            f"git clone/checkout failed (exit={proc.returncode}): "
            f"stdout={stdout_b.decode(errors='replace')} "
            f"stderr={stderr_b.decode(errors='replace')}"
        )

    if downloaded:
        try:
            bundle_path.unlink()
        except OSError:
            pass

    return working_tree


async def materialize_working_tree(
    *,
    dataset_kind: DatasetKind,
    artifact_url: str,
    head_sha: str | None,
    scratch_dir: Path,
    fetch_timeout_s: int = DEFAULT_FETCH_TIMEOUT_S,
    fetch_headers: Optional[Dict[str, str]] = None,
) -> Path:
    """Dispatch to the kind-specific materializer and return the working tree path."""
    if dataset_kind == "fileset":
        return await materialize_fileset(
            artifact_url=artifact_url,
            scratch_dir=scratch_dir,
            fetch_timeout_s=fetch_timeout_s,
            fetch_headers=fetch_headers,
        )
    if dataset_kind == "githistory":
        if not head_sha:
            raise ValueError("githistory rows require verifier_metadata.head_sha")
        return await materialize_githistory(
            artifact_url=artifact_url,
            head_sha=head_sha,
            scratch_dir=scratch_dir,
            fetch_timeout_s=fetch_timeout_s,
            fetch_headers=fetch_headers,
        )
    raise ValueError(f"unknown dataset_kind: {dataset_kind!r}")


@dataclass
class _SharedTree:
    directory: Path
    refs: int = 0
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    path: Optional[Path] = None
    error: Optional[BaseException] = None


class SharedWorkingTrees:
    """Reference-counted working trees shared by concurrent rollouts of the same task.

    The first rollout to `acquire` a key builds the tree into a fresh directory under
    `root`; rollouts arriving while it is in use wait for that build and reuse it. The
    directory is removed when the last holder releases it, so idle trees never pile up.
    A failed build fails every rollout waiting on it; the next acquire retries.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self._entries: Dict[str, _SharedTree] = {}
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def acquire(self, key: str, build: Callable[[Path], Awaitable[Path]]) -> AsyncIterator[Path]:
        async with self._lock:
            entry = self._entries.get(key)
            builder = entry is None
            if builder:
                self.root.mkdir(parents=True, exist_ok=True)
                entry = _SharedTree(directory=Path(tempfile.mkdtemp(prefix="tree_", dir=self.root)))
                self._entries[key] = entry
            entry.refs += 1
        try:
            if builder:
                try:
                    entry.path = await build(entry.directory)
                except BaseException as e:
                    entry.error = e
                    raise
                finally:
                    entry.ready.set()
            else:
                await entry.ready.wait()
                if entry.error is not None:
                    raise RuntimeError(f"shared working tree build failed: {entry.error!r}")
            assert entry.path is not None
            yield entry.path
        finally:
            async with self._lock:
                entry.refs -= 1
                last = entry.refs == 0
                if last and self._entries.get(key) is entry:
                    del self._entries[key]
            if last:
                await asyncio.to_thread(shutil.rmtree, entry.directory, True)
