# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-rollout artifact fetch + working-tree materialization.

Two `dataset_kind`s:

- `fileset`: artifact is a `.tar.gz` of a repo at a specific SHA. Extract it
  into the scratch dir and use the resulting directory as the working tree.
- `githistory`: artifact is a git bundle. Clone it into the scratch dir and
  `git checkout <head_sha>`.

Everything is per-rollout. There is no shared cross-rollout cache: refetching
artifacts on each rollout removes a class of bugs around stale state and
concurrent writes, at the cost of duplicate downloads under repeats. If
artifact storage becomes the bottleneck, add a content-addressed cache
keyed on `artifact_key`.
"""

from __future__ import annotations

import asyncio
import shutil
import shlex
from pathlib import Path
from typing import Dict, Literal, Optional
from urllib.parse import unquote, urlparse

from nemo_gym.server_utils import raise_for_status, request


DatasetKind = Literal["fileset", "githistory"]

DEFAULT_FETCH_TIMEOUT_S = 120


async def fetch_artifact(
    url: str,
    dest_path: Path,
    timeout_s: int = DEFAULT_FETCH_TIMEOUT_S,
    headers: Optional[Dict[str, str]] = None,
) -> None:
    """Stream-download an artifact via aiohttp into `dest_path`.

    `nemo_gym.server_utils.request` retries indefinitely on `ClientOSError`
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
    """Download a tarball and extract it into `<scratch_dir>/working_tree/`."""
    archive_path = scratch_dir / "source.tar.gz"
    working_tree = scratch_dir / "working_tree"
    working_tree.mkdir(parents=True, exist_ok=True)

    await fetch_artifact(artifact_url, archive_path, timeout_s=fetch_timeout_s, headers=fetch_headers)

    proc = await asyncio.create_subprocess_exec(
        "tar",
        "-xzf",
        str(archive_path),
        "-C",
        str(working_tree),
        "--strip-components=1",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        _, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=300)
    except asyncio.TimeoutError:
        proc.kill()
        raise RuntimeError("tar extraction timed out")

    if proc.returncode != 0:
        proc2 = await asyncio.create_subprocess_exec(
            "tar",
            "-xzf",
            str(archive_path),
            "-C",
            str(working_tree),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr2_b = await asyncio.wait_for(proc2.communicate(), timeout=300)
        if proc2.returncode != 0:
            raise RuntimeError(
                f"tar extraction failed: {stderr_b.decode(errors='replace')} / {stderr2_b.decode(errors='replace')}"
            )

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
    bundle_path = scratch_dir / "source.bundle"
    working_tree = scratch_dir / "working_tree"

    await fetch_artifact(artifact_url, bundle_path, timeout_s=fetch_timeout_s, headers=fetch_headers)

    cmd = (
        f"git clone --quiet {shlex.quote(str(bundle_path))} {shlex.quote(str(working_tree))} "
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
