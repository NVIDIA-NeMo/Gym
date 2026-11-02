# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download a task's reference files for staging into its sandbox."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def download_reference_files(
    reference_files: list[str],
    reference_file_urls: list[str],
    dest_dir: Path,
) -> list[str]:
    """Materialize reference files into ``dest_dir``.

    Each entry in ``reference_file_urls`` is one of:
    - an HTTP(S) URL — downloaded with HF auth + retry-on-429/5xx (anonymous
      HuggingFace downloads hit an aggressive rate limit when the agent runs
      tasks concurrently; the previous urlretrieve-based version silently
      dropped files, leaving subsequent judging/scoring without the reference
      context — we now pass ``HF_TOKEN`` as a bearer token and retry with
      exponential backoff);
    - an absolute filesystem path or ``file://`` URL — copied directly with
      ``shutil.copy2``, bypassing the urllib retry loop. Benchmarks whose
      reference files already live on the local filesystem (rather than on
      the HuggingFace Hub) use this path.
    """
    import shutil as _shutil
    import time
    import urllib.error
    import urllib.request

    if not reference_files or not reference_file_urls:
        return []

    token = os.environ.get("HF_TOKEN")
    max_attempts = 6  # backoffs: 1, 2, 4, 8, 16, 30 seconds
    downloaded = []
    for file_path, url in zip(reference_files, reference_file_urls):
        rel_path = file_path.lstrip("/")
        local_path = dest_dir / rel_path
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if not (url.startswith("http://") or url.startswith("https://")):
            src = url[len("file://") :] if url.startswith("file://") else url
            try:
                _shutil.copy2(src, local_path)
                downloaded.append(rel_path)
            except Exception as e:
                print(f"Warning: failed to copy local file {src} -> {rel_path}: {e}", flush=True)
            continue

        req = urllib.request.Request(url)
        if token:
            req.add_header("Authorization", f"Bearer {token}")

        last_err: Optional[Exception] = None
        for attempt in range(max_attempts):
            try:
                with urllib.request.urlopen(req, timeout=60) as resp, open(local_path, "wb") as f_out:
                    _shutil.copyfileobj(resp, f_out)
                downloaded.append(rel_path)
                last_err = None
                break
            except urllib.error.HTTPError as e:
                last_err = e
                if e.code != 429 and not (500 <= e.code < 600):
                    break  # non-retryable (404, 403, etc.)
                if attempt < max_attempts - 1:
                    time.sleep(min(2**attempt, 30))
            except Exception as e:  # URLError, timeout, socket errors
                last_err = e
                if attempt < max_attempts - 1:
                    time.sleep(min(2**attempt, 30))

        if last_err is not None:
            print(f"Warning: failed to download {url} -> {rel_path}: {last_err}", flush=True)

    return downloaded
