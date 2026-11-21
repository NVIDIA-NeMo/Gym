# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read publisher checksums for an opt-in harness release before staging it in the sandbox."""

import re

from aiohttp import ClientTimeout

from nemo_gym.client import request


_RELEASES: dict[str, dict[str, str]] = {}


async def release_checksums(repository: str, version: str) -> dict[str, str]:
    """Return verified-format SHA-256 values by filename for a tagged release."""
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?", version):
        raise ValueError(f"Expected an exact release, got {version!r}")
    kimi = repository == "moonshot-ai/kimi-code"
    url = (
        f"https://code.kimi.com/kimi-code/binaries/{version}/manifest.json"
        if kimi
        else f"https://api.github.com/repos/{repository}/releases/tags/v{version}"
    )
    if url not in _RELEASES:
        async with await request("GET", url, timeout=ClientTimeout(total=30)) as response:
            response.raise_for_status()
            metadata = await response.json(content_type=None)
        if kimi and metadata["version"] == version:
            candidates = {asset["filename"]: asset["checksum"] for asset in metadata["platforms"].values()}
        elif not kimi and metadata["tag_name"] == f"v{version}":
            candidates = {
                asset["name"]: (asset.get("digest") or "").removeprefix("sha256:")
                for asset in metadata["assets"]
                if (asset.get("digest") or "").startswith("sha256:")
            }
        else:
            raise ValueError(f"Unexpected version in release metadata for {repository} {version}")
        digests = {
            name: digest
            for name, digest in candidates.items()
            if isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest)
        }
        if not digests:
            raise ValueError(f"No published SHA-256 checksums for {repository} {version}")
        _RELEASES[url] = digests
    return _RELEASES[url]
