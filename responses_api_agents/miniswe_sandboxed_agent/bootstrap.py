# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-side cache of pinned bootstrap assets, shared by all task sandboxes."""

import asyncio
import hashlib
from pathlib import Path
from tempfile import gettempdir
from uuid import uuid4

from aiohttp import ClientTimeout

from nemo_gym.server_utils import request


CACHE_DIR = Path(gettempdir()) / "nemo-gym-miniswe-assets"
_DOWNLOAD_LOCK = asyncio.Lock()
# Matches uv 0.10.12's Python download manifest.
PYTHON_SHA256 = {
    ("aarch64", "gnu"): (
        "0ebc0049121318b5de80b887d22abaed55dd302014f73cd6811c9981a83d960e"  # pragma: allowlist secret
    ),
    ("aarch64", "musl"): (
        "88fb902adca37099176fe5c94bb4483b8eb3242606e0a74fd50616a7c83bce63"  # pragma: allowlist secret
    ),
    ("x86_64", "gnu"): (
        "904adc9bc4371c01b20f7e75a19b10f07fb577889be02bf21f7f449229b97611"  # pragma: allowlist secret
    ),
    ("x86_64", "musl"): (
        "3c9db1ed094d6d08e474600b7a4eab7cab655b60f7f74c3a41f1d995eb53611a"  # pragma: allowlist secret
    ),
}


async def cached_download(url: str, *, sha256: str | None = None) -> Path:
    """Download once per cache, serializing concurrent first-task requests."""
    async with _DOWNLOAD_LOCK:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = CACHE_DIR / url.rsplit("/", 1)[1]
        if path.is_file() and (sha256 is None or hashlib.sha256(path.read_bytes()).hexdigest() == sha256):
            return path
        temporary = path.with_name(path.name + "." + uuid4().hex)
        try:
            async with await request("GET", url, timeout=ClientTimeout(total=120)) as response:
                response.raise_for_status()
                data = await response.read()
            if sha256 and hashlib.sha256(data).hexdigest() != sha256:
                raise RuntimeError(f"Bootstrap checksum mismatch: {url}")
            temporary.write_bytes(data)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return path


async def bootstrap_assets(arch: str, libc: str) -> tuple[Path, Path]:
    """Select Linux binaries using the sandbox architecture and libc."""
    if (arch, libc) not in PYTHON_SHA256:
        raise RuntimeError(f"Unsupported mini-SWE sandbox platform: {arch}/{libc}")
    uv = await cached_download(
        f"https://github.com/astral-sh/uv/releases/download/0.10.12/uv-{arch}-unknown-linux-musl.tar.gz"
    )
    python = await cached_download(
        "https://github.com/astral-sh/python-build-standalone/releases/download/20260310/"
        f"cpython-3.13.12%2B20260310-{arch}-unknown-linux-{libc}-install_only_stripped.tar.gz",
        sha256=PYTHON_SHA256[arch, libc],
    )
    return uv, python
