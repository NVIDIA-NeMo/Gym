# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-side cache of pinned bootstrap assets, shared by all task sandboxes.

``prepare`` (``python -m responses_api_agents.miniswe_sandboxed_agent.bootstrap prepare --dest DIR``) fills a
directory once, on a host with network, with the same uv and Python archives and, per sandbox platform, a wheelhouse
for mini-SWE and the MCP client. An agent configured with ``bootstrap_dir: DIR`` then sets up every sandbox from it,
so sandboxes with a deny-all network policy can run mini-SWE.
"""

import argparse
import asyncio
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
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


async def cached_download(url: str, *, sha256: str | None = None, cache_dir: Path | None = None) -> Path:
    """Download once per cache, serializing concurrent first-task requests."""
    async with _DOWNLOAD_LOCK:
        cache_dir = cache_dir or CACHE_DIR
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / url.rsplit("/", 1)[1]
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


def _uv_url(arch: str) -> str:
    return f"https://github.com/astral-sh/uv/releases/download/0.10.12/uv-{arch}-unknown-linux-musl.tar.gz"


def _python_url(arch: str, libc: str) -> str:
    return (
        "https://github.com/astral-sh/python-build-standalone/releases/download/20260310/"
        f"cpython-3.13.12%2B20260310-{arch}-unknown-linux-{libc}-install_only_stripped.tar.gz"
    )


async def bootstrap_assets(arch: str, libc: str) -> tuple[Path, Path]:
    """Select Linux binaries using the sandbox architecture and libc."""
    if (arch, libc) not in PYTHON_SHA256:
        raise RuntimeError(f"Unsupported mini-SWE sandbox platform: {arch}/{libc}")
    uv = await cached_download(_uv_url(arch))
    python = await cached_download(_python_url(arch, libc), sha256=PYTHON_SHA256[arch, libc])
    return uv, python


# What the sandbox installs; the wheelhouse holds these and all their dependencies.
OFFLINE_REQUIREMENTS = ("mini-swe-agent==2.4.6", "mcp==1.29.0", "httpx-aiohttp==0.2.0")
# Wheel platform tags per sandbox platform, newest first: a sandbox's uv picks the best one its libc supports.
WHEEL_PLATFORMS = {
    ("x86_64", "gnu"): ("manylinux_2_28_x86_64", "manylinux2014_x86_64"),
    ("x86_64", "musl"): ("musllinux_1_2_x86_64", "musllinux_1_1_x86_64"),
    ("aarch64", "gnu"): ("manylinux_2_28_aarch64", "manylinux2014_aarch64"),
    ("aarch64", "musl"): ("musllinux_1_2_aarch64", "musllinux_1_1_aarch64"),
}


def _wheels_name(arch: str, libc: str) -> str:
    return f"wheels-{arch}-{libc}.tar.gz"


def offline_assets(bootstrap_dir: Path, arch: str, libc: str) -> tuple[Path, Path, Path]:
    """uv, Python and the wheelhouse for one sandbox platform from a prepared directory (see ``prepare``)."""
    if (arch, libc) not in PYTHON_SHA256:
        raise RuntimeError(f"Unsupported mini-SWE sandbox platform: {arch}/{libc}")
    bootstrap_dir = Path(bootstrap_dir)
    uv = bootstrap_dir / _uv_url(arch).rsplit("/", 1)[1]
    python = bootstrap_dir / _python_url(arch, libc).rsplit("/", 1)[1]
    wheels = bootstrap_dir / _wheels_name(arch, libc)
    for path in (uv, python, wheels):
        if not path.is_file():
            raise RuntimeError(f"Offline mini-SWE bootstrap is missing {path}; run bootstrap.py prepare")
    if hashlib.sha256(python.read_bytes()).hexdigest() != PYTHON_SHA256[arch, libc]:
        raise RuntimeError(f"Bootstrap checksum mismatch: {python}")
    return uv, python, wheels


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fetch(url: str, directory: Path, *, sha256: str | None = None) -> Path:
    """Download `url` into `directory` once (a plain download: the CLI runs outside a Gym server)."""
    path = directory / url.rsplit("/", 1)[1]
    if path.is_file() and (sha256 is None or _sha256(path) == sha256):
        return path
    partial = path.with_name(path.name + ".partial")
    with urllib.request.urlopen(url, timeout=300) as response, partial.open("wb") as stream:
        shutil.copyfileobj(response, stream)
    if sha256 and _sha256(partial) != sha256:
        partial.unlink()
        raise RuntimeError(f"Bootstrap checksum mismatch: {url}")
    partial.replace(path)
    return path


def prepare(dest: Path, *, platforms: list[tuple[str, str]], pip: list[str]) -> dict:
    """Fill ``dest`` with uv, Python and a wheelhouse per platform; write and return ``manifest.json``."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "requirements": list(OFFLINE_REQUIREMENTS),
        "python": "3.13.12",
        "uv": "0.10.12",
        "platforms": {},
    }
    for arch, libc in platforms:
        _fetch(_uv_url(arch), dest)
        _fetch(_python_url(arch, libc), dest, sha256=PYTHON_SHA256[arch, libc])
        with tempfile.TemporaryDirectory(prefix="miniswe-wheels-") as scratch:
            wheel_dir = Path(scratch) / "wheels"
            command = [*pip, "download", "--dest", str(wheel_dir), "--only-binary=:all:", "--python-version", "3.13"]
            command += ["--implementation", "cp", "--abi", "cp313", "--abi", "abi3", "--abi", "none"]
            for tag in WHEEL_PLATFORMS[arch, libc]:
                command += ["--platform", tag]
            subprocess.run([*command, *OFFLINE_REQUIREMENTS], check=True)
            archive = dest / _wheels_name(arch, libc)
            partial = archive.with_name(archive.name + ".partial")
            names = sorted(path.name for path in wheel_dir.iterdir())
            with tarfile.open(partial, "w:gz") as tar:
                for name in names:
                    tar.add(wheel_dir / name, arcname=name)
            partial.replace(archive)
            manifest["platforms"][f"{arch}-{libc}"] = {
                "uv": _uv_url(arch).rsplit("/", 1)[1],
                "python": _python_url(arch, libc).rsplit("/", 1)[1],
                "wheels": archive.name,
                "wheels_sha256": _sha256(archive),
                "wheel_files": names,
            }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the offline mini-SWE sandbox bootstrap directory")
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("prepare")
    command.add_argument("--dest", type=Path, required=True)
    command.add_argument("--platform", action="append", default=None, help="arch-libc, e.g. x86_64-gnu (repeatable)")
    command.add_argument("--pip", default="python3 -m pip", help="pip command used to download wheels")
    args = parser.parse_args()
    platforms = [tuple(p.split("-", 1)) for p in (args.platform or ["x86_64-gnu", "x86_64-musl"])]
    print(json.dumps(prepare(args.dest, platforms=platforms, pip=args.pip.split()), indent=2))


if __name__ == "__main__":
    main()
