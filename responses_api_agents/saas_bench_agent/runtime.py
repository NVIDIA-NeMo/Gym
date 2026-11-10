# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bundled benchmark locations and explicit writable runtime paths."""

import os
import socket
from pathlib import Path


BENCHMARK_ROOT = Path(__file__).resolve().parent / "benchmark"
DEFAULT_BUNDLE_ROOT = Path(__file__).resolve().parent / "data/runtime"
HARNESS_VERSIONS = {"opencode": "2.1.1", "browser_use": "browser-use-0.13.7"}


def configure_cache(root: str) -> Path:
    cache = Path(root).resolve()
    if any(cache == Path(p) or cache.is_relative_to(p) for p in ("/root", "/home/pritamb")):
        raise ValueError("SaaS cache must be outside forbidden home directories")
    cache = cache / socket.gethostname().split(".")[0]
    directories = {
        "HOME": "home",
        "XDG_CACHE_HOME": "cache",
        "XDG_CONFIG_HOME": "config",
        "XDG_DATA_HOME": "data",
        "XDG_STATE_HOME": "state",
        "HF_HOME": "huggingface",
        "UV_CACHE_DIR": "uv",
        "PIP_CACHE_DIR": "pip",
        "NPM_CONFIG_CACHE": "npm",
        "NPM_CONFIG_PREFIX": "npm-prefix",
        "BUN_INSTALL_CACHE_DIR": "bun",
        "ENROOT_CACHE_PATH": "enroot-cache",
        "ENROOT_CONFIG_PATH": "enroot-config",
        "ENROOT_DATA_PATH": "enroot-data",
        "ENROOT_RUNTIME_PATH": "enroot-runtime",
        "ENROOT_TEMP_PATH": "enroot-tmp",
    }
    for variable, suffix in directories.items():
        path = cache / suffix
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
    helper = os.environ.get("SAAS_ENROOT_RUNTIME_DIR")
    paths = [str(cache / "npm-prefix/bin")]
    if helper:
        paths.extend(str(Path(helper) / directory) for directory in ("usr/bin", "usr/sbin"))
        libraries = str(Path(helper) / "usr/lib/x86_64-linux-gnu")
        existing_libraries = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = libraries + (os.pathsep + existing_libraries if existing_libraries else "")
    os.environ["PATH"] = os.pathsep.join(paths) + os.pathsep + os.environ.get("PATH", "")
    # Native refresh sockets must fit Linux's 108-byte Unix socket limit.
    # Browser profiles and sockets are transient; persistent caches stay above.
    os.environ["TMPDIR"] = "/tmp"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["SAAS_UID_SHIM"] = str(cache / "saas-uid-shim.so")
    return cache


def materialize_bundle(data_dir: str, destination: str | None = None) -> Path:
    """Join shipped verifier/runtime code with a hash-verified external data release."""
    import fcntl
    import hashlib
    import json
    import shutil
    import tempfile

    data = Path(data_dir).resolve(strict=True)
    output = Path(destination) if destination else DEFAULT_BUNDLE_ROOT
    if not output.is_absolute():
        output = Path(__file__).resolve().parents[2] / output
    output = output.resolve()
    expected = json.loads((BENCHMARK_ROOT / "task-data-manifest.json").read_text())
    for relative, digest in expected["files"].items():
        source = (data / relative).resolve(strict=True)
        if not source.is_relative_to(data) or hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise ValueError("Benchmark data differs from pinned release: " + relative)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if output.exists():
            # A prepared bundle is immutable; validate its code and data before reuse.
            for relative, digest in expected["files"].items():
                if hashlib.sha256((output / relative).read_bytes()).hexdigest() != digest:
                    raise ValueError("Existing bundle data differs; select a fresh run directory")
            provenance = json.loads((BENCHMARK_ROOT / "provenance.json").read_text())
            for item in provenance["files"]:
                if (
                    hashlib.sha256((output / item["source_path"]).read_bytes()).hexdigest()
                    != hashlib.sha256((BENCHMARK_ROOT / item["source_path"]).read_bytes()).hexdigest()
                ):
                    raise ValueError("Existing bundle code differs; select a fresh run directory")
            return output
        temporary = Path(tempfile.mkdtemp(prefix=".saas-bundle-", dir=output.parent))
        try:
            shutil.copytree(BENCHMARK_ROOT, temporary, dirs_exist_ok=True)
            for relative in expected["files"]:
                target = temporary / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(data / relative, target)
            temporary.rename(output)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return output
