# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bundled benchmark locations and explicit writable runtime paths."""

import atexit
import errno
import hashlib
import json
import os
import shutil
import socket
import tempfile
from pathlib import Path


BENCHMARK_ROOT = Path(__file__).resolve().parent / "benchmark"
DEFAULT_BUNDLE_ROOT = Path(__file__).resolve().parent / "data/runtime"
_SYSTEM_ENROOT_CONFIG = Path("/etc/enroot")
_GPU_ENROOT_HOOKS = {"hooks.d/50-mig-config.sh", "hooks.d/98-nvidia.sh"}
HARNESS_VERSIONS = {"opencode": "2.1.1", "browser_use": "browser-use-0.13.7"}


def _cpu_enroot_config(cache: Path) -> Path:
    files = {
        str(path.relative_to(_SYSTEM_ENROOT_CONFIG)): path.read_bytes()
        for path in sorted(_SYSTEM_ENROOT_CONFIG.rglob("*"))
        if path.is_file() and str(path.relative_to(_SYSTEM_ENROOT_CONFIG)) not in _GPU_ENROOT_HOOKS
    }
    manifest = {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:24]
    base = cache / "enroot-system"
    base.mkdir(exist_ok=True)
    target = base / digest
    if not target.exists():
        temporary = Path(tempfile.mkdtemp(prefix=".config-", dir=base))
        try:
            for name, content in files.items():
                path = temporary / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
                path.chmod((_SYSTEM_ENROOT_CONFIG / name).stat().st_mode & 0o777)
            try:
                temporary.rename(target)
            except OSError as exc:
                if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                    raise
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    for name, content in files.items():
        if (target / name).read_bytes() != content:
            raise ValueError("Cached CPU Enroot configuration differs: " + name)
    return target


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
    # Extracted app filesystems are platform runtime state, not library caches.
    # Shared NFS storage cannot provide the namespace-local ownership semantics.
    platform_root = os.environ.get("SAAS_ENROOT_PLATFORM_ROOT")
    if platform_root:
        platform = Path(platform_root).resolve()
        if platform != Path("/raid/enroot"):
            raise ValueError("SVG Enroot platform root must be /raid/enroot")
        identity = "saas-" + os.environ.get("SLURM_JOB_ID", "interactive") + "-" + str(os.getpid())
        host_uid = os.environ.get("SLURM_JOB_UID", str(os.getuid()))
        if not host_uid.isdecimal():
            raise ValueError("SLURM_JOB_UID must identify the platform directory owner")
        for variable, directory in (("ENROOT_DATA_PATH", "data"), ("ENROOT_RUNTIME_PATH", "runtime")):
            location = platform / directory / ("user-" + host_uid) / identity
            location.mkdir(parents=True, exist_ok=True)
            os.environ[variable] = str(location)
    os.environ["ENROOT_SYSCONF_PATH"] = str(_cpu_enroot_config(cache))
    helper = os.environ.get("SAAS_ENROOT_RUNTIME_DIR")
    paths = [str(cache / "npm-prefix/bin")]
    if helper:
        paths.extend(str(Path(helper) / directory) for directory in ("usr/bin", "usr/sbin"))
        libraries = str(Path(helper) / "usr/lib/x86_64-linux-gnu")
        existing_libraries = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = libraries + (os.pathsep + existing_libraries if existing_libraries else "")
        shell = Path(helper) / "usr/bin/busybox"
        if shell.is_file() and os.access(shell, os.X_OK):
            os.environ["SAAS_STATIC_SHELL"] = str(shell)
    os.environ["PATH"] = os.pathsep.join(paths) + os.pathsep + os.environ.get("PATH", "")
    if os.environ.get("SAAS_NRT_ENROOT", "").lower() in {"1", "true", "yes"}:
        required = [name for name in ("enroot", "nsenter", "unshare", "socat", "gcc") if not shutil.which(name)]
        for name in ("usr/bin/busybox", "usr/bin/slirp4netns"):
            candidate = Path(helper or "") / name
            if not helper or not candidate.is_file() or not os.access(candidate, os.X_OK):
                required.append(str(candidate))
        if required:
            raise RuntimeError("Missing SaaS Enroot dependencies before episode admission: " + ", ".join(required))
    # Native refresh sockets need short names; their storage stays workspace-backed.
    temporary = cache / "tmp"
    temporary.mkdir(exist_ok=True)
    if len(str(temporary).encode()) > 60:
        digest = hashlib.sha256(str(temporary).encode()).hexdigest()[:8]
        alias = Path(f"/tmp/saas-{os.getuid()}-{os.getpid()}-{digest}")
        if alias.is_symlink():
            if alias.resolve() != temporary:
                raise ValueError("Temporary alias points outside this workspace cache")
        else:
            alias.symlink_to(temporary, target_is_directory=True)

        def remove_alias() -> None:
            if alias.is_symlink() and os.readlink(alias) == str(temporary):
                alias.unlink()

        atexit.register(remove_alias)
        os.environ["TMPDIR"] = str(alias)
    else:
        os.environ["TMPDIR"] = str(temporary)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["SAAS_UID_SHIM"] = str(cache / "saas-uid-shim.so")
    os.environ["SAAS_VERIFIER_LAUNCHER"] = str(Path(__file__).with_name("verifier_transport.py"))
    return cache


def materialize_bundle(data_dir: str, destination: str | None = None) -> Path:
    """Join shipped verifier/runtime code with a hash-verified external data release."""
    import fcntl

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
