# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Internal startup/cache-transfer helper for sbatch_external_vllm.sh.

The launcher passes this source to Python on each node, so collecting the cache
does not require a shared cache mount or an extra mount for this helper.
"""

import argparse
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import BinaryIO


CACHE_ROOT = Path("/opt/super-vl-evals/flashinfer-autotune-cache")
JIT_WORKSPACE = Path("/opt/super-vl-evals/flashinfer-warmup-workspace")
VLLM_CACHE_ROOT = Path("/opt/super-vl-evals/vllm-cache")


def wait_for_engines(urls: list[str], timeout: int) -> None:
    """Wait for every API endpoint, including replicas on the same node."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + timeout
    while True:
        pending = []
        for url in urls:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"vLLM startup timed out after {timeout}s")
            try:
                with opener.open(f"{url}/health", timeout=min(3, remaining)) as response:
                    if response.status != 200:
                        pending.append(url)
            except (urllib.error.URLError, TimeoutError, OSError):
                pending.append(url)
        if not pending:
            print(f"All {len(urls)} vLLM endpoints are ready", flush=True)
            return
        time.sleep(min(5, max(0, deadline - time.monotonic())))


def validate_cache(config_hash: str, contents: str) -> None:
    """Check the vLLM directory key and reject empty or corrupt cache files."""
    if not re.fullmatch(r"[0-9a-f]{64}", config_hash):
        raise ValueError(f"Invalid vLLM cache configuration hash: {config_hash!r}")
    data = json.loads(contents)
    if not isinstance(data, dict) or not any(not key.startswith("_") for key in data):
        raise ValueError(f"No autotuning entries in cache {config_hash}")


def collect_caches(root: Path, *, allow_empty: bool) -> dict[str, str]:
    """Collect complete cache files after startup, preserving FlashInfer metadata."""
    started_at = (root / ".warmup-start").stat().st_mtime_ns
    caches = {}
    updated = False
    for path in sorted(root.glob("*/autotune_configs.json")):
        contents = path.read_text()
        validate_cache(path.parent.name, contents)
        caches[path.parent.name] = contents
        updated |= path.stat().st_mtime_ns >= started_at
    # Re-warming an exported image may reuse every tactic without rewriting
    # its JSON while still compiling missing kernels or creating startup plans.
    if not allow_empty and not updated and not (caches and (root / ".baked").is_file()):
        raise ValueError(
            "No autotuning caches were written during startup; check that FlashInfer autotuning is enabled"
        )
    return caches


def install_caches(root: Path, payloads: list[str]) -> None:
    """Install the union of configuration directories from all warmed nodes."""
    caches: dict[str, str] = {}
    for payload in payloads:
        node_caches = json.loads(payload)
        if not isinstance(node_caches, dict):
            raise ValueError("Expected a cache mapping from each node")
        for config_hash, contents in node_caches.items():
            if not isinstance(contents, str):
                raise ValueError(f"Invalid cache contents for {config_hash!r}")
            validate_cache(config_hash, contents)
            # Replicas of the same configuration may pick different tactics.
            # Keep one complete file, including its metadata and generation hash,
            # rather than hand-merging FlashInfer's internal serialization.
            if config_hash in caches:
                existing_keys = set(json.loads(caches[config_hash]))
                incoming_keys = set(json.loads(contents))
                if existing_keys < incoming_keys:
                    caches[config_hash] = contents
                elif not incoming_keys <= existing_keys:
                    raise ValueError(f"Replicas produced incompatible cache entry sets for {config_hash}")
            else:
                caches[config_hash] = contents
    if not caches:
        raise ValueError("No autotuning caches collected; refusing to export an unwarmed image")
    for config_hash, contents in caches.items():
        directory = root / config_hash
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "autotune_configs.json").write_text(contents)
    (root / ".warmup-start").unlink(missing_ok=True)
    (root / ".baked").write_text("Loaded automatically by sbatch_external_vllm.sh\n")
    print(f"Installed {len(caches)} autotuning configuration caches in {root}", flush=True)


def dump_warmup(
    cache_root: Path, jit_workspace: Path, startup_root: Path, output: BinaryIO, *, allow_empty: bool
) -> None:
    """Stream tuning results, compiled kernels, and startup plans from one node."""
    caches = json.dumps(collect_caches(cache_root, allow_empty=allow_empty)).encode()
    with tarfile.open(fileobj=output, mode="w|") as archive:
        entry = tarfile.TarInfo("autotune.json")
        entry.size = len(caches)
        archive.addfile(entry, io.BytesIO(caches))
        # Each worker's workspace includes the FlashInfer version and CUDA arch;
        # custom images also namespace it by the installed source-build hash.
        # Keep loadable libraries, not generated sources, objects, or Ninja state.
        for library in sorted(jit_workspace.glob("**/cached_ops/*/*.so")):
            if library.name != f"{library.parent.name}.so" or library.is_symlink():
                raise ValueError(f"Unexpected FlashInfer library: {library}")
            archive.add(library, arcname=f"jit/{library.name}", recursive=False)
        for plan in sorted(startup_root.glob("startup_plan_*.json")):
            archive.add(plan, arcname=f"startup/{plan.name}", recursive=False)
        compile_root = startup_root.parent / "torch_compile_cache"
        for path in sorted(compile_root.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Unexpected symlink in Torch compilation cache: {path}")
            if path.is_file():
                archive.add(path, arcname=f"compile/{path.relative_to(compile_root)}", recursive=False)


def compile_cache_group(name: str) -> str:
    """Validate a relative cache path and identify its complete per-rank cache."""
    if any(part in ("", ".", "..") for part in name.split("/")):
        raise ValueError(f"Invalid Torch compilation cache path: {name}")
    match = re.fullmatch(r"((?:[0-9a-f]{10}|torch_aot_compile/[0-9a-f]{64})/rank_[0-9]+_[0-9]+)/.+", name)
    if match is None:
        raise ValueError(f"Invalid Torch compilation cache path: {name}")
    return match[1]


def install_warmup(cache_root: Path, aot_root: Path, startup_root: Path, source: BinaryIO) -> list[Path]:
    """Merge concatenated node archives into one image, including both PD roles."""
    caches = []
    libraries: dict[str, Path] = {}
    plans: dict[str, dict] = {}
    compile_groups: dict[str, int] = {}
    compile_root = startup_root.parent / "torch_compile_cache"
    # Each dump is a standalone tar stream. ignore_zeros reads past its end
    # markers into the next node's stream without buffering large libraries.
    with tarfile.open(fileobj=source, mode="r|", ignore_zeros=True) as archive:
        for entry in archive:
            if not entry.isfile():
                raise ValueError(f"Unexpected warmup archive entry: {entry.name}")
            contents = archive.extractfile(entry)
            assert contents is not None
            with contents:
                if entry.name == "autotune.json":
                    caches.append(contents.read().decode())
                elif entry.name.startswith("compile/"):
                    relative = entry.name.removeprefix("compile/")
                    group = compile_cache_group(relative)
                    # Keep a whole rank's cache from one replica. Mixing its
                    # metadata with another replica's binaries is unsafe.
                    if group not in compile_groups:
                        compile_groups[group] = len(caches)
                        shutil.rmtree(compile_root / group, ignore_errors=True)
                    if compile_groups[group] != len(caches):
                        continue
                    destination = compile_root / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with destination.open("wb") as output:
                        shutil.copyfileobj(contents, output)
                elif match := re.fullmatch(r"jit/([A-Za-z0-9_][A-Za-z0-9_.-]*)\.so", entry.name):
                    name = match[1]
                    if entry.size == 0:
                        raise ValueError(f"Empty FlashInfer library: {name}")
                    if name in libraries:
                        continue
                    # Package-local AOT modules are resolved before the runtime
                    # JIT workspace, even when HOME or workspace paths change.
                    destination = aot_root / name / f"{name}.so"
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with destination.open("wb") as output:
                        shutil.copyfileobj(contents, output)
                    libraries[name] = destination
                elif match := re.fullmatch(r"startup/startup_plan_([0-9a-f]{16})\.json", entry.name):
                    plan = json.load(contents)
                    if (
                        not isinstance(plan, dict)
                        or plan.get("fingerprint") != match[1]
                        or plan.get("schema") != 1
                        or any(
                            type(plan.get(key)) is not int or plan[key] <= 0
                            for key in ("kv_cache_memory_bytes", "free_memory_baseline")
                        )
                    ):
                        raise ValueError(f"Invalid startup plan: {entry.name}")
                    # Keep one complete plan per fingerprint. Prefer the smaller
                    # allocation if replicas measured different memory usage;
                    # vLLM still checks its recorded free-memory baseline.
                    previous = plans.get(match[1])
                    if previous is None or plan["kv_cache_memory_bytes"] < previous["kv_cache_memory_bytes"]:
                        plans[match[1]] = plan
                else:
                    raise ValueError(f"Unexpected warmup archive entry: {entry.name}")
    for fingerprint, plan in plans.items():
        startup_root.mkdir(parents=True, exist_ok=True)
        (startup_root / f"startup_plan_{fingerprint}.json").write_text(json.dumps(plan))
    install_caches(cache_root, caches)
    if plans:
        (cache_root / ".startup-plans-baked").write_text("Loaded automatically by sbatch_external_vllm.sh\n")
    print(f"Installed {len(libraries)} FlashInfer JIT modules and {len(plans)} startup plans", flush=True)
    print(f"Installed {len(compile_groups)} Torch compilation caches (configuration/rank pairs)", flush=True)
    return list(libraries.values())


def verify_libraries(libraries: list[Path]) -> None:
    """Require baked libraries to resolve through AOT and load without compiling."""
    os.environ["FLASHINFER_DISABLE_JIT"] = "1"
    import tvm_ffi
    from flashinfer.jit import env as jit_env

    for library in libraries:
        if jit_env.get_aot_path(library.stem) != library:
            raise ValueError(f"FlashInfer does not resolve baked module: {library}")
        tvm_ffi.load_module(str(library))
        print(f"Verified baked FlashInfer module: {library.stem}", flush=True)


def write_manifest(cache_root: Path, aot_root: Path, startup_root: Path, libraries: list[Path]) -> None:
    """Record what was baked so later jobs can identify and check the actual image."""
    path = cache_root / "warmed-manifest.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    names = set(previous.get("jit_modules", {})) | {library.stem for library in libraries}
    manifest = {
        "schema": 1,
        "build_id": uuid.uuid4().hex,
        "autotune": {
            cache.parent.name: hashlib.sha256(cache.read_bytes()).hexdigest()
            for cache in sorted(cache_root.glob("*/autotune_configs.json"))
        },
        "startup_plans": {
            plan.name: hashlib.sha256(plan.read_bytes()).hexdigest()
            for plan in sorted(startup_root.glob("startup_plan_*.json"))
        },
        "jit_modules": {name: (aot_root / name / f"{name}.so").stat().st_size for name in sorted(names)},
        "torch_compile_files": {
            str(path.relative_to(startup_root.parent / "torch_compile_cache")): path.stat().st_size
            for path in sorted((startup_root.parent / "torch_compile_cache").rglob("*"))
            if path.is_file()
        },
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def inspect_manifest(cache_root: Path, aot_root: Path, startup_root: Path) -> list[Path]:
    """Check baked artifacts before startup, without loading CUDA or compiling."""
    manifest = json.loads((cache_root / "warmed-manifest.json").read_text())
    if manifest.get("schema") != 1 or not manifest.get("autotune"):
        raise ValueError("Invalid warmed-container manifest; rebuild with SAVE_WARMED_CONTAINER")
    for name, digest in manifest["autotune"].items():
        if not re.fullmatch(r"[0-9a-f]{64}", name):
            raise ValueError(f"Invalid baked autotune hash: {name}")
        path = cache_root / name / "autotune_configs.json"
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Baked autotune cache missing or changed: {path}")
    for name, digest in manifest["startup_plans"].items():
        if not re.fullmatch(r"startup_plan_[0-9a-f]{16}\.json", name):
            raise ValueError(f"Invalid baked startup-plan name: {name}")
        path = startup_root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Baked startup plan missing or changed: {path}")
    libraries = []
    for name, size in manifest["jit_modules"].items():
        if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", name):
            raise ValueError(f"Invalid baked module name: {name}")
        path = aot_root / name / f"{name}.so"
        if size <= 0 or not path.is_file() or path.stat().st_size != size:
            raise ValueError(f"Baked FlashInfer module missing or changed: {path}")
        libraries.append(path)
    compile_files = manifest.get("torch_compile_files", {})
    for name, size in compile_files.items():
        compile_cache_group(name)
        path = startup_root.parent / "torch_compile_cache" / name
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError(f"Baked Torch compilation cache missing or changed: {path}")
    print(
        f"Warmed image build={manifest['build_id']}: {len(manifest['autotune'])} autotune caches, "
        f"{len(manifest['startup_plans'])} startup plans, {len(libraries)} baked JIT modules, "
        f"{len(compile_files)} Torch compilation files",
        flush=True,
    )
    print(f"Baked autotune hashes: {', '.join(sorted(manifest['autotune']))}", flush=True)
    print(f"Baked startup plans: {', '.join(sorted(manifest['startup_plans']))}", flush=True)
    print(f"Baked JIT modules: {', '.join(sorted(manifest['jit_modules']))}", flush=True)
    return libraries


def main() -> None:
    """Run the requested readiness or cache-transfer operation."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    wait = commands.add_parser("wait")
    wait.add_argument("--timeout", type=int, required=True)
    wait.add_argument("urls", nargs="+")
    dump = commands.add_parser("dump")
    dump.add_argument("--allow-empty", action="store_true")
    commands.add_parser("install")
    commands.add_parser("inspect")
    commands.add_parser("verify")
    args = parser.parse_args()
    if args.command == "wait":
        wait_for_engines(args.urls, args.timeout)
    elif args.command == "dump":
        dump_warmup(
            CACHE_ROOT,
            JIT_WORKSPACE,
            VLLM_CACHE_ROOT / "startup_plan",
            sys.stdout.buffer,
            allow_empty=args.allow_empty,
        )
    elif args.command == "install":
        from flashinfer.jit import env as jit_env

        libraries = install_warmup(
            CACHE_ROOT, jit_env.FLASHINFER_AOT_DIR, VLLM_CACHE_ROOT / "startup_plan", sys.stdin.buffer
        )
        verify_libraries(libraries)
        write_manifest(CACHE_ROOT, jit_env.FLASHINFER_AOT_DIR, VLLM_CACHE_ROOT / "startup_plan", libraries)
        # All collected libraries now live in the package. Leave no disposable
        # compilation workspace (or object files) in the exported image.
        shutil.rmtree(JIT_WORKSPACE, ignore_errors=True)
    else:
        # Locating the package does not import FlashInfer/PyTorch or initialize
        # CUDA. Normal serving jobs only perform cheap file checks here.
        spec = importlib.util.find_spec("flashinfer")
        if spec is None or spec.origin is None:
            raise RuntimeError("Warmed image is missing the FlashInfer Python package")
        aot_root = Path(spec.origin).parent / "data" / "aot"
        libraries = inspect_manifest(CACHE_ROOT, aot_root, VLLM_CACHE_ROOT / "startup_plan")
        if args.command == "verify":
            verify_libraries(libraries)


if __name__ == "__main__":
    main()
