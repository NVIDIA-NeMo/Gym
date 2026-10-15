# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Internal startup/cache-transfer helper for sbatch_external_vllm.sh.

The launcher passes this source to Python on each node, so collecting the cache
does not require a shared cache mount or an extra mount for this helper.
"""

import argparse
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


def install_warmup(cache_root: Path, aot_root: Path, startup_root: Path, source: BinaryIO) -> list[Path]:
    """Merge concatenated node archives into one image, including both PD roles."""
    caches = []
    libraries: dict[str, Path] = {}
    plans: dict[str, dict] = {}
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
    else:
        from flashinfer.jit import env as jit_env

        libraries = install_warmup(
            CACHE_ROOT, jit_env.FLASHINFER_AOT_DIR, VLLM_CACHE_ROOT / "startup_plan", sys.stdin.buffer
        )
        verify_libraries(libraries)
        # All collected libraries now live in the package. Leave no disposable
        # compilation workspace (or object files) in the exported image.
        shutil.rmtree(JIT_WORKSPACE, ignore_errors=True)


if __name__ == "__main__":
    main()
