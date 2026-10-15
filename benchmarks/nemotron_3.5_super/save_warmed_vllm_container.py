# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Internal startup/cache-transfer helper for sbatch_external_vllm.sh.

The launcher passes this source to Python on each node, so collecting the cache
does not require a shared cache mount or an extra mount for this helper.
"""

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


CACHE_ROOT = Path("/opt/super-vl-evals/flashinfer-autotune-cache")


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
        print(f"Waiting for {len(pending)}/{len(urls)} endpoints: {', '.join(pending)}", flush=True)
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
    if not allow_empty and not updated:
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
        print(json.dumps(collect_caches(CACHE_ROOT, allow_empty=args.allow_empty)))
    else:
        install_caches(CACHE_ROOT, list(sys.stdin))


if __name__ == "__main__":
    main()
