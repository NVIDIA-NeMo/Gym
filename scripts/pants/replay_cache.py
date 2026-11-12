# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bound only the pilot branch's experiment caches; --dry-run never deletes."""

import argparse
import json
import re
import subprocess
from pathlib import Path


REF = "refs/heads/glovisotto/pants-pilot"
API = "repos/NVIDIA-NeMo/Gym/actions/caches"
LIMIT = 5 * 1024**3
ENGINE = "pants-full-engine-"
RESULTS = "pants-full-results-"


def scoped(cache):
    return cache["ref"] == REF and cache["key"].startswith("pants-full-")


def plan_removals(
    caches,
    *,
    contract=None,
    engine_prefix="pants-full-engine-v4-",
    results_prefix="pants-full-results-v1-",
    enforce_limit=False,
    namespace=None,
):
    remaining = [c for c in caches if scoped(c)]
    removed = []

    def remove(cache, reason):
        remaining.remove(cache)
        removed.append({"reason": reason, "cache": cache})

    prefixes = (engine_prefix, results_prefix, "pants-full-v2-uv-", "pants-full-launcher-v1-")
    if namespace:
        for cache in remaining[:]:
            prefix = next((value for value in prefixes if cache["key"].startswith(value)), None)
            if prefix and not cache["key"].startswith(prefix + namespace + "-"):
                remove(cache, "obsolete experiment cache namespace")

    # Keep only the latest dependency generation for each arm/platform/version.
    uv_groups = {}
    for cache in remaining:
        if cache["key"].startswith("pants-full-v2-uv-"):
            stem, _, digest = cache["key"].rpartition("-")
            if len(digest) == 64 and all(c in "0123456789abcdef" for c in digest):
                uv_groups.setdefault((stem, cache["version"]), []).append(cache)
    for entries in uv_groups.values():
        for cache in sorted(entries, key=lambda c: (c["created_at"], c["id"]), reverse=True)[1:]:
            remove(cache, "older dependency generation for arm/platform/version")

    for cache in remaining[:]:
        key = cache["key"]
        if key.startswith((ENGINE, RESULTS)) and contract:
            prefix = results_prefix if key.startswith(RESULTS) else engine_prefix
            if not key.startswith(prefix) or "-" + contract + "-" not in key:
                remove(cache, "obsolete archive layout or implementation contract")

    # The full stem includes result shard, dependency digest, implementation
    # contract, OS/arch, and job. Keep distinct action cache path versions too.
    groups = {}
    for cache in remaining:
        if cache["key"].startswith((engine_prefix, results_prefix)):
            stem, separator, sha = cache["key"].rpartition("-")
            if separator and len(sha) == 40 and all(c in "0123456789abcdef" for c in sha):
                groups.setdefault((stem, cache["version"]), []).append(cache)
    for entries in groups.values():
        for cache in sorted(entries, key=lambda c: (c["created_at"], c["id"]), reverse=True)[1:]:
            remove(cache, "older source archive for job/shard/dependency key/version")

    if enforce_limit:
        total = sum(c["size_in_bytes"] for c in remaining)
        # Optional wrapper dependency archives go first. Actual native results
        # are retained preferentially; uv and shared launcher remain untouched.
        candidates = [c for c in remaining if c["key"].startswith((ENGINE, RESULTS))]
        for cache in sorted(candidates, key=lambda c: (c["key"].startswith(RESULTS), c["last_accessed_at"], c["id"])):
            if total <= LIMIT:
                break
            remove(cache, "5 GiB bound: optional engine archive before reusable test results")
            total -= cache["size_in_bytes"]
    return removed, remaining


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract")
    parser.add_argument("--namespace")
    parser.add_argument("--engine-prefix", default="pants-full-engine-v4-")
    parser.add_argument("--results-prefix", default="pants-full-results-v1-")
    parser.add_argument("--enforce-limit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--snapshot", type=Path, help="Offline snapshot; requires --dry-run")
    args = parser.parse_args()
    if args.namespace and not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", args.namespace):
        parser.error("--namespace must match the workflow cache namespace format")
    if args.contract and not re.fullmatch(r"[a-f0-9]{64}", args.contract):
        parser.error("--contract must be a full SHA256 digest")
    if args.snapshot and not args.dry_run:
        parser.error("--snapshot requires --dry-run")
    args.output.mkdir(parents=True, exist_ok=True)

    def snapshot():
        data = json.loads(
            subprocess.check_output(["gh", "api", API + "?ref=refs%2Fheads%2Fglovisotto%2Fpants-pilot&per_page=100"])
        )
        if data["total_count"] > 100:
            raise RuntimeError("Ref has more than 100 caches; refusing partial inventory")
        return data

    before = json.loads(args.snapshot.read_text()) if args.snapshot else snapshot()
    (args.output / "before.json").write_text(json.dumps(before, indent=2))
    planned, remaining = plan_removals(
        before["actions_caches"],
        contract=args.contract,
        engine_prefix=args.engine_prefix,
        results_prefix=args.results_prefix,
        enforce_limit=args.enforce_limit,
        namespace=args.namespace,
    )
    (args.output / "planned.json").write_text(json.dumps(planned, indent=2))
    removed = []
    for entry in planned:
        if args.dry_run:
            continue
        subprocess.run(["gh", "api", "--method", "DELETE", API + "/" + str(entry["cache"]["id"])], check=True)
        removed.append(entry)
        (args.output / "removed.json").write_text(json.dumps(removed, indent=2))
    after = before if args.dry_run else snapshot()
    (args.output / "after.json").write_text(json.dumps(after, indent=2))
    (args.output / "removed.json").write_text(json.dumps(removed, indent=2))
    actual_total = sum(c["size_in_bytes"] for c in after["actions_caches"] if scoped(c))
    projected_total = sum(c["size_in_bytes"] for c in remaining)
    print(
        json.dumps(
            {
                "dry_run": args.dry_run,
                "removed": len(removed),
                "planned": len(planned),
                "scoped_GiB": actual_total / 2**30,
                "projected_GiB": projected_total / 2**30,
                "within_limit": (projected_total if args.dry_run else actual_total) <= LIMIT,
            }
        )
    )
    if args.enforce_limit and (projected_total if args.dry_run else actual_total) > LIMIT:
        raise RuntimeError("Protected uv/launcher caches exceed 5 GiB; no broad deletion allowed")


if __name__ == "__main__":
    main()
