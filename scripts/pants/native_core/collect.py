# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Record the complete pytest collection before selecting native Pants targets."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


# Loaded as a pytest plugin only in the baseline's exact environment. Keeping the
# partitioner itself in scripts/ci avoids a second sharding algorithm.
def pytest_sessionfinish(session, exitstatus):
    output = os.environ.get("GYM_NATIVE_COLLECTION_OUTPUT")
    if output is None:
        return
    from scripts.ci.core_test_shards import DURATIONS_PATH, partition_files

    root = Path(session.config.rootpath)
    files: dict[str, list[str]] = {}
    for item in session.items:
        path = item.path.relative_to(root).as_posix()
        files.setdefault(path, []).append(item.nodeid)
    durations = json.loads(DURATIONS_PATH.read_text())["files"]
    assignments = partition_files(list(files), durations, 4)
    Path(output).write_text(
        json.dumps(
            {
                "valid": int(exitstatus) == 0 and bool(files),
                "exit_code": int(exitstatus),
                "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                "marker": os.environ["GYM_NATIVE_COLLECTION_MARKER"],
                "durations_sha256": hashlib.sha256(DURATIONS_PATH.read_bytes()).hexdigest(),
                "files": files,
                "assignments": assignments,
                "testcase_count": sum(len(items) for items in files.values()),
            },
            indent=2,
        )
        + "\n"
    )


def targets(manifest: dict[str, object], *, resolve: str, shard: int | None) -> list[str]:
    """Select from one global collection; never repartition isolated Pants batches."""
    if not manifest["valid"]:
        raise ValueError("Collection failed; refusing partial native execution")
    if resolve not in {"core", "sandbox"}:
        raise ValueError("Unknown resolve")
    if resolve == "core" and (shard is None or not 0 <= shard < 4):
        raise ValueError("Core requires a shard in [0, 4)")
    if resolve == "sandbox" and shard is not None:
        raise ValueError("Sandbox runs once, without core sharding")
    marker = "not sandbox" if resolve == "core" else "sandbox"
    if manifest["marker"] != marker:
        raise ValueError("Collection marker differs from the requested workload")
    return [
        f"//{path}:{'../' * len(Path(path).parent.parts)}{resolve}-tests"
        for path in sorted(manifest["files"])
        if shard is None or manifest["assignments"][path] == shard
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect = subparsers.add_parser("collect")
    collect.add_argument("--repo", type=Path, required=True)
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--resolve", choices=["core", "sandbox"], required=True)
    collect.add_argument("--uv", default="uv")
    select = subparsers.add_parser("targets")
    select.add_argument("manifest", type=Path)
    select.add_argument("--resolve", choices=["core", "sandbox"], required=True)
    select.add_argument("--shard", type=int)
    args = parser.parse_args()
    if args.command == "targets":
        print("\n".join(targets(json.loads(args.manifest.read_text()), resolve=args.resolve, shard=args.shard)))
        return
    from prepare import EXTRAS, require_uv_version

    require_uv_version(args.uv)
    marker = "not sandbox" if args.resolve == "core" else "sandbox"
    command = [args.uv, "run", "--locked"]
    for extra in EXTRAS[args.resolve]:
        command.extend(["--extra", extra])
    command.extend(
        ["python", "-m", "pytest", "--collect-only", "-q", "-m", marker, "-p", "scripts.pants.native_core.collect"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    # Do not inherit selection overrides; match the CI scripts' collection.
    for key in ("PYTEST_ADDOPTS", "PYTHONPATH", "NEMO_GYM_EXTRA_ROOTS", "NEMO_GYM_CONFIG_DICT"):
        environment.pop(key, None)
    environment.update(GYM_NATIVE_COLLECTION_OUTPUT=str(args.output.resolve()), GYM_NATIVE_COLLECTION_MARKER=marker)
    result = subprocess.run(command, cwd=args.repo, env=environment)
    if result.returncode or not args.output.exists():
        raise SystemExit(result.returncode or 1)
    manifest = json.loads(args.output.read_text())
    if not manifest["valid"]:
        raise SystemExit(1)
    print(f"Collected {manifest['testcase_count']} cases in {len(manifest['files'])} files")


if __name__ == "__main__":
    main()
