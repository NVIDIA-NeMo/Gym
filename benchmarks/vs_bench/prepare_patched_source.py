#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Create a checksum-verified VS-Bench source export with the KAZ fix.

The input is a clean export of the pinned upstream revision. The input and any
existing destination are never modified. This is a local source-preparation
step; it neither fetches nor publishes upstream code.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from benchmarks.vs_bench.reference_identity import AUTHOR_REVISION, KAZ_ARRAY_FIX_SHA256


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(source: Path, destination: Path) -> None:
    source = source.resolve(strict=True)
    destination = destination.absolute()
    manifest_path = Path(__file__).with_name("reference_source_manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if manifest["revision"] != AUTHOR_REVISION:
        raise ValueError("Unexpected upstream manifest revision")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Destination already exists: {destination}")
    staging = destination.with_name(destination.name + ".staging")
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(f"Staging directory already exists: {staging}")
    for name, expected in manifest["sha256"].items():
        path = (source / name).resolve(strict=True)
        if not path.is_relative_to(source) or not path.is_file() or digest(path) != expected:
            raise ValueError(f"Input differs from pinned upstream: {name}")
    staging.mkdir(parents=True)
    for name in manifest["sha256"]:
        target = staging / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    patch_path = Path(__file__).with_name("patches") / "kaz_numpy_labels.patch"
    subprocess.run(["patch", "--batch", "--forward", "-p1", "-i", str(patch_path.resolve())], cwd=staging, check=True)
    for name, expected in manifest["sha256"].items():
        if name == "envs/knights_archers_zombies.py":
            expected = KAZ_ARRAY_FIX_SHA256
        if digest(staging / name) != expected:
            raise ValueError(f"Prepared source failed checksum: {name}")
    (staging / "vsbench-source-provenance.json").write_text(
        json.dumps(
            {
                "upstream_revision": AUTHOR_REVISION,
                "patch": "kaz_numpy_labels.patch",
                "patch_sha256": digest(patch_path),
                "patched_file_sha256": KAZ_ARRAY_FIX_SHA256,
            },
            indent=2,
        )
        + "\n"
    )
    staging.rename(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    prepare(args.source, args.destination)
