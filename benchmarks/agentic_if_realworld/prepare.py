# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare exact wrapper content; private criteria stay in the dataset directory."""

import json
import os
import tempfile
from pathlib import Path

from resources_servers.agentic_if_realworld.benchmark.core import DEFAULT_DATA_DIR, Dataset, source_preflight
from resources_servers.agentic_if_realworld.benchmark.registry import fetch_release, private_path


def prepare() -> Path:
    synthetic = os.environ.get("AGENTIC_IF_SYNTHETIC") == "1"
    directory = Path(os.environ.get("AGENTIC_IF_DATA_DIR") or DEFAULT_DATA_DIR)
    if not synthetic:
        fetch_release(directory)
    dataset = Dataset(directory, synthetic=synthetic)
    if not dataset.synthetic:
        source_preflight(
            os.environ.get("AGENTIC_IF_SPLIT_REGISTRY"),
            os.environ.get("AGENTIC_IF_GROUND_TRUTH"),
            directory=dataset.directory,
        )
    path = private_path(Path(os.environ.get("AGENTIC_IF_PREPARED_PATH") or DEFAULT_DATA_DIR / "prepared.jsonl"))
    if path.name.startswith("prepared_prepare.") or path.name in {
        "prepared_metrics.json",
        "prepared_metrics_conflict.json",
    }:
        raise ValueError("Prepared input must not use a filename reserved for a Gym collation sidecar")
    data_dir = Path(__file__).parent / "data"
    if (data_dir.exists() or data_dir.is_symlink()) and (
        not data_dir.is_symlink() or data_dir.resolve() != path.parent
    ):
        raise ValueError("Prepared-data directory already exists or points elsewhere; reconcile it first")
    alias = path.parent / "prepared.jsonl"
    if (
        alias != path
        and (alias.exists() or alias.is_symlink())
        and (not alias.is_symlink() or alias.resolve() != path)
    ):
        raise ValueError("Private prepared.jsonl conflicts with the configured destination; reconcile it first")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            for row in dataset.prepared():
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    if alias != path and not alias.is_symlink():
        alias.symlink_to(path)
    if not data_dir.is_symlink():
        data_dir.symlink_to(path.parent, target_is_directory=True)
    return data_dir / "prepared.jsonl"


if __name__ == "__main__":
    prepare()
