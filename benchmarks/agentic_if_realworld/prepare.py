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
    link = Path(__file__).parent / "data/prepared.jsonl"
    if (link.exists() or link.is_symlink()) and (not link.is_symlink() or link.resolve() != path):
        raise ValueError("Prepared-data link already points elsewhere; reconcile it first")
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
    link.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not link.is_symlink():
        link.symlink_to(path)
    return link


if __name__ == "__main__":
    prepare()
