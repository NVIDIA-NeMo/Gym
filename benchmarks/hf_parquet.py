# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read the rows of a pinned Hugging Face dataset config from its files, without background readers.

``datasets.load_dataset(..., streaming=True).take(n)`` leaves pyarrow dataset-scanner threads alive that call back
into Python file objects while the interpreter finalizes, so a capped smoke run hangs at process exit. These helpers
resolve a config's files the way ``datasets`` does (the ``configs`` of the dataset card, files sorted) and read
parquet shards one row group at a time with ``pyarrow.parquet.ParquetFile``: single-threaded, no read-ahead, and every
file closed when the iterator finishes or is closed, so nothing survives a capped read.
"""

from __future__ import annotations

import json
import re
from typing import Iterator

import yaml


BLOCK_SIZE = 8 << 20  # one range request per block when reading shards through the Hub file system
BATCH_ROWS = 64  # rows decoded at a time within a row group


def _root(repo: str, revision: str) -> str:
    return f"datasets/{repo}@{revision}"


def config_files(repo: str, revision: str, config: str | None, split: str) -> list[str]:
    """The files of a dataset config split at a pinned revision, in ``datasets`` order (sorted by path).

    ``config`` and ``split`` are looked up in the ``configs`` of the dataset card; ``None`` selects the only config.
    Returns repository-relative paths.
    """
    from huggingface_hub import HfFileSystem, hf_hub_download

    card_path = hf_hub_download(repo, "README.md", repo_type="dataset", revision=revision)
    with open(card_path, encoding="utf-8") as card:
        text = card.read()
    header = re.match(r"---\n(.*?)\n---[ \t]*(?:\n|$)", text, re.DOTALL)  # comments may contain "---" mid-line
    if header is None:
        raise ValueError(f"{repo}@{revision}: README.md has no metadata header")
    configs = (yaml.safe_load(header.group(1)) or {}).get("configs") or []
    matching = [entry for entry in configs if config is None or entry.get("config_name") == config]
    if len(matching) != 1:
        raise ValueError(f"{repo}@{revision}: expected one config {config!r}, found {len(matching)}")
    patterns = []
    for data_file in matching[0].get("data_files") or []:
        if data_file.get("split") == split:
            paths = data_file["path"]
            patterns.extend([paths] if isinstance(paths, str) else paths)
    root = _root(repo, revision)
    filesystem = HfFileSystem()
    files = sorted({match[len(root) + 1 :] for pattern in patterns for match in filesystem.glob(f"{root}/{pattern}")})
    if not files:
        raise ValueError(f"{repo}@{revision}: no files for config {config!r} split {split!r}")
    return files


def iter_rows(repo: str, revision: str, files: list[str], *, limit: int | None = None) -> Iterator[dict]:
    """Yield the rows of the parquet ``files`` in order, at most ``limit``, as dicts (audio stays ``{bytes, path}``).

    A shard is read one row group at a time in small batches and closed as soon as the rows needed from it are
    consumed; once ``limit`` rows were yielded no further row group is touched.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    filesystem = HfFileSystem()
    remaining = limit
    for name in files:
        if remaining is not None and remaining <= 0:
            return
        with filesystem.open(f"{_root(repo, revision)}/{name}", "rb", block_size=BLOCK_SIZE) as stream:
            parquet = pq.ParquetFile(stream, pre_buffer=False)
            for group in range(parquet.num_row_groups):
                for batch in parquet.iter_batches(batch_size=BATCH_ROWS, row_groups=[group], use_threads=False):
                    rows = batch.to_pylist()
                    if remaining is not None:
                        rows = rows[:remaining]
                        remaining -= len(rows)
                    yield from rows
                    if remaining is not None and remaining <= 0:
                        return


def iter_json_rows(repo: str, revision: str, files: list[str]) -> Iterator[dict]:
    """Yield the rows of pinned JSON Lines ``files`` of a dataset repository."""
    from huggingface_hub import hf_hub_download

    for name in files:
        with open(hf_hub_download(repo, name, repo_type="dataset", revision=revision), encoding="utf-8") as lines:
            for line in lines:
                if line.strip():
                    yield json.loads(line)
