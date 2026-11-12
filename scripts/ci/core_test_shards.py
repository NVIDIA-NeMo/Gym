# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in, duration-balanced whole-file pytest sharding for isolated CI jobs."""

import json
from collections import Counter
from pathlib import Path
from statistics import median

import pytest


DURATIONS_PATH = Path(__file__).with_name("core_test_durations.json")


def partition_files(files: list[str], durations: dict[str, float], num_shards: int) -> dict[str, int]:
    """Assign each file once, longest first; ties and new-file estimates are deterministic."""
    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    fallback = median(durations.values()) if durations else 1.0
    loads = [0.0] * num_shards
    assignments = {}
    for file in sorted(set(files), key=lambda file: (-durations.get(file, fallback), file)):
        shard = min(range(num_shards), key=lambda index: (loads[index], index))
        assignments[file] = shard
        loads[shard] += durations.get(file, fallback)
    return assignments


def pytest_addoption(parser):
    group = parser.getgroup("core CI sharding")
    group.addoption("--core-shard-index", type=int, default=None)
    group.addoption("--core-num-shards", type=int, default=None)


def pytest_configure(config):
    index = config.getoption("--core-shard-index")
    count = config.getoption("--core-num-shards")
    if index is None and count is None:
        return
    if index is None or count is None or not 0 <= index < count:
        raise pytest.UsageError("core sharding requires 0 <= --core-shard-index < --core-num-shards")


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    index = config.getoption("--core-shard-index")
    if index is None:
        return
    count = config.getoption("--core-num-shards")
    files = [item.path.relative_to(config.rootpath).as_posix() for item in items]
    durations = json.loads(DURATIONS_PATH.read_text())["files"]
    assignments = partition_files(files, durations, count)
    selected, deselected = [], []
    for item, file in zip(items, files):
        (selected if assignments[file] == index else deselected).append(item)
    config.hook.pytest_deselected(items=deselected)
    items[:] = selected
    reporter = config.pluginmanager.get_plugin("terminalreporter")
    if reporter:
        counts = Counter(assignments.values())
        reporter.write_line(f"Core shard {index}/{count}: {counts[index]} files, {len(selected)} tests")
