# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A benchmark config that sets `max_steps` for a simple agent sets an integer.

`SimpleAgentConfig.max_steps` is `int = None`: leaving it out is fine, but an explicit `null` fails validation and the
agent server does not start.
"""

from pathlib import Path

import pytest
import yaml


BENCHMARKS = Path(__file__).resolve().parents[3] / "benchmarks"


def _simple_agent_blocks(node):
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "simple_agent" and isinstance(value, dict):
                yield value
            yield from _simple_agent_blocks(value)
    elif isinstance(node, list):
        for item in node:
            yield from _simple_agent_blocks(item)


def _configs_with_max_steps():
    for path in sorted(BENCHMARKS.rglob("*.yaml")):
        with open(path) as f:
            try:
                documents = list(yaml.safe_load_all(f))
            except yaml.YAMLError:
                continue
        for document in documents:
            for block in _simple_agent_blocks(document):
                if "max_steps" in block:
                    yield pytest.param(path, block["max_steps"], id=str(path.relative_to(BENCHMARKS)))


@pytest.mark.parametrize("path,max_steps", list(_configs_with_max_steps()))
def test_max_steps_is_an_integer(path, max_steps):
    assert isinstance(max_steps, int) and not isinstance(max_steps, bool), (
        f"{path}: simple_agent max_steps is {max_steps!r}; use an integer or leave the key out"
    )
