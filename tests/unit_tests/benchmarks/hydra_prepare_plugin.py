# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pytest plugin that makes every benchmark ``prepare()`` receive its arguments the way the Gym CLI passes them.

``gym eval prepare +prepare_script_args.profiles=[a,b]`` hands ``prepare()`` an OmegaConf ``ListConfig`` (and a
``DictConfig`` for a mapping), not a ``list``; the offline tests call ``prepare()`` with plain Python values and so
never see the difference. Loaded with ``-p hydra_prepare_plugin`` (see ``test_prepare_hydra_arguments.py``), this
plugin wraps every ``benchmarks.<dir>.prepare.prepare`` after collection so that the existing offline tests of each
benchmark run with Hydra-typed containers, and checks that a run leaves a prepared JSONL plus a stamp that parses as JSON.
"""

import json
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf


def hydra_typed(value):
    """``value`` as Hydra delivers it: lists and tuples become ``ListConfig``, mappings ``DictConfig``."""
    if isinstance(value, (list, tuple, dict)):
        return OmegaConf.create({"value": value})["value"]
    return value


def _wrap(prepare):
    def prepare_as_hydra(*args, **kwargs):
        output = prepare(*args, **{key: hydra_typed(value) for key, value in kwargs.items()})
        if isinstance(output, Path) and output.is_file():
            assert output.read_text().strip(), f"{output} has no rows"
            stamp = json.loads(output.with_name(output.name + ".stamp").read_text())
            assert set(stamp) == {"code", "options"}, stamp
        return output

    prepare_as_hydra.__wrapped__ = prepare
    return prepare_as_hydra


@pytest.hookimpl(trylast=True)
def pytest_collection_finish(session):
    for name, module in list(sys.modules.items()):
        if name.startswith("benchmarks.") and name.endswith(".prepare") and callable(getattr(module, "prepare", None)):
            if not hasattr(module.prepare, "__wrapped__"):
                module.prepare = _wrap(module.prepare)
