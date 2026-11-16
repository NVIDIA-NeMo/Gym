# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every audio benchmark ``prepare()`` survives the argument types the Gym CLI really passes.

``gym eval prepare --benchmark X +prepare_script_args.k=v`` parses ``v`` with the Hydra override grammar and hands
``prepare()`` its OmegaConf value: ``ListConfig`` for ``[a,b]``, ``int`` for ``2``, ``str`` for ``card``. The offline
tests of each benchmark call ``prepare()`` with plain Python values, which is how a ``ListConfig`` that ``json`` cannot
encode went unnoticed until a full run. This module closes the gap: every offline ``prepare()`` test of every
benchmark (``test_<name>_benchmark.py``, discovered, not listed) is re-run with its list arguments delivered as
``ListConfig`` (``hydra_prepare_plugin``), asserting the run ends in rows plus a stamp that parses as JSON.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from hydra.core.override_parser.overrides_parser import OverridesParser
from omegaconf import DictConfig, ListConfig

from tests.unit_tests.benchmarks import hydra_prepare_plugin as plugin


BENCHMARK_TESTS = Path(__file__).parent
REPOSITORY = BENCHMARK_TESTS.parents[2]


def _parsed(value: str):
    """The value Hydra's override grammar gives ``+prepare_script_args.k=<value>``."""
    override = OverridesParser.create().parse_override(f"+prepare_script_args.k={value}")
    assert not override.is_sweep_override(), f"{value!r} is a Hydra sweep, not a value"
    return override.value()


def test_a_bracketed_override_is_a_list_that_arrives_as_a_listconfig() -> None:
    assert _parsed("[a,b]") == ["a", "b"]
    assert isinstance(plugin.hydra_typed(_parsed("[a,b]")), ListConfig)
    assert isinstance(plugin.hydra_typed({"a": 1}), DictConfig)
    assert plugin.hydra_typed("card") == "card" and plugin.hydra_typed(2) == 2


def test_a_bare_comma_separated_override_is_a_hydra_sweep_so_lists_need_brackets() -> None:
    # `+prepare_script_args.subsets=a,b` aborts `gym eval prepare` with "Ambiguous value": pass `[a,b]`.
    override = OverridesParser.create().parse_override("+prepare_script_args.subsets=a,b")
    assert override.is_sweep_override()
    assert _parsed("[a,b]") == ["a", "b"]


def test_every_offline_prepare_test_passes_with_hydra_typed_arguments() -> None:
    """Re-run all ``test_<name>_benchmark.py`` offline ``prepare()`` tests with list arguments as ``ListConfig``."""
    tests = sorted(BENCHMARK_TESTS.glob("test_*_benchmark.py"))
    if not tests:
        pytest.skip("no benchmark has a prepare() test yet")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(REPOSITORY), os.environ.get("PYTHONPATH", "")])}
    command = [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "-o", "addopts="]
    if env.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD"):
        command += ["-p", "pytest_asyncio.plugin", "-p", "anyio"]
    command += ["-p", "tests.unit_tests.benchmarks.hydra_prepare_plugin", *map(str, tests)]
    result = subprocess.run(command, cwd=REPOSITORY, env=env, capture_output=True, text=True, timeout=1800)
    assert result.returncode == 0, result.stdout[-6000:] + result.stderr[-2000:]
