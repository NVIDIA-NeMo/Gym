# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Session hygiene for the core unit tests."""

import logging
import sys

import pytest


@pytest.fixture(autouse=True)
def fresh_global_config():
    """The merged run config is a process-wide cache. A test that parses one through ``get_global_config_dict``
    (the Harbor CLI tests do, through ``gym eval run`` and ``gym dataset validate``) would otherwise hand it to
    the next test in the session, which then reads another test's servers or misses its own override. Every
    test starts and ends with the cache empty."""
    # Cleanup must not load configuration (and its dependencies) for unrelated tests.
    # Look up again at teardown: the test may import the module for the first time.
    if config := sys.modules.get("nemo_gym.config.global_config"):
        config.reset_global_config_dict()
    yield
    if config := sys.modules.get("nemo_gym.config.global_config"):
        config.reset_global_config_dict()


@pytest.fixture(autouse=True)
def fresh_run_failure():
    """``mark_run_failed`` sets a process-wide flag that exporters read when they close their run. A CLI test whose
    collection fails would otherwise leave it set, and the next test's exporter would close its run FAILED. Every
    test starts and ends with the flag clear."""
    if exporters := sys.modules.get("nemo_gym.artifacts.exporters"):
        exporters._RUN_FAILED = False
    yield
    if exporters := sys.modules.get("nemo_gym.artifacts.exporters"):
        exporters._RUN_FAILED = False


@pytest.fixture(autouse=True)
def fresh_root_logging():
    """``gym --verbose`` (and the ``verbose`` config key) call ``logging.basicConfig``, which installs a root
    ``StreamHandler`` bound to whatever ``sys.stderr`` is at that moment: under pytest, the capture stream of the
    test that ran the command. That stream is closed when the test ends, and every later test that logs anything
    then prints a ``--- Logging error ---`` traceback to the real stderr, so a test asserting a clean stderr
    (the package-store CLI tests do) fails only when it runs after a verbose CLI test. Every test ends with
    the root logger's handlers and level exactly as it found them."""
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    for handler in list(root.handlers):
        if handler not in handlers:
            root.removeHandler(handler)
            handler.close()
    for handler in handlers:
        if handler not in root.handlers:
            root.addHandler(handler)
    root.setLevel(level)
