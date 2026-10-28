# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Session hygiene for the core unit tests."""

import logging

import pytest

from nemo_gym.config.global_config import reset_global_config_dict


@pytest.fixture(autouse=True)
def fresh_global_config():
    """The merged run config is a process-wide cache. A test that parses one through ``get_global_config_dict``
    (the Harbor CLI tests do, through ``gym eval run`` and ``gym dataset validate``) would otherwise hand it to
    the next test in the session, which then reads another test's servers or misses its own override. Every
    test starts and ends with the cache empty."""
    reset_global_config_dict()
    yield
    reset_global_config_dict()


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
