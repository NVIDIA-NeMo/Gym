# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Session hygiene for the core unit tests."""

import logging
import urllib.request

import pytest

from nemo_gym.global_config import reset_global_config_dict


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


@pytest.fixture(autouse=True)
def no_real_container_registry(request, monkeypatch):
    """No Harbor unit test reaches a container registry. Preparing a task records its base images through
    ``RegistryClient``, and the shared ``write_task`` helper writes ``FROM ubuntu:24.04``, so a test that did not
    ask for ``fake_registry`` went to Docker Hub and failed on its rate limit (HTTP 429) whatever the code under
    test did. The guard used to live in ``test_harbor_tasks.py``, where it covered only the modules that imported
    it; here it covers every ``test_harbor_*`` module. A test that serves the registry over a faked HTTP layer
    instead (``patch_opener`` in ``test_harbor_image_configs.py`` replaces ``OpenerDirector.open``) gets the real
    client, which then talks to that fake."""
    if not request.module.__name__.rpartition(".")[2].startswith("test_harbor"):
        return
    from nemo_gym.tasks.harbor import image_configs
    from tests.unit_tests.test_harbor_tasks import FakeRegistryClient

    real_client = image_configs.RegistryClient
    real_open = urllib.request.OpenerDirector.open

    def client(*args, **kwargs):
        if urllib.request.OpenerDirector.open is not real_open:
            return real_client(*args, **kwargs)
        return FakeRegistryClient(*args, **kwargs)

    monkeypatch.setattr(image_configs, "RegistryClient", client)
    monkeypatch.setattr(FakeRegistryClient, "missing", set())
    monkeypatch.setattr(FakeRegistryClient, "resolved", [])
