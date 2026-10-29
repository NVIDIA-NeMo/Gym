# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Session hygiene for the core unit tests."""

import logging

import pytest

import nemo_gym.artifacts.exporters as exporters_module
from nemo_gym.config.global_config import reset_global_config_dict
from tests.unit_tests.harbor_registry_fakes import FakeRegistryClient


def _use_fake_registry(monkeypatch: pytest.MonkeyPatch) -> type[FakeRegistryClient]:
    from nemo_gym.data.tasks.harbor import image_configs as image_configs_module

    monkeypatch.setattr(image_configs_module, "RegistryClient", FakeRegistryClient)
    monkeypatch.setattr(FakeRegistryClient, "missing", set())
    monkeypatch.setattr(FakeRegistryClient, "resolved", [])
    return FakeRegistryClient


@pytest.fixture
def fake_registry(monkeypatch):
    """``prepare`` records base images through the fake registry client, which its ``missing`` and ``resolved``
    state lets a test fail a reference or read what was asked for."""
    return _use_fake_registry(monkeypatch)


@pytest.fixture(autouse=True)
def no_real_registry(request):
    """No Harbor test reaches a registry. A task's default Dockerfile is ``FROM ubuntu:24.04`` and prepare records
    base images, so a test that did not ask for ``fake_registry`` went to Docker Hub and failed on its rate limit
    (HTTP 429) whatever the code under test did. It is autouse here, and not imported into each module, so a new
    Harbor test module cannot forget it. A test that needs another client patches ``RegistryClient`` itself, which
    wins over this one, and the ones that exercise the real client build it from ``nemo_gym.data.tasks.harbor.registry``.

    ``monkeypatch`` is requested only for Harbor modules: requesting it for every test sets it up before the other
    autouse fixtures, so it is torn down after them, and a fixture such as the telemetry state reset would then run
    its teardown while the test's patches are still in place."""
    if request.module.__name__.rpartition(".")[2].startswith("test_harbor"):
        _use_fake_registry(request.getfixturevalue("monkeypatch"))


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
def fresh_run_failure():
    """``mark_run_failed`` sets a process-wide flag that exporters read when they close their run. A CLI test whose
    collection fails would otherwise leave it set, and the next test's exporter would close its run FAILED. Every
    test starts and ends with the flag clear."""
    exporters_module._RUN_FAILED = False
    yield
    exporters_module._RUN_FAILED = False


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
