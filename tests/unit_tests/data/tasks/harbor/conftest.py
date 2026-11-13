# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep Harbor registry isolation local to the Harbor test subtree."""

import pytest

from tests.unit_tests.data.tasks.harbor.harbor_registry_fakes import FakeRegistryClient


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
