# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest

from nemo_gym.web import BrowserSessionHandle, ComposedWebBackend, RemoteWebBackend, WebObservation
from resources_servers.webarena_browser.backend import WebArenaBrowserDriver, webarena_backend_factory
from resources_servers.webarena_browser.evaluators import WebArenaTaskEvaluator
from resources_servers.webarena_browser.tests.test_contract import _browser_lease, _config, _visualwebarena_task


def test_local_backend_keeps_the_site_driver_and_evaluator():
    backend = webarena_backend_factory(_config(), "local", object(), _browser_lease())
    assert isinstance(backend, ComposedWebBackend)
    assert isinstance(backend.driver, WebArenaBrowserDriver)
    assert isinstance(backend.evaluator, WebArenaTaskEvaluator)


@pytest.mark.parametrize("provider", ["local_instance_pool", "external_instance_provider"])
def test_instance_lease_runs_the_same_webarena_factory_in_its_owner_process(monkeypatch, provider):
    config = _config()
    handle = BrowserSessionHandle(
        provider_name=provider,
        transport="browser_instance",
        endpoint="http://127.0.0.1:32000",
        metadata={"token": "test-only"},
    )
    backend = webarena_backend_factory(config, "pooled", object(), handle)
    assert isinstance(backend, RemoteWebBackend)
    observation = WebObservation()
    rpc = Mock(return_value={"observation": observation.model_dump(mode="json"), "info": {"owner": "instance"}})
    monkeypatch.setattr(backend, "_call", rpc)
    task = _visualwebarena_task()
    actual, info = backend.reset(task)
    assert actual == observation and info == {"owner": "instance"}
    rpc.assert_called_once_with(
        "/reset",
        {
            "session_id": "pooled",
            "backend_factory": "resources_servers.webarena_browser.backend:webarena_backend_factory",
            "config_class": "resources_servers.webarena_browser.config:WebArenaBrowserResourcesServerConfig",
            "config": config.model_dump(mode="json"),
            "task": task.model_dump(mode="json"),
        },
    )
