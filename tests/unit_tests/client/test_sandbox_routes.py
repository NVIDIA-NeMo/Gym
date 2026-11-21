# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nemo_gym.client import sandbox_routes
from nemo_gym.config.config_types import BaseServerConfig


def config(**overrides):
    return {"model": {"responses_api_models": {"test": {"host": "model-host", "port": 8000, **overrides}}}}


def test_service_without_override_keeps_direct_networking():
    assert BaseServerConfig(host="model-host", port=8000).sandbox_url is None
    assert sandbox_routes.sandbox_service_url("model", global_config=config()) is None
    assert sandbox_routes.sandbox_service_url("model", global_config=config(sandbox_url=None)) is None


@pytest.mark.parametrize("url", ["http://127.0.0.1:18000", "https://relay.example:8443/run/model"])
def test_static_service_url_needs_no_provider_configuration(url):
    assert BaseServerConfig(host="model-host", port=8000, sandbox_url=url).sandbox_url == url
    assert sandbox_routes.sandbox_service_url("model", global_config=config(sandbox_url=url)) == url


def test_explicit_config_takes_precedence_over_process_config(monkeypatch):
    monkeypatch.setattr(sandbox_routes, "maybe_get_global_config_dict", lambda: config(sandbox_url="https://other"))
    assert sandbox_routes.sandbox_service_url("model") == "https://other"
    assert sandbox_routes.sandbox_service_url("model", global_config=config()) is None


def test_service_name_must_exist_even_without_an_override():
    with pytest.raises(ValueError):
        sandbox_routes.sandbox_service_url("missing", global_config=config())


@pytest.mark.parametrize(
    "url",
    [
        "",
        "file:///etc/passwd",
        "https:///model",
        "https://user:password@model.example",
        "https://model.example?key=value",
        "https://model.example#fragment",
        "http://model.example:invalid",
        42,
    ],
)
def test_invalid_configured_url_is_rejected(url):
    with pytest.raises(ValueError):
        sandbox_routes.sandbox_service_url("model", global_config=config(sandbox_url=url))
