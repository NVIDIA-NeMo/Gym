# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
from pydantic import ValidationError

from nemo_gym.web.models import WebBenchmark
from nemo_gym.web.resource_config import WebResourcesServerConfig


def _config(**updates) -> WebResourcesServerConfig:
    values = {
        "name": "web",
        "host": "localhost",
        "port": 8000,
        "entrypoint": "app.py",
        "domain": "agent",
    }
    values.update(updates)
    return WebResourcesServerConfig(**values)


def test_web_resources_config_defaults_and_paths(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    config = _config(artifact_dir="~/web-artifacts")

    assert config.max_sessions == 1
    assert config.site_pool_mode == "unmanaged"
    assert config.allowed_benchmarks == list(WebBenchmark)
    assert config.resolved_artifact_dir() == os.path.abspath(os.path.expanduser("~/web-artifacts"))
    assert config.auth_token() == ""

    monkeypatch.setenv(config.auth_token_env, "  bearer-secret  ")
    assert config.auth_token() == "bearer-secret"


@pytest.mark.parametrize("num_workers", [0, 2, 8])
def test_web_resources_config_requires_one_process(num_workers: int) -> None:
    with pytest.raises(ValidationError, match="num_workers must be 1"):
        _config(num_workers=num_workers)


def test_web_resources_config_rejects_duplicate_benchmarks() -> None:
    with pytest.raises(ValidationError, match="allowed_benchmarks must not contain duplicates"):
        _config(allowed_benchmarks=[WebBenchmark.WEBARENA, WebBenchmark.WEBARENA])


def test_web_resources_config_accepts_explicit_single_worker() -> None:
    assert _config(num_workers=1).num_workers == 1


def test_web_resources_config_requires_one_browser_provider() -> None:
    with pytest.raises(ValidationError, match="select exactly one provider"):
        _config(browser_session_provider={})


@pytest.mark.parametrize(
    "updates",
    [
        {
            "browser_lease_ttl_seconds": 60,
            "browser_heartbeat_interval_seconds": 60,
        },
        {
            "browser_lease_ttl_seconds": 60,
            "browser_heartbeat_timeout_seconds": 60,
        },
    ],
)
def test_web_resources_config_heartbeat_must_fit_inside_provider_ttl(updates) -> None:
    with pytest.raises(ValidationError, match="shorter than the provider lease TTL"):
        _config(**updates)
