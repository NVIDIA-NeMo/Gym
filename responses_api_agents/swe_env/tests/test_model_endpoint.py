# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the model-server egress primitive that resolves a model endpoint per provider."""

from __future__ import annotations

import pytest

from responses_api_agents.swe_env.model_endpoint import ModelEgressUnavailable, ModelEndpoint, resolve


def test_apptainer_uses_host_loopback_by_default():
    """Apptainer resolves to the host loopback base URL when none is configured."""
    ep = resolve("apptainer", {"model": "qwen"})
    assert ep.base_url == "http://127.0.0.1:8000/v1"
    assert ep.model == "qwen"


def test_docker_uses_configured_base_when_present():
    """Docker uses the explicitly configured base URL."""
    ep = resolve("docker", {"base_url": "http://10.0.0.5:8000/v1"})
    assert ep.base_url == "http://10.0.0.5:8000/v1"


def test_opensandbox_requires_service_url():
    """Opensandbox raises when no reachable service URL is supplied."""
    with pytest.raises(ModelEgressUnavailable):
        resolve("opensandbox", {"base_url": "http://127.0.0.1:8000/v1"})


def test_opensandbox_with_service_url_ok():
    """Opensandbox resolves to the provided service URL."""
    ep = resolve("opensandbox", {"model": "m"}, opensandbox_service_url="http://gym-model.svc.cluster.local/v1")
    assert ep.base_url == "http://gym-model.svc.cluster.local/v1"


def test_to_sandbox_env_is_minimal():
    """The sandbox env carries only the base URL, API key, and model name."""
    ak_value = "abc-test"
    env = ModelEndpoint(base_url="http://h/v1", api_key=ak_value, model="m").to_sandbox_env()
    assert env["OPENAI_BASE_URL"] == "http://h/v1"
    assert env["OPENAI_API_KEY"] == ak_value
    assert env["NEMO_GYM_MODEL"] == "m"
    # never leaks a full global-config dict
    assert "NEMO_GYM_CONFIG_DICT" not in env
