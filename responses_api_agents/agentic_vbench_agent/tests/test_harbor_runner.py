# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from responses_api_agents.agentic_vbench_agent.harbor_runner import container_endpoint


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "[::1]"])
def test_host_proxy_uses_rootless_gateway(host: str) -> None:
    assert container_endpoint(f"http://{host}:8123/v1/") == "http://10.0.2.2:8123/v1"


def test_remote_model_endpoint_is_preserved() -> None:
    assert container_endpoint("https://model.example/v1") == "https://model.example/v1"


@pytest.mark.parametrize("endpoint", ["file:///tmp/model", "http://user:secret@localhost:8000/v1", "localhost"])
def test_invalid_or_credentialed_endpoint_is_rejected(endpoint: str) -> None:
    with pytest.raises(ValueError):
        container_endpoint(endpoint)
