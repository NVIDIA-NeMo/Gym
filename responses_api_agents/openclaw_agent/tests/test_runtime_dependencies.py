# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import timedelta

import pytest

from nemo_gym.sandbox.providers.opensandbox import OpenSandboxProvider


@pytest.mark.asyncio
async def test_opensandbox_sdk_is_available_in_agent_environment() -> None:
    # Run unmocked in this server's venv: session fakes never load the provider SDK.
    provider = OpenSandboxProvider(connection={"domain": "sandbox.example:8080/", "protocol": "http"})
    try:
        connection = provider._connection_config(request_timeout_s=30)
        assert connection.domain == "sandbox.example:8080"
        assert connection.request_timeout == timedelta(seconds=30)
    finally:
        await provider.aclose()
