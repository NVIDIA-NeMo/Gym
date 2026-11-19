# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import timedelta


def test_opensandbox_sdk_is_available_in_agent_environment() -> None:
    # Session tests fake the provider, so only this test loads the SDK in this server's venv.
    # Import here so a missing extra fails this test instead of aborting collection.
    from opensandbox.config import ConnectionConfig

    connection = ConnectionConfig(
        domain="sandbox.example:8080", protocol="http", request_timeout=timedelta(seconds=30)
    )
    assert connection.domain == "sandbox.example:8080"
    assert connection.request_timeout == timedelta(seconds=30)
