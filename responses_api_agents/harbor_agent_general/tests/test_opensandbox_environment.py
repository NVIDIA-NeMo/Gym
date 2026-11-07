# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ssl
from datetime import timedelta
from pathlib import Path

import pytest
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths
from opensandbox.config import ConnectionConfig
from opensandbox.transport import RetryAsyncTransport

from responses_api_agents.harbor_agent_general.envs.opensandbox import OpenSandboxEnvironment


@pytest.mark.parametrize("tls_verify", [None, True, False], ids=["default", "enabled", "disabled"])
async def test_tls_context_reaches_sdk_transport(tmp_path: Path, tls_verify: bool | None) -> None:
    environment = OpenSandboxEnvironment(
        environment_dir=tmp_path,
        environment_name="tls-test",
        session_id="tls-test",
        trial_paths=TrialPaths(trial_dir=tmp_path / "trial"),
        task_env_config=EnvironmentConfig(docker_image="unused:test"),
        domain="sandbox.example:443",
        api_key="test-key",
        protocol="https",
        use_server_proxy=True,
        request_timeout_sec=17,
        **({} if tls_verify is None else {"tls_verify": tls_verify}),
    )
    config = environment._build_connection_config({"ConnectionConfig": ConnectionConfig})
    connection = config.with_transport_if_missing()
    try:
        transport = connection.transport
        if isinstance(transport, RetryAsyncTransport):
            transport = transport.inner
        # Inspect the SDK-created connection pool, rather than just the config attribute.
        context = transport._pool._ssl_context
        verify = tls_verify is not False
        assert context.verify_mode == (ssl.CERT_REQUIRED if verify else ssl.CERT_NONE)
        assert context.check_hostname is verify
        assert context is config._ssl_context
        assert connection.disable_metrics is not verify
        assert connection.domain == "sandbox.example:443"
        assert connection.api_key == "test-key"
        assert connection.protocol == "https"
        assert connection.use_server_proxy is True
        assert connection.request_timeout == timedelta(seconds=17)
        assert connection.owns_transport is True
    finally:
        await connection.close_transport_if_owned()
