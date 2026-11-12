# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from app import CompCodingResourcesServer, CompCodingResourcesServerConfig
from pydantic import ValidationError
from sandbox_app import CompCodingSandboxResourcesServer

from nemo_gym.client import ServerClient
from nemo_gym.server_runtime import ReverifyMode


@pytest.mark.parametrize("entrypoint", ["app", "sandbox_app"])
def test_sandbox_entrypoint_imports_without_ray(entrypoint):
    script = """
import importlib.abc
import sys

class NoRay(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'ray' or fullname.startswith('ray.'):
            raise ModuleNotFoundError('Ray unavailable in this environment')

sys.meta_path.insert(0, NoRay())
__import__(sys.argv[1])
assert 'ray' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", script, entrypoint],
        cwd=Path(__file__).parents[1],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("server_type", [CompCodingResourcesServer, CompCodingSandboxResourcesServer])
def test_sandbox_entrypoint_requires_provider(server_type):
    kwargs = dict(
        host="127.0.0.1",
        port=8080,
        name="code_gen",
        entrypoint="app.py",
        num_processes=1,
        unit_test_timeout_secs=10,
        debug=False,
        sandbox_config={"image": "code-gen-verifier:test"},
    )
    with pytest.raises(ValidationError, match="sandbox_provider"):
        CompCodingResourcesServerConfig(**kwargs)
    with pytest.raises(ValidationError, match="sandbox_provider"):
        CompCodingResourcesServerConfig(**kwargs, sandbox_provider=None)
    config = CompCodingResourcesServerConfig(**kwargs, sandbox_provider="sandbox")
    server = server_type(config=config, server_client=MagicMock(spec=ServerClient))
    assert server.ray_enabled is False


@pytest.mark.asyncio
async def test_code_gen_advertises_stateless_reverification():
    config = CompCodingResourcesServerConfig(
        host="127.0.0.1",
        port=8080,
        name="code_gen",
        entrypoint="sandbox_app.py",
        num_processes=1,
        unit_test_timeout_secs=10,
        debug=False,
        sandbox_provider="sandbox",
        sandbox_config={"image": "code-gen-verifier:test"},
    )
    server = CompCodingSandboxResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    assert await server.get_reverify_mode() == ReverifyMode.STATELESS
