# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from app import CompCodingResourcesServerConfig
from omegaconf import OmegaConf
from pydantic import ValidationError
from sandbox_app import CompCodingSandboxResourcesServer

from nemo_gym.base_resources_server import ReverifyMode
from nemo_gym.cli.env import _ray_server_names
from nemo_gym.server_utils import ServerClient, _connect_server_to_ray


def test_sandbox_entrypoint_imports_without_ray():
    script = """
import importlib.abc
import sys

class NoRay(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'ray' or fullname.startswith('ray.'):
            raise ModuleNotFoundError('Ray unavailable in this environment')

sys.meta_path.insert(0, NoRay())
import sandbox_app
assert 'ray' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[1],
        check=True,
        capture_output=True,
        text=True,
    )


def test_sandbox_entrypoint_does_not_require_parent_ray(monkeypatch):
    def unexpected_ray_check():
        pytest.fail("sandbox-only server must not check or initialize Ray")

    monkeypatch.setattr("nemo_gym.cli.env.ray_is_installed", unexpected_ray_check)
    monkeypatch.setattr("nemo_gym.server_utils.initialize_ray", unexpected_ray_check)
    server = SimpleNamespace(
        top_level_path="code_gen",
        dir_path=Path(__file__).parents[1],
        entrypoint_fpath=Path("sandbox_app.py"),
    )
    assert _ray_server_names([server]) == []
    _connect_server_to_ray(CompCodingSandboxResourcesServer, OmegaConf.create({}))


@pytest.mark.parametrize("provider", [None, "sandbox"])
def test_sandbox_entrypoint_requires_provider(provider):
    config = CompCodingResourcesServerConfig(
        host="127.0.0.1",
        port=8080,
        name="code_gen",
        entrypoint="sandbox_app.py",
        num_processes=1,
        unit_test_timeout_secs=10,
        debug=False,
        sandbox_provider=provider,
        sandbox_config={"image": "code-gen-verifier:test"},
    )
    kwargs = {"config": config, "server_client": MagicMock(spec=ServerClient)}
    if provider is None:
        with pytest.raises(ValidationError, match="requires sandbox_provider"):
            CompCodingSandboxResourcesServer(**kwargs)
    else:
        assert CompCodingSandboxResourcesServer(**kwargs).config.sandbox_provider == "sandbox"


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
