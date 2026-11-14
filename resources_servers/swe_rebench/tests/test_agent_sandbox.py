# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""swe_rebench's agent network restriction and sandbox setup, through the server's startup and endpoints."""

from pathlib import Path
from runpy import run_path
from unittest.mock import MagicMock

import pytest

import resources_servers.swe_rebench.app as app
from nemo_gym.client.server_client import ServerClient
from nemo_gym.config import ModelServerRef
from nemo_gym.sandbox.providers.base import SandboxExecResult


# gym env test runs from the server directory, whose local tests package shadows the root tests package.
check_swe_agent_sandbox = run_path(str(Path(__file__).resolve().parents[3] / "tests/helpers/swe_agent_sandbox.py"))[
    "check_swe_agent_sandbox"
]

SETUP_FAILURES = [
    SandboxExecResult(return_code=1, stdout="", stderr="chmod failed"),
    SandboxExecResult(return_code=0, stdout="", stderr="", error_type="timeout"),
]


@pytest.mark.parametrize("entry", ["legacy", "episode"])
def test_agent_sandbox_is_restricted_and_evaluation_keeps_its_network(monkeypatch, tmp_path, entry) -> None:
    check_swe_agent_sandbox(monkeypatch, tmp_path, "swe_rebench", "SWERebench", entry=entry, setup_result={})


@pytest.mark.parametrize("entry", ["legacy", "episode"])
@pytest.mark.parametrize("failure", SETUP_FAILURES, ids=["exit-1", "timeout"])
def test_a_failed_agent_setup_keeps_no_sandbox(monkeypatch, tmp_path, entry, failure) -> None:
    check_swe_agent_sandbox(
        monkeypatch, tmp_path, "swe_rebench", "SWERebench", entry=entry, setup_result={"agent": failure}
    )


@pytest.mark.parametrize("failure", SETUP_FAILURES, ids=["exit-1", "timeout"])
def test_a_failed_evaluation_setup_is_retried_then_incomplete(monkeypatch, tmp_path, failure) -> None:
    check_swe_agent_sandbox(
        monkeypatch, tmp_path, "swe_rebench", "SWERebench", entry="episode", setup_result={"evaluation": failure}
    )


def _server(monkeypatch, tmp_path, **config) -> None:
    monkeypatch.setattr(app, "get_global_config_dict", lambda: {})
    monkeypatch.setattr(app, "resolve_provider_config", lambda *_: {"opensandbox": {}})
    getattr(app, "SWERebenchResourcesServer")(
        config=getattr(app, "SWERebenchResourcesServerConfig")(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="swe_rebench",
            sandbox_provider="opensandbox",
            **{"sandbox_config": {}, **config},
        ),
        server_client=MagicMock(spec=ServerClient),
    )


MODEL = ModelServerRef(type="responses_api_models", name="policy_model")


def test_enforcing_without_a_model_server_fails_at_startup(monkeypatch, tmp_path) -> None:
    with pytest.raises(ValueError, match="sandbox_model_server"):
        _server(monkeypatch, tmp_path, enforce_agent_no_network=True)


@pytest.mark.parametrize(
    "policy",
    [
        {"defaultAction": "allow"},
        {"defaultAction": "deny", "egress": [{"action": "allow", "target": "github.com"}]},
        {},
        None,
    ],
    ids=["allow-default", "allow-rule", "empty", "none"],
)
def test_enforcing_with_a_configured_network_policy_fails_at_startup(monkeypatch, tmp_path, policy) -> None:
    with pytest.raises(ValueError, match="not both"):
        _server(
            monkeypatch,
            tmp_path,
            enforce_agent_no_network=True,
            sandbox_model_server=MODEL,
            sandbox_config={"provider_options": {"network_policy": policy}},
        )


def test_a_missing_files_source_fails_at_startup(monkeypatch, tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        _server(monkeypatch, tmp_path, sandbox_config={"files": {"/tmp/x": str(tmp_path / "missing")}})
