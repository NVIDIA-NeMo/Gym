# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_gym.client import ServerClient
from nemo_gym.config import BaseServerConfig
from resources_servers.mcp_atlas.credentials import (
    SERVER_CREDENTIALS,
    SERVICE_ENV_NAMES,
    CredentialConfigurationError,
    configured_dataset_servers,
    task_environment,
    validate_provider_environment,
)
from resources_servers.mcp_atlas.tests.test_app import judge_client, make_server


ROOT = Path(__file__).resolve().parents[3]


def agent_config(path: Path, **dataset_fields) -> dict:
    return {
        "agent": {
            "responses_api_agents": {
                "simple_agent": {
                    "resources_server": {"name": "mcp_atlas"},
                    "datasets": [{"jsonl_fpath": str(path), **dataset_fields}],
                }
            }
        }
    }


def test_credential_registry_covers_the_catalog_without_aliasing_weather() -> None:
    catalog = json.loads((ROOT / "resources_servers/mcp_atlas/tool_catalog.json").read_text())
    assert set(SERVER_CREDENTIALS) == set(catalog["servers"])
    assert len(SERVICE_ENV_NAMES) == len(set(SERVICE_ENV_NAMES)) == 21
    assert SERVER_CREDENTIALS["weather"] == ()
    assert SERVER_CREDENTIALS["weather-data"] == ("WEATHER_API_KEY",)


def test_task_environment_uses_only_enabled_server_credentials_including_overrides() -> None:
    environment = {
        "GITHUB_TOKEN": "fake github credential",
        "NOTION_TOKEN": "fake unrelated notion credential",
        "ALLOW_SHELL_OPERATORS": "true",
    }
    assert task_environment(environment, ["github", "cli-mcp-server"]) == {
        "GITHUB_TOKEN": "fake github credential",
        "ALLOW_SHELL_OPERATORS": "true",
    }
    assert task_environment(environment, ["cli-mcp-server"]) == {"ALLOW_SHELL_OPERATORS": "true"}


@pytest.mark.parametrize("value", ["", "   "])
def test_every_required_placeholder_must_be_nonblank(value: str) -> None:
    environment = {name: "fake configured value" for name in SERVICE_ENV_NAMES}
    environment["GOOGLE_REFRESH_TOKEN"] = value
    with pytest.raises(CredentialConfigurationError, match="GOOGLE_REFRESH_TOKEN"):
        task_environment(environment, ["google-workspace"])


def test_unknown_enabled_server_cannot_bypass_credential_validation() -> None:
    with pytest.raises(CredentialConfigurationError, match="unknown MCP-Atlas servers"):
        task_environment({}, ["unknown"])


def test_dataset_preflight_ignores_unrelated_agents_and_reads_required_servers(tmp_path) -> None:
    source = tmp_path / "tasks.jsonl"
    source.write_text(json.dumps({"verifier_metadata": {"servers": ["exa", "weather"]}}) + "\n")
    config = agent_config(source)
    config["unrelated"] = agent_config(tmp_path / "absent.jsonl")["agent"]
    config["unrelated"]["responses_api_agents"]["simple_agent"]["resources_server"]["name"] = "other"
    required = configured_dataset_servers(config, resources_server="mcp_atlas", root=tmp_path)
    assert required.servers == {"exa", "weather"}
    assert required.deferred_inputs == ()
    assert task_environment({"EXA_API_KEY": "fake exa credential"}, sorted(required.servers)) == {
        "EXA_API_KEY": "fake exa credential"
    }


def test_missing_requested_dataset_is_rejected_but_declared_prepare_is_explicitly_deferred(tmp_path) -> None:
    source = tmp_path / "not-prepared.jsonl"
    with pytest.raises(CredentialConfigurationError, match="dataset does not exist"):
        configured_dataset_servers(agent_config(source), resources_server="mcp_atlas", root=tmp_path)
    required = configured_dataset_servers(
        agent_config(source, prepare_script="benchmarks/mcp_atlas/prepare.py"),
        resources_server="mcp_atlas",
        root=tmp_path,
    )
    assert required.servers == frozenset()
    assert required.deferred_inputs == (source,)


def test_actual_five_examples_require_no_service_credential_at_startup(monkeypatch, tmp_path) -> None:
    for name in SERVICE_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    client = judge_client()
    client.global_config_dict = agent_config(ROOT / "resources_servers/mcp_atlas/data/example.jsonl")
    required = configured_dataset_servers(client.global_config_dict, resources_server="mcp_atlas", root=ROOT)
    assert len(required.servers) == 13
    assert not {name for server in required.servers for name in SERVER_CREDENTIALS[server]}
    image = tmp_path / "development.sif"
    image.touch()
    server = make_server("", client, sandbox_image=str(image), sandbox_provider={"apptainer": {}})
    assert server._pool is not None
    assert task_environment(server._pool.sandbox_env, sorted(required.servers)) == {}


def test_startup_rejects_missing_actual_dataset_credentials(monkeypatch, tmp_path) -> None:
    for name in SERVICE_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    source = tmp_path / "tasks.jsonl"
    source.write_text(json.dumps({"verifier_metadata": {"servers": ["github"]}}) + "\n")
    client = judge_client()
    client.global_config_dict = agent_config(source)
    image = tmp_path / "development.sif"
    image.touch()
    with pytest.raises(ValueError, match="GITHUB_TOKEN"):
        make_server("", client, sandbox_image=str(image), sandbox_provider={"apptainer": {}})


@pytest.mark.parametrize("name", ["SLACK_MCP_XOXC_TOKEN", "OPENSANDBOX_API_KEY", "INFERENCE_API_KEY", "NVAUTH_TOKEN"])
def test_provider_setup_cannot_restore_scoped_or_host_credentials(name: str) -> None:
    provider = {
        "opensandbox": {"connection": {"api_key": "fake host identity"}, "setup": {"env": {name: "fake value"}}}
    }
    with pytest.raises(CredentialConfigurationError, match="provider setup.env") as rejected:
        validate_provider_environment(provider)
    assert name in str(rejected.value) and "fake value" not in str(rejected.value)
    provider["opensandbox"]["setup"]["env"] = {"PYTHONUNBUFFERED": "1"}
    validate_provider_environment(provider)
    assert provider["opensandbox"]["connection"] == {"api_key": "fake host identity"}
    assert provider["opensandbox"]["setup"]["env"] == {"PYTHONUNBUFFERED": "1"}


@pytest.mark.parametrize(
    "name",
    [
        "OPENSANDBOX_API_KEY",
        "INFERENCE_API_KEY",
        "NVAUTH_TOKEN",
        "CUSTOM_JUDGE_TOKEN",
        "GOOGLE_APPLICATION_CREDENTIALS",
    ],
)
def test_explicit_workload_environment_cannot_forward_host_credentials(name: str) -> None:
    with pytest.raises(CredentialConfigurationError, match="host credentials") as rejected:
        task_environment({name: "fake host sentinel"}, ["calculator"])
    assert "fake host sentinel" not in str(rejected.value)


@pytest.mark.parametrize("name", ["SLACK_MCP_XOXC_TOKEN", "INFERENCE_API_KEY"])
def test_provider_setup_credential_route_is_rejected_by_real_resources_startup(name: str) -> None:
    provider = {
        "opensandbox": {
            "connection": {"api_key": "fake host identity"},
            "setup": {"env": {name: "fake provider inherited sentinel"}},
        }
    }
    with pytest.raises(ValueError, match="provider setup.env") as rejected:
        make_server(
            "",
            judge_client(),
            sandbox_image="ghcr.io/scaleapi/mcp-atlas:1.2.7",
            sandbox_provider=provider,
            sandbox_options={"network_policy": {"defaultAction": "deny", "egress": []}},
            sandbox_development_mode=False,
        )
    assert "fake provider inherited sentinel" not in str(rejected.value)
    assert "fake host identity" not in str(rejected.value)


def test_startup_error_does_not_dump_other_credentials_from_the_real_client_configuration(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    source = tmp_path / "tasks.jsonl"
    source.write_text(json.dumps({"verifier_metadata": {"servers": ["github"]}}) + "\n")
    config = agent_config(source)
    config["model_api_key"] = "fake model credential sentinel"
    client = ServerClient(
        head_server_config=BaseServerConfig(host="127.0.0.1", port=0), global_config_dict=OmegaConf.create(config)
    )
    image = tmp_path / "development.sif"
    image.touch()
    with pytest.raises(ValueError, match="GITHUB_TOKEN") as rejected:
        make_server("", client, sandbox_image=str(image), sandbox_provider={"apptainer": {}})
    assert "fake model credential sentinel" not in str(rejected.value)
