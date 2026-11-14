# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

from nemo_gym.client import ServerClient
from nemo_gym.config import (
    INPUT_READ_ROOTS_KEY_NAME,
    BaseServerConfig,
    GlobalConfigDictParser,
    GlobalConfigDictParserConfig,
)
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
CLEAN_APPTAINER_PROVIDER = {
    "apptainer": {"create": {"extra_start_args": ["--cleanenv"]}, "exec": {"extra_exec_args": ["--cleanenv"]}}
}


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


def controller_config(config: dict) -> DictConfig:
    """Use the real controller loader, then the same serialized config boundary as a worker."""
    config = OmegaConf.create(config)
    config.setdefault("allow_openai_version_skew", True)
    config.setdefault("mcp_atlas", {"resources_servers": {"mcp_atlas": {"entrypoint": "app.py", "domain": "other"}}})
    for component in config.values():
        if not isinstance(component, dict) and not OmegaConf.is_dict(component):
            continue
        for server_type in ("resources_servers", "responses_api_agents"):
            for server in (component.get(server_type) or {}).values():
                server.setdefault("entrypoint", "app.py")
                for index, dataset in enumerate(server.get("datasets") or []):
                    dataset.setdefault("name", f"tasks-{index}")
                    dataset.setdefault("type", "benchmark")
    loaded = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=config,
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    return OmegaConf.create(OmegaConf.to_yaml(loaded))


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


def test_empty_enabled_servers_cannot_select_upstream_auto_mode() -> None:
    with pytest.raises(CredentialConfigurationError, match="nonempty"):
        task_environment({}, [])


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


def test_dataset_inventory_includes_bound_resource_and_agent_declarations(tmp_path) -> None:
    own = tmp_path / "own.jsonl"
    own.write_text(json.dumps({"verifier_metadata": {"servers": ["notion"]}}) + "\n")
    agent = tmp_path / "agent.jsonl"
    agent.write_text(json.dumps({"verifier_metadata": {"servers": ["exa"]}}) + "\n")
    config = agent_config(agent)
    config["mcp_atlas"] = {"resources_servers": {"mcp_atlas": {"datasets": [{"jsonl_fpath": str(own)}]}}}
    config["unrelated"] = {
        "resources_servers": {"mcp_atlas": {"datasets": [{"jsonl_fpath": str(tmp_path / "absent.jsonl")}]}}
    }
    required = configured_dataset_servers(config, resources_server="mcp_atlas", root=tmp_path)
    assert required.servers == {"exa", "notion"}
    assert required.deferred_inputs == ()


@pytest.mark.parametrize("materialized,canonical", [(False, False), (False, True), (True, False), (True, True)])
def test_dataset_inventory_accepts_supported_task_containers(tmp_path, materialized: bool, canonical: bool) -> None:
    metadata = {"verifier_metadata": {"servers": ["exa"]}}
    task_input = {"task_data": metadata} if canonical else metadata
    row = (
        {"task_id": {"taskset": "mcp_atlas", "task_id": "task"}, "task_input": task_input}
        if materialized
        else task_input
    )
    source = tmp_path / "tasks.jsonl"
    source.write_text(json.dumps(row) + "\n")
    required = configured_dataset_servers(agent_config(source), resources_server="mcp_atlas", root=tmp_path)
    assert required.servers == {"exa"}


@pytest.mark.parametrize("duplicate_equal", [True, False])
def test_dataset_inventory_preserves_canonical_task_field_conflict_rules(tmp_path, duplicate_equal: bool) -> None:
    metadata = {"servers": ["exa"]}
    canonical = metadata if duplicate_equal else {"servers": ["exa"], "claims": ["different"]}
    source = tmp_path / "tasks.jsonl"
    source.write_text(
        json.dumps({"verifier_metadata": metadata, "task_data": {"verifier_metadata": canonical}}) + "\n"
    )
    if duplicate_equal:
        assert configured_dataset_servers(
            agent_config(source), resources_server="mcp_atlas", root=tmp_path
        ).servers == {"exa"}
    else:
        with pytest.raises(CredentialConfigurationError, match="conflicting task_data fields: verifier_metadata"):
            configured_dataset_servers(agent_config(source), resources_server="mcp_atlas", root=tmp_path)


@pytest.mark.parametrize("task_input", [None, {"task_data": None}])
def test_dataset_inventory_rejects_malformed_materialized_containers(tmp_path, task_input: object) -> None:
    source = tmp_path / "tasks.jsonl"
    source.write_text(
        json.dumps({"task_id": {"taskset": "mcp_atlas", "task_id": "task"}, "task_input": task_input}) + "\n"
    )
    with pytest.raises(CredentialConfigurationError, match="must be an object"):
        configured_dataset_servers(agent_config(source), resources_server="mcp_atlas", root=tmp_path)


def test_dataset_inventory_uses_controller_precedence_after_worker_cwd_and_import_roots_change(
    monkeypatch, tmp_path
) -> None:
    controller = tmp_path / "controller"
    extra = tmp_path / "extra"
    component = tmp_path / "component"
    worker = component / "resources_servers" / "mcp_atlas"
    worker.mkdir(parents=True)
    for directory, server in [(controller, "notion"), (extra, "exa"), (component, "github"), (worker, "weather-data")]:
        directory.mkdir(exist_ok=True)
        (directory / "tasks.jsonl").write_text(json.dumps({"verifier_metadata": {"servers": [server]}}) + "\n")
    monkeypatch.chdir(controller)
    monkeypatch.setenv("NEMO_GYM_EXTRA_ROOTS", str(extra))
    extra_config = controller_config(agent_config(Path("tasks.jsonl")))
    monkeypatch.delenv("NEMO_GYM_EXTRA_ROOTS")
    controller_only_config = controller_config(agent_config(Path("tasks.jsonl")))
    monkeypatch.chdir(worker)
    monkeypatch.setenv("NEMO_GYM_EXTRA_ROOTS", str(component))
    required = configured_dataset_servers(extra_config, resources_server="mcp_atlas", root=component)
    assert required.servers == {"exa"}
    assert configured_dataset_servers(
        controller_only_config, resources_server="mcp_atlas", root=component
    ).servers == {"notion"}


def test_dataset_inventory_missing_prepared_input_keeps_controller_origin(monkeypatch, tmp_path) -> None:
    origin = tmp_path / "controller"
    origin.mkdir()
    monkeypatch.chdir(origin)
    monkeypatch.delenv("NEMO_GYM_EXTRA_ROOTS", raising=False)
    config = controller_config(agent_config(Path("pending.jsonl"), prepare_script="prepare.py"))
    required = configured_dataset_servers(config, resources_server="mcp_atlas", root=tmp_path)
    assert required.servers == frozenset()
    assert required.deferred_inputs == (origin / "pending.jsonl",)
    config["agent"]["responses_api_agents"]["simple_agent"]["datasets"][0].pop("prepare_script")
    with pytest.raises(CredentialConfigurationError, match="absolute resource-host-visible.*--search-dir"):
        configured_dataset_servers(config, resources_server="mcp_atlas", root=tmp_path)


def test_dataset_inventory_does_not_fall_through_a_shadowing_directory(monkeypatch, tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    (first / "tasks.jsonl").mkdir(parents=True)
    second.mkdir()
    (second / "tasks.jsonl").write_text(json.dumps({"verifier_metadata": {"servers": ["exa"]}}) + "\n")
    monkeypatch.chdir(second)
    monkeypatch.setenv("NEMO_GYM_EXTRA_ROOTS", str(first))
    config = controller_config(agent_config(Path("tasks.jsonl"), prepare_script="prepare.py"))
    with pytest.raises(CredentialConfigurationError, match="dataset is not readable.*absolute resource-host-visible"):
        configured_dataset_servers(config, resources_server="mcp_atlas", root=tmp_path)


def test_dataset_inventory_direct_call_uses_explicit_origin(tmp_path) -> None:
    (tmp_path / "tasks.jsonl").write_text(json.dumps({"verifier_metadata": {"servers": ["exa"]}}) + "\n")
    required = configured_dataset_servers(
        agent_config(Path("tasks.jsonl")), resources_server="mcp_atlas", root=tmp_path
    )
    assert required.servers == {"exa"}


@pytest.mark.parametrize("roots", ["not-a-list", [], ["relative"]])
def test_invalid_read_roots_require_controller_config_reload(tmp_path, roots) -> None:
    (tmp_path / "tasks.jsonl").write_text(json.dumps({"verifier_metadata": {"servers": ["exa"]}}) + "\n")
    config = agent_config(Path("tasks.jsonl")) | {INPUT_READ_ROOTS_KEY_NAME: roots}
    with pytest.raises(CredentialConfigurationError, match="reload the config in the controller"):
        configured_dataset_servers(config, resources_server="mcp_atlas", root=tmp_path)


def test_dataset_inventory_finds_input_prepared_after_controller_config_load(monkeypatch, tmp_path) -> None:
    controller = tmp_path / "controller"
    extra = tmp_path / "extra"
    worker = tmp_path / "worker"
    for directory in (controller, extra, worker):
        directory.mkdir()
    monkeypatch.chdir(controller)
    monkeypatch.setenv("NEMO_GYM_EXTRA_ROOTS", str(extra))
    config = controller_config(agent_config(Path("pending.jsonl"), prepare_script="prepare.py"))
    (controller / "pending.jsonl").write_text(json.dumps({"verifier_metadata": {"servers": ["exa"]}}) + "\n")
    (worker / "pending.jsonl").write_text(json.dumps({"verifier_metadata": {"servers": ["notion"]}}) + "\n")
    monkeypatch.chdir(worker)
    monkeypatch.setenv("NEMO_GYM_EXTRA_ROOTS", str(worker))
    required = configured_dataset_servers(config, resources_server="mcp_atlas", root=worker)
    assert required.servers == {"exa"}
    assert required.deferred_inputs == ()


def test_dataset_inventory_uses_current_relative_path_after_config_load(monkeypatch, tmp_path) -> None:
    controller = tmp_path / "controller"
    worker = tmp_path / "worker"
    controller.mkdir()
    worker.mkdir()
    for name, server in (("before.jsonl", "exa"), ("after.jsonl", "notion")):
        (controller / name).write_text(json.dumps({"verifier_metadata": {"servers": [server]}}) + "\n")
    monkeypatch.chdir(controller)
    monkeypatch.delenv("NEMO_GYM_EXTRA_ROOTS", raising=False)
    config = controller_config(agent_config(Path("before.jsonl")))
    config["agent"]["responses_api_agents"]["simple_agent"]["datasets"][0]["jsonl_fpath"] = "after.jsonl"
    monkeypatch.chdir(worker)
    required = configured_dataset_servers(config, resources_server="mcp_atlas", root=worker)
    assert required.servers == {"notion"}


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
    server = make_server("", client, sandbox_image=str(image), sandbox_provider=CLEAN_APPTAINER_PROVIDER)
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
        make_server("", client, sandbox_image=str(image), sandbox_provider=CLEAN_APPTAINER_PROVIDER)


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
        make_server("", client, sandbox_image=str(image), sandbox_provider=CLEAN_APPTAINER_PROVIDER)
    assert "fake model credential sentinel" not in str(rejected.value)
