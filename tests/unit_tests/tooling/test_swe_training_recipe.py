# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recipe contract checks without a model download, sandbox, Slurm job or training worker."""

import copy
import hashlib
import json
import pprint
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf
from scripts import prepare_swe_single_agent_turn_recipe as recipe

from nemo_gym.config.global_config import (
    ALLOW_UNSUPPORTED_PAIRING_ENV_VAR_NAME,
    ALLOW_UNSUPPORTED_PAIRING_KEY_NAME,
    GlobalConfigDictParser,
    GlobalConfigDictParserConfig,
)
from nemo_gym.rollouts.rollout_collection import NG_ENVIRONMENT_SERVER_KEY, RolloutCollectionHelper
from tests.unit_tests.tooling.swe_training_fixtures import NEXT_ROW


@pytest.fixture
def codex_catalog(tmp_path):
    directory = tmp_path / "export"
    directory.mkdir()
    fallback = {
        "slug": "gym-policy-model",
        "base_instructions": "Keep these exact instructions.",
        "context_window": 272000,
        "max_context_window": 272000,
        "supports_parallel_tool_calls": False,
        "truncation_policy": {"mode": "bytes", "limit": 10000},
    }
    descriptor = {key: value for key, value in fallback.items() if key not in ("context_window", "max_context_window")}
    (directory / "models.json").write_text(json.dumps({"models": [descriptor]}))
    (directory / "fallback.json").write_text(json.dumps(fallback))
    provenance = recipe.CODEX_EXPORT | {
        f"{name}_sha256": hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in ("models.json", "fallback.json")
    }
    (directory / "provenance.json").write_text(json.dumps(provenance))
    return directory / "models.json"


@pytest.mark.parametrize("harnesses", [["atom"], [], ["pi", "pi"], ["pi", "hermes"], ["nooa", "codex"]])
def test_unsupported_or_reordered_blends_fail_before_preparation(harnesses):
    with pytest.raises(ValueError):
        recipe.harness_assets(harnesses, codex_catalog=None)


@pytest.mark.parametrize(
    "mutation",
    [
        "source_revision",
        "codex_version",
        "exporter_sha256",
        "status",
        "bytes",
        "alias",
        "duplicate",
        "context",
        "prompt",
        "tools",
    ],
)
def test_codex_export_rejects_mismatched_provenance_and_controls(codex_catalog, mutation):
    provenance_path = codex_catalog.with_name("provenance.json")
    provenance = json.loads(provenance_path.read_text())
    catalog = json.loads(codex_catalog.read_text())
    descriptor = catalog["models"][0]
    if mutation in ("source_revision", "codex_version", "exporter_sha256", "status"):
        provenance[mutation] = "unreviewed"
    elif mutation == "bytes":
        codex_catalog.write_text(codex_catalog.read_text() + " ")
    else:
        if mutation == "alias":
            descriptor["slug"] = "/checkpoint/path"
        elif mutation == "duplicate":
            catalog["models"].append(copy.deepcopy(descriptor))
        elif mutation == "context":
            descriptor["auto_compact_token_limit"] = 32000
        elif mutation == "prompt":
            descriptor["base_instructions"] = "Changed prompt"
        elif mutation == "tools":
            descriptor["supports_parallel_tool_calls"] = True
        codex_catalog.write_text(json.dumps(catalog))
        # A matching receipt alone cannot conceal changed non-context model behavior.
        provenance["models.json_sha256"] = hashlib.sha256(codex_catalog.read_bytes()).hexdigest()
    provenance_path.write_text(json.dumps(provenance))
    with pytest.raises(ValueError):
        recipe.harness_assets(["codex"], codex_catalog=codex_catalog)


def test_codex_export_is_required_only_when_selected(codex_catalog):
    with pytest.raises(ValueError, match="requires --codex-catalog"):
        recipe.harness_assets(["codex"], codex_catalog=None)
    with pytest.raises(ValueError, match="requires selecting codex"):
        recipe.harness_assets(["opencode"], codex_catalog=codex_catalog)


def test_atom_settings_are_staged_only_when_selected():
    import yaml

    assets = recipe.harness_assets(["nooa"], codex_catalog=None)
    assert set(assets) == {"atom-settings.yaml"}
    assert yaml.safe_load(assets["atom-settings.yaml"]) == {
        "atom": {"summarization": {"policy": "none"}, "context_error_policy": "raise"}
    }
    assert recipe.harness_assets(["opencode", "hermes", "pi"], codex_catalog=None) == {}


@pytest.mark.parametrize("route", ["single-agent-turn", "harbor"])
def test_atom_recipe_uses_the_actual_config_and_harness_without_ignored_controls(tmp_path, monkeypatch, route):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    pytest.importorskip("nooa", reason="NOOA server dependencies are not installed in the core environment")
    import yaml
    from nooa_atom.gym import AtomHarness
    from nooa_atom.workspace.options import AtomOptions

    from responses_api_agents.nooa_agent.config import NOOAAgentConfig, validate_invocation

    config = OmegaConf.load(recipe.ROOT / "examples/training/swe_single_agent_turn.yaml")
    recipe.configure_gym(
        config,
        harnesses=["nooa"],
        route=route,
        output=tmp_path,
        prepared=SimpleNamespace(tasks=[], folder=tmp_path / "tasks"),
    )
    gym = OmegaConf.merge(
        config.env.nemo_gym,
        {
            "policy_model_name": "test-model",
            "policy_base_url": "http://127.0.0.1:9000/v1",
            "policy_api_key": "unused",
            "sandbox": {"opensandbox": {"connection": {"domain": "https://example.invalid", "api_key": "unused"}}},
        },
    )
    resolved = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=gym, skip_load_from_cli=True, skip_load_from_dotenv=True, offline=True
        )
    )
    data = OmegaConf.to_container(resolved.nooa_agent.responses_api_agents.nooa_agent, resolve=True)
    # Startup adds the registered name. Keep every emitted adapter field for strict validation.
    parsed = NOOAAgentConfig.model_validate(data | {"name": "nooa_agent"})
    agent_class, adapter = validate_invocation(parsed.nooa)
    assert agent_class is AtomHarness and adapter.__name__ == "invoke_atom"
    assert parsed.context_window == 32768 and parsed.sandbox_runner_timeout_seconds == 480
    assert parsed.settings_file == tmp_path / "atom-settings.yaml"
    assert parsed.max_policy_calls is None and parsed.nooa.execution_mode == "sandboxed"
    assert config.policy.generation.max_new_tokens == 1024
    settings = yaml.safe_load(recipe.harness_assets(["nooa"], codex_catalog=None)["atom-settings.yaml"])
    options = AtomOptions.model_validate(settings["atom"])
    assert options.summarization.policy == "none" and options.context_error_policy == "raise"


def test_task_order_and_verifier_payload_are_preserved_without_prompt_leakage():
    rows = [
        copy.deepcopy(NEXT_ROW) | {"instance_id": f"task-{index}", "harbor_task_id": f"prepared-{index}"}
        for index in range(16)
    ]
    original = copy.deepcopy(rows)
    harnesses = ["opencode", "hermes", "pi"]
    records = recipe.materialize_training_rows(rows, harnesses=harnesses, route="single-agent-turn")
    assert len(records) == 48
    assert [r["task_id"] for r in records] == [
        {"taskset": f"swe_next_{harness}", "task_id": row["instance_id"]} for row in rows for harness in harnesses
    ]
    for record, row in zip(records, (row for row in rows for _ in harnesses)):
        payload = record["task_input"]
        assert payload["responses_create_params"] == row["responses_create_params"]
        assert payload["task_data"] == {key: value for key, value in row.items() if key != "responses_create_params"}
        assert row["patch"] not in json.dumps(payload["responses_create_params"])
    assert rows == original


def test_old_run_uses_the_original_flat_agent_contract():
    row = copy.deepcopy(NEXT_ROW)
    records = recipe.materialize_training_rows([row], harnesses=["opencode"], route="old-run")
    assert records == [
        row
        | {
            "agent_ref": {"type": "responses_api_agents", "name": "opencode_sandboxed_agent"},
            "task_source": "swe_next_opencode",
        }
    ]
    assert "task_input" not in records[0]


@pytest.mark.parametrize("route", ["old-run", "single-agent-turn", "harbor"])
def test_sandbox_credentials_resolve_only_inside_gym(monkeypatch, tmp_path, route):
    sandbox_key = "synthetic-swe-sandbox-key"
    sandbox_domain = "https://sandbox.example.invalid"
    monkeypatch.setenv("OPENSANDBOX_API_KEY", sandbox_key)
    monkeypatch.setenv("OPENSANDBOX_DOMAIN", sandbox_domain)
    config = OmegaConf.load(recipe.ROOT / "examples/training/swe_single_agent_turn.yaml")
    recipe.configure_gym(
        config,
        harnesses=["opencode"],
        route=route,
        output=tmp_path,
        prepared=SimpleNamespace(tasks=[], folder=tmp_path / "tasks"),
    )
    # RL resolves and prints this mapping before Gym loads its provider config.
    rl_env = OmegaConf.to_container(config.env, resolve=True)
    assert sandbox_key not in pprint.pformat(rl_env)
    assert sandbox_domain not in pprint.pformat(rl_env)
    gym = OmegaConf.merge(
        rl_env["nemo_gym"],
        {"policy_model_name": "test-model", "policy_base_url": "http://127.0.0.1:9000/v1", "policy_api_key": "unused"},
    )
    resolved = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=gym, skip_load_from_cli=True, skip_load_from_dotenv=True, offline=True
        )
    )
    assert resolved.sandbox.opensandbox.connection.api_key == sandbox_key
    assert resolved.sandbox.opensandbox.connection.domain == sandbox_domain


@pytest.mark.parametrize(
    "route,harnesses",
    [(route, list(recipe.OLD_AGENTS)) for route in ("old-run", "single-agent-turn", "harbor")]
    + [(route, list(recipe.HARNESSES)) for route in ("single-agent-turn", "harbor")],
)
def test_generated_gym_configuration_resolves_offline(monkeypatch, tmp_path, route, harnesses):
    monkeypatch.delenv(ALLOW_UNSUPPORTED_PAIRING_ENV_VAR_NAME, raising=False)
    config = OmegaConf.load(recipe.ROOT / "examples/training/swe_single_agent_turn.yaml")
    config.policy.model_name = "test-model"
    recipe.configure_gym(
        config,
        harnesses=harnesses,
        route=route,
        output=tmp_path,
        prepared=SimpleNamespace(tasks=[], folder=tmp_path / "tasks"),
    )
    gym = OmegaConf.merge(
        OmegaConf.to_container(config.env.nemo_gym, resolve=True),
        {
            "policy_model_name": "test-model",
            "policy_base_url": "http://127.0.0.1:9000/v1",
            "policy_api_key": "unused",
            "sandbox": {"opensandbox": {"connection": {"domain": "https://example.invalid", "api_key": "unused"}}},
        },
    )
    resolved = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=gym,
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    assert resolved.environment_routing_mode == ("agent" if route == "old-run" else "taskset")
    assert resolved[ALLOW_UNSUPPORTED_PAIRING_KEY_NAME] is True
    if route != "harbor":
        assert resolved.swe_resources.resources_servers.swe_next.verification_log_dir == str(tmp_path / "verifier")
        assert resolved.swe_resources.resources_servers.swe_next.allowed_agents == ["opencode_sandboxed_agent"]
    if route == "old-run":
        for implementation in recipe.OLD_AGENTS.values():
            assert resolved[implementation].responses_api_agents[implementation].sandbox_provider == "sandbox"
        agent = resolved.opencode_sandboxed_agent.responses_api_agents.opencode_sandboxed_agent
        assert agent.opencode_env.OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX == "1024"
        model = agent.opencode_config.provider.nemo_gym.models.dummy_model
        assert model.limit.output == 1024 and model.interleaved.field == "reasoning_content"
    for harness in harnesses:
        name = f"swe_next_{harness}_environment"
        impl = {"old-run": "legacy_agent", "single-agent-turn": "single_agent_turn", "harbor": "harbor"}[route]
        env = resolved[name].environment_servers[impl]
        assert env.default_episode_timeout_seconds == 1200
        agent_name = recipe.OLD_AGENTS[harness] if route == "old-run" else f"{harness}_agent"
        assert env.agent_server.name == agent_name
        agent = resolved[agent_name].responses_api_agents[agent_name]
        assert agent.token_id_capture and agent.num_workers == 1
        assert agent.get("datasets") == (None if harness == "nooa" else [])
        assert agent.resources_server == (
            {"type": "resources_servers", "name": "swe_resources"} if route == "old-run" else None
        )
        if harness == "opencode":
            assert agent.concurrency == 8
            assert agent.report_agent_sandbox_oom is True
            if route == "old-run":
                assert agent.sandbox_timeout == 480
            else:
                assert agent.sandbox_install_timeout_seconds == 480 and agent.timeout == 480
        elif harness == "claude_code":
            assert agent.claude_code_version == "2.1.108" and agent.timeout == 480
            assert agent.model_server.name == "policy_model" and agent.anthropic_api_key == ""
            assert agent.model == "claude-sonnet-4-6" and agent.bare and agent.thinking == "enabled"
            assert agent.settings == str(tmp_path / "claude-settings.json")
        elif harness == "codex":
            assert agent.codex_version == "0.144.4" and agent.timeout == 480
            assert agent.model == "gym-policy-model" and agent.model_server.name == "policy_model"
            assert agent.model_context_window is None and agent.model_auto_compact_token_limit is None
            assert agent.model_catalog_json == str(tmp_path / "codex/models.json")
            assert agent.features == {"multi_agent": False, "code_mode": False, "token_budget": False}
        elif harness == "hermes":
            assert agent.context_length == 32768 and agent.max_tokens == 1024 and not agent.compression_enabled
            assert agent.max_turns == 30 and agent.terminal_timeout == 60 and agent.concurrency == 32
            if route == "old-run":
                assert agent.runtime_python is None and agent.sandbox_timeout == 480
                assert agent.chat_template_kwargs == {"truncate_history_thinking": False}
                assert agent.model == "test-model"
        elif harness == "pi":
            assert agent.pi_version == "0.80.2" and agent.timeout == 480
            assert agent.context_window == 32768 and agent.max_output_tokens == 1024
            assert agent.output_token_policy == "fixed" and not agent.auto_compaction
            if route == "old-run":
                assert not agent.execution_failure_reward_zero and agent.network_access == "inherit"
                assert agent.model_server.name == "policy_model" and agent.model == "test-model"
                assert not any("pi_sandboxed_agent/configs" in path for path in resolved.config_paths)
        elif harness == "nooa":
            assert agent.nooa == {
                "agent_class": "nooa_atom.gym:AtomHarness",
                "invocation_adapter": "responses_api_agents.nooa_agent.atom_adapter:invoke_atom",
                "execution_mode": "sandboxed",
                "init_kwargs": {"turn_method": "handle_batch"},
            }
            assert agent.context_window == 32768 and agent.sandbox_runner_timeout_seconds == 480
            assert agent.max_policy_calls is None and agent.settings_file == str(tmp_path / "atom-settings.yaml")
            assert agent.model_server.name == "policy_model"
            assert "nooa_environment_server" not in resolved and "nooa_atom_agent" not in resolved
            assert not any(key in agent for key in ("model", "concurrency", "max_output_tokens"))
        if route == "harbor":
            assert env.verifier_completion_key == "evaluation_completed"
            assert env.inconclusive_verification_retries == 1
            assert env.sandbox_resources_override.memory_mib == 30720
        # Run the actual dispatch admission check, beyond resolving the composed YAML.
        row = copy.deepcopy(NEXT_ROW)
        prepared = None
        if route == "harbor":
            task = SimpleNamespace(
                task_id=row["harbor_task_id"],
                task_data={"instance_id": row["instance_id"]},
                digest="a" * 64,
                instruction=row["responses_create_params"]["input"][0]["content"],
                turns=[],
                config=SimpleNamespace(agent=SimpleNamespace(timeout_sec=480)),
            )
            prepared = SimpleNamespace(tasks=[task])
        records = recipe.materialize_training_rows([row], harnesses=[harness], route=route, prepared=prepared)
        if route == "old-run":
            validate = RolloutCollectionHelper._validate_agent_pairings
        else:
            records[0][NG_ENVIRONMENT_SERVER_KEY] = resolved.environment_server_routes[f"swe_next_{harness}"]
            validate = RolloutCollectionHelper._validate_environment_servers
        validate(records, resolved)
        without_override = copy.deepcopy(resolved)
        without_override[ALLOW_UNSUPPORTED_PAIRING_KEY_NAME] = False
        if route == "harbor" or (route == "old-run" and harness == "opencode"):
            validate(records, without_override)
        else:
            with pytest.raises(ValueError, match="accepts only: opencode_sandboxed_agent"):
                validate(records, without_override)
        if route == "harbor":
            # The supplied task fixture declares no allowlist. A typed dataset declaration, when present,
            # is checked by the same guard even though Harbor has no resources_server binding.
            without_override[agent_name].responses_api_agents[agent_name].datasets = [
                {
                    "name": f"swe_next_{harness}",
                    "taskset": f"swe_next_{harness}",
                    "type": "benchmark",
                    "verified": False,
                    "jsonl_fpath": "tasks.jsonl",
                    "allowed_agents": ["opencode_sandboxed_agent"],
                }
            ]
            with pytest.raises(ValueError, match="admits only: opencode_sandboxed_agent"):
                validate(records, without_override)
            without_override[ALLOW_UNSUPPORTED_PAIRING_KEY_NAME] = True
            validate(records, without_override)
        if route != "old-run":
            mismatched = copy.deepcopy(records)
            mismatched[0]["agent_ref"] = {"name": "unbound_agent"}
            with pytest.raises(ValueError, match="is not an agent of environment server"):
                validate(mismatched, resolved)
