# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recipe contract checks without a model download, sandbox, Slurm job or training worker."""

import copy
import hashlib
import json
import os
import pprint
import subprocess
import sys
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf
from scripts import check_swe_training_recipe as check
from scripts import prepare_swe_single_agent_turn_recipe as recipe

from nemo_gym.config.global_config import (
    ALLOW_UNSUPPORTED_PAIRING_ENV_VAR_NAME,
    ALLOW_UNSUPPORTED_PAIRING_KEY_NAME,
    GlobalConfigDictParser,
    GlobalConfigDictParserConfig,
)
from nemo_gym.rollouts.rollout_collection import NG_ENVIRONMENT_SERVER_KEY, RolloutCollectionHelper
from tests.unit_tests.tooling.test_swe_rows_to_harbor import REBENCH_ROW, parser_cache  # noqa: F401


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
        f"{name}_sha256": check.file_sha256(directory / name) for name in ("models.json", "fallback.json")
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
        provenance["models.json_sha256"] = check.file_sha256(codex_catalog)
    provenance_path.write_text(json.dumps(provenance))
    with pytest.raises(ValueError):
        recipe.harness_assets(["codex"], codex_catalog=codex_catalog)


def test_codex_export_is_required_only_when_selected(codex_catalog):
    with pytest.raises(ValueError, match="requires --codex-catalog"):
        recipe.harness_assets(["codex"], codex_catalog=None)
    with pytest.raises(ValueError, match="requires selecting codex"):
        recipe.harness_assets(["opencode"], codex_catalog=codex_catalog)


def test_atom_rejects_old_run_before_reading_sources_or_preparing_assets(tmp_path, monkeypatch):
    monkeypatch.setattr(recipe, "clean_revision", lambda _: pytest.fail("source read before route validation"))
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="Atom .* has no old-run route"):
        recipe.prepare(
            SimpleNamespace(rl_root=tmp_path, output_dir=output, harnesses=list(recipe.HARNESSES), route="old-run")
        )
    assert not output.exists()
    with pytest.raises(ValueError, match="has no old-run route"):
        recipe.materialize_training_rows([REBENCH_ROW], harnesses=["nooa"], route="old-run")


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
        parser_cache=tmp_path / "parsers",
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
    rows = [copy.deepcopy(REBENCH_ROW) | {"instance_id": f"task-{index}"} for index in range(16)]
    original = copy.deepcopy(rows)
    harnesses = ["opencode", "hermes", "pi"]
    records = recipe.materialize_training_rows(rows, harnesses=harnesses, route="single-agent-turn")
    assert len(records) == 48
    assert [r["task_id"] for r in records] == [
        {"taskset": f"swe_rebench_{harness}", "task_id": row["instance_id"]} for row in rows for harness in harnesses
    ]
    for record, row in zip(records, (row for row in rows for _ in harnesses)):
        payload = record["task_input"]
        assert payload["responses_create_params"] == row["responses_create_params"]
        assert payload["task_data"] == {key: value for key, value in row.items() if key != "responses_create_params"}
        assert row["patch"] not in json.dumps(payload["responses_create_params"])
    assert rows == original


def test_old_run_uses_the_original_flat_agent_contract():
    row = copy.deepcopy(REBENCH_ROW)
    records = recipe.materialize_training_rows([row], harnesses=["opencode"], route="old-run")
    assert records == [
        row
        | {
            "agent_ref": {"type": "responses_api_agents", "name": "opencode_sandboxed_agent"},
            "task_source": "swe_rebench_opencode",
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
        parser_cache=tmp_path / "parsers",
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
        parser_cache=tmp_path / "parsers",
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
        assert resolved.swe_resources.resources_servers.swe_rebench.verification_log_dir == str(tmp_path / "verifier")
        assert resolved.swe_resources.resources_servers.swe_rebench.allowed_agents == ["opencode_sandboxed_agent"]
    if route == "old-run":
        for implementation in recipe.OLD_AGENTS.values():
            assert resolved[implementation].responses_api_agents[implementation].sandbox_provider == "sandbox"
        agent = resolved.opencode_sandboxed_agent.responses_api_agents.opencode_sandboxed_agent
        assert agent.opencode_env.OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX == "1024"
        model = agent.opencode_config.provider.nemo_gym.models.dummy_model
        assert model.limit.output == 1024 and model.interleaved.field == "reasoning_content"
    for harness in harnesses:
        name = f"swe_rebench_{harness}_environment"
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
        row = copy.deepcopy(REBENCH_ROW)
        prepared = None
        if route == "harbor":
            task = SimpleNamespace(
                task_id=recipe.task_folder_name("swe_rebench", row["instance_id"]),
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
            records[0][NG_ENVIRONMENT_SERVER_KEY] = resolved.environment_server_routes[f"swe_rebench_{harness}"]
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
            # Converted tasks currently declare no allowlist. A typed dataset declaration, when present,
            # is checked by the same guard even though Harbor has no resources_server binding.
            without_override[agent_name].responses_api_agents[agent_name].datasets = [
                {
                    "name": f"swe_rebench_{harness}",
                    "taskset": f"swe_rebench_{harness}",
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


@pytest.fixture
def assets(tmp_path, parser_cache):
    prepared, root, model = (tmp_path / name for name in ("prepared", "source", "a" * 40))
    for directory in (prepared, root, model):
        directory.mkdir()
    (prepared / "config.yaml").write_text("seed: 42\n")
    (root / "capture.mjs").write_text("export const capture = true;\n")
    (model / "model.safetensors").write_bytes(b"fake weight bytes, never loaded")
    config = b'{"model_type":"test"}'
    (model / "config.json").write_bytes(config)
    parser_root = parser_cache / "SWE-rebench-V2"
    revision = subprocess.check_output(["git", "-C", str(parser_root), "rev-parse", "HEAD"], text=True).strip()
    manifest = {
        "files_sha256": {"config.yaml": check.file_sha256(prepared / "config.yaml")},
        "runtime_files_sha256": {"gym/capture.mjs": check.file_sha256(root / "capture.mjs")},
        "model": {
            "path": str(model),
            "revision": model.name,
            "files": {
                "model.safetensors": {"algorithm": "sha256", "digest": check.file_sha256(model / "model.safetensors")},
                "config.json": {
                    "algorithm": "git_blob_sha1",
                    "digest": hashlib.sha1(b"blob 21\0" + config).hexdigest(),
                },
            },
        },
        "parser": {"path": str(parser_root), "commit": revision},
    }
    return prepared, root, model, manifest


@pytest.mark.parametrize("mutation", [None, "config", "helper", "weights", "model-config", "missing-weights"])
def test_preflight_checks_source_helpers_and_actual_model_bytes(assets, mutation):
    prepared, root, model, manifest = assets
    if mutation == "config":
        (prepared / "config.yaml").write_text("seed: 0\n")
    elif mutation == "helper":
        (root / "capture.mjs").write_text("export const capture = false;\n")
    elif mutation == "weights":
        (model / "model.safetensors").write_bytes(b"different bytes")
    elif mutation == "model-config":
        (model / "config.json").write_text("{}")
    elif mutation == "missing-weights":
        (model / "model.safetensors").unlink()
    if mutation is None:
        check.check_files(prepared, {"gym": root}, manifest)
    else:
        with pytest.raises(ValueError):
            check.check_files(prepared, {"gym": root}, manifest)


def test_model_provenance_records_hub_hashes_without_downloading(monkeypatch, tmp_path):
    info = SimpleNamespace(
        sha="a" * 40,
        siblings=[
            SimpleNamespace(rfilename="config.json", lfs=None, blob_id="b" * 40),
            SimpleNamespace(rfilename="model.safetensors", lfs=SimpleNamespace(sha256="c" * 64)),
        ],
    )
    calls = []

    def metadata(repo, **kwargs):
        calls.append((repo, kwargs))
        return info

    monkeypatch.setattr(recipe, "HfApi", lambda: SimpleNamespace(model_info=metadata))
    manifest = recipe.model_manifest("example/model", info.sha, tmp_path)
    assert calls == [("example/model", {"revision": info.sha, "files_metadata": True})]
    assert manifest["files"]["config.json"] == {"algorithm": "git_blob_sha1", "digest": "b" * 40}
    assert manifest["files"]["model.safetensors"] == {"algorithm": "sha256", "digest": "c" * 64}
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("model,expert_parallel_size", [("4b", 1), ("30b", 2)])
@pytest.mark.parametrize("hardware_profile,gpus,generation_gpus", [("arm64-b300", 4, 2), ("x86_64-h100", 8, 4)])
@pytest.mark.parametrize(
    "steps,count,harnesses,route",
    [
        (1, 8, ["opencode"], "single-agent-turn"),
        (5, 8, ["opencode"], "single-agent-turn"),
        (20, 8, ["opencode"], "single-agent-turn"),
        (20, 32, list(recipe.OLD_AGENTS), "old-run"),
        (20, 32, list(recipe.OLD_AGENTS), "single-agent-turn"),
        (20, 32, list(recipe.OLD_AGENTS), "harbor"),
        (20, 32, list(recipe.HARNESSES), "single-agent-turn"),
        (20, 32, list(recipe.HARNESSES), "harbor"),
    ],
)
def test_preparation_materializes_the_requested_update_budget(
    monkeypatch,
    tmp_path,
    codex_catalog,
    model,
    expert_parallel_size,
    steps,
    count,
    harnesses,
    route,
    hardware_profile,
    gpus,
    generation_gpus,
):
    rl_root = tmp_path / "rl"
    rl_root.mkdir()
    (rl_root / "uv.lock").write_text("# test lock\n")
    roster = tmp_path / "roster.json"
    roster.write_text(json.dumps({"source_uncompressed_sha256": "source-fixture"}))
    rows = [copy.deepcopy(REBENCH_ROW) | {"instance_id": f"task-{index}"} for index in range(count)]
    revision = "a" * 40
    # Stub provenance/network reads; exercise the real materializer, route configuration and budget calculation.
    monkeypatch.setattr(
        recipe, "clean_revision", lambda path: recipe.PARSER_REVISION if path.name == "SWE-rebench-V2" else revision
    )
    monkeypatch.setattr(
        recipe.subprocess, "check_output", lambda command, **kwargs: "" if "ls-files" in command else revision
    )
    monkeypatch.setattr(recipe, "select_rows", lambda *args, **kwargs: (rows, [row["instance_id"] for row in rows]))
    monkeypatch.setattr(recipe, "model_manifest", lambda repo, commit, path: {"revision": commit, "files": {}})
    monkeypatch.setattr(
        recipe,
        "load_rl_config",
        lambda root: OmegaConf.load(recipe.ROOT / "examples/training/swe_single_agent_turn.yaml"),
    )
    if route == "harbor":
        # Image/registry preparation is covered separately; exercise the real Harbor row materializer here.
        folder = tmp_path / "harbor-tasks"
        tasks = [
            SimpleNamespace(
                task_id=recipe.task_folder_name("swe_rebench", row["instance_id"]),
                path=folder / recipe.task_folder_name("swe_rebench", row["instance_id"]),
                task_data={"instance_id": row["instance_id"]},
                digest="a" * 64,
                instruction=row["responses_create_params"]["input"][0]["content"],
                turns=[],
                config=SimpleNamespace(agent=SimpleNamespace(timeout_sec=480)),
            )
            for row in rows
        ]
        monkeypatch.setattr(recipe, "convert", lambda *args, **kwargs: None)
        monkeypatch.setattr(
            recipe, "prepare_target", lambda *args, **kwargs: SimpleNamespace(tasks=tasks, folder=folder, skipped=[])
        )
    args = SimpleNamespace(
        rl_root=rl_root,
        rl_revision=revision,
        parser_cache=tmp_path / "parsers",
        source=tmp_path / "source",
        roster=roster,
        image_pins=roster,
        route=route,
        harnesses=harnesses,
        model=model,
        model_path=tmp_path / recipe.MODELS[model][1],
        output_dir=tmp_path / "prepared",
        count=count,
        steps=steps,
        wandb_project=None,
        codex_catalog=codex_catalog if "codex" in harnesses else None,
        hardware_profile=hardware_profile,
    )
    manifest = recipe.prepare(args)
    config = OmegaConf.load(args.output_dir / "config.yaml")
    # The RL driver resolves the whole prepared config before Gym injects any aliases.
    resolved_config = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    if route == "old-run":
        for harness in set(harnesses) & {"hermes", "pi"}:
            name = recipe.OLD_AGENTS[harness]
            agent = resolved_config["env"]["nemo_gym"][name]["responses_api_agents"][name]
            assert agent["model"] == resolved_config["policy"]["model_name"] == str(args.model_path)
    train = [json.loads(line) for line in (args.output_dir / "train.jsonl").read_text().splitlines()]
    assert manifest["hardware_profile"] == hardware_profile
    assert config.cluster.num_nodes == 1 and config.cluster.gpus_per_node == gpus
    assert config.policy.generation.colocated.resources.gpus_per_node == generation_gpus
    assert config.policy.megatron_cfg.tensor_model_parallel_size == 2
    assert config.policy.generation.vllm_cfg.tensor_parallel_size == 2
    width = len(harnesses)
    assert len(train) == count * width and manifest["episodes"] == steps * 32 * width
    assert config.grpo.max_num_steps == steps and config.grpo.max_num_epochs == steps * 8 // count
    assert config.grpo.num_prompts_per_step == 8 * width and config.grpo.num_generations_per_prompt == 4
    assert config.policy.train_global_batch_size == 32 * width
    assert config.async_rl.max_inflight_prompts == (4 if steps in (1, 5) else 8 if width == 1 else 16)
    expected_ids = [(row["instance_id"], f"swe_rebench_{harness}") for row in rows for harness in harnesses]
    if route == "old-run":
        assert [(row["instance_id"], row["task_source"]) for row in train] == expected_ids
        assert [row["agent_ref"]["name"] for row in train] == [recipe.OLD_AGENTS[name] for name in harnesses] * count
    else:
        assert [
            (row["task_input"]["task_data"]["instance_id"], row["task_id"]["taskset"]) for row in train
        ] == expected_ids
    if width in (5, 6):
        assert count == 32 and steps == 20 and config.grpo.max_num_epochs == 5
        assert len(train) == 32 * width and config.policy.train_global_batch_size == 32 * width
        for offset in range(0, len(train), 8 * width):
            assert expected_ids[offset : offset + 8 * width] == [
                (f"task-{task}", f"swe_rebench_{harness}")
                for task in range(offset // width, offset // width + 8)
                for harness in harnesses
            ]
        for name in ("models.json", "fallback.json", "provenance.json"):
            staged = args.output_dir / "codex" / name
            assert staged.read_bytes() == codex_catalog.with_name(name).read_bytes()
            assert manifest["files_sha256"][f"codex/{name}"] == check.file_sha256(staged)
        settings = args.output_dir / "claude-settings.json"
        assert json.loads(settings.read_text()) == {
            "env": {
                "DISABLE_COMPACT": "1",
                "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "32768",
                "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "1024",
            }
        }
        assert manifest["files_sha256"][settings.name] == check.file_sha256(settings)
        if width == 6:
            assert config.grpo.num_prompts_per_step == 48 and manifest["episodes"] == 3840
            assert config.async_rl.max_inflight_prompts == 16
            atom = args.output_dir / "atom-settings.yaml"
            assert OmegaConf.load(atom).atom == {"summarization": {"policy": "none"}, "context_error_policy": "raise"}
            assert manifest["files_sha256"][atom.name] == check.file_sha256(atom)
    assert config.checkpointing.save_period == (1 if steps == 1 else 10)
    assert (
        config.checkpointing.save_optimizer
        and config.policy.megatron_cfg.expert_model_parallel_size == expert_parallel_size
    )
    assert not config.grpo.val_at_start and not config.grpo.val_at_end and config.grpo.val_period == 0
    assert config.data.validation is None and not config.data.shuffle and not config.data.use_multiple_dataloader
    assert config.async_rl.sampler == {"name": "in_order", "max_lookahead_versions": 0}
    assert config.async_rl.min_groups_for_streaming_train == config.async_rl.max_buffered_rollouts == 8 * width
    assert config.async_rl.rollout_failure.min_step_batch_fraction == 1
    assert config.async_rl.rollout_failure.max_consecutive_dropped_prompts == 0
    assert config.async_rl.rollout_failure.max_skipped_prompts == 0
    assert not config.grpo.use_dynamic_sampling and config.grpo.masked_reward_policy == "include"
    assert config.env.should_mask_flagged_samples and config.grpo.overlong_filtering
    assert config.policy.max_total_sequence_length == config.policy.generation.vllm_cfg.max_model_len == 32768
    assert config.policy.generation.max_new_tokens == 1024
    assert config.policy.generation.temperature == config.policy.generation.top_p == 1
    assert config.env.nemo_gym.policy_model.responses_api_models.vllm_model.chat_template_kwargs == {
        "enable_thinking": True,
        "truncate_history_thinking": False,
    }


@pytest.mark.parametrize(
    "name",
    [
        "claude-settings.json",
        "codex/models.json",
        "codex/fallback.json",
        "codex/provenance.json",
        "atom-settings.yaml",
    ],
)
def test_preflight_covers_staged_harness_assets(assets, codex_catalog, name):
    prepared, root, _, manifest = assets
    for relative, data in recipe.harness_assets(list(recipe.HARNESSES), codex_catalog=codex_catalog).items():
        path = prepared / relative
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(data)
        manifest["files_sha256"][relative] = check.file_sha256(path)
    check.check_files(prepared, {"gym": root}, manifest)
    (prepared / name).write_text("changed")
    with pytest.raises(ValueError, match="Prepared file changed"):
        check.check_files(prepared, {"gym": root}, manifest)


@pytest.mark.parametrize("gpus", [4, 8])
@pytest.mark.parametrize("dry_run", [True, False])
@pytest.mark.parametrize("preflight_ok", [True, False])
def test_launcher_uses_checked_allocation_and_stops_before_submission_on_failure(
    tmp_path, gpus, dry_run, preflight_ok
):
    # Bypass preflight only in this command-construction test; real launches always run it.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "uv").write_text(f"#!/bin/sh\nprintf '{gpus}\\n'\n" if preflight_ok else "#!/bin/sh\nexit 72\n")
    (bin_dir / "uv").chmod(0o755)
    receipt = tmp_path / "submission-receipt"
    (bin_dir / "sbatch").write_text('#!/bin/sh\nprintf "%s\\n" "$GPUS_PER_NODE" "$@" > "$SUBMISSION_RECEIPT"\n')
    (bin_dir / "sbatch").chmod(0o755)
    env = os.environ | {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "NEMO_RL_ROOT": str(tmp_path / "rl"),
        "CONFIG_PATH": str(tmp_path / "config.yaml"),
        "CONTAINER": str(tmp_path / "image.sqsh"),
        "CONTAINER_PROVENANCE": str(tmp_path / "image.json"),
        "SHARED_ROOT": "/",
        "SLURM_ACCOUNT": "test-account",
        "SLURM_PARTITION": "test-partition",
        "OPENSANDBOX_DOMAIN": "https://example.invalid",
        "OPENSANDBOX_API_KEY": "not-a-real-key",
        "SUBMISSION_RECEIPT": str(receipt),
    }
    result = subprocess.run(
        [
            "bash",
            str(recipe.ROOT / "scripts/launch_swe_single_agent_turn_training.sh"),
            *(["--dry-run"] if dry_run else []),
        ],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if not preflight_ok:
        assert result.returncode == 72 and not receipt.exists()
        assert "Submit:" not in result.stdout
        return
    assert result.returncode == 0, result.stderr
    assert "--nodes=1" in result.stdout and "--exclusive" in result.stdout
    assert f"--gres=gpu:{gpus}" in result.stdout
    assert "run_grpo_single_controller.py" in result.stdout
    assert "check_swe_training_recipe.py" in result.stdout
    assert "not-a-real-key" not in result.stdout
    if dry_run:
        assert not receipt.exists()
    else:
        submitted = receipt.read_text().splitlines()
        assert submitted[0] == str(gpus) and f"--gres=gpu:{gpus}" in submitted


@pytest.fixture
def hardware_assets(assets):
    prepared, root, _, manifest = assets
    manifest.update({key: "a" * 64 for key in ("gym_commit", "rl_commit", "gym_lock_sha256", "rl_lock_sha256")})
    return prepared, root, manifest


def write_hardware_assets(prepared, manifest, profile_name):
    if profile_name is not None:
        manifest["hardware_profile"] = profile_name
    profile = check.HARDWARE_PROFILES[profile_name or check.DEFAULT_HARDWARE_PROFILE]
    config = OmegaConf.load(recipe.ROOT / "examples/training/swe_single_agent_turn.yaml")
    config.cluster.gpus_per_node = profile.gpus_per_node
    config.policy.generation.colocated.resources.gpus_per_node = profile.generation_gpus
    OmegaConf.save(config, prepared / "config.yaml")
    manifest["files_sha256"]["config.yaml"] = check.file_sha256(prepared / "config.yaml")
    (prepared / "provenance.json").write_text(json.dumps(manifest))


@pytest.mark.parametrize(
    "profile_name,machine,names,valid",
    [
        (None, "aarch64", ["NVIDIA B300"] * 4, True),
        (None, "aarch64", ["NVIDIA GB300"] * 4, True),
        ("arm64-b300", "aarch64", ["NVIDIA B300"] * 4, True),
        ("arm64-b300", "aarch64", ["NVIDIA GB300"] * 4, True),
        ("arm64-b300", "aarch64", ["NVIDIA GB200"] * 4, False),
        ("arm64-b300", "x86_64", ["NVIDIA H100 80GB HBM3"] * 8, False),
        ("x86_64-h100", "x86_64", ["NVIDIA H100 80GB HBM3"] * 8, True),
        ("x86_64-h100", "aarch64", ["NVIDIA H100 80GB HBM3"] * 8, False),
        ("x86_64-h100", "x86_64", ["NVIDIA H100 80GB HBM3"] * 4, False),
        ("x86_64-h100", "x86_64", ["NVIDIA H100 80GB HBM3"] * 7 + ["NVIDIA H200"], False),
        ("x86_64-h100", "x86_64", ["NVIDIA H1000"] * 8, False),
    ],
)
def test_runtime_enforces_prepared_hardware_and_records_observations(
    hardware_assets, monkeypatch, profile_name, machine, names, valid
):
    prepared, root, manifest = hardware_assets
    write_hardware_assets(prepared, manifest, profile_name)
    monkeypatch.setattr(
        sys,
        "argv",
        ["check", "--prepared", str(prepared), "--rl-root", str(root), "--gym-root", str(root), "--runtime"],
    )
    monkeypatch.setattr(check.platform, "machine", lambda: machine)
    for label in ("gym", "rl"):
        monkeypatch.setitem(sys.modules, f"nemo_{label}", SimpleNamespace(__file__=str(root / "__init__.py")))
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            __version__="test-only",
            cuda=SimpleNamespace(device_count=lambda: len(names), get_device_name=names.__getitem__),
        ),
    )
    if not valid:
        with pytest.raises(ValueError, match="Hardware profile"):
            check.main()
        assert not (prepared / "runtime-check.json").exists()
        return
    check.main()
    receipt = json.loads((prepared / "runtime-check.json").read_text())
    assert receipt["hardware_profile"] == (profile_name or "arm64-b300")
    assert receipt["machine"] == machine and receipt["gpu_names"] == names
    assert receipt["model_content_checked_against_hub"] == manifest["model"]["revision"]


@pytest.mark.parametrize(
    "profile_name,image_platform,mutation,expected_gpus",
    [
        (None, "linux/arm64", None, 4),
        ("arm64-b300", "linux/arm64", None, 4),
        ("x86_64-h100", "linux/amd64", None, 8),
        ("x86_64-h100", "linux/arm64", None, None),
        ("arm64-b300", "linux/amd64", None, None),
        ("x86_64-h100", "linux/amd64", "mutable-image", None),
        ("x86_64-h100", "linux/amd64", "container-bytes", None),
        ("x86_64-h100", "linux/amd64", "config-split", None),
        ("x86_64-h100", "linux/amd64", "unknown-profile", None),
        ("arm64-b300", "linux/arm64", "derived", 4),
        ("arm64-b300", "linux/arm64", "derived-mutable-base", None),
        ("arm64-b300", "linux/arm64", "derived-ambiguous", None),
        ("arm64-b300", "linux/arm64", "derived-missing-audit", None),
        ("arm64-b300", "linux/arm64", "derived-container-bytes", None),
        ("arm64-b300", "linux/amd64", "derived", None),
        ("arm64-b300", "linux/arm64", "unknown-format", None),
        ("arm64-b300", "linux/arm64", "oci-ambiguous", None),
    ],
)
def test_host_checks_profile_image_and_config_before_reporting_allocation(
    hardware_assets, monkeypatch, capsys, profile_name, image_platform, mutation, expected_gpus
):
    prepared, root, manifest = hardware_assets
    write_hardware_assets(prepared, manifest, profile_name)
    container = prepared / "image.sqsh"
    container.write_bytes(b"test container bytes")
    image = {key: manifest[key] for key in ("gym_commit", "rl_commit", "gym_lock_sha256", "rl_lock_sha256")}
    image.update(
        platform=image_platform, image="example/image@sha256:" + "b" * 64, squashfs_sha256=check.file_sha256(container)
    )
    if mutation and mutation.startswith("derived"):
        image["image_format"] = "derived-squashfs"
        image["base_image"] = image.pop("image")
        for key in ("base_squashfs_sha256", "source_archive_sha256", "derivation_sha256", "environment_audit_sha256"):
            image[key] = "c" * 64
        if mutation == "derived-mutable-base":
            image["base_image"] = "example/image:latest"
        elif mutation == "derived-ambiguous":
            image["image"] = image["base_image"]
        elif mutation == "derived-missing-audit":
            del image["environment_audit_sha256"]
        elif mutation == "derived-container-bytes":
            container.write_bytes(b"changed")
    elif mutation == "unknown-format":
        image["image_format"] = "unreviewed"
    elif mutation == "oci-ambiguous":
        image["base_image"] = image["image"]
    if mutation == "mutable-image":
        image["image"] = "example/image:latest"
    elif mutation == "container-bytes":
        container.write_bytes(b"changed")
    elif mutation == "config-split":
        config = OmegaConf.load(prepared / "config.yaml")
        config.policy.generation.colocated.resources.gpus_per_node = 2
        OmegaConf.save(config, prepared / "config.yaml")
        manifest["files_sha256"]["config.yaml"] = check.file_sha256(prepared / "config.yaml")
    elif mutation == "unknown-profile":
        manifest["hardware_profile"] = "any-gpu"
    (prepared / "provenance.json").write_text(json.dumps(manifest))
    image_path = prepared / "image.json"
    image_path.write_text(json.dumps(image))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check",
            "--prepared",
            str(prepared),
            "--rl-root",
            str(root),
            "--gym-root",
            str(root),
            "--container",
            str(container),
            "--container-provenance",
            str(image_path),
            "--print-gpus-per-node",
        ],
    )
    if expected_gpus is None:
        with pytest.raises(ValueError):
            check.main()
        assert capsys.readouterr().out == ""
    else:
        check.main()
        assert capsys.readouterr().out == f"{expected_gpus}\n"
