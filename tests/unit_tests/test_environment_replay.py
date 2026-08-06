# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace

import orjson
import pytest
from omegaconf import OmegaConf

import nemo_gym.environment_execution as environment_execution
import nemo_gym.global_config as global_config_module
from nemo_gym.config_types import ConfigError
from nemo_gym.environment_manifest import EnvironmentManifest, dump_manifest
from nemo_gym.environment_replay import infer_replay_paths, replay_environment_rollouts, verifier_only_config
from nemo_gym.global_config import (
    AGENT_REF_KEY_NAME,
    ROLLOUT_INDEX_KEY_NAME,
    TASK_INDEX_KEY_NAME,
    StaticValidationConfigParser,
)
from nemo_gym.path_utils import failures_path_for
from nemo_gym.trajectory_bundle import (
    DEFAULT_TRAJECTORY_IDENTITY_FIELDS,
    CapturedEnvironment,
    FailureReplaySelection,
    bundle_path_for,
    write_trajectory_bundle,
)


@pytest.fixture(autouse=True)
def _component_root_is_test_checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(environment_execution, "component_search_roots", lambda: [tmp_path])
    monkeypatch.setenv("NEMO_GYM_EXTRA_ROOTS", str(tmp_path))
    for group, implementation in (
        ("resources_servers", "scorer"),
        ("responses_api_agents", "simple_agent"),
        ("responses_api_models", "policy_impl"),
        ("responses_api_models", "judge_impl"),
    ):
        source_dir = tmp_path / group / implementation
        source_dir.mkdir(parents=True)
        (source_dir / "app.py").write_text("# fixture entrypoint\n")


def _manifest(tmp_path: Path, **updates) -> tuple[Path, EnvironmentManifest]:
    data = {
        "name": "replay_fixture",
        "version": "1.0.0",
        "kind": "environment",
        "integration_profile": "stock-loop",
        "domain": "other",
        "description": "A verifier-only trajectory replay fixture.",
        "modality": "text",
        "licensing": "Apache-2.0",
        "authors": ["fixture-owner"],
        "reward": {"range": [0, 1], "higher_is_better": True},
        "determinism": "seeded",
        "resources_server": "scorer",
        "agent_server": "simple_agent",
        "model_server": "policy_model",
        "datasets": [{"name": "fixture", "type": "example", "jsonl_fpath": "fixture.jsonl"}],
        **updates,
    }
    manifest = EnvironmentManifest.model_validate(data)
    path = tmp_path / "environments" / "replay_fixture" / "manifest.yaml"
    path.parent.mkdir(parents=True)
    dump_manifest(manifest, path)
    return path, manifest


def _resolved_config():
    return OmegaConf.create(
        {
            "config_paths": ["environments/replay_fixture/config.yaml"],
            "selected_verifier": {
                "resources_servers": {
                    "scorer": {
                        "entrypoint": "app.py",
                        "model_server": {"type": "responses_api_models", "name": "judge_model"},
                        "provides": ["verification"],
                    }
                }
            },
            "agent": {
                "responses_api_agents": {
                    "simple_agent": {
                        "entrypoint": "app.py",
                        "requires": ["verification"],
                        "resources_server": {"type": "resources_servers", "name": "selected_verifier"},
                        "model_server": {"type": "responses_api_models", "name": "policy_model"},
                        "datasets": [
                            {
                                "name": "fixture",
                                "type": "example",
                                "jsonl_fpath": "fixture.jsonl",
                            }
                        ],
                    }
                }
            },
            "policy_model": {
                "responses_api_models": {"policy_impl": {"entrypoint": "app.py", "provides": ["text-model"]}}
            },
            "judge_model": {"responses_api_models": {"judge_impl": {"entrypoint": "app.py"}}},
            "default_host": "127.0.0.1",
        }
    )


def _captured_run(tmp_path: Path, name: str = "run") -> tuple[Path, Path]:
    rollouts = tmp_path / f"{name}.jsonl"
    materialized = tmp_path / f"{name}_materialized_inputs.jsonl"
    identity = {
        TASK_INDEX_KEY_NAME: 0,
        ROLLOUT_INDEX_KEY_NAME: 0,
        AGENT_REF_KEY_NAME: {"name": "simple_agent"},
    }
    rollouts.write_text(orjson.dumps({**identity, "response": {"output": []}}).decode() + "\n")
    materialized.write_text(
        orjson.dumps({**identity, "responses_create_params": {"input": "fixture"}}).decode() + "\n"
    )
    write_trajectory_bundle(
        rollouts_path=rollouts,
        materialized_inputs_path=materialized,
        environment=CapturedEnvironment(
            name="replay_fixture",
            kind="environment",
            version="1.0.0",
            composition_hash="a" * 64,
            integration_profile="stock-loop",
            resources_server="scorer",
        ),
    )
    return rollouts, materialized


def test_replay_paths_infer_materialized_inputs_and_protect_outputs(tmp_path: Path, monkeypatch) -> None:
    rollouts, materialized = _captured_run(tmp_path)

    inferred = infer_replay_paths(rollouts)
    assert inferred.rollouts == rollouts
    assert inferred.materialized_inputs == materialized
    assert inferred.bundle == bundle_path_for(rollouts)
    assert inferred.output == tmp_path / "run_replayed.jsonl"

    monkeypatch.chdir(tmp_path)
    explicit = infer_replay_paths(rollouts, "custom.jsonl")
    assert explicit.output == tmp_path / "custom.jsonl"

    explicit.output.write_text("do not overwrite\n")
    with pytest.raises(ConfigError, match="never overwrites"):
        infer_replay_paths(rollouts, explicit.output)
    explicit.output.unlink()
    failures_path_for(explicit.output).write_text("prior failure\n")
    with pytest.raises(ConfigError, match="never overwrites"):
        infer_replay_paths(rollouts, explicit.output)


def test_replay_paths_require_capture_sibling_and_distinct_output(tmp_path: Path) -> None:
    rollouts = tmp_path / "run.jsonl"
    rollouts.write_text("{}\n")
    with pytest.raises(ConfigError, match="run_materialized_inputs.jsonl"):
        infer_replay_paths(rollouts)

    materialized = tmp_path / "run_materialized_inputs.jsonl"
    materialized.write_text("{}\n")
    with pytest.raises(ConfigError, match="must differ"):
        infer_replay_paths(rollouts, rollouts)


def test_verifier_only_config_excludes_agent_policy_and_unrelated_services(tmp_path: Path) -> None:
    _, manifest = _manifest(tmp_path)
    config = _resolved_config()
    config.selected_verifier.resources_servers.unused_scorer = {
        "entrypoint": "app.py",
        "model_server": {"type": "responses_api_models", "name": "policy_model"},
    }
    config.other_verifier = {"resources_servers": {"other_scorer": {"entrypoint": "app.py"}}}

    filtered, started = verifier_only_config(config, manifest)

    assert started == ("judge_model", "selected_verifier")
    assert "selected_verifier" in filtered
    assert "judge_model" in filtered
    assert "agent" not in filtered
    assert "policy_model" not in filtered
    assert "other_verifier" not in filtered
    assert tuple(filtered.selected_verifier.resources_servers) == ("scorer",)
    assert filtered.default_host == "127.0.0.1"


def test_verifier_only_config_strips_co_located_agent_groups(tmp_path: Path) -> None:
    _, manifest = _manifest(tmp_path)
    config = _resolved_config()
    co_located_agent = {
        "simple_agent": {
            "entrypoint": "app.py",
            "model_server": {"type": "responses_api_models", "name": "policy_model"},
        }
    }
    config.selected_verifier.responses_api_agents = co_located_agent
    config.selected_verifier.shared_setting = "keep-me"
    config.judge_model.responses_api_agents = co_located_agent

    filtered, started = verifier_only_config(config, manifest)

    assert started == ("judge_model", "selected_verifier")
    assert list(filtered.selected_verifier) == ["resources_servers", "shared_setting"]
    assert filtered.selected_verifier.shared_setting == "keep-me"
    assert list(filtered.judge_model) == ["responses_api_models"]
    assert "responses_api_agents" not in filtered.selected_verifier
    assert "responses_api_agents" not in filtered.judge_model
    assert "policy_model" not in filtered


def test_verifier_only_config_refuses_agent_dependency(tmp_path: Path) -> None:
    _, manifest = _manifest(tmp_path)
    config = _resolved_config()
    config.selected_verifier.resources_servers.scorer.agent = {
        "type": "responses_api_agents",
        "name": "agent",
    }

    with pytest.raises(ConfigError, match="refuses to start an agent"):
        verifier_only_config(config, manifest)


def test_run_helper_starts_prevalidated_verifier_projection(tmp_path: Path, monkeypatch) -> None:
    import nemo_gym.cli.env as cli_env

    manifest_path, manifest = _manifest(tmp_path)
    config = _resolved_config()
    config.manifest_path = str(manifest_path)
    filtered, _started = verifier_only_config(config, manifest)

    # A verifier-only projection intentionally no longer matches the complete
    # manifest composition, but every service it will launch must still exist.
    with pytest.raises(ConfigError):
        environment_execution.preflight_manifest_execution(filtered)

    class ReachedRay(RuntimeError):
        pass

    def reached_ray() -> None:
        raise ReachedRay

    monkeypatch.setattr(cli_env, "initialize_ray", reached_ray)
    with pytest.raises(ReachedRay):
        cli_env.RunHelper().start(
            None,
            global_config_dict=filtered,
            preflight_mode="launch-sources-only",
        )


def test_replay_delegates_to_existing_helper_under_filtered_global_config(tmp_path: Path, monkeypatch) -> None:
    import nemo_gym.rollout_reverification as rollout_reverification

    manifest_path, _ = _manifest(tmp_path)
    rollouts, materialized = _captured_run(tmp_path)
    raw = _resolved_config()
    raw.environment_ref = "replay_fixture@1.0.0"
    raw.manifest_path = str(manifest_path)
    raw.replay_rollouts_path = str(rollouts)
    raw.output_jsonl_fpath = str(tmp_path / "rescored.jsonl")
    raw.force = True
    raw.limit = 3
    raw.num_samples_in_parallel = 2
    raw.unrelated_model = {
        "responses_api_models": {"unrelated_impl": {"entrypoint": "app.py"}},
    }
    original_global = OmegaConf.create({"original": True})
    monkeypatch.setattr(global_config_module, "_GLOBAL_CONFIG_DICT", original_global)
    events = []
    captured = {}

    def fingerprint(config, *, resources_server_name):
        captured["fingerprint_config"] = config
        assert resources_server_name == "selected_verifier"
        return "f" * 64

    monkeypatch.setattr(rollout_reverification, "reverification_fingerprint", fingerprint)

    class FakeRunner:
        def start(self, parser_config, *, preflight_mode) -> None:
            assert parser_config is None
            assert preflight_mode == "launch-sources-only"
            assert set(global_config_module._GLOBAL_CONFIG_DICT) >= {"selected_verifier", "judge_model"}
            assert "agent" not in global_config_module._GLOBAL_CONFIG_DICT
            events.append("start")

        def shutdown(self) -> None:
            events.append("shutdown")

    class FakeReverificationHelper:
        async def run_from_config(self, config, *, resources_server_name):
            captured["config"] = config
            captured["resources_server_name"] = resources_server_name
            assert "policy_model" not in global_config_module._GLOBAL_CONFIG_DICT
            events.append("reverify")
            return [{"reward": 1.0}, {"reward": 0.0}]

    result = replay_environment_rollouts(
        raw,
        run_helper_factory=FakeRunner,
        reverification_helper_factory=FakeReverificationHelper,
    )

    assert events == ["start", "reverify", "shutdown"]
    assert global_config_module._GLOBAL_CONFIG_DICT is original_global
    assert result.rows == 2
    assert result.started_components == ("judge_model", "selected_verifier")
    assert captured["config"].materialized_inputs_jsonl_fpath == str(materialized)
    assert captured["config"].rollouts_jsonl_fpath == str(rollouts)
    assert captured["config"].output_jsonl_fpath == str(tmp_path / "rescored.jsonl")
    assert captured["config"].force is True
    assert captured["config"].limit == 3
    assert captured["config"].num_samples_in_parallel == 2
    assert captured["config"].upload_rollouts_to_wandb is False
    assert captured["config"].failure_trajectories == FailureReplaySelection.LATEST_REPLAYABLE
    assert captured["config"].trajectory_identity_fields == DEFAULT_TRAJECTORY_IDENTITY_FIELDS
    assert captured["config"].verifier_fingerprint == "f" * 64
    assert captured["resources_server_name"] == "selected_verifier"
    assert "agent" not in captured["fingerprint_config"]
    assert "policy_model" not in captured["fingerprint_config"]
    assert "unrelated_model" not in captured["fingerprint_config"]


def test_replay_cli_defers_source_checks_until_after_filtering(tmp_path: Path, monkeypatch) -> None:
    import nemo_gym.cli.env as cli_env
    import nemo_gym.environment_replay as environment_replay

    manifest_path, _ = _manifest(tmp_path)
    raw = _resolved_config()
    raw.manifest_path = str(manifest_path)
    raw.unrelated_model = {
        "responses_api_models": {"unrelated_impl": {"entrypoint": "app.py"}},
    }
    captured = {}

    def parsed_config(*, global_config_dict_parser_config, global_config_dict_parser_cls):
        captured["parser"] = global_config_dict_parser_cls
        assert global_config_dict_parser_config is not None
        global_config_dict_parser_cls()._initialize_runtime_integrations(raw)
        return raw

    def replay(config):
        captured["config"] = config
        return SimpleNamespace(rows=0, paths=SimpleNamespace(output=tmp_path / "replayed.jsonl"))

    monkeypatch.setattr(cli_env, "get_global_config_dict", parsed_config)
    monkeypatch.setattr(environment_replay, "replay_environment_rollouts", replay)

    cli_env.replay_rollouts.__wrapped__()

    assert captured == {"parser": StaticValidationConfigParser, "config": raw}


def test_replay_reports_unsafe_output_selected_by_shared_helper(tmp_path: Path) -> None:
    manifest_path, _ = _manifest(tmp_path)
    rollouts, _ = _captured_run(tmp_path)
    raw = _resolved_config()
    raw.environment_ref = "replay_fixture"
    raw.manifest_path = str(manifest_path)
    raw.replay_rollouts_path = str(rollouts)
    raw.output_jsonl_fpath = str(tmp_path / "rescored.jsonl")

    class FakeRunner:
        def start(self, parser_config, *, preflight_mode) -> None:
            assert parser_config is None
            assert preflight_mode == "launch-sources-only"

        def shutdown(self) -> None:
            pass

    class FakeReverificationHelper:
        async def run_from_config(self, config, *, resources_server_name):
            assert resources_server_name == "selected_verifier"
            Path(config.output_jsonl_fpath).with_name("unsafe_" + Path(config.output_jsonl_fpath).name).touch()
            return []

    result = replay_environment_rollouts(
        raw,
        run_helper_factory=FakeRunner,
        reverification_helper_factory=FakeReverificationHelper,
    )

    assert result.paths.output == tmp_path / "unsafe_rescored.jsonl"


def test_replay_rejects_capability_supplied_only_by_filtered_policy_before_start(tmp_path: Path) -> None:
    manifest_path, _ = _manifest(tmp_path, model_server="policy_model")
    rollouts, _ = _captured_run(tmp_path)
    raw = _resolved_config()
    raw.manifest_path = str(manifest_path)
    raw.environment_ref = "replay_fixture"
    raw.replay_rollouts_path = str(rollouts)
    raw.selected_verifier.resources_servers.scorer.requires = ["judge-capability"]
    raw.policy_model.responses_api_models.policy_impl.provides.append("judge-capability")
    starts = []

    class FakeRunner:
        def start(self, _parser_config, *, preflight_mode) -> None:
            assert preflight_mode == "launch-sources-only"
            starts.append("start")

    with pytest.raises(ConfigError, match="judge-capability"):
        replay_environment_rollouts(raw, run_helper_factory=FakeRunner)

    assert starts == []


def test_replay_rejects_resources_server_without_verification_protocol_before_start(tmp_path: Path) -> None:
    manifest_path, _ = _manifest(tmp_path)
    rollouts, _ = _captured_run(tmp_path)
    raw = _resolved_config()
    raw.manifest_path = str(manifest_path)
    raw.environment_ref = "replay_fixture"
    raw.replay_rollouts_path = str(rollouts)
    del raw.selected_verifier.resources_servers.scorer.provides
    starts = []

    class FakeRunner:
        def start(self, _parser_config, *, preflight_mode) -> None:
            assert preflight_mode == "launch-sources-only"
            starts.append("start")

    with pytest.raises(ConfigError, match=r"provides: \[verification\]"):
        replay_environment_rollouts(raw, run_helper_factory=FakeRunner)

    assert starts == []
