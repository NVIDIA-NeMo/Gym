# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from scripts.harness_conformance import episode, runner
from scripts.harness_conformance.provider import Probe
from scripts.harness_conformance.scenarios import SCENARIOS

from nemo_gym.core import ModelCallSampling, encode_model_sampling
from nemo_gym.rollouts.harness_capabilities import digest_file, json_rows


PORTS = {"policy_model": 10001, "probe_resources": 10002, "probe_agent": 10003, "probe_environment": 10004}


def test_claude_is_explicit_opt_in_and_does_not_change_default_cli_or_ci(tmp_path, monkeypatch):
    from scripts.harness_conformance.ci import select_harnesses
    from scripts.harness_conformance.registry import HARNESSES, PROBE_HARNESSES

    default = ["simple", "opencode", "pi", "codex", "hermes", "kilocode", "stirrup", "openclaw", "nooa"]
    assert list(HARNESSES) == default
    assert PROBE_HARNESSES["claude_code"] == "ClaudeCodeAgent"
    assert select_harnesses(["nemo_gym/models/model_call_capture.py"]) == default
    assert select_harnesses(["responses_api_agents/claude_code_agent/app.py"]) == []
    calls = []

    def run(**kwargs):
        calls.append(kwargs)
        return {}, 0

    monkeypatch.setattr(runner, "run_suite", run)
    assert runner.main(["--output", str(tmp_path / "default")]) == 0
    assert calls[-1]["harnesses"] == default
    assert calls[-1]["scenarios"] == SCENARIOS
    assert runner.main(["--harness", "claude_code", "--output", str(tmp_path / "claude")]) == 0
    assert calls[-1]["harnesses"] == ["claude_code"] and calls[-1]["scenarios"] == SCENARIOS


async def test_prepared_claude_settings_and_model_survive_local_auth_rewiring(tmp_path, monkeypatch):
    from unittest.mock import MagicMock

    from fastapi import HTTPException

    from nemo_gym.agents import AgentSeedSessionRequest
    from nemo_gym.client import ServerClient
    from responses_api_agents.claude_code_agent import app

    runtime_settings = tmp_path / "settings.json"
    runtime_settings.write_text(
        json.dumps({"env": {"DISABLE_COMPACT": "1", "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "32768"}})
    )
    source = tmp_path / "prepared.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "config_paths": [
                    str(
                        Path(__file__).resolve().parents[3]
                        / "responses_api_agents/claude_code_agent/configs/claude_code_agent.yaml"
                    )
                ],
                "claude_code_agent": {
                    "responses_api_agents": {
                        "claude_code_agent": {
                            "model": "${policy_model_name}",
                            "claude_code_version": "2.1.108",
                            "bare": True,
                            "settings": str(runtime_settings),
                            "max_turns": 17,
                            "anthropic_base_url": "${unavailable_remote_endpoint}",
                        }
                    }
                },
            }
        )
    )
    settings = episode.prepared_settings(
        harness="claude_code",
        config_path=source,
        agent_name="claude_code_agent",
        sampling_path=None,
        policy_model_name="served-policy-alias",
    )
    assert settings["agent"]["model"] == "served-policy-alias"
    assert settings["agent"]["claude_code_version"] == "2.1.108"
    assert settings["agent"]["bare"] is True and settings["agent"]["max_turns"] == 17
    assert "anthropic_api_key" not in settings["agent"] and "anthropic_base_url" not in settings["agent"]
    assert settings["referenced_files"] == {str(runtime_settings): digest_file(runtime_settings)}
    config = episode._config(
        "claude_code", tmp_path, PORTS, 10, use_agent_sessions=True, prepared_agent=settings["agent"]
    )
    effective = config["probe_agent"]["responses_api_agents"]["claude_code_agent"]
    parsed = app.ClaudeCodeAgentConfig.model_validate(effective)
    assert parsed.model == "served-policy-alias" and parsed.claude_code_version == "2.1.108"
    assert parsed.settings == str(runtime_settings) and parsed.bare and parsed.max_turns == 17
    assert parsed.anthropic_api_key == "conformance" and parsed.anthropic_base_url is None
    assert parsed.model_server.name == "policy_model"
    assert episode._supports_agent_sessions(app.ClaudeCodeAgent)
    assert set(config["probe_environment"]["environment_servers"]) == {"single_agent_turn_legacy"}
    monkeypatch.setattr(episode.shutil, "which", lambda _: pytest.fail("session mode must not inspect host runtime"))
    assert episode.check_runtime("claude_code", use_agent_sessions=True) == {"installation": "agent_session"}

    # No pin is invented when the published adapter default is null, and setup
    # rejects it before any provider or runtime work.
    unpinned = episode._config("claude_code", tmp_path, PORTS, 10, use_agent_sessions=True)
    unpinned_config = app.ClaudeCodeAgentConfig.model_validate(
        unpinned["probe_agent"]["responses_api_agents"]["claude_code_agent"]
    )
    assert unpinned_config.claude_code_version is None
    agent = app.ClaudeCodeAgent(config=unpinned_config, server_client=MagicMock(spec=ServerClient))
    monkeypatch.setattr(
        app, "create_provider", lambda *a, **kw: pytest.fail("provider must not be created without a pin")
    )
    with pytest.raises(HTTPException, match="exact claude_code_version"):
        await agent._seed_agent_session_state(
            AgentSeedSessionRequest(
                agent_session_id="session",
                episode_id={"rollout_id": "rollout"},
                task_id={"taskset": "fixture", "task_id": "task"},
            )
        )
    runtime_settings.write_text("{}")
    monkeypatch.setattr(
        episode.importlib, "import_module", lambda _: pytest.fail("runtime imported before hash check")
    )
    with pytest.raises(ValueError, match="Prepared behavior file changed"):
        await episode.run_episode(
            harness="claude_code", scenario_name="tool_success", directory=tmp_path, timeout=1, settings=settings
        )


def write_config(path, agent, **global_values):
    path.write_text(yaml.safe_dump(global_values | {"selected": {"responses_api_agents": {"opencode_agent": agent}}}))
    return path


def test_prepared_config_resolves_nested_profiles_and_omits_unrelated_credentials(tmp_path):
    inner = write_config(tmp_path / "inner.yaml", {"opencode_config": {"permission": {"bash": "deny"}, "x": 1}})
    outer = write_config(
        tmp_path / "outer.yaml",
        {"model": "${policy_model_name}", "opencode_config": {"permission": {"bash": "allow"}}},
        config_paths=[str(inner)],
    )
    sibling = write_config(tmp_path / "sibling.yaml", {"opencode_config": {"x": 2}})
    source = tmp_path / "prepared.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "env": {
                    "nemo_gym": {
                        "config_paths": [str(outer), str(sibling)],
                        "context": 32768,
                        "policy_model_name": "gym-policy-model",
                        "unused_provider": {"api_key": "DO_NOT_RETAIN", "domain": "${oc.env:UNSET_PROVIDER_DOMAIN}"},
                        "selected": {
                            "responses_api_agents": {
                                "opencode_agent": {
                                    "context_window": "${context}",
                                    "max_output_tokens": 1024,
                                    "timeout": 480,
                                    "token_id_capture": True,
                                    "opencode_config": {"compaction": {"auto": False, "prune": False}},
                                }
                            }
                        },
                    }
                }
            }
        )
    )
    result = episode.prepared_settings(
        harness="opencode", config_path=source, agent_name="selected", sampling_path=None
    )
    agent = result["agent"]
    assert agent == {
        "model": "gym-policy-model",
        "context_window": 32768,
        "max_output_tokens": 1024,
        "timeout": 480,
        "token_id_capture": True,
        "opencode_config": {"permission": {"bash": "allow"}, "x": 2, "compaction": {"auto": False, "prune": False}},
    }
    assert result["sources"] == {str(path): digest_file(path) for path in (source, inner, outer, sibling)}
    assert len(result["selected_agent_sha256"]) == 64
    assert "DO_NOT_RETAIN" not in json.dumps(result)
    config = episode._config("opencode", tmp_path, PORTS, 10, use_agent_sessions=True, prepared_agent=agent)
    launched = config["probe_agent"]["responses_api_agents"]["opencode_agent"]
    assert all(launched[key] == value for key, value in agent.items())
    assert launched["model"] == "gym-policy-model"
    assert launched["sandbox_config"]["workdir"] == str(tmp_path / "workspace")


@pytest.mark.parametrize(
    "harness,preset_field", [("hermes", "max_turns"), ("simple", "max_steps"), ("opencode", "opencode_config")]
)
def test_prepared_input_does_not_inherit_behavior_presets(tmp_path, harness, preset_field):
    prepared = {"token_id_capture": True}
    config = episode._config(harness, tmp_path, PORTS, 10, use_agent_sessions=True, prepared_agent=prepared)
    agent = config["probe_agent"]["responses_api_agents"][f"{harness}_agent"]
    assert preset_field not in agent
    assert agent["token_id_capture"] is True
    assert prepared == {"token_id_capture": True}


@pytest.mark.parametrize("sampling", [{"max_output_tokens": 0}, {"max_output_tokens": "1024"}, {"extra": 1}])
def test_invalid_sampling_fails_before_output_or_launch(tmp_path, sampling):
    path = tmp_path / "sampling.json"
    path.write_text(json.dumps(sampling))
    output = tmp_path / "run"
    assert runner.main(["--harness", "opencode", "--model-sampling", str(path), "--output", str(output)]) == 2
    assert not output.exists()


def test_runner_freezes_inputs_and_keeps_selected_compaction_scenario(tmp_path, monkeypatch):
    source = write_config(
        tmp_path / "prepared.yaml",
        {"model": "${policy_model_name}", "context_window": 32768, "opencode_config": {"compaction": {"auto": False}}},
    )
    sampling = tmp_path / "sampling.yaml"
    sampling.write_text("max_output_tokens: 1024\ntemperature: 1.0\n")
    launched = []

    def run(command, *, directory, timeout):
        settings = json.loads((tmp_path / "run/prepared_settings.json").read_text())
        assert command[-2:] == ["--prepared-settings", str(tmp_path / "run/prepared_settings.json")]
        assert settings["agent"]["opencode_config"]["compaction"]["auto"] is False
        assert settings["agent"]["model"] == settings["policy_model_name"] == "served-alias"
        assert settings["model_sampling"] == {"max_output_tokens": 1024, "temperature": 1.0}
        launched.append(directory.name)
        return {"returncode": 0, "timed_out": False}

    monkeypatch.setattr(runner, "run_process", run)
    scenario = next(s for s in SCENARIOS if s.name == "context_compaction")
    result, code = runner.run_suite(
        harnesses=["opencode"],
        scenarios=(scenario,),
        output=tmp_path / "run",
        timeout=1,
        gate_tier="P1",
        prepared_config=source,
        agent_name="selected",
        model_sampling=sampling,
        policy_model_name="served-alias",
    )
    assert launched == ["context_compaction"]
    assert code == 1 and result["full_suite"] is False
    checks = result["harnesses"]["opencode"]["scenarios"][0]["checks"]
    assert next(c for c in checks if c["id"] == "compaction.event")["status"] == "fail"
    manifest = json.loads((tmp_path / "run/suite.json").read_text())
    assert manifest["prepared_settings_sha256"] == digest_file(tmp_path / "run/prepared_settings.json")


@pytest.mark.parametrize(
    "api,limit_field",
    [("chat/completions", "max_completion_tokens"), ("responses", "max_output_tokens"), ("messages", "max_tokens")],
)
def test_required_sampling_reaches_effective_witness_and_capture_without_training_tokens(tmp_path, api, limit_field):
    scenario = next(s for s in SCENARIOS if s.name == "judge_verifier")
    probe = Probe(scenario, tmp_path)
    app = probe.model_app()
    episode._install_sampling(app, tmp_path, name="policy_model", required=True)
    sampling = ModelCallSampling(temperature=1.0, top_p=1.0, max_output_tokens=1024)
    segment = encode_model_sampling(sampling)
    body = {"model": "conformance-model", "temperature": 0.2, "top_p": 0.4, limit_field: 4096}
    body["input" if api == "responses" else "messages"] = []
    with TestClient(app) as client:
        assert client.post(f"/ng-rollout/0-0/v1/{api}", json=body).status_code == 422
        response = client.post(f"/ng-rollout/0-0/ng-sampling/{segment}/v1/{api}", json=body)
        assert response.status_code == 200
    assert len(probe.attempts) == 1
    effective = probe.attempts[0]["request"]
    assert (effective[limit_field], effective["temperature"], effective["top_p"]) == (1024, 1.0, 1.0)
    [(_, captured)] = list(json_rows(tmp_path / "capture/0-0.capture.jsonl"))
    assert captured["request"] == effective
    assert "token_ids" not in json.dumps(captured)
    paths = [row for _, row in json_rows(tmp_path / "sampling_requests.jsonl")]
    assert [row["status"] for row in paths] == [422, 200]
    assert paths[1]["path"] == f"/ng-rollout/0-0/ng-sampling/{segment}/v1/{api}"


def test_judge_null_sampling_keeps_its_own_limit(tmp_path):
    probe = Probe(next(s for s in SCENARIOS if s.name == "judge_verifier"), tmp_path)
    app = probe.judge_app()
    episode._install_sampling(app, tmp_path, name="judge_model", required=False)
    body = {"messages": [], "max_tokens": 16, "temperature": 0.2}
    with TestClient(app) as client:
        response = client.post("/ng-rollout/0-0/ng-sampling/null/v1/chat/completions", json=body)
    assert response.status_code == 200
    assert probe.judge_calls[0]["request"] == body


@pytest.mark.parametrize("model", ["gym-policy-model", None])
async def test_prepared_model_alias_and_catalog_reference_survive_local_rewiring(tmp_path, monkeypatch, model):
    catalog = tmp_path / "models.json"
    catalog.write_text('{"models": [{"slug": "gym-policy-model"}]}')
    source = tmp_path / "prepared.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "selected": {
                    "responses_api_agents": {
                        "codex_agent": {
                            "model": model,
                            "model_catalog_json": str(catalog),
                            "extra_config": {"web_search": "disabled"},
                        }
                    }
                }
            }
        )
    )
    settings = episode.prepared_settings(
        harness="codex", config_path=source, agent_name="selected", sampling_path=None
    )
    config = episode._config("codex", tmp_path, PORTS, 10, use_agent_sessions=True, prepared_agent=settings["agent"])
    assert config["probe_agent"]["responses_api_agents"]["codex_agent"]["model"] == model
    assert settings["referenced_files"] == {str(catalog): digest_file(catalog)}
    catalog.write_text('{"models": [{"slug": "changed"}]}')
    monkeypatch.setattr(
        episode.importlib, "import_module", lambda _: pytest.fail("runtime imported before validation")
    )
    with pytest.raises(ValueError, match="Prepared behavior file changed"):
        await episode.run_episode(
            harness="codex", scenario_name="tool_success", directory=tmp_path, timeout=1, settings=settings
        )


def test_unresolved_required_reference_rejects_before_launch(tmp_path):
    source = write_config(tmp_path / "prepared.yaml", {"model": "${runtime_model_alias}"})
    output = tmp_path / "run"
    assert (
        runner.main(
            [
                "--harness",
                "opencode",
                "--prepared-config",
                str(source),
                "--agent-name",
                "selected",
                "--output",
                str(output),
            ]
        )
        == 2
    )
    assert not output.exists()


@pytest.mark.parametrize(
    "model,expected",
    [("${policy_model_name}", "served-alias"), ("explicit-agent-alias", "explicit-agent-alias"), (None, None)],
)
def test_explicit_runtime_global_resolves_prepared_rl_without_changing_agent_identity(tmp_path, model, expected):
    source = tmp_path / "prepared.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "env": {
                    "nemo_gym": {
                        "selected": {"responses_api_agents": {"opencode_agent": {"model": model}}},
                    }
                }
            }
        )
    )
    settings = episode.prepared_settings(
        harness="opencode",
        config_path=source,
        agent_name="selected",
        sampling_path=None,
        policy_model_name="served-alias",
    )
    assert settings["agent"]["model"] == expected
    assert settings["policy_model_name"] == "served-alias"
    assert settings["sources"][str(source)] == digest_file(source)


@pytest.mark.parametrize(
    "configured,provided,valid",
    [("served-alias", "served-alias", True), ("different", "served-alias", False), ("served-alias", "", False)],
)
def test_runtime_global_rejects_conflicts_and_empty_alias(tmp_path, configured, provided, valid):
    source = write_config(tmp_path / "prepared.yaml", {"model": "${policy_model_name}"}, policy_model_name=configured)
    if valid:
        settings = episode.prepared_settings(
            harness="opencode",
            config_path=source,
            agent_name="selected",
            sampling_path=None,
            policy_model_name=provided,
        )
        assert settings["agent"]["model"] == configured
    else:
        output = tmp_path / "run"
        assert (
            runner.main(
                [
                    "--harness",
                    "opencode",
                    "--prepared-config",
                    str(source),
                    "--agent-name",
                    "selected",
                    "--policy-model-name",
                    provided,
                    "--output",
                    str(output),
                ]
            )
            == 2
        )
        assert not output.exists()


def test_top_level_source_hash_follows_loaded_cwd_file_despite_extra_root_shadow(tmp_path, monkeypatch):
    extra = tmp_path / "extra"
    extra.mkdir()
    loaded = write_config(tmp_path / "prepared.yaml", {"context_window": 32768})
    shadow = write_config(extra / "prepared.yaml", {"context_window": 65536})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NEMO_GYM_EXTRA_ROOTS", str(extra))
    settings = episode.prepared_settings(
        harness="opencode", config_path=Path("prepared.yaml"), agent_name="selected", sampling_path=None
    )
    assert settings["agent"]["context_window"] == 32768
    assert settings["sources"] == {str(loaded): digest_file(loaded)}
    assert str(shadow) not in settings["sources"]


def test_retained_settings_resolve_sibling_and_outer_rl_references_without_unrelated_credentials(tmp_path):
    source = tmp_path / "prepared.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "policy": {"generation": {"max_new_tokens": 1024}, "unused": "${oc.env:UNSET_POLICY_SECRET}"},
                "env": {
                    "nemo_gym": {
                        "provider": {"api_key": "${oc.env:UNSET_PROVIDER_SECRET}"},
                        "selected": {
                            "responses_api_agents": {
                                "opencode_agent": {
                                    "model_server": {
                                        "name": "served-alias",
                                        "unused": "${oc.env:UNSET_SERVER_SECRET}",
                                    },
                                    "model": "${.model_server.name}",
                                    "max_output_tokens": "${policy.generation.max_new_tokens}",
                                    "api_key": "${oc.env:UNSET_AGENT_SECRET}",
                                    "sandbox_config": {
                                        "workdir": "/prepared/workspace",
                                        "env": {"ORIGINAL_WORKDIR": "${..workdir}"},
                                    },
                                }
                            }
                        },
                    }
                },
            }
        )
    )
    settings = episode.prepared_settings(
        harness="opencode", config_path=source, agent_name="selected", sampling_path=None
    )
    assert settings["agent"] == {
        "model": "served-alias",
        "max_output_tokens": 1024,
        "sandbox_config": {"env": {"ORIGINAL_WORKDIR": "/prepared/workspace"}},
    }
    assert "UNSET_" not in json.dumps(settings)
    assert settings["sources"] == {str(source): digest_file(source)}
