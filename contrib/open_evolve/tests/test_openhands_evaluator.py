from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from openevolve_gym.openhands_evaluator import (
    DEFAULT_INPUT_JSONL,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_SWE_AGENTS_CONFIG,
    OpenHandsEvaluationConfig,
    build_collect_rollouts_command,
    build_ng_run_command,
    evaluate_candidate,
    parse_rollout_scores,
    render_openhands_candidate,
    run_gym_rollout,
)


def test_render_openhands_candidate_writes_codex_prompt_overlay(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.md"
    candidate.write_text("Always reproduce the failure before patching.\n")

    rendered = render_openhands_candidate(candidate, tmp_path / "rendered")

    assert "Always reproduce" in rendered.system_prompt.read_text()
    assert "{{ workspace_path }}" in rendered.user_prompt.read_text()

    overlay = yaml.safe_load(rendered.config_overlay.read_text())
    swe_agents_config = overlay["swe_agents"]["responses_api_agents"]["swe_agents"]
    override = swe_agents_config["agent_prompt_overrides"][0]
    assert override["agent_cls"] == "CodexAgent"
    assert override["system_prompt_template"] == str(rendered.system_prompt)
    assert override["user_prompt_template"] == str(rendered.user_prompt)
    assert swe_agents_config["agent_prompt_override_random"] is False
    assert swe_agents_config["concurrency"] == 1


def test_build_commands_point_at_gym_cli_and_candidate_overlay(tmp_path: Path) -> None:
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text("swe_agents: {}\n")
    output_jsonl = tmp_path / "rollouts.jsonl"
    config = OpenHandsEvaluationConfig(
        nemo_gym_repo=Path("/gym"),
        config_overlay=overlay,
        output_jsonl=output_jsonl,
        policy_base_url="http://model.invalid/v1",
        policy_api_key="dummy",
        policy_model_name="dummy-model",
    )

    run_command = build_ng_run_command(config)
    collect_command = build_collect_rollouts_command(config)

    assert run_command[0] == "/gym/.venv/bin/ng_run"
    assert str(overlay) in run_command[1]
    assert "+policy_base_url=http://model.invalid/v1" in run_command
    assert collect_command == [
        "/gym/.venv/bin/ng_collect_rollouts",
        "+agent_name=swe_agents",
        "+input_jsonl_fpath=responses_api_agents/swe_agents/data/example.jsonl",
        f"+output_jsonl_fpath={output_jsonl}",
        "+limit=1",
        "+num_samples_in_parallel=1",
        "+upload_rollouts_to_wandb=false",
    ]


def test_parse_rollout_scores_uses_combined_score_when_present(tmp_path: Path) -> None:
    output_jsonl = tmp_path / "rollouts.jsonl"
    output_jsonl.write_text(
        json.dumps(
            {
                "reward": 1.0,
                "resolved": True,
                "metadata": {
                    "metrics": {
                        "combined_score": 0.84,
                        "tests_passed_ratio": 0.75,
                    },
                },
            },
        )
        + "\n",
    )

    score = parse_rollout_scores(output_jsonl)

    assert score["combined_score"] == 0.84
    assert score["metrics"]["mean_reward"] == 1.0
    assert score["metrics"]["task_passed_rate"] == 1.0
    assert score["metrics"]["tests_passed_ratio"] == 0.75
    assert score["artifacts"]["rollout_path"] == str(output_jsonl)


def test_evaluate_candidate_dry_run_returns_rendered_artifacts(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.md"
    candidate.write_text("Use failing-test evidence before editing.\n")
    config = OpenHandsEvaluationConfig(nemo_gym_repo=Path("/gym"))

    score = evaluate_candidate(candidate, tmp_path / "candidate-run", config=config, execute=False)

    assert score["combined_score"] == 0.0
    assert score["metrics"]["dry_run"] is True
    assert Path(score["artifacts"]["system_prompt_path"]).exists()
    assert Path(score["artifacts"]["user_prompt_path"]).exists()
    assert Path(score["artifacts"]["config_overlay_path"]).exists()
    assert score["artifacts"]["ng_run_command"][0] == "/gym/.venv/bin/ng_run"
    assert score["artifacts"]["collect_rollouts_command"][0] == "/gym/.venv/bin/ng_collect_rollouts"


def test_run_gym_rollout_gates_missing_live_prerequisites_before_starting(
    tmp_path: Path,
) -> None:
    config = OpenHandsEvaluationConfig(
        nemo_gym_repo=tmp_path / "missing-gym",
        config_overlay=tmp_path / "missing-overlay.yaml",
        output_jsonl=tmp_path / "rollouts.jsonl",
    )
    popen_called = False

    def fail_if_started(*args, **kwargs):
        nonlocal popen_called
        popen_called = True
        raise AssertionError("ng_run should not start before readiness passes")

    with pytest.raises(RuntimeError, match="Live OpenEvolve evaluation is not ready") as error:
        run_gym_rollout(config, popen_factory=fail_if_started)

    assert popen_called is False
    message = str(error.value)
    assert "nemo_gym_repo" in message
    assert "policy_api_key" in message
    assert "policy_model_name" in message


def test_run_gym_rollout_starts_server_collects_and_cleans_up(tmp_path: Path) -> None:
    output_jsonl = tmp_path / "rollouts.jsonl"
    nemo_gym_repo = _ready_gym_repo(tmp_path)
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text("swe_agents: {}\n")
    config = OpenHandsEvaluationConfig(
        nemo_gym_repo=nemo_gym_repo,
        config_overlay=overlay,
        output_jsonl=output_jsonl,
        policy_api_key="test-key",
        policy_model_name="test-model",
    )
    calls = []

    class FakeProcess:
        returncode = None

        def __init__(self) -> None:
            self.terminated = False

        def poll(self):
            return None

        def terminate(self) -> None:
            self.terminated = True
            calls.append("terminate")

        def wait(self, timeout=None):
            calls.append(("wait", timeout))

    fake_process = FakeProcess()

    def fake_popen(command, **kwargs):
        calls.append(("popen", command, kwargs["cwd"]))
        return fake_process

    def fake_run(command, **kwargs):
        calls.append(("run", command, kwargs["cwd"], kwargs["check"]))
        output_jsonl.write_text(json.dumps({"reward": 1.0, "resolved": True}) + "\n")

    def fake_wait_for_servers(*args, **kwargs):
        calls.append(("ready", args, kwargs))

    score = run_gym_rollout(
        config,
        popen_factory=fake_popen,
        run_command=fake_run,
        wait_for_servers=fake_wait_for_servers,
    )

    assert score["combined_score"] == 1.0
    assert calls[0][0] == "popen"
    assert calls[1][0] == "ready"
    assert calls[2][0] == "run"
    assert calls[-2:] == ["terminate", ("wait", 10)]


def _ready_gym_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "gym"
    bin_dir = repo / ".venv" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "ng_run").write_text("#!/bin/sh\n")
    (bin_dir / "ng_collect_rollouts").write_text("#!/bin/sh\n")
    (repo / "responses_api_agents/swe_agents/configs").mkdir(parents=True)
    (repo / "responses_api_models/openai_model/configs").mkdir(parents=True)
    (repo / DEFAULT_SWE_AGENTS_CONFIG).write_text("swe_agents: {}\n")
    (repo / DEFAULT_MODEL_CONFIG).write_text("model: {}\n")
    (repo / "responses_api_agents/swe_agents/data").mkdir(parents=True)
    (repo / DEFAULT_INPUT_JSONL).write_text("{}\n")
    return repo
