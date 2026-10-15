# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from pathlib import Path

import pytest
import yaml

import nemo_gym.cli.main as main_module
import nemo_gym.global_config as global_config
from nemo_gym.path_utils import failures_path_for
from nemo_gym.tasks.harbor.cli import build_run, prepare_target, resolve_agent, summarize_validation
from tests.unit_tests.test_harbor_tasks import write_task


def result(task="a", *, index=0, repeat=0, **fields):
    return {
        "_ng_task_id": {"taskset": "ds", "task_id": task},
        "_ng_task_index": index,
        "_ng_rollout_index": repeat,
        "reward": 1.0,
        **fields,
    }


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.mark.parametrize("missing", [result("b", index=1), result(repeat=1)])
def test_summary_requires_each_selected_task_and_repeat(tmp_path, missing):
    path = tmp_path / "rollouts.jsonl"
    write_jsonl(path, [result()])
    report = summarize_validation(path, [], expected=[result(), missing])
    assert not report.ok and "missing result" in report.text


def test_success_supersedes_only_failures_for_the_same_task_repeat(tmp_path):
    path = tmp_path / "rollouts.jsonl"
    write_jsonl(path, [result(_ng_attempt_index=1)])
    write_jsonl(failures_path_for(path), [result(_ng_attempt_index=0, _ng_failure_message="old failure")])
    report = summarize_validation(path, [], expected=[result()])
    assert report.ok and "old failure" not in report.text
    write_jsonl(failures_path_for(path), [result(repeat=1, _ng_failure_message="unresolved repeat")])
    report = summarize_validation(path, [], expected=[result(), result(repeat=1)])
    assert not report.ok and "unresolved repeat" in report.text


def test_failure_sidecar_is_reported_without_a_results_file(tmp_path):
    path = tmp_path / "rollouts.jsonl"
    write_jsonl(failures_path_for(path), [result(_ng_failure_message="seed failed")])
    report = summarize_validation(path, [], expected=[result()])
    assert not report.ok and "seed failed" in report.text


def test_results_from_other_tasksets_do_not_cover_selected_tasks(tmp_path):
    path = tmp_path / "rollouts.jsonl"
    write_jsonl(path, [result(_ng_task_id={"taskset": "other", "task_id": "a"})])
    report = summarize_validation(path, [], expected=[result()])
    assert not report.ok and "missing result" in report.text


@pytest.mark.parametrize("reward", [None, 0.0, float("nan"), float("inf")])
def test_invalid_or_failing_reward_never_passes(tmp_path, reward):
    path = tmp_path / "rollouts.jsonl"
    write_jsonl(path, [result(reward=reward)])
    assert not summarize_validation(path, [], expected=[result()]).ok


@pytest.fixture
def validation_cli(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(global_config, "_GLOBAL_CONFIG_DICT", None)

    def invoke(*flags):
        monkeypatch.setattr(
            sys, "argv", ["gym", "dataset", "validate", str(tmp_path / "ds"), "--sandbox", "docker", *flags]
        )
        main_module.main()

    return invoke


@pytest.mark.parametrize("source", ["cli", "file"])
def test_validate_reads_effective_output_and_preserves_evaluation_artifacts(
    tmp_path, monkeypatch, validation_cli, source
):
    write_task(tmp_path / "ds" / "a")
    evaluated = prepare_target(str(tmp_path / "ds"))
    config_path, _ = build_run(evaluated, resolve_agent("hermes"), sandbox="docker", overrides=[])
    original_config = config_path.read_bytes()
    stale = evaluated.output_dir / "rollouts.jsonl"
    write_jsonl(stale, [result()])
    custom = tmp_path / "custom output.jsonl"
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(yaml.safe_dump({"output_jsonl_fpath": str(custom)}))
    seen = []

    def collect(target, tokens):
        monkeypatch.setattr(sys, "argv", ["e2e", *tokens])
        seen.append(target)
        config = global_config.get_global_config_dict()
        assert Path(config.output_jsonl_fpath) == custom
        write_jsonl(custom, [result(reward=0.0)])

    monkeypatch.setattr(main_module, "dispatch", collect)
    flags = [f"+output_jsonl_fpath={json.dumps(str(custom))}"] if source == "cli" else ["--config", str(overlay)]
    with pytest.raises(SystemExit) as exc:
        validation_cli(*flags)
    assert exc.value.code == 1
    assert seen == ["nemo_gym.cli.eval:e2e_rollout_collection"]
    assert config_path.read_bytes() == original_config
    assert json.loads(stale.read_text())["reward"] == 1.0
    assert (tmp_path / "results/harbor/validation/ds/run_config.yaml").is_file()


@pytest.mark.parametrize("oracle_status", ["solved", "failed"])
def test_validate_requires_solution_success_even_when_verifier_passes(
    tmp_path, monkeypatch, validation_cli, capsys, oracle_status
):
    write_task(tmp_path / "ds" / "a")

    def collect(target, tokens):
        config = global_config.get_global_config_dict()
        write_jsonl(Path(config.output_jsonl_fpath), [result(response={"metadata": {"oracle": oracle_status}})])

    monkeypatch.setattr(main_module, "dispatch", collect)
    if oracle_status == "failed":
        with pytest.raises(SystemExit) as exc:
            validation_cli()
        assert exc.value.code == 1
    else:
        validation_cli()
    assert f"oracle {oracle_status}\t1.0" in capsys.readouterr().out


@pytest.mark.parametrize("missing_repeat", [False, True])
def test_validate_filters_solutionless_before_limit_and_checks_repeats(
    tmp_path, monkeypatch, capsys, validation_cli, missing_repeat
):
    for name in ("a", "b", "c"):
        write_task(tmp_path / "ds" / name)
    (tmp_path / "ds/a/solution/solve.sh").unlink()

    def collect(target, tokens):
        monkeypatch.setattr(sys, "argv", ["e2e", *tokens])
        config = global_config.get_global_config_dict()
        agent = config.harbor_ds_agent.responses_api_agents.oracle_agent
        rows = [json.loads(line) for line in Path(agent.datasets[0].jsonl_fpath).read_text().splitlines()]
        assert [row["task_id"]["task_id"] for row in rows] == ["b", "c"]
        repeats = [result("b", repeat=0)] if missing_repeat else [result("b", repeat=i) for i in range(2)]
        write_jsonl(Path(config.output_jsonl_fpath), repeats)

    monkeypatch.setattr(main_module, "dispatch", collect)
    if missing_repeat:
        with pytest.raises(SystemExit) as exc:
            validation_cli("--limit", "1", "+num_repeats=2")
        assert exc.value.code == 1
    else:
        validation_cli("--limit", "1", "+num_repeats=2")
    output = capsys.readouterr().out
    assert "a\tunvalidated" in output
    assert "c [repeat" not in output
    assert ("missing result" in output) is missing_repeat


@pytest.mark.parametrize("with_solution", [False, True])
def test_validate_fails_for_skipped_tasks_even_with_no_solutions(
    tmp_path, monkeypatch, capsys, validation_cli, with_solution
):
    write_task(tmp_path / "ds" / "a")
    write_task(tmp_path / "ds" / "bad", toml="invalid toml")
    if not with_solution:
        (tmp_path / "ds/a/solution/solve.sh").unlink()

    def collect(target, tokens):
        monkeypatch.setattr(sys, "argv", ["e2e", *tokens])
        assert with_solution
        write_jsonl(Path(global_config.get_global_config_dict().output_jsonl_fpath), [result()])

    monkeypatch.setattr(main_module, "dispatch", collect)
    with pytest.raises(SystemExit) as exc:
        validation_cli()
    assert exc.value.code == 1
    assert "bad\tskipped:" in capsys.readouterr().out


def test_all_solutionless_tasks_are_reported_without_dispatch(tmp_path, monkeypatch, capsys, validation_cli):
    write_task(tmp_path / "ds" / "a")
    (tmp_path / "ds/a/solution/solve.sh").unlink()
    monkeypatch.setattr(main_module, "dispatch", lambda *args: pytest.fail("must not dispatch solutionless tasks"))
    validation_cli()
    assert "a\tunvalidated" in capsys.readouterr().out


@pytest.mark.parametrize("cli_override", [False, True])
def test_network_default_is_below_explicit_config_and_cli(tmp_path, monkeypatch, cli_override):
    write_task(tmp_path / "ds" / "a")
    prepared = prepare_target(str(tmp_path / "ds"), output_root=tmp_path / "out")
    overlay = tmp_path / "network.yaml"
    overlay.write_text("use_absolute_ip: false\n")
    from nemo_gym import PARENT_DIR
    from nemo_gym.cli.main import _merge_config_paths

    _, tokens = build_run(
        prepared,
        resolve_agent("hermes"),
        sandbox="docker",
        overrides=[
            f"+config_paths=[{PARENT_DIR / 'responses_api_models/openai_model/configs/openai_model.yaml'},{overlay}]",
            "+policy_base_url=http://127.0.0.1:1/v1",
            "+policy_api_key=unused",
            "+policy_model_name=test",
            *(["+use_absolute_ip=true"] if cli_override else []),
        ],
    )
    monkeypatch.setattr(sys, "argv", ["e2e", *_merge_config_paths(tokens)])
    monkeypatch.setattr(global_config, "_GLOBAL_CONFIG_DICT", None)
    monkeypatch.setattr(global_config, "gethostbyname", lambda _: "192.0.2.20")
    assert global_config.get_global_config_dict().use_absolute_ip is cli_override


@pytest.mark.parametrize("source", ["cli", "file", "partial", "dotenv", "none"])
def test_validate_needs_no_model_even_with_unused_endpoint_settings(tmp_path, monkeypatch, validation_cli, source):
    write_task(tmp_path / "ds" / "a")
    settings = {
        "policy_base_url": "http://127.0.0.1:8123/v1",
        "policy_model_name": "existing",
        "policy_api_key": "unused",
    }
    if source == "partial":
        settings = {"policy_model_name": "existing"}
    if source == "none":
        settings = {}
    if source == "dotenv":
        (tmp_path / "env.yaml").write_text(yaml.safe_dump(settings))
        flags = []
    elif source == "file":
        overlay = tmp_path / "model settings.yaml"
        overlay.write_text(yaml.safe_dump(settings))
        flags = ["--config", str(overlay)]
    else:
        flags = [f"+{key}={value}" for key, value in settings.items()]

    def collect(target, tokens):
        config = global_config.get_global_config_dict()
        from nemo_gym.cli import env

        assert "policy_model" not in config
        for key, value in settings.items():
            assert config[key] == value
        monkeypatch.setattr(env, "_probe_endpoint", lambda *args: pytest.fail("Oracle must not probe an unused model"))
        env.RunHelper().wait_for_model_endpoints(config)
        write_jsonl(Path(config.output_jsonl_fpath), [result()])

    monkeypatch.setattr(main_module, "dispatch", collect)
    validation_cli(*flags)


@pytest.mark.parametrize("directory", ["custom", "responses_api_models"])
def test_validate_accepts_custom_model_config_in_any_directory(tmp_path, monkeypatch, validation_cli, directory):
    from nemo_gym import PARENT_DIR

    write_task(tmp_path / "ds" / "a")
    model_path = tmp_path / directory / "model.yaml"
    model_path.parent.mkdir()
    model_path.write_text((PARENT_DIR / "responses_api_models/vllm_model/configs/vllm_model.yaml").read_text())

    def collect(target, tokens):
        config = global_config.get_global_config_dict()
        assert list(config.policy_model.responses_api_models) == ["vllm_model"]
        assert config.policy_model.responses_api_models.vllm_model.base_url == "http://127.0.0.1:8123/v1"
        write_jsonl(Path(config.output_jsonl_fpath), [result()])

    monkeypatch.setattr(main_module, "dispatch", collect)
    validation_cli(
        "--config",
        str(model_path),
        "+policy_base_url=http://127.0.0.1:8123/v1",
        "+policy_api_key=unused",
        "+policy_model_name=existing",
    )


def test_validate_reports_missing_sandbox_environment_cleanly(tmp_path, monkeypatch, validation_cli, capsys):
    write_task(tmp_path / "ds" / "a")
    monkeypatch.delenv("OPENSANDBOX_API_KEY", raising=False)
    monkeypatch.setattr(main_module, "dispatch", lambda *args: pytest.fail("invalid config must not dispatch"))
    with pytest.raises(SystemExit) as exc:
        validation_cli("--sandbox", "opensandbox")
    assert exc.value.code == 1
    output = capsys.readouterr()
    assert "Error:" in output.out and "OPENSANDBOX_API_KEY" in output.out
    assert "Traceback" not in output.out + output.err


def test_validate_reports_preflight_config_errors_cleanly(tmp_path, monkeypatch, validation_cli, capsys):
    write_task(tmp_path / "ds" / "a")
    monkeypatch.setattr(main_module, "dispatch", lambda *args: pytest.fail("invalid config must not dispatch"))
    with pytest.raises(SystemExit) as exc:
        validation_cli("--model-type", "vllm_model")
    assert exc.value.code == 1
    output = capsys.readouterr()
    assert "Error:" in output.out and "policy_base_url" in output.out
    assert "Traceback" not in output.out + output.err


def test_validate_resume_rejects_changed_digest_and_preserves_unchanged_retries(
    tmp_path, monkeypatch, validation_cli, capsys
):
    """Real CLI, preparation and collector; replace only server startup and rollout execution."""
    from types import SimpleNamespace

    from nemo_gym.cli.eval import RunHelper
    from nemo_gym.path_utils import materialized_path_for
    from nemo_gym.rollout_collection import RolloutCollectionHelper, _CompletedRollout

    task = write_task(tmp_path / "ds" / "a")
    dispatched = []
    first_attempt = True

    async def complete(row):
        dispatched.append(row)
        if first_attempt:
            payload = result(_ng_failure_class="agent_request_failed", _ng_failure_message="temporary failure")
        elif "exit 99" in (task / "solution/solve.sh").read_text():
            payload = result(reward=0.0, response={"metadata": {"oracle": "failed"}})
        else:
            payload = result(response={"metadata": {"oracle": "solved"}})
        return _CompletedRollout(row=row, result=payload, rollout_latency_ms=1.0)

    monkeypatch.setattr(RunHelper, "start", lambda *args: None)
    monkeypatch.setattr(RunHelper, "shutdown", lambda *args: None)
    monkeypatch.setattr(
        RolloutCollectionHelper,
        "setup_server_client",
        lambda *args: SimpleNamespace(global_config_dict=global_config.get_global_config_dict()),
    )
    monkeypatch.setattr(
        RolloutCollectionHelper,
        "_run_examples_with_metadata",
        lambda self, rows, **kwargs: (complete(row) for row in rows),
    )

    def invoke(*flags):
        monkeypatch.setattr(global_config, "_GLOBAL_CONFIG_DICT", None)
        validation_cli("+disable_aggregation=true", "+resume_from_cache=true", *flags)

    with pytest.raises(RuntimeError, match="None of the 1 dispatched rollouts produced a result"):
        invoke()
    first_attempt = False
    invoke()
    assert len(dispatched) == 2 and dispatched[-1]["_ng_attempt_index"] == 1
    rollouts = tmp_path / "results/harbor/validation/ds/rollouts.jsonl"
    cached_inputs = materialized_path_for(rollouts).read_bytes()
    cached_results = rollouts.read_bytes()
    invoke()
    assert len(dispatched) == 2

    (task / "solution/solve.sh").write_text("#!/bin/bash\nexit 99\n")
    with pytest.raises(SystemExit) as exc:
        invoke()
    assert exc.value.code == 1
    assert len(dispatched) == 2
    assert "task digests have changed" in capsys.readouterr().out
    assert materialized_path_for(rollouts).read_bytes() == cached_inputs
    assert rollouts.read_bytes() == cached_results

    # An explicit fresh run replaces the stale cache and reports the new failing solution.
    monkeypatch.setattr(global_config, "_GLOBAL_CONFIG_DICT", None)
    with pytest.raises(SystemExit) as exc:
        validation_cli("+disable_aggregation=true", "+resume_from_cache=false")
    assert exc.value.code == 1 and len(dispatched) == 3
    assert json.loads(rollouts.read_text())["reward"] == 0.0
    assert materialized_path_for(rollouts).read_bytes() != cached_inputs
