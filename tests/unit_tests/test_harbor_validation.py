# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

import nemo_gym.cli.main as main_module
import nemo_gym.global_config as global_config
from nemo_gym.path_utils import failures_path_for, materialized_path_for
from nemo_gym.tasks.harbor.cli import build_run, prepare_target, resolve_agent
from nemo_gym.tasks.harbor.validation_report import summarize_validation
from tests.unit_tests.test_harbor_tasks import no_real_registry, write_task  # noqa: F401 (autouse fixture)


def result(task="a", *, index=0, repeat=0, **fields):
    return {
        "_ng_task_id": {"taskset": "ds", "task_id": task},
        "_ng_task_index": index,
        "_ng_rollout_index": repeat,
        "reward": 1.0,
        # What the oracle agent stamps on every response; a row another agent produced has no such key.
        "response": {"metadata": {"oracle": "solved"}},
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
    # `gym -v` installs a root logging handler bound to that run's stderr. pytest swaps the stream per
    # test, so a handler left behind by an earlier test would write a "Logging error" traceback here.
    root = logging.getLogger()
    saved = root.handlers[:]
    root.handlers.clear()

    def invoke(*flags):
        monkeypatch.setattr(
            sys, "argv", ["gym", "dataset", "validate", str(tmp_path / "ds"), "--sandbox", "docker", *flags]
        )
        main_module.main()

    yield invoke
    root.handlers[:] = saved


@pytest.mark.parametrize("source", ["cli", "file", "flag"])
def test_validate_reads_effective_output_and_preserves_evaluation_artifacts(
    tmp_path, monkeypatch, validation_cli, source
):
    write_task(tmp_path / "ds" / "a")
    evaluated = prepare_target(str(tmp_path / "ds"))
    config_path, _ = build_run(evaluated, resolve_agent("hermes_agent"), sandbox="docker", overrides=[])
    original_config = config_path.read_bytes()
    stale = evaluated.output_dir / "rollouts.jsonl"
    write_jsonl(stale, [result()])
    # `--output` is translated to a quoted Hydra override, so parentheses, commas, Unicode and backslashes
    # survive the real parse and collection is pointed at the file the user named.
    custom = tmp_path / (r"résumé(1),back\slash.jsonl" if source == "flag" else "custom output.jsonl")
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
    flags = {
        "cli": [f"+output_jsonl_fpath='{custom}'"],
        "file": ["--config", str(overlay)],
        "flag": ["--output", str(custom)],
    }[source]
    with pytest.raises(SystemExit) as exc:
        validation_cli(*flags)
    assert exc.value.code == 1
    assert seen == ["nemo_gym.cli.eval:e2e_rollout_collection"]
    assert config_path.read_bytes() == original_config
    assert json.loads(stale.read_text())["reward"] == 1.0
    assert (tmp_path / "results/harbor/validation/ds/run_config_harbor_oracle_agent.yaml").is_file()


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
        agent = config.harbor_ds_agent.responses_api_agents.harbor_oracle_agent
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


@pytest.mark.parametrize("override", [None, "+num_repeats=2"])
def test_validate_plays_each_solution_once_unless_the_command_line_says_otherwise(
    tmp_path, monkeypatch, capsys, validation_cli, override
):
    """A dataset's `[gym.defaults].num_repeats` is an evaluation default; the oracle's result does not vary, so
    `gym dataset validate` adds `+num_repeats=1` unless the command line sets a count, and sizes its report so."""
    write_task(tmp_path / "ds" / "a")
    (tmp_path / "ds" / "dataset.toml").write_text("[gym.defaults]\nnum_repeats = 3\n")
    seen = {}

    def collect(target, tokens):
        monkeypatch.setattr(sys, "argv", ["e2e", *tokens])
        config = global_config.get_global_config_dict()
        seen["num_repeats"] = config.num_repeats
        seen["tokens"] = [token for token in tokens if token.lstrip("+").startswith("num_repeats=")]
        write_jsonl(Path(config.output_jsonl_fpath), [result(repeat=i) for i in range(config.num_repeats)])

    monkeypatch.setattr(main_module, "dispatch", collect)
    validation_cli(*([override] if override else []))
    assert seen["num_repeats"] == (2 if override else 1)
    assert seen["tokens"] == ([override] if override else ["+num_repeats=1"])
    output = capsys.readouterr().out
    assert f"validating {seen['num_repeats']} oracle rollout(s)" in output
    assert ("a [repeat 1]" in output) is bool(override) and "a [repeat 2]" not in output


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
    # Nothing was checked, so `gym dataset validate X && publish` must not go ahead.
    with pytest.raises(SystemExit) as exc:
        validation_cli()
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "a\tunvalidated" in out and "-\t0 tasks validated (no selected task has a solution/)" in out


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
        resolve_agent("hermes_agent"),
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


def _server_hosts(config) -> dict[tuple[str, str], str]:
    """``(instance, server type) -> host`` for every server instance in the merged configuration."""
    hosts = {}
    for name, block in config.items():
        if not hasattr(block, "items"):
            continue
        for type_key in ("responses_api_models", "responses_api_agents", "environment_servers", "resources_servers"):
            if type_key in block:
                (impl,) = block[type_key].keys()
                hosts[(name, type_key)] = block[type_key][impl]["host"]
    return hosts


@pytest.mark.parametrize("in_sandbox", [True, False])
def test_sandbox_model_access_moves_only_the_model_server(tmp_path, monkeypatch, in_sandbox):
    """With otherwise identical inputs, a harness that calls the model from inside the sandbox gets a model
    server on the node's resolved address while the head server (its unauthenticated /global_config_dict_yaml
    endpoint returns the merged configuration, credentials included), the agent server and the environment
    server stay on loopback as on main. A harness that calls the model from the host moves nothing."""
    write_task(tmp_path / "ds" / "a")
    prepared = prepare_target(str(tmp_path / "ds"), output_root=tmp_path / "out")
    from nemo_gym import PARENT_DIR
    from nemo_gym.cli.main import _merge_config_paths

    # The same harness, config and overrides; only the declaration differs (the real one for Hermes).
    agent = resolve_agent("hermes_agent")
    assert agent.runs_in_sandbox is True
    agent = replace(agent, runs_in_sandbox=in_sandbox)
    _, tokens = build_run(
        prepared,
        agent,
        sandbox="docker",
        overrides=[
            f"+config_paths=[{PARENT_DIR / 'responses_api_models/openai_model/configs/openai_model.yaml'}]",
            "+policy_base_url=http://127.0.0.1:1/v1",
            "+policy_api_key=unused",
            "+policy_model_name=test",
        ],
    )
    monkeypatch.setattr(sys, "argv", ["e2e", *_merge_config_paths(tokens)])
    monkeypatch.setattr(global_config, "_GLOBAL_CONFIG_DICT", None)
    resolved = []
    monkeypatch.setattr(global_config, "gethostbyname", lambda _: resolved.append("192.0.2.20") or "192.0.2.20")
    config = global_config.get_global_config_dict()

    hosts = _server_hosts(config)
    assert config.use_absolute_ip is agent.runs_in_sandbox
    assert config.head_server.host == "127.0.0.1"
    model_hosts = {key: host for key, host in hosts.items() if key[1] == "responses_api_models"}
    other_hosts = {key: host for key, host in hosts.items() if key[1] != "responses_api_models"}
    assert model_hosts == {
        ("policy_model", "responses_api_models"): "192.0.2.20" if agent.runs_in_sandbox else "127.0.0.1"
    }
    assert {("harbor_ds_agent", "responses_api_agents"), ("harbor_ds_environment", "environment_servers")} <= set(
        other_hosts
    )
    assert set(other_hosts.values()) == {"127.0.0.1"}
    assert resolved == (["192.0.2.20"] if agent.runs_in_sandbox else [])


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


@pytest.fixture
def oracle_collection(monkeypatch, validation_cli):
    """Real CLI, preparation and collector; replace only server startup and rollout execution.

    ``payload(row)`` is what each dispatched rollout returns, ``dispatched`` every row that ran, and
    ``invoke(*flags)`` a resuming ``gym dataset validate`` of ``ds``.
    """
    from types import SimpleNamespace

    from nemo_gym.cli.eval import RunHelper
    from nemo_gym.rollout_collection import RolloutCollectionHelper, _CompletedRollout

    collection = SimpleNamespace(dispatched=[], payload=lambda row: result())

    async def complete(row):
        collection.dispatched.append(row)
        return _CompletedRollout(row=row, result=collection.payload(row), rollout_latency_ms=1.0)

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

    collection.invoke = invoke
    return collection


def test_validate_resume_rejects_changed_digest_and_preserves_unchanged_retries(
    tmp_path, monkeypatch, validation_cli, capsys, oracle_collection
):
    task = write_task(tmp_path / "ds" / "a")
    dispatched, invoke = oracle_collection.dispatched, oracle_collection.invoke
    first_attempt = True

    def payload(row):
        if first_attempt:
            return result(_ng_failure_class="agent_request_failed", _ng_failure_message="temporary failure")
        if "exit 99" in (task / "solution/solve.sh").read_text():
            return result(reward=0.0, response={"metadata": {"oracle": "failed"}})
        return result(response={"metadata": {"oracle": "solved"}})

    oracle_collection.payload = payload
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


def test_validate_resume_repairs_an_interrupted_final_record(tmp_path, monkeypatch, capsys, oracle_collection):
    """A run killed while appending its last result leaves a half-written final line. The resume guard reads
    the cache with the collector's tolerance, so the collector's own repair runs and only the lost task reruns.
    A complete unterminated final record still counts (and is still rejected when another agent wrote it), and
    a damaged interior record is refused by file and line with the cache left as it was."""
    from nemo_gym.tasks.harbor.validation_report import _validation_key

    write_task(tmp_path / "ds" / "a")
    write_task(tmp_path / "ds" / "b")
    dispatched, invoke = oracle_collection.dispatched, oracle_collection.invoke
    # Non-ASCII output, as solve.sh logs produce, so the cut below can land inside a multibyte character.
    oracle_collection.payload = lambda row: result(
        _validation_key(row)[1], index=_validation_key(row)[2], output="日本語の出力"
    )
    invoke()
    rollouts = tmp_path / "results/harbor/validation/ds/rollouts.jsonl"
    first, second = rollouts.read_text().splitlines()
    kept, lost = (json.loads(line)["_ng_task_id"]["task_id"] for line in (first, second))
    assert len(dispatched) == 2 and {kept, lost} == {"a", "b"}

    # Interrupted mid-append: one complete record and part of the next, cut one byte into 日, no newline.
    cut = second.encode().index("日".encode()) + 1
    rollouts.write_bytes(first.encode() + b"\n" + second.encode()[:cut])
    invoke()
    assert [_validation_key(row)[1] for row in dispatched[2:]] == [lost]
    rows = [json.loads(line) for line in rollouts.read_text().splitlines()]
    assert rows[0] == json.loads(first) and [row["_ng_task_id"]["task_id"] for row in rows] == [kept, lost]
    out = capsys.readouterr().out
    assert "Dropped a truncated final line" in out and "oracle solved\t1.0" in out

    # A complete final record without a newline is a result: nothing reruns.
    rollouts.write_text(rollouts.read_text().rstrip("\n"))
    invoke()
    assert len(dispatched) == 3

    # Still a result when another agent wrote it: the resume is refused as before.
    foreign = dict(rows[1])
    del foreign["response"]
    rollouts.write_text(first + "\n" + json.dumps(foreign))
    with pytest.raises(SystemExit) as exc:
        invoke()
    assert exc.value.code == 1 and len(dispatched) == 3
    assert "were not produced by the oracle agent" in " ".join(capsys.readouterr().out.split())
    assert rollouts.read_text() == first + "\n" + json.dumps(foreign)

    # Damage the collector does not repair is named, not resumed over, and the cache is left alone.
    rollouts.write_text('{"half\n' + second + "\n")
    with pytest.raises(SystemExit) as exc:
        invoke()
    assert exc.value.code == 1 and len(dispatched) == 3
    out = " ".join(capsys.readouterr().out.split())
    assert "results/harbor/validation/ds/rollouts.jsonl, line 1: not a JSON record (Unterminated string" in out
    assert rollouts.read_text() == '{"half\n' + second + "\n"


def test_validate_resume_rejects_results_another_agent_wrote(
    tmp_path, monkeypatch, validation_cli, capsys, oracle_collection
):
    """A model evaluation pointed at the same rollouts file leaves reward-1.0 rows without oracle metadata. The
    task keys and digests still match, so only the rows tell: a resume refuses them, while the oracle's own
    cache (the control) resumes without dispatching anything."""
    write_task(tmp_path / "ds" / "a")
    dispatched, invoke = oracle_collection.dispatched, oracle_collection.invoke
    invoke()
    rollouts = tmp_path / "results/harbor/validation/ds/rollouts.jsonl"
    cached_inputs = materialized_path_for(rollouts).read_bytes()
    invoke()
    assert len(dispatched) == 1

    model_row = json.loads(rollouts.read_text())
    del model_row["response"]
    write_jsonl(rollouts, [model_row])
    with pytest.raises(SystemExit) as exc:
        invoke()
    assert exc.value.code == 1 and len(dispatched) == 1
    out = " ".join(capsys.readouterr().out.split())  # the error is printed wrapped
    assert "1 cached result(s) were not produced by the oracle agent (no oracle metadata): a." in out
    assert "+resume_from_cache=false" in out and "+output_jsonl_fpath" in out
    assert materialized_path_for(rollouts).read_bytes() == cached_inputs
    assert json.loads(rollouts.read_text()) == model_row

    # The report never passes such a row either, should one reach it without the resume guard.
    assert not summarize_validation(rollouts, [], expected=[result()]).ok
    # An explicit fresh run replaces the foreign rows with the oracle's.
    monkeypatch.setattr(global_config, "_GLOBAL_CONFIG_DICT", None)
    validation_cli("+disable_aggregation=true", "+resume_from_cache=false")
    assert len(dispatched) == 2
    assert json.loads(rollouts.read_text())["response"]["metadata"]["oracle"] == "solved"


@pytest.mark.usefixtures("validation_cli")
def test_validate_renders_a_template_where_eval_run_does(tmp_path, monkeypatch, capsys):
    """A template is rendered into results/harbor/<taskset>/rendered/ by `gym dataset validate` as by `gym eval run`:
    the rendered folder is the dataset root prepare records into, so one copy means one manifest and a template's
    image built and pushed once, while the oracle's rows and results still live under results/harbor/validation/."""
    from nemo_gym.tasks.harbor.records import read_built_images
    from tests.unit_tests.test_harbor_authoring import write_template

    write_template(tmp_path / "ds")
    monkeypatch.setattr(global_config, "gethostbyname", lambda _: "192.0.2.20")
    outputs = []

    def collect(target, tokens):
        monkeypatch.setattr(sys, "argv", ["e2e", *tokens])
        output = Path(global_config.get_global_config_dict().output_jsonl_fpath)
        outputs.append(output.resolve())
        write_jsonl(output, [result("q-1"), result("q-2", index=1)])

    monkeypatch.setattr(main_module, "dispatch", collect)
    monkeypatch.setattr(sys, "argv", ["gym", "dataset", "validate", str(tmp_path / "ds"), "--sandbox", "docker"])
    main_module.main()
    rendered = (tmp_path / "results/harbor/ds/rendered").resolve()
    assert sorted(p.name for p in rendered.iterdir() if p.is_dir()) == ["q-1", "q-2"]
    assert not (tmp_path / "results/harbor/validation/ds/rendered").exists()
    assert outputs == [(tmp_path / "results/harbor/validation/ds/rollouts.jsonl").resolve()]
    out = capsys.readouterr().out
    assert (
        f"Rendered 2 task(s) from {tmp_path / 'ds'} into {rendered}" in out and "q-2 [repeat 0]\toracle solved" in out
    )
    manifest = (rendered / "manifest.toml").read_bytes()
    built = {"abc": {"image": "registry.invalid/hb__abc", "tasks": ["q-1", "q-2"], "tool": "buildah", "pushed": True}}
    (rendered / "built-images.json").write_text(json.dumps(built))
    inodes = {p.name: p.stat().st_ino for p in rendered.iterdir()}

    # The model run prepares the same template into the same folder: the manifest, the built image and every rendered
    # task folder are the ones validate left, and its rows and results are its own.
    monkeypatch.setattr(global_config, "_GLOBAL_CONFIG_DICT", None)
    model = ["gym", "eval", "run", str(tmp_path / "ds"), "--agent", "hermes_agent", "--sandbox", "docker"]
    model += ["--model-type", "vllm_model", "+policy_model_name=m", "+policy_base_url=http://localhost:1"]
    monkeypatch.setattr(sys, "argv", [*model, "+policy_api_key=none"])
    main_module.main()
    assert outputs[-1] == (tmp_path / "results/harbor/ds/rollouts.jsonl").resolve()
    assert (rendered / "manifest.toml").read_bytes() == manifest and read_built_images(rendered) == built
    assert {p.name: p.stat().st_ino for p in rendered.iterdir()} == inodes
    assert sorted(p.name for p in (tmp_path / "results/harbor").iterdir()) == ["ds", "validation"]


@pytest.mark.usefixtures("validation_cli")
def test_eval_run_with_the_oracle_is_a_validation_run(tmp_path, monkeypatch, capsys):
    """`gym eval run --agent harbor_oracle` writes under results/harbor/validation/ like `gym dataset validate`, so a
    model run on the same target, resumed or not, never finds the reference solution's rewards as its own."""
    write_task(tmp_path / "ds" / "a")
    monkeypatch.setattr(global_config, "gethostbyname", lambda _: "192.0.2.20")
    outputs = []

    def collect(target, tokens):
        monkeypatch.setattr(sys, "argv", ["e2e", *tokens])
        output = Path(global_config.get_global_config_dict().output_jsonl_fpath)
        outputs.append(output.resolve())
        assert not output.exists() and not materialized_path_for(output).exists()
        write_jsonl(output, [result()])

    monkeypatch.setattr(main_module, "dispatch", collect)
    oracle = ["gym", "eval", "run", str(tmp_path / "ds"), "--agent", "harbor_oracle", "--sandbox", "docker"]
    monkeypatch.setattr(sys, "argv", oracle)
    main_module.main()
    validation = (tmp_path / "results/harbor/validation/ds").resolve()
    assert outputs == [validation / "rollouts.jsonl"]
    assert any(validation.glob("run_config*.yaml"))  # the oracle's run config sits beside its results
    assert not (tmp_path / "results/harbor/ds").exists()
    assert "a [repeat 0]\toracle solved\t1.0" in capsys.readouterr().out

    # A model run, resumed, reads its own folder, where the oracle left nothing to resume from.
    monkeypatch.setattr(global_config, "_GLOBAL_CONFIG_DICT", None)
    model = ["gym", "eval", "run", str(tmp_path / "ds"), "--agent", "hermes_agent", "--sandbox", "docker", "--resume"]
    model += ["--model-type", "vllm_model", "+policy_model_name=m", "+policy_base_url=http://localhost:1"]
    monkeypatch.setattr(sys, "argv", [*model, "+policy_api_key=none"])
    main_module.main()
    evaluation = (tmp_path / "results/harbor/ds").resolve()
    assert outputs[-1] == evaluation / "rollouts.jsonl"
    assert not materialized_path_for(evaluation / "rollouts.jsonl").exists()
    assert json.loads((validation / "rollouts.jsonl").read_text())["response"]["metadata"]["oracle"] == "solved"


def test_validate_keeps_a_multi_step_task_whose_solutions_are_per_step(tmp_path, monkeypatch, capsys, validation_cli):
    """G7 round 24, attempt 4: the MemoryArena task keeps its reference solutions in `steps/<name>/solution/` and has
    no top-level `solution/`, so `gym dataset validate` filtered it out as "no solution/" before any server started.
    A multi-step task counts as validatable when every step has a solution, its own or the task's; a step with
    neither leaves the task unvalidated, as a single-turn task without `solution/` is."""
    from tests.unit_tests.test_harbor_tasks import write_multi_step_task

    def multi_step(name: str, *, own: tuple[str, ...], top_level: bool) -> None:
        root = write_multi_step_task(tmp_path / "ds" / name)
        if not top_level:
            (root / "solution" / "solve.sh").unlink()
            (root / "solution").rmdir()
        for step in own:
            (root / "steps" / step / "solution").mkdir()
            (root / "steps" / step / "solution" / "solve.sh").write_text("#!/bin/bash\necho ok\n")

    multi_step("per_step", own=("01_plan", "02_implement"), top_level=False)
    multi_step("mixed", own=("01_plan",), top_level=True)
    multi_step("incomplete", own=("01_plan",), top_level=False)

    def collect(target, tokens):
        monkeypatch.setattr(sys, "argv", ["e2e", *tokens])
        config = global_config.get_global_config_dict()
        agent = config.harbor_ds_agent.responses_api_agents.harbor_oracle_agent
        rows = [json.loads(line) for line in Path(agent.datasets[0].jsonl_fpath).read_text().splitlines()]
        assert [row["task_id"]["task_id"] for row in rows] == ["mixed", "per_step"]
        assert all(len(row["task_input"]["turns"]) == 2 for row in rows)
        write_jsonl(Path(config.output_jsonl_fpath), [result("mixed", index=0), result("per_step", index=1)])

    monkeypatch.setattr(main_module, "dispatch", collect)
    validation_cli()
    out = capsys.readouterr().out
    # Listed as unvalidated, as a single-turn task without solution/ is (the report's existing rule for them).
    assert "incomplete\tunvalidated (no solution/)" in out
    assert "per_step [repeat 0]\toracle solved\t1.0" in out and "mixed [repeat 0]\toracle solved\t1.0" in out
