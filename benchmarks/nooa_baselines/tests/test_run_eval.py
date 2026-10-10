# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.nooa_baselines import run_eval
from benchmarks.nooa_baselines.run_eval import pipeline_result_valid, reconcile_coverage
from nemo_gym.path_utils import failures_path_for


@pytest.fixture(autouse=True)
def policy_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_POLICY_KEY", "fixture-secret")


def identity(name: str, taskset: str = "tb") -> dict[str, str]:
    return {"taskset": taskset, "task_id": name}


def result(name: str, **extra: object) -> dict[str, object]:
    return {"_ng_task_id": identity(name), "evaluation_completed": True, "reward": 0, **extra}


def inputs(*names: str) -> list[dict[str, object]]:
    return [{"task_id": identity(name)} for name in names]


def test_all_attempted_preserves_one_ungraded_failure_without_a_synthetic_reward() -> None:
    failed = {"_ng_task_id": identity("b"), "_ng_failure_class": "environment_server_failed"}
    coverage = reconcile_coverage(inputs("a", "b"), [result("a")], [failed], "tb")
    assert coverage["coverage_complete"]
    assert coverage["expected_tasks"] == coverage["unique_attempted"] == 2
    assert coverage["valid_results"] == coverage["ungraded_tasks"] == 1
    assert coverage["completed_result_ids"] == [identity("a")]
    assert coverage["ungraded_task_ids"] == [identity("b")]
    assert coverage["failed_attempt_ids"] == [{"task_id": identity("b"), "attempt_index": 0, "sidecar_line": 1}]
    assert "reward" not in failed


def test_missing_task_and_unexpected_identity_prevent_full_completion() -> None:
    coverage = reconcile_coverage(inputs("a", "b"), [result("a"), result("c")], [], "tb")
    assert not coverage["coverage_complete"]
    assert coverage["missing_ids"] == [identity("b")]
    assert coverage["unexpected_ids"] == [identity("c")]
    assert coverage["unique_attempted"] == 1


@pytest.mark.parametrize("kind", ["result", "failure", "cross_file"])
def test_duplicate_attempts_never_inflate_coverage(kind: str) -> None:
    failed = {"_ng_task_id": identity("a"), "_ng_failure_class": "environment_server_failed"}
    rows = [result("a"), result("a")] if kind == "result" else [result("a")] if kind == "cross_file" else []
    failures = [failed, failed] if kind == "failure" else [failed] if kind == "cross_file" else []
    coverage = reconcile_coverage(inputs("a"), rows, failures, "tb")
    assert coverage["unique_attempted"] == 1
    assert not coverage["coverage_complete"]
    assert coverage["duplicates"][0]["records"] == 2


def test_native_transport_failure_uses_only_valid_exact_input_index() -> None:
    failed = {"_ng_task_index": 1, "_ng_failure_class": "agent_request_failed"}
    coverage = reconcile_coverage(inputs("a", "b"), [result("a")], [failed], "tb")
    assert coverage["coverage_complete"]
    assert coverage["failure_identity_from_input_index_lines"] == [1]
    assert coverage["failed_attempt_ids"][0]["task_id"] == identity("b")
    for index in (-1, 2, True, "1"):
        invalid = reconcile_coverage(inputs("a", "b"), [result("a")], [{**failed, "_ng_task_index": index}], "tb")
        assert not invalid["coverage_complete"]
        assert invalid["missing_ids"] == [identity("b")]
    conflict = reconcile_coverage(inputs("a", "b"), [result("a")], [{**failed, "_ng_task_id": identity("a")}], "tb")
    assert conflict["invalid_rows"] == [{"source": "failures", "line": 1}]


def test_masked_verifier_result_is_attempted_but_not_graded_or_canary_valid() -> None:
    row = result("a", evaluation_completed=False, mask_sample=True, failure_kind="verifier_error")
    coverage = reconcile_coverage(inputs("a"), [row], [], "tb")
    assert coverage["coverage_complete"]
    assert coverage["valid_results"] == 0
    assert coverage["ungraded_result_ids"] == [identity("a")]
    assert not pipeline_result_valid(row, "tb")
    assert pipeline_result_valid(result("a"), "tb")


def test_taskset_is_part_of_identity_and_duplicate_inputs_are_rejected() -> None:
    expected = [{"task_id": identity("a", taskset)} for taskset in ("one", "two")]
    rows = [result("a", _ng_task_id=row["task_id"]) for row in expected]
    assert reconcile_coverage(expected, rows, [], "tb")["unique_attempted"] == 2
    with pytest.raises(ValueError, match="unique native"):
        reconcile_coverage(inputs("a", "a"), [], [], "tb")


def test_canonical_failure_sidecar_path_and_missing_failure_marker(tmp_path: Path) -> None:
    assert failures_path_for(tmp_path / "rollouts.jsonl") == tmp_path / "rollouts_failures.jsonl"
    coverage = reconcile_coverage(inputs("a"), [], [{"_ng_task_id": identity("a")}], "tb")
    assert not coverage["coverage_complete"]
    assert coverage["missing_ids"] == [identity("a")]


@pytest.mark.parametrize("with_failure,exit_code,passed", [(True, 0, True), (False, 0, False), (True, 7, False)])
def test_full_completion_requires_all_identities_and_successful_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_failure: bool, exit_code: int, passed: bool
) -> None:
    for folder in ("model", "data", "private", "tb/canary"):
        (tmp_path / folder).mkdir(parents=True)
    model_path = tmp_path / "model/model.json"
    model_path.write_text(
        json.dumps(
            {
                "job_id": "7",
                "base_url": "http://model.invalid/v1",
                "served_model": "fixture-policy",
                "api_key_env": "TEST_POLICY_KEY",
            }
        )
    )
    (tmp_path / "model/preflight.json").write_text(
        json.dumps(
            {
                "passed": True,
                "job_id": "7",
                "model_manifest_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            }
        )
    )
    source_path = tmp_path / "source-manifest.json"
    source_path.write_text("{}")
    input_path = tmp_path / "data/terminal-bench-2.1-nooa-89.jsonl"
    input_path.write_text("".join(json.dumps(row) + "\n" for row in inputs("a", "b")))
    (tmp_path / "private/opensandbox.json").write_text(json.dumps({"domain": "unused.invalid", "api_key": "fixture"}))
    protocol = {
        "overlays_sha256": hashlib.sha256(b"").hexdigest(),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "model_manifest_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "source_manifest_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
    }
    (tmp_path / "tb/canary/completion.json").write_text(json.dumps({"pipeline_passed": True, "protocol": protocol}))
    (tmp_path / "tb/rollouts.jsonl").write_text(json.dumps(result("a")) + "\n")
    if with_failure:
        (tmp_path / "tb/rollouts_failures.jsonl").write_text(
            json.dumps({"_ng_task_id": identity("b"), "_ng_failure_class": "environment_server_failed"}) + "\n"
        )

    def subprocess_result(command, **kwargs):
        import yaml

        config = yaml.safe_load(Path(command[-1]).read_text())
        assert config["server_spinup_timeout_seconds"] == 3600
        assert config["skip_venv_if_present"] is False
        assert kwargs["env"]["UV_LINK_MODE"] == "hardlink"
        assert "UV_VENV_CLEAR" not in kwargs["env"]
        return SimpleNamespace(returncode=exit_code)

    monkeypatch.setenv("UV_LINK_MODE", "copy")
    monkeypatch.setenv("UV_VENV_CLEAR", "true")
    monkeypatch.setattr(run_eval, "prepare_full_resume", lambda *args, **kwargs: {"coverage_fixture": True})
    monkeypatch.setattr(run_eval.subprocess, "run", subprocess_result)
    monkeypatch.setattr("sys.argv", ["run_eval", "--run-root", str(tmp_path), "--benchmark", "tb", "--phase", "full"])
    with pytest.raises(SystemExit) as stopped:
        run_eval.main()
    completion = json.loads((tmp_path / "tb/full/completion.json").read_text())
    assert completion["pipeline_passed"] is passed
    assert completion["coverage"]["valid_results"] == 1
    assert completion["coverage"]["unique_attempted"] == (2 if with_failure else 1)
    assert (stopped.value.code == 0) is passed


@pytest.mark.parametrize("benchmark,count", [("swe", 731), ("tb", 89), ("gdp", 220)])
@pytest.mark.parametrize("collection_fails", [False, True])
def test_real_cli_uses_exact_prepared_native_rows_and_owns_shutdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, benchmark: str, count: int, collection_fails: bool
) -> None:
    """Exercise actual CLI dispatch/config checks; replace only runtime side effects."""
    import importlib

    import yaml

    from nemo_gym.cli import eval as cli_eval
    from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
    from nemo_gym.rollout_collection import RolloutCollectionHelper
    from nemo_gym.train_data_utils import TrainDataProcessor

    cli_main = importlib.import_module("nemo_gym.cli.main")
    recipe, _, environment, _, _ = run_eval.BENCHMARKS[benchmark]
    for key, value in {
        "POLICY_BASE_URL": "http://model.invalid/v1",
        "POLICY_API_KEY": "fixture",
        "POLICY_MODEL_NAME": "fixture",
        "NOOA_SWE_BASELINE_RUN_DIR": str(tmp_path),
        "OPENSANDBOX_DOMAIN": "sandbox.invalid",
        "OPENSANDBOX_API_KEY": "fixture",
        "GDPVAL_CONTAINER_PATH": "/images/audited.sif",
        "PERSIST_DELIVERABLES_DIR": str(tmp_path / "deliverables"),
    }.items():
        monkeypatch.setenv(key, value)
    include = [recipe]
    if benchmark != "swe":
        include.append("responses_api_models/vllm_model/configs/nooa_vllm_model.yaml")
    if benchmark == "tb":
        include.append("nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml")
    input_path = tmp_path / "native.jsonl"
    original = b"".join(
        (json.dumps({"task_id": identity(str(index), benchmark), "task_input": {}}) + "\n").encode()
        for index in range(count)
    )
    input_path.write_bytes(original)
    config, prepared = run_eval.prepare_e2e_config(
        {
            "config_paths": include,
            "output_jsonl_fpath": str(tmp_path / "rollouts.jsonl"),
            "agent_name": environment + "_agent",
            "policy_base_url": "http://model.invalid/v1",
            "policy_api_key": "fixture",
            "policy_model_name": "fixture",
            "limit": 1,
            "num_samples_in_parallel": 1,
            "max_resident_rollout_tasks": 1,
            "resume_from_cache": True,
            "disable_health_check": True,
            "server_spinup_timeout_seconds": 3600,
            "uv_venv_dir": str(tmp_path / "server-venvs" / benchmark),
        },
        input_path=input_path,
    )
    assert "input_jsonl_fpath" not in config and "config_paths" not in config
    assert prepared.read_bytes() == original
    readiness = run_eval.server_venv_readiness(config)
    assert len(readiness) == (5 if benchmark == "gdp" else 4)
    assert not any(readiness.values())
    config["skip_venv_if_present"] = all(readiness.values())
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    events = []

    def parse_actual_cli():
        return GlobalConfigDictParser().parse(GlobalConfigDictParserConfig(skip_load_from_dotenv=True, offline=True))

    class Servers:
        def start(self, config):
            events.append("ready")

        def shutdown(self):
            events.append("shutdown")

    async def collect(self, rollout):
        assert events == ["ready"]
        assert Path(rollout.input_jsonl_fpath) == prepared
        assert Path(rollout.input_jsonl_fpath).read_bytes() == original
        assert rollout.limit == 1 and rollout.resume_from_cache
        events.append("collect")
        if collection_fails:
            raise RuntimeError("collection failed")

    def forbid_repreparation(*args, **kwargs):
        pytest.fail("Immutable native input must not be regenerated")

    monkeypatch.setattr(cli_eval, "get_global_config_dict", parse_actual_cli)
    monkeypatch.setattr(cli_eval, "RunHelper", Servers)
    monkeypatch.setattr(RolloutCollectionHelper, "run_from_config", collect)
    monkeypatch.setattr(TrainDataProcessor, "run", forbid_repreparation)
    monkeypatch.setattr("sys.argv", ["gym", "eval", "run", "--config", str(config_path)])
    if collection_fails:
        with pytest.raises(RuntimeError, match="collection failed"):
            cli_main.main()
    else:
        cli_main.main()
    assert events == ["ready", "collect", "shutdown"]
    assert input_path.read_bytes() == prepared.read_bytes() == original


def test_prepared_input_mismatch_refuses_to_overwrite_existing_rows(tmp_path: Path) -> None:
    source = tmp_path / "native.jsonl"
    source.write_bytes(b'{"task_id": "original"}\n')
    config = {"config_paths": [], "output_jsonl_fpath": str(tmp_path / "rollouts.jsonl")}
    _, prepared = run_eval.prepare_e2e_config(config, input_path=source)
    source.write_bytes(b'{"task_id": "changed"}\n')
    with pytest.raises(ValueError, match="differs from the immutable"):
        run_eval.prepare_e2e_config(config, input_path=source)
    assert prepared.read_bytes() == b'{"task_id": "original"}\n'


def test_reuse_requires_every_server_completion_marker_and_interpreter(tmp_path: Path) -> None:
    from nemo_gym.cli._venv_setup import SETUP_COMPLETE_MARKER, setup_environment

    config = {
        "uv_venv_dir": str(tmp_path),
        "one": {"responses_api_models": {"vllm_model": {"entrypoint": "app.py"}}},
        "two": {"environment_servers": {"single_agent_turn": {"entrypoint": "app.py"}}},
    }
    readiness = run_eval.server_venv_readiness(config)
    assert len(readiness) == 2 and not any(readiness.values())
    for name in readiness:
        venv = Path(name)
        (venv / "bin").mkdir(parents=True)
        (venv / "bin/python").touch()
        (venv / "bin/activate").touch()
        # Match an interrupted install: interpreter, activation and some packages.
        (venv / "preserved-package").write_text("existing bytes")
    assert not any(run_eval.server_venv_readiness(config).values())
    first, second = map(Path, sorted(readiness))
    assert setup_environment(first, "exit 7", skip_if_ready=False) == 7
    assert not (first / SETUP_COMPLETE_MARKER).exists()
    assert not any(run_eval.server_venv_readiness(config).values())
    assert setup_environment(first, "true", skip_if_ready=False) == 0
    assert list(run_eval.server_venv_readiness(config).values()) == [True, False]
    assert setup_environment(second, "true", skip_if_ready=False) == 0
    assert all(run_eval.server_venv_readiness(config).values())
    for venv in (first, second):
        assert (venv / "preserved-package").read_text() == "existing bytes"
    (second / "bin/python").unlink()
    assert not all(run_eval.server_venv_readiness(config).values())


@pytest.mark.parametrize("count", [731, 89, 220])
def test_full_resume_expands_real_collector_schedule_and_never_repeats_canary(tmp_path: Path, count: int) -> None:
    from nemo_gym.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper

    source = tmp_path / "native.jsonl"
    source.write_text(
        "".join(json.dumps({"task_id": identity(str(i)), "task_input": {"index": i}}) + "\n" for i in range(count))
    )
    output = tmp_path / "rollouts.jsonl"
    config = {
        "output_jsonl_fpath": str(output),
        "environment_server_routes": {"tb": "environment"},
        "num_repeats": 1,
        "resume_from_cache": True,
    }
    collector = RolloutCollectionHelper()
    limited = RolloutCollectionConfig.model_validate(config | {"input_jsonl_fpath": str(source), "limit": 1})
    canary_inputs = collector._preprocess_rows_from_config(limited)
    old_schedule = (json.dumps(canary_inputs[0]) + "\n").encode()
    limited.materialized_jsonl_fpath.write_bytes(old_schedule)
    output.write_text(json.dumps(result("0", _ng_task_index=0, _ng_rollout_index=0)) + "\n")
    old_output = output.read_bytes()
    capture = tmp_path / "0-0.capture.jsonl"
    capture.write_bytes(b'{"status_code":200}\n')
    phase = tmp_path / "full"
    phase.mkdir()
    receipt = run_eval.prepare_full_resume(config, input_path=source, phase_dir=phase, benchmark="tb")
    assert receipt["previous_rows"] == 1 and receipt["full_rows"] == count
    assert (phase / "materialized-inputs-before-full.jsonl").read_bytes() == old_schedule
    remaining, _, completed, _ = collector._load_from_cache(limited.model_copy(update={"limit": None}))
    assert len(remaining) == count - 1 and len(completed) == 1
    assert [row["_ng_task_index"] for row in remaining] == list(range(1, count))
    assert [row["task_id"] for row in remaining] == [identity(str(i)) for i in range(1, count)]
    assert output.read_bytes() == old_output and capture.read_bytes() == b'{"status_code":200}\n'
    again = run_eval.prepare_full_resume(config, input_path=source, phase_dir=phase, benchmark="tb")
    assert again["previous_rows"] == count and "preserved_previous_path" not in again


@pytest.mark.parametrize("conflict", ["task_input", "result_identity", "missing_cache", "invalid_canary"])
def test_full_resume_refuses_conflicting_history_without_mutation(tmp_path: Path, conflict: str) -> None:
    from nemo_gym.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper

    source = tmp_path / "native.jsonl"
    source.write_text(json.dumps({"task_id": identity("a"), "task_input": {"original": True}}) + "\n")
    output = tmp_path / "rollouts.jsonl"
    config = {"output_jsonl_fpath": str(output), "environment_server_routes": {"tb": "environment"}}
    collection = RolloutCollectionConfig.model_validate(config | {"input_jsonl_fpath": str(source)})
    row = RolloutCollectionHelper()._preprocess_rows_from_config(collection)[0]
    if conflict == "task_input":
        row["task_input"] = {"changed": True}
    if conflict != "missing_cache":
        collection.materialized_jsonl_fpath.write_text(json.dumps(row) + "\n")
    output.write_text(
        json.dumps(
            result(
                "wrong" if conflict == "result_identity" else "a",
                _ng_task_index=0,
                _ng_rollout_index=0,
                evaluation_completed=conflict != "invalid_canary",
            )
        )
        + "\n"
    )
    old = output.read_bytes()
    before = collection.materialized_jsonl_fpath.read_bytes() if collection.materialized_jsonl_fpath.exists() else None
    with pytest.raises(ValueError):
        run_eval.prepare_full_resume(config, input_path=source, phase_dir=tmp_path, benchmark="tb")
    assert output.read_bytes() == old
    assert (
        collection.materialized_jsonl_fpath.read_bytes() if collection.materialized_jsonl_fpath.exists() else None
    ) == before
    assert not (tmp_path / "materialized-inputs-before-full.jsonl").exists()


@pytest.mark.parametrize("mode", ["explicit", "missing", "wrong_manifest", "different_model", "different_overlay"])
def test_full_resume_source_bridge_is_explicit_and_keeps_input_and_model_pinned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    import yaml

    from nemo_gym.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper

    for folder in ("model", "data", "gdp/canary"):
        (tmp_path / folder).mkdir(parents=True)
    model = tmp_path / "model/model.json"
    model.write_text(
        json.dumps(
            {
                "job_id": "7",
                "base_url": "http://model.invalid/v1",
                "served_model": "fixture-policy",
                "api_key_env": "TEST_POLICY_KEY",
            }
        )
    )
    (model.parent / "preflight.json").write_text(
        json.dumps(
            {"passed": True, "job_id": "7", "model_manifest_sha256": hashlib.sha256(model.read_bytes()).hexdigest()}
        )
    )
    preserved = tmp_path / "old-source-manifest.json"
    preserved.write_text('{"source":"canary"}')
    (tmp_path / "source-manifest.json").write_text('{"source":"reviewed-launcher-fix"}')
    source = tmp_path / "data/gdpval-nooa-220.jsonl"
    source.write_text(
        "".join(json.dumps({"task_id": identity(str(i), "gdpval-nooa"), "task_input": {}}) + "\n" for i in range(2))
    )
    protocol = {
        "overlays_sha256": hashlib.sha256(b"").hexdigest(),
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "model_manifest_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "source_manifest_sha256": hashlib.sha256(preserved.read_bytes()).hexdigest(),
    }
    (tmp_path / "gdp/canary/completion.json").write_text(json.dumps({"pipeline_passed": True, "protocol": protocol}))
    generation = tmp_path / "generation.json"
    generation.write_text("{}")
    output = tmp_path / "gdp/rollouts.jsonl"
    output.write_text(
        json.dumps(
            {
                "_ng_task_id": identity("0", "gdpval-nooa"),
                "_ng_task_index": 0,
                "_ng_rollout_index": 0,
                "execute_only": True,
                "generation_manifest": str(generation),
            }
        )
        + "\n"
    )
    limited = RolloutCollectionConfig(
        input_jsonl_fpath=str(source),
        output_jsonl_fpath=str(output),
        limit=1,
        environment_server_routes={"gdpval-nooa": "gdpval_nooa"},
    )
    limited.materialized_jsonl_fpath.write_text(
        json.dumps(RolloutCollectionHelper()._preprocess_rows_from_config(limited)[0]) + "\n"
    )
    if mode == "wrong_manifest":
        preserved.write_text('{"wrong":"manifest"}')
    if mode == "different_model":
        model.write_text(json.dumps({"job_id": "7", "base_url": "http://different.invalid/v1"}))
    argv = ["run_eval", "--run-root", str(tmp_path), "--benchmark", "gdp", "--phase", "full", "--concurrency", "16"]
    if mode == "different_overlay":
        overlay = tmp_path / "changed.yaml"
        overlay.write_text("policy_model_name: different\n")
        argv += ["--config", str(overlay)]
    if mode != "missing":
        argv += ["--canary-source-manifest", str(preserved)]
    called = []

    def launch(command, **kwargs):
        resolved = yaml.safe_load(Path(command[-1]).read_text())
        assert resolved["limit"] is None
        assert resolved["num_samples_in_parallel"] == resolved["max_resident_rollout_tasks"] == 16
        called.append(True)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("sys.argv", argv)
    monkeypatch.setattr(run_eval.subprocess, "run", launch)
    if mode == "explicit":
        with pytest.raises(SystemExit):
            run_eval.main()
        receipt = json.loads((tmp_path / "gdp/full/launch.json").read_text())
        assert receipt["canary_source_bridge"]["canary_source_manifest_sha256"] == protocol["source_manifest_sha256"]
        assert receipt["resume_schedule"]["full_rows"] == 2 and called == [True]
        assert json.loads((tmp_path / "gdp/canary/completion.json").read_text())["protocol"] == protocol
    else:
        with pytest.raises(RuntimeError):
            run_eval.main()
        assert not called


@pytest.mark.parametrize("separator", ["\u0085", "\u2028", "\u2029"])
def test_physical_lf_preserves_unicode_inside_json_strings(tmp_path: Path, separator: str) -> None:
    path = tmp_path / "rows.jsonl"
    rows = [{"text": "before" + separator + "after"}, {"text": "next"}]
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    assert list(run_eval.read_jsonl(path)) == rows


@pytest.mark.parametrize("data", [b'{"partial":true}', b"[]\n", b"\n", b"{broken}\n"])
def test_jsonl_rejects_incomplete_or_nonobject_records(tmp_path: Path, data: bytes) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_bytes(data)
    with pytest.raises(ValueError):
        list(run_eval.read_jsonl(path))


def test_native_exit_survives_reporting_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for folder in ("model", "data"):
        (tmp_path / folder).mkdir()
    model = tmp_path / "model/model.json"
    model.write_text(
        json.dumps(
            {"base_url": "http://model.invalid/v1", "served_model": "fixture-policy", "api_key_env": "TEST_POLICY_KEY"}
        )
    )
    (model.parent / "preflight.json").write_text(
        json.dumps({"passed": True, "model_manifest_sha256": hashlib.sha256(model.read_bytes()).hexdigest()})
    )
    (tmp_path / "source-manifest.json").write_text("{}")
    source = tmp_path / "data/terminal-bench-2.1-nooa-89.jsonl"
    source.write_text(json.dumps({"task_id": identity("a"), "task_input": {}}) + "\n")
    monkeypatch.setattr(
        "sys.argv", ["run_eval", "--run-root", str(tmp_path), "--benchmark", "tb", "--phase", "canary"]
    )

    def run(command, **kwargs):
        assert kwargs["env"]["POLICY_API_KEY"] == "fixture-secret"
        assert kwargs["env"]["POLICY_MODEL_NAME"] == "fixture-policy"
        assert "NEMO_GYM_USER" not in kwargs["env"]
        (tmp_path / "tb/rollouts.jsonl").write_bytes(b"{broken}\n")
        return SimpleNamespace(returncode=7)

    monkeypatch.delenv("NEMO_GYM_USER", raising=False)
    monkeypatch.setattr(run_eval.subprocess, "run", run)
    with pytest.raises(json.JSONDecodeError):
        run_eval.main()
    receipt = json.loads((tmp_path / "tb/canary/native-exit.json").read_text())
    assert receipt["exit_code"] == 7
    assert not (tmp_path / "tb/canary/completion.json").exists()
    assert "fixture-secret" not in (tmp_path / "tb/canary/launch.json").read_text()
    with pytest.raises(FileExistsError):
        run_eval.main()


def test_outcome_metadata_discards_large_bodies_without_changing_coverage() -> None:
    row = result("a", _ng_task_index=0, output=[{"body": "large"}], response={"output": "large"})
    compact = run_eval.outcome_metadata(row)
    assert "output" not in compact and "response" not in compact
    assert reconcile_coverage(inputs("a"), [compact], [], "tb") == reconcile_coverage(inputs("a"), [row], [], "tb")


@pytest.mark.parametrize("indices", [[20], [3, 7, 731]])
def test_coverage_preserves_sparse_global_input_indices(indices):
    source = [{"task_id": identity(str(index)), "_ng_task_index": index} for index in indices]
    rows = [result(str(index), _ng_task_index=index) for index in indices[:-1]]
    failure = {"_ng_task_index": indices[-1], "_ng_failure_class": "environment_server_failed"}
    coverage = reconcile_coverage(source, rows, [failure], "tb")
    assert coverage["coverage_complete"] and coverage["unique_attempted"] == len(indices)
    assert coverage["ungraded_task_ids"] == [identity(str(indices[-1]))]
    conflict = reconcile_coverage(source, [result("wrong", _ng_task_index=indices[0])], [], "tb")
    assert not conflict["coverage_complete"] and conflict["invalid_rows"]


@pytest.mark.parametrize("indices", [[20, 20], [-1], [True], ["20"], [None]])
def test_input_index_mapping_rejects_invalid_or_duplicate_indices(indices):
    source = [{"task_id": identity(str(position)), "_ng_task_index": index} for position, index in enumerate(indices)]
    with pytest.raises(ValueError, match="unique nonnegative"):
        reconcile_coverage(source, [], [], "tb")


def test_explicit_index_cannot_collide_with_implicit_input_position():
    source = [{"task_id": identity("a"), "_ng_task_index": 1}, {"task_id": identity("b")}]
    with pytest.raises(ValueError, match="unique nonnegative"):
        reconcile_coverage(source, [], [], "tb")


def test_server_inventory_keeps_launch_secrets_unresolved(monkeypatch, tmp_path: Path) -> None:
    from omegaconf import OmegaConf

    from nemo_gym.global_config import GlobalConfigDictParser

    for name in ("POLICY_BASE_URL", "POLICY_API_KEY", "POLICY_MODEL_NAME", "NOOA_SWE_BASELINE_RUN_DIR"):
        monkeypatch.delenv(name, raising=False)
    _, configs = GlobalConfigDictParser().load_extra_config_paths(["benchmarks/swebench/pro/nooa_baseline.yaml"])
    config = OmegaConf.to_container(OmegaConf.merge(*configs), resolve=False)
    config["uv_venv_dir"] = str(tmp_path / "venvs")
    before = json.dumps(config, sort_keys=True)
    inventory = run_eval.server_venv_readiness(config)
    assert len(inventory) == 4
    assert not any(inventory.values())
    assert any("nooa_single_agent_turn" in path for path in inventory)
    assert config["policy_api_key"] == "${oc.env:POLICY_API_KEY}"
    assert json.dumps(config, sort_keys=True) == before
