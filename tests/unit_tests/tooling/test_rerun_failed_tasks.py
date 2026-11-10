# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fictional original-task reruns retain native scoring inputs and attempt identity."""

import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "rerun_failed_tasks.py"
spec = importlib.util.spec_from_file_location("original_task_rerun_tests", SCRIPT)
utility = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = utility
spec.loader.exec_module(utility)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def source(tmp_path: Path, monkeypatch) -> dict[str, Any]:
    # Some workstations initialize their entire shared temp directory as a Git
    # worktree. Isolate that unrelated marker while retaining this test's own
    # Git markers and exercising the real destination guard below.
    actual_exists = Path.exists
    shared_temp_marker = Path(tempfile.gettempdir()) / ".git"
    monkeypatch.setattr(Path, "exists", lambda path: False if path == shared_temp_marker else actual_exists(path))
    gym_root = tmp_path / "fictional-gym"
    gym_root.mkdir()
    config = gym_root / "original.yaml"
    config.write_text(
        "fictional_native_grader:\n  resources_servers:\n    fictional_original:\n      entrypoint: app.py\n"
        "fictional_original_agent:\n  responses_api_agents:\n    simple_agent:\n      entrypoint: app.py\n"
        "      max_steps: 13\n      resources_server:\n        type: resources_servers\n"
        "        name: fictional_native_grader\n",
        encoding="utf-8",
    )
    (gym_root / ".git").mkdir()
    revision = "a" * 40
    monkeypatch.setattr(utility, "_git_revision", lambda root: revision)
    dataset = tmp_path / "original_dataset.jsonl"
    original = [
        {
            "responses_create_params": {
                "input": [
                    {"role": "system", "content": "Keep the original fictional evaluation protocol."},
                    {"role": "user", "content": "Use the evidence and native rubric for case A."},
                ],
                "tools": [
                    {
                        "type": "function",
                        "name": "search_original",
                        "strict": False,
                        "parameters": {
                            "type": "object",
                            "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
                            "required": ["queries"],
                            "additionalProperties": False,
                        },
                    }
                ],
                "temperature": 0.6,
                "top_p": 1.0,
                "max_output_tokens": 4096,
                "tool_choice": "auto",
                "parallel_tool_calls": False,
            },
            "verifier_metadata": {
                "reference_answer": "Fictional reference A",
                "rubric": [{"id": "A1", "criterion": "Keep the source format"}],
            },
            "ground_truth": {"case": "A", "answer": 23},
            "native_custom_field": ["preserve", {"nested": True}],
        },
        {
            "responses_create_params": {
                "input": [{"role": "user", "content": "Fictional original case B"}],
                "tools": [],
                "temperature": 1.0,
            },
            "verifier_metadata": {
                "reference_answer": "Fictional reference B",
                "expected_function": "native_lookup",
                "expected_params": {"limit": 8},
            },
            "native_custom_field": {"reference_artifact": "fictional-evidence.txt"},
        },
    ]
    write_jsonl(dataset, original)
    routes = {
        "schema_version": "original-task-rerun/routes/v1",
        "gym_revision": revision,
        "benchmarks": {
            "fictional_benchmark": {
                "dataset_path": dataset.name,
                "dataset_sha256": file_hash(dataset),
                "config_paths": [str(config)],
                "agent_name": "fictional_original_agent",
            },
        },
    }
    selection = [
        {
            "benchmark": "fictional_benchmark",
            "orig_trajectory_id": identifier,
            "orig_task_index": task_index,
            "orig_rollout_index": rollout_index,
            "dataset_row_index": row_index,
            "expected_task_sha256": utility.canonical_sha256(original[row_index]),
        }
        for identifier, task_index, rollout_index, row_index in [
            ("original-attempt-A0", 11, 0, 0),
            ("original-attempt-A2", 11, 2, 0),
            ("original-attempt-B1", 29, 1, 1),
        ]
    ]
    routes_path, selection_path = tmp_path / "routes.json", tmp_path / "selection.jsonl"
    write_json(routes_path, routes)
    write_jsonl(selection_path, selection)
    policy = tmp_path / "policy.json"
    write_json(policy, {"policy_model": {"responses_api_models": {"vllm_model": {"model": "fictional-checkpoint"}}}})
    return {
        "root": tmp_path,
        "gym_root": gym_root,
        "revision": revision,
        "dataset": dataset,
        "original": original,
        "config": config,
        "routes": routes,
        "routes_path": routes_path,
        "selection": selection,
        "selection_path": selection_path,
        "policy": policy,
    }


def prepare(source: dict[str, Any], directory: str = "prepared") -> tuple[dict[str, Any], Path]:
    out = source["root"] / directory
    manifest = utility.prepare_dataset(source["selection_path"], source["routes_path"], out)
    return manifest, out


def test_complete_original_rows_and_repeated_attempts_are_preserved(source):
    manifest, out = prepare(source)
    assert manifest["num_rows"] == 3
    assert manifest["num_task_groups"] == 2
    assert manifest["selection_sha256"] == file_hash(source["selection_path"])
    assert manifest["routes_sha256"] == file_hash(source["routes_path"])
    job = manifest["jobs"][0]
    emitted = read_jsonl(out / job["input_path"])
    mapping = read_jsonl(out / job["mapping_path"])
    assert len(emitted) == len(mapping) == 3
    assert [row["orig_trajectory_id"] for row in emitted] == [row["orig_trajectory_id"] for row in source["selection"]]
    for row, selected in zip(emitted, source["selection"]):
        original = source["original"][selected["dataset_row_index"]]
        for key, value in original.items():
            assert row[key] == value, key
        assert row["agent_ref"]["name"] == "fictional_original_agent"
        assert row["orig_task_index"] == selected["orig_task_index"]
        assert row["orig_rollout_index"] == selected["orig_rollout_index"]
    assert emitted[0]["_ng_task_index"] == emitted[1]["_ng_task_index"]
    assert emitted[0]["_ng_rollout_index"] != emitted[1]["_ng_rollout_index"]
    assert emitted[0]["_ng_task_index"] != emitted[2]["_ng_task_index"]
    assert job["config_files"][0]["sha256"] == file_hash(source["config"])
    assert job["gym_revision"] == source["revision"]


@pytest.mark.parametrize(
    "tamper", ["file_hash", "row_hash", "row_index", "config_removed", "agent_conflict", "debug_collision"]
)
def test_binding_failures_are_refused_before_any_output_is_published(source, tamper):
    if tamper == "file_hash":
        source["dataset"].write_text(source["dataset"].read_text() + "\n", encoding="utf-8")
    elif tamper == "row_hash":
        source["selection"][0]["expected_task_sha256"] = "0" * 64
    elif tamper == "row_index":
        source["selection"][0]["dataset_row_index"] = 1
    elif tamper == "config_removed":
        source["config"].unlink()
    else:
        source["original"][0]["agent_ref" if tamper == "agent_conflict" else "orig_trajectory_id"] = (
            {"name": "different_agent"} if tamper == "agent_conflict" else "conflicting-source-id"
        )
        write_jsonl(source["dataset"], source["original"])
        source["routes"]["benchmarks"]["fictional_benchmark"]["dataset_sha256"] = file_hash(source["dataset"])
        for selected in source["selection"]:
            selected["expected_task_sha256"] = utility.canonical_sha256(
                source["original"][selected["dataset_row_index"]]
            )
        write_json(source["routes_path"], source["routes"])
    write_jsonl(source["selection_path"], source["selection"])
    out = source["root"] / "must_not_publish"
    with pytest.raises(ValueError):
        utility.prepare_dataset(source["selection_path"], source["routes_path"], out)
    assert not (out / "prepared_manifest.json").exists()


@pytest.mark.parametrize(
    "change",
    [
        "missing_id",
        "duplicate_id",
        "missing_original_index",
        "negative_index",
        "boolean_index",
        "duplicate_original_attempt",
    ],
)
def test_invalid_original_identities_and_indices_are_not_guessed(source, change):
    selected = source["selection"]
    if change == "missing_id":
        selected[0].pop("orig_trajectory_id")
    elif change == "duplicate_id":
        selected[1]["orig_trajectory_id"] = selected[0]["orig_trajectory_id"]
    elif change == "missing_original_index":
        selected[0].pop("orig_task_index")
    elif change == "negative_index":
        selected[0]["orig_rollout_index"] = -1
    elif change == "boolean_index":
        selected[0]["dataset_row_index"] = True
    else:
        selected[1]["orig_rollout_index"] = selected[0]["orig_rollout_index"]
    write_jsonl(source["selection_path"], selected)
    with pytest.raises(ValueError):
        prepare(source)


def test_private_preparation_refuses_git_checkout_and_existing_destination(source):
    with pytest.raises(ValueError):
        utility.prepare_dataset(source["selection_path"], source["routes_path"], source["gym_root"] / "private-data")
    _, out = prepare(source)
    before = (out / "prepared_manifest.json").read_bytes()
    with pytest.raises(ValueError):
        utility.prepare_dataset(source["selection_path"], source["routes_path"], out)
    assert (out / "prepared_manifest.json").read_bytes() == before


def test_dry_run_delegates_to_original_configs_agent_and_verifier(source):
    _, prepared = prepare(source)
    output = source["root"] / "fresh-run"
    plan = utility.plan_run(prepared / "prepared_manifest.json", source["gym_root"], source["policy"], output)
    assert not output.exists()
    assert len(plan["jobs"]) == 1
    job = plan["jobs"][0]
    command = job["command"]
    rendered = " ".join(command)
    assert command[:3] == [sys.executable, str(SCRIPT), "_worker"]
    assert command[3:] == [
        "--job",
        job["worker_path"],
        "--expected-job-sha256",
        utility.canonical_sha256(job["worker_spec"]),
    ]
    assert job["worker_spec"]["agent_name"] == "fictional_original_agent"
    assert job["worker_spec"]["config_files"] == [
        {"path": str(source["config"]), "sha256": file_hash(source["config"])}
    ]
    assert job["worker_spec"]["policy_config"] == str(source["policy"])
    assert job["worker_spec"]["input_path"] == job["input_path"]
    assert "captured_instruction_following" not in rendered
    assert "single_turn_text_replay" not in rendered
    result = utility.run_prepared(
        prepared / "prepared_manifest.json", source["gym_root"], source["policy"], output, execute=False
    )
    assert result["schema_version"] == "original-task-rerun/run/v1"
    assert not output.exists(), "Dry run must not start Gym or create rollout artifacts"


@pytest.mark.parametrize("changed", ["config", "prepared_input", "prepared_mapping", "policy_hash"])
def test_runtime_revalidates_preparation_and_configuration_pins(source, changed):
    manifest, prepared = prepare(source)
    if changed == "config":
        source["config"].write_text(source["config"].read_text() + "# modified after preparation\n")
    elif changed == "prepared_input":
        path = prepared / manifest["jobs"][0]["input_path"]
        path.write_text(path.read_text() + "\n")
    elif changed == "prepared_mapping":
        path = prepared / manifest["jobs"][0]["mapping_path"]
        path.write_text(path.read_text() + "\n")
    else:
        # A prepared manifest cannot silently point at another revision's dataset.
        manifest["jobs"][0]["gym_revision"] = "f" * 40
        write_json(prepared / "prepared_manifest.json", manifest)
    with pytest.raises(ValueError):
        utility.plan_run(
            prepared / "prepared_manifest.json", source["gym_root"], source["policy"], source["root"] / "run"
        )


@pytest.mark.parametrize(
    "forbidden", ["top_level_grader", "top_level_sampling", "nested_tools", "nested_resources", "nested_agent"]
)
def test_checkpoint_overlay_cannot_replace_original_grading_prompts_or_tools(source, forbidden):
    _, prepared = prepare(source)
    policy = {"policy_model": {"responses_api_models": {"vllm_model": {"model": "fictional-checkpoint"}}}}
    if forbidden == "top_level_grader":
        policy["fictional_native_grader"] = {"resources_servers": {"new_grader": {"reward": 1}}}
    elif forbidden == "top_level_sampling":
        policy["responses_create_params"] = {"temperature": 0, "tools": []}
    else:
        key = {
            "nested_tools": "tools",
            "nested_resources": "resources_servers",
            "nested_agent": "responses_api_agents",
        }[forbidden]
        policy["policy_model"]["responses_api_models"]["vllm_model"][key] = {"must_not_override": True}
    write_json(source["policy"], policy)
    with pytest.raises(ValueError):
        utility.plan_run(
            prepared / "prepared_manifest.json",
            source["gym_root"],
            source["policy"],
            source["root"] / "forbidden-policy",
        )


def test_rehashed_input_still_cannot_change_original_reference_or_prompt(source):
    manifest, prepared = prepare(source)
    path = prepared / manifest["jobs"][0]["input_path"]
    values = read_jsonl(path)
    values[0]["verifier_metadata"]["reference_answer"] = "Invented replacement reference"
    write_jsonl(path, values)
    manifest["jobs"][0]["input_sha256"] = file_hash(path)
    write_json(prepared / "prepared_manifest.json", manifest)
    with pytest.raises(ValueError, match="whole original row"):
        utility.plan_run(
            prepared / "prepared_manifest.json",
            source["gym_root"],
            source["policy"],
            source["root"] / "must-not-use-new-reference",
        )


def test_source_bookkeeping_is_retained_without_changing_original_attempt_mapping(source):
    source["original"][0].update(_ng_task_index=71, _ng_rollout_index=9)
    write_jsonl(source["dataset"], source["original"])
    source["routes"]["benchmarks"]["fictional_benchmark"]["dataset_sha256"] = file_hash(source["dataset"])
    for selected in source["selection"]:
        selected["expected_task_sha256"] = utility.canonical_sha256(source["original"][selected["dataset_row_index"]])
    write_json(source["routes_path"], source["routes"])
    write_jsonl(source["selection_path"], source["selection"])
    manifest, prepared = prepare(source)
    mappings = read_jsonl(prepared / manifest["jobs"][0]["mapping_path"])
    inputs = read_jsonl(prepared / manifest["jobs"][0]["input_path"])
    assert mappings[0]["source_bookkeeping"] == {"_ng_task_index": 71, "_ng_rollout_index": 9}
    assert mappings[1]["source_bookkeeping"] == mappings[0]["source_bookkeeping"]
    assert [mapping["orig_rollout_index"] for mapping in mappings[:2]] == [0, 2]
    assert inputs[0]["_ng_rollout_index"] != inputs[1]["_ng_rollout_index"]


@pytest.mark.parametrize("returncode", [0, 1])
def test_explicit_execution_delegates_only_to_original_gym_and_records_process_status(source, monkeypatch, returncode):
    _, prepared = prepare(source)
    calls = []

    def fake_original_gym(command, **kwargs):
        calls.append((command, kwargs))
        assert command[:3] == [sys.executable, str(SCRIPT), "_worker"]
        assert kwargs["cwd"] == str(source["gym_root"])
        job = json.loads(Path(command[command.index("--job") + 1]).read_text())
        assert job["agent_name"] == "fictional_original_agent"
        assert job["config_files"][0]["path"] == str(source["config"])
        assert "--temperature" not in command and "--max-output-tokens" not in command
        inputs = read_jsonl(Path(job["input_path"]))
        assert inputs[0]["verifier_metadata"] == source["original"][0]["verifier_metadata"]
        if "--preflight-only" in command:
            return SimpleNamespace(returncode=0)
        if returncode == 0:
            write_jsonl(
                Path(job["output_path"]),
                [
                    {
                        "_ng_task_index": row["_ng_task_index"],
                        "_ng_rollout_index": row["_ng_rollout_index"],
                        "reward": 0.75,
                        "original_grader_diagnostics": {"source": "fictional native criterion"},
                    }
                    for row in inputs
                ],
            )
        return SimpleNamespace(returncode=returncode)

    monkeypatch.setattr(utility.subprocess, "run", fake_original_gym)
    output = source["root"] / "executed-with-mock-gym"
    result = utility.run_prepared(
        prepared / "prepared_manifest.json", source["gym_root"], source["policy"], output, execute=True
    )
    assert len(calls) == 2
    assert calls[0][0][-1] == "--preflight-only"
    assert "--preflight-only" not in calls[1][0]
    assert result["execute"] is True
    assert result["jobs"][0]["status"] == ("completed" if returncode == 0 else "process_failed")
    assert json.loads((output / "run_manifest.json").read_text())["jobs"][0]["returncode"] == returncode
    with pytest.raises(ValueError):
        utility.run_prepared(
            prepared / "prepared_manifest.json", source["gym_root"], source["policy"], output, execute=True
        )
    assert len(calls) == 2


def test_all_original_jobs_preflight_before_any_model_or_resource_server_starts(source, monkeypatch):
    second = copy.deepcopy(source["selection"][-1])
    second["benchmark"] = "fictional_second_benchmark"
    second["orig_trajectory_id"] = "independent-second-benchmark-attempt"
    source["selection"].append(second)
    source["routes"]["benchmarks"][second["benchmark"]] = copy.deepcopy(
        source["routes"]["benchmarks"]["fictional_benchmark"]
    )
    write_json(source["routes_path"], source["routes"])
    write_jsonl(source["selection_path"], source["selection"])
    _, prepared = prepare(source)
    commands = []

    def fake_preflight(command, **kwargs):
        commands.append(command)
        assert command[-1] == "--preflight-only", "A model run started before every original subset was validated"
        assert kwargs["cwd"] == str(source["gym_root"])
        return SimpleNamespace(returncode=0 if len(commands) == 1 else 1)

    monkeypatch.setattr(utility.subprocess, "run", fake_preflight)
    output = source["root"] / "second-preflight-fails"
    result = utility.run_prepared(
        prepared / "prepared_manifest.json", source["gym_root"], source["policy"], output, execute=True
    )
    assert len(commands) == 2
    assert result["jobs"][1]["status"] == "preflight_failed"
    assert not list(output.glob("*.rollouts.jsonl"))
    assert not list(output.glob("*.stdout.log"))


def synthetic_run(source: dict[str, Any]) -> tuple[dict[str, Any], Path, list[dict[str, Any]]]:
    manifest, prepared = prepare(source)
    run_dir = source["root"] / "completed-run"
    plan = utility.plan_run(prepared / "prepared_manifest.json", source["gym_root"], source["policy"], run_dir)
    run_dir.mkdir()
    inputs = read_jsonl(prepared / manifest["jobs"][0]["input_path"])
    results = [
        {
            "_ng_task_index": row["_ng_task_index"],
            "_ng_rollout_index": row["_ng_rollout_index"],
            "reward": reward,
            "mask_sample": False,
            "response": {
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": f"Fresh fictional answer {index}"}],
                    }
                ]
            },
            "original_grader_diagnostics": {"reference": f"fictional-{index}", "criteria_passed": reward},
        }
        for index, (row, reward) in enumerate(zip(inputs, [1.0, 0.0, 0.0], strict=True))
    ]
    plan["execute"] = True
    plan["jobs"][0]["status"] = "completed"
    return plan, run_dir, results


def save_run(plan: dict[str, Any], run_dir: Path, results: list[dict[str, Any]] | None) -> Path:
    if results is not None:
        path = Path(plan["jobs"][0]["output_path"])
        write_jsonl(path, results)
        plan["jobs"][0]["output_sha256"] = file_hash(path)
    path = run_dir / "run_manifest.json"
    write_json(path, plan)
    return path


def test_summary_retains_actual_original_grader_results_and_identity(source):
    plan, run_dir, results = synthetic_run(source)
    run_manifest = save_run(plan, run_dir, results)
    destination = source["root"] / "joined-summary"
    summary = utility.summarize_results(run_manifest, destination)
    assert summary["expected"] == summary["observed"] == 3
    assert summary["missing"] == 0
    assert summary["comparison_ready"] is True
    job = summary["jobs"][0]
    assert job["recorded_reward_zero"] == job["scored_reward_zero"] == 2
    assert job["recorded_reward_positive"] == job["scored_reward_positive"] == 1
    matched = read_jsonl(destination / "matched_results.jsonl")
    assert [row["original_gym_result"] for row in matched] == results
    assert [row["source_identity"]["orig_trajectory_id"] for row in matched] == [
        row["orig_trajectory_id"] for row in source["selection"]
    ]
    assert matched[0]["source_identity"]["orig_rollout_index"] == 0
    assert matched[1]["source_identity"]["orig_rollout_index"] == 2


@pytest.mark.parametrize(
    "problem",
    [
        "missing_file",
        "missing_row",
        "duplicate_row",
        "unknown_indices",
        "masked",
        "legacy_masked",
        "skip",
        "error",
        "nested_verifier_error",
        "nested_response_error",
        "missing_reward",
        "failed_process",
    ],
)
def test_incomplete_or_unscored_run_is_never_presented_as_measured_zero_failures(source, problem):
    plan, run_dir, results = synthetic_run(source)
    if problem == "missing_file":
        results = None
    elif problem == "missing_row":
        results.pop()
    elif problem == "duplicate_row":
        results.append(copy.deepcopy(results[0]))
    elif problem == "unknown_indices":
        results.append({"_ng_task_index": 999, "_ng_rollout_index": 999, "reward": 0})
    elif problem == "masked":
        results[-1]["mask_sample"] = True
    elif problem == "legacy_masked":
        results[-1]["reward_mask"] = 0
    elif problem == "skip":
        results[-1]["skip"] = True
    elif problem == "error":
        results[-1]["agent_timeout_error"] = "fictional transport timeout"
    elif problem == "nested_verifier_error":
        results[-1]["response"]["status"] = "completed"
        results[-1]["metadata"] = {"exception_info": {"exception_type": "VerifierTimeoutError"}}
    elif problem == "nested_response_error":
        results[-1]["response"]["error"] = {"code": "provider_error", "message": "fictional unavailable provider"}
    elif problem == "missing_reward":
        results[-1].pop("reward")
    else:
        plan["jobs"][0]["status"] = "process_failed"
        plan["jobs"][0]["returncode"] = 1
    run_manifest = save_run(plan, run_dir, results)
    summary = utility.summarize_results(run_manifest, source["root"] / "not-a-complete-score")
    assert summary["comparison_ready"] is False
    job = summary["jobs"][0]
    if problem.startswith("missing_"):
        if problem == "missing_reward":
            assert job["missing_reward"] == 1
        else:
            assert summary["missing"] > 0
    if problem in {
        "masked",
        "legacy_masked",
        "skip",
        "error",
        "nested_verifier_error",
        "nested_response_error",
        "missing_reward",
    }:
        assert job["scored_reward_zero"] == 1, "Unscored placeholder zero must not become a measured model failure"
    if problem in {"masked", "legacy_masked", "skip"}:
        assert job["masked"] == 1
    if problem in {"error", "nested_verifier_error", "nested_response_error"}:
        assert job["errors"] == 1


def test_result_and_mapping_checksums_protect_summary_from_stale_outputs(source):
    plan, run_dir, results = synthetic_run(source)
    run_manifest = save_run(plan, run_dir, results)
    path = Path(plan["jobs"][0]["output_path"])
    changed = copy.deepcopy(results)
    changed[0]["reward"] = 0
    write_jsonl(path, changed)
    with pytest.raises(ValueError):
        utility.summarize_results(run_manifest, source["root"] / "must-not-score-tampered-output")
    write_jsonl(path, results)
    mapping = Path(plan["jobs"][0]["mapping_path"])
    mapping.write_text(mapping.read_text() + "\n")
    with pytest.raises(ValueError):
        utility.summarize_results(run_manifest, source["root"] / "must-not-score-tampered-mapping")


def install_original_runtime(
    source,
    monkeypatch,
    *,
    collector_raises=False,
    imported_elsewhere=False,
    preprocessed_prompt_override=False,
    config_overrides=None,
    runtime_asset_mutation=None,
):
    """Model original APIs, with full-split collation poisoned rather than mocked away."""
    from omegaconf import OmegaConf

    calls = []
    parser_configs = []
    monkeypatch.chdir(source["gym_root"])
    monkeypatch.delenv("NEMO_GYM_CONFIG_DICT", raising=False)
    monkeypatch.delenv("NEMO_GYM_CONFIG_PATH", raising=False)
    # Worker changes its import priority intentionally; restore it after each test.
    monkeypatch.setattr(sys, "path", list(sys.path))

    def forbidden_collation(*args, **kwargs):
        raise AssertionError("Full benchmark collation must never replace selected original rows")

    class OriginalParserConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            parser_configs.append(self)

    def original_global_config(*, global_config_dict_parser_config):
        parser = global_config_dict_parser_config
        assert parser.skip_load_from_cli is True
        assert parser.skip_load_from_dotenv is True
        options = OmegaConf.to_container(parser.initial_global_config_dict, resolve=True)
        assert options["agent_name"] == "fictional_original_agent"
        assert options["num_repeats"] == 1
        assert options["config_paths"] == [str(source["config"])]
        assert options["policy_model"]["responses_api_models"]["vllm_model"]["model"] == "fictional-checkpoint"
        assert options["upload_rollouts_to_wandb"] is False
        assert options["wandb_api_key"] is None
        options.update(config_overrides or {})
        return OmegaConf.create(options)

    class OriginalRunHelper:
        def start(self, config):
            assert config is parser_configs[-1]
            if runtime_asset_mutation:
                runtime_asset_mutation(OmegaConf.to_container(config.initial_global_config_dict, resolve=True))
            calls.append("original_server_start")

        def shutdown(self):
            calls.append("original_server_shutdown")

    class OriginalRolloutConfig:
        @classmethod
        def model_validate(cls, value):
            assert isinstance(value, dict)
            return SimpleNamespace(**value)

    class OriginalCollector:
        def _preprocess_rows_from_config(self, config):
            calls.append("original_subset_preprocess")
            selected = read_jsonl(Path(config.input_jsonl_fpath))
            # A mistaken original e2e call would replace these selected three
            # attempts with the two rows in the complete fictional dataset.
            assert len(read_jsonl(source["dataset"])) == 2
            assert len(selected) == 3
            assert [row["orig_trajectory_id"] for row in selected] == [
                row["orig_trajectory_id"] for row in source["selection"]
            ]
            assert selected[0]["verifier_metadata"] == source["original"][0]["verifier_metadata"]
            assert [(row["_ng_task_index"], row["_ng_rollout_index"]) for row in selected] == [(0, 0), (0, 1), (1, 0)]
            for row in selected:
                row["responses_create_params"].update(getattr(config, "responses_create_params", None) or {})
            if preprocessed_prompt_override:
                selected[0]["responses_create_params"]["input"][-1]["content"] = (
                    "Changed by an inherited prompt config"
                )
            return selected

        async def run_from_config(self, config):
            calls.append("original_subset_collector")
            assert config.num_repeats == 1
            assert len(read_jsonl(Path(config.input_jsonl_fpath))) == 3
            if collector_raises:
                raise RuntimeError("Fictional original collector error")
            return []

    modules = {}
    for name in (
        "nemo_gym",
        "nemo_gym.global_config",
        "nemo_gym.cli",
        "nemo_gym.cli.env",
        "nemo_gym.cli.eval",
        "nemo_gym.rollout_collection",
        "nemo_gym.train_data_processor",
    ):
        module = ModuleType(name)
        root = source["root"] / "wrong-install" if imported_elsewhere else source["gym_root"]
        module.__file__ = str(root / Path(*name.split(".")) / "__init__.py")
        module.__path__ = [str(Path(module.__file__).parent)]
        modules[name] = module
        monkeypatch.setitem(sys.modules, name, module)
    modules["nemo_gym.global_config"].GlobalConfigDictParserConfig = OriginalParserConfig
    modules["nemo_gym.global_config"].get_global_config_dict = original_global_config
    modules["nemo_gym.cli.env"].RunHelper = OriginalRunHelper
    modules["nemo_gym.rollout_collection"].RolloutCollectionConfig = OriginalRolloutConfig
    modules["nemo_gym.rollout_collection"].RolloutCollectionHelper = OriginalCollector
    modules["nemo_gym.cli.eval"].e2e_rollout_collection = forbidden_collation
    modules["nemo_gym.train_data_processor"].TrainDataProcessor = forbidden_collation
    return calls


@pytest.mark.parametrize("collector_raises", [False, True])
def test_native_worker_dispatches_prepared_subset_without_full_dataset_collation(
    source, monkeypatch, collector_raises
):
    """Original e2e CLI must never silently replace the explicitly selected subset."""
    _, prepared = prepare(source)
    plan = utility.plan_run(
        prepared / "prepared_manifest.json", source["gym_root"], source["policy"], source["root"] / "native-worker"
    )
    job = plan["jobs"][0]["worker_spec"]
    calls = install_original_runtime(source, monkeypatch, collector_raises=collector_raises)
    if collector_raises:
        with pytest.raises(RuntimeError, match="Fictional original collector error"):
            utility.execute_original_job(job)
    else:
        assert utility.execute_original_job(job)["status"] == "completed"
    assert calls == [
        "original_subset_preprocess",
        "original_server_start",
        "original_subset_collector",
        "original_server_shutdown",
    ]


def test_native_preflight_only_reads_selected_rows_without_starting_servers(source, monkeypatch):
    _, prepared = prepare(source)
    plan = utility.plan_run(
        prepared / "prepared_manifest.json", source["gym_root"], source["policy"], source["root"] / "preflight-only"
    )
    calls = install_original_runtime(source, monkeypatch)
    receipt = utility.execute_original_job(plan["jobs"][0]["worker_spec"], preflight_only=True)
    assert receipt["status"] == "preflight_passed" and receipt["num_rows"] == 3
    assert calls == ["original_subset_preprocess"]
    assert not Path(plan["output_dir"]).exists()


def test_worker_refuses_import_from_another_installed_gym_before_servers_start(source, monkeypatch):
    _, prepared = prepare(source)
    plan = utility.plan_run(
        prepared / "prepared_manifest.json", source["gym_root"], source["policy"], source["root"] / "wrong-checkout"
    )
    calls = install_original_runtime(source, monkeypatch, imported_elsewhere=True)
    with pytest.raises(ValueError, match="different Gym checkout"):
        utility.execute_original_job(plan["jobs"][0]["worker_spec"])
    assert calls == []


def test_worker_cli_rejects_changed_job_specification_before_importing_or_starting_gym(source, monkeypatch):
    _, prepared = prepare(source)
    plan = utility.plan_run(
        prepared / "prepared_manifest.json", source["gym_root"], source["policy"], source["root"] / "bound-worker"
    )
    job = plan["jobs"][0]
    worker_file = source["root"] / "tampered-worker.json"
    changed = copy.deepcopy(job["worker_spec"])
    changed["agent_name"] = "unapproved_replacement_agent"
    write_json(worker_file, changed)

    def must_not_execute(*args, **kwargs):
        pytest.fail("A modified worker specification reached original Gym")

    monkeypatch.setattr(utility, "execute_original_job", must_not_execute)
    with pytest.raises(SystemExit) as error:
        utility.main(
            [
                "_worker",
                "--job",
                str(worker_file),
                "--expected-job-sha256",
                job["worker_spec_sha256"],
                "--preflight-only",
            ]
        )
    assert error.value.code == 2


def test_worker_refuses_preprocessing_that_changes_prompt_without_changing_count_or_indices(source, monkeypatch):
    _, prepared = prepare(source)
    plan = utility.plan_run(
        prepared / "prepared_manifest.json", source["gym_root"], source["policy"], source["root"] / "altered-input"
    )
    calls = install_original_runtime(source, monkeypatch, preprocessed_prompt_override=True)
    with pytest.raises(ValueError):
        utility.execute_original_job(plan["jobs"][0]["worker_spec"], preflight_only=True)
    assert calls == ["original_subset_preprocess"]


@pytest.mark.parametrize(
    "config_override",
    [
        {"prompt_config": {"system_prompt": "Changed instruction"}},
        {"skills": ["fictional unrecorded skill"]},
        {"responses_create_params": {"input": [{"role": "user", "content": "Replacement"}]}},
        {"responses_create_params": {"tools": []}},
        {"responses_create_params": {"instructions": "Replacement"}},
    ],
)
def test_worker_refuses_inherited_prompt_tool_and_skill_overrides(source, monkeypatch, config_override):
    _, prepared = prepare(source)
    plan = utility.plan_run(
        prepared / "prepared_manifest.json", source["gym_root"], source["policy"], source["root"] / "unsafe-config"
    )
    calls = install_original_runtime(source, monkeypatch, config_overrides=config_override)
    with pytest.raises(ValueError):
        utility.execute_original_job(plan["jobs"][0]["worker_spec"], preflight_only=True)
    assert "original_server_start" not in calls


def test_worker_records_only_explicit_original_sampling_overrides(source, monkeypatch):
    _, prepared = prepare(source)
    plan = utility.plan_run(
        prepared / "prepared_manifest.json", source["gym_root"], source["policy"], source["root"] / "sampling"
    )
    sampling = {
        "temperature": 0.25,
        "top_p": 0.9,
        "max_output_tokens": 2048,
        "reasoning": {"effort": "low"},
        "seed": 17,
    }
    calls = install_original_runtime(source, monkeypatch, config_overrides={"responses_create_params": sampling})
    receipt = utility.execute_original_job(plan["jobs"][0]["worker_spec"], preflight_only=True)
    assert receipt["sampling_overrides"] == sampling
    assert receipt["preprocessed_rows_sha256"] != utility.canonical_sha256(
        read_jsonl(Path(plan["jobs"][0]["input_path"]))
    )
    assert calls == ["original_subset_preprocess"]


def test_original_required_assets_are_pinned_and_rechecked_before_execution(source):
    asset = source["root"] / "fictional-reference.txt"
    asset.write_text("The original fictional verifier reference", encoding="utf-8")
    route = source["routes"]["benchmarks"]["fictional_benchmark"]
    route["required_files"] = [{"path": str(asset), "sha256": file_hash(asset)}]
    write_json(source["routes_path"], source["routes"])
    manifest, prepared = prepare(source)
    assert manifest["jobs"][0]["required_files"] == route["required_files"]
    asset.write_text("Changed reference", encoding="utf-8")
    with pytest.raises(ValueError, match="required asset"):
        utility.plan_run(
            prepared / "prepared_manifest.json",
            source["gym_root"],
            source["policy"],
            source["root"] / "changed-reference",
        )


def test_native_runtime_mutates_fresh_asset_copies_while_pinned_sources_remain_valid(source, monkeypatch):
    assets = source["root"] / "immutable-original-assets"
    (assets / "fictional-task").mkdir(parents=True)
    task = assets / "fictional-task" / "task.toml"
    task.write_text('instruction = "Original fictional task"\n', encoding="utf-8")
    source_sha256 = file_hash(task)
    config_key = "fictional_native_grader.resources_servers.fictional_original.task_folder"
    source["routes"]["benchmarks"]["fictional_benchmark"]["runtime_assets"] = [
        {"source_dir": str(assets), "config_key": config_key}
    ]
    write_json(source["routes_path"], source["routes"])
    manifest, prepared = prepare(source)
    assert manifest["jobs"][0]["runtime_assets"][0]["entries"][1]["sha256"] == source_sha256
    plans = [
        utility.plan_run(
            prepared / "prepared_manifest.json",
            source["gym_root"],
            source["policy"],
            source["root"] / f"isolated-run-{index}",
        )
        for index in range(2)
    ]
    modified_copies = []

    def original_resource_rewrites_its_task_mirror(initial):
        location = Path(initial["fictional_native_grader"]["resources_servers"]["fictional_original"]["task_folder"])
        assert location != assets
        assert file_hash(location / "fictional-task" / "task.toml") == source_sha256
        (location / "fictional-task" / "task.toml").write_text(
            "Mutated by original fictional runtime\n", encoding="utf-8"
        )
        modified_copies.append(location)

    install_original_runtime(source, monkeypatch, runtime_asset_mutation=original_resource_rewrites_its_task_mirror)
    for plan in plans:
        job = plan["jobs"][0]["worker_spec"]
        assert utility.execute_original_job(job, preflight_only=True)["status"] == "preflight_passed"
        assert not Path(plan["output_dir"]).exists(), "Preflight must never copy or rewrite asset trees"
        assert utility.execute_original_job(job)["status"] == "completed"
        assert file_hash(task) == source_sha256
        utility._recheck_plan_inputs(plan)
    assert len(set(modified_copies)) == 2
    assert all(location.is_relative_to(source["root"]) for location in modified_copies)


def test_changed_runtime_asset_tree_is_refused_before_any_copy(source):
    assets = source["root"] / "original-assets"
    assets.mkdir()
    (assets / "reference.txt").write_text("Original fictional reference", encoding="utf-8")
    source["routes"]["benchmarks"]["fictional_benchmark"]["runtime_assets"] = [
        {"source_dir": str(assets), "config_key": "fictional_resource.task_folder"}
    ]
    write_json(source["routes_path"], source["routes"])
    _, prepared = prepare(source)
    (assets / "injected.txt").write_text("An unpinned extra reference", encoding="utf-8")
    with pytest.raises(ValueError):
        utility.plan_run(
            prepared / "prepared_manifest.json", source["gym_root"], source["policy"], source["root"] / "must-not-copy"
        )
