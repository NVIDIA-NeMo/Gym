# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import shutil
from pathlib import Path

import pytest
import yaml

from benchmarks.gdpval.prepare_nooa_judging import JUDGE_GYM_COMMIT, prepare_judging
from nemo_gym.config_types import BenchmarkDatasetConfig


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def artifacts(tmp_path):
    source = tmp_path / "canonical.jsonl"
    source.write_text(
        json.dumps({"task_id": "one", "prompt": "Make a report", "responses_create_params": {"input": []}}) + "\n"
    )
    refs = tmp_path / "references.json"
    refs.write_text(json.dumps({"reference": {"elo": 1000, "deliverables_dir": "/refs/reference"}}))
    root = tmp_path / "generation"
    session = root / "gdp-one"
    outputs = session / "deliverables-one"
    reference = outputs / "reference_files" / "input.txt"
    reference.parent.mkdir(parents=True)
    reference.write_bytes(b"original input\x00\xff")
    report = outputs / "report.txt"
    report.write_bytes(b"candidate\x00\xff")
    (outputs / "finish_params.json").write_text(
        json.dumps(
            {"summary": "Done", "paths": [str(report)], "submission_method": "nooa_final_response_output_directory"}
        )
    )
    (session / "artifacts.json").write_text(
        json.dumps([{"name": "report.txt", "size": report.stat().st_size, "sha256": digest(report)}])
    )
    receipt = {
        "schema_version": 1,
        "task_id": {"taskset": "gdp", "task_id": "one"},
        "episode_id": {"rollout_id": "generation-one", "attempt": 2},
        "resources_session_id": "resource-one",
        "verify_request": {
            "task_id": "one",
            "prompt": "Make a report",
            "deliverables_dir": str(outputs),
            "response": {"output": []},
        },
    }
    (session / "generation.json").write_text(json.dumps(receipt))
    return {
        "source": source,
        "generation_roots": [root],
        "reference_models": refs,
        "output": tmp_path / "judging",
        "expected_tasks": 1,
    }


def test_copy_preserves_artifacts_canonical_rows_and_generation_identity(artifacts):
    root = artifacts["generation_roots"][0]
    before = {str(p.relative_to(root)): digest(p) for p in root.rglob("*") if p.is_file()}
    output = prepare_judging(**artifacts)
    manifest = json.loads((output / "preparation.json").read_text())
    assert manifest["judge_gym_commit"] == JUDGE_GYM_COMMIT
    assert not manifest["judging_performed"] and not manifest["policy_inference_performed"]
    assert manifest["tasks"][0]["generation_episode_id"] == {"rollout_id": "generation-one", "attempt": 2}
    assert manifest["tasks"][0]["judging_repeat_index"] == 0
    assert (output / "gdpval_benchmark.jsonl").read_bytes() == artifacts["source"].read_bytes()
    for name, expected in manifest["tasks"][0]["files_sha256"].items():
        assert digest(output / "deliverables/task_one/repeat_0" / name) == expected
    assert before == {str(p.relative_to(root)): digest(p) for p in root.rglob("*") if p.is_file()}
    overlay = yaml.safe_load((output / "judge_data.yaml").read_text())
    assert overlay["gdpval_stirrup_agent"]["responses_api_agents"]["stirrup_agent"]["persist_deliverables_dir"] == str(
        output / "deliverables"
    )
    dataset = BenchmarkDatasetConfig.model_validate(
        overlay["gdpval_stirrup_agent"]["responses_api_agents"]["stirrup_agent"]["datasets"][0]
    )
    assert dataset.jsonl_fpath == output / "gdpval_benchmark.jsonl"
    assert dataset.prepare_script == Path("benchmarks/gdpval/prepare.py")
    with pytest.raises(FileExistsError):
        prepare_judging(**artifacts)


def test_completed_empty_deliverables_remain_a_real_judging_input(artifacts):
    session = artifacts["generation_roots"][0] / "gdp-one"
    delivered = session / "deliverables-one"
    (delivered / "report.txt").unlink()
    (session / "artifacts.json").write_text("[]")
    marker_path = delivered / "finish_params.json"
    marker = json.loads(marker_path.read_text())
    marker["paths"] = []
    marker_path.write_text(json.dumps(marker))
    output = prepare_judging(**artifacts)
    candidate = output / "deliverables/task_one/repeat_0"
    assert json.loads((candidate / "finish_params.json").read_text()) == marker
    assert not (candidate / "report.txt").exists()
    manifest = json.loads((output / "preparation.json").read_text())
    assert manifest["task_count"] == 1
    assert set(manifest["tasks"][0]["files_sha256"]) == {"finish_params.json", "reference_files/input.txt"}
    assert not manifest["judging_performed"]


@pytest.mark.parametrize(
    "problem",
    [
        "missing",
        "duplicate",
        "identity",
        "prompt",
        "hash",
        "unlisted",
        "symlink",
        "marker",
        "outside",
        "nested",
        "source_duplicate",
        "reference",
    ],
)
def test_refuses_ambiguous_or_changed_inputs(artifacts, problem):
    session = artifacts["generation_roots"][0] / "gdp-one"
    receipt_path = session / "generation.json"
    receipt = json.loads(receipt_path.read_text())
    outputs = session / "deliverables-one"
    if problem == "missing":
        receipt_path.unlink()
    elif problem == "duplicate":
        copy = session.with_name("gdp-two")
        shutil.copytree(session, copy)
    elif problem == "identity":
        receipt["verify_request"]["task_id"] = "different"
    elif problem == "prompt":
        receipt["verify_request"]["prompt"] = "different"
    elif problem == "hash":
        (outputs / "report.txt").write_bytes(b"changed bytes")
    elif problem == "unlisted":
        (outputs / "extra.txt").write_text("unlisted")
    elif problem == "symlink":
        (outputs / "reference_files/link").symlink_to(outputs / "report.txt")
    elif problem == "marker":
        (outputs / "finish_params.json").unlink()
    elif problem == "outside":
        receipt["verify_request"]["deliverables_dir"] = str(session.parent)
    elif problem == "nested":
        (outputs / "nested").mkdir()
        (outputs / "nested/file").write_text("extra")
    elif problem == "source_duplicate":
        artifacts["source"].write_text(artifacts["source"].read_text() * 2)
    elif problem == "reference":
        artifacts["reference_models"].write_text('{"ref":{"elo":1000,"deliverables_dir":"relative"}}')
    if problem in {"identity", "prompt", "outside"}:
        receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        prepare_judging(**artifacts)
    assert not artifacts["output"].exists()


def test_copy_failure_cleans_own_staging_only(artifacts, monkeypatch):
    real_copy = shutil.copytree

    def changed_copy(source, target, *args, **kwargs):
        result = real_copy(source, target, *args, **kwargs)
        if (Path(target) / "report.txt").exists():
            (Path(target) / "report.txt").write_text("changed during transfer")
        return result

    monkeypatch.setattr(shutil, "copytree", changed_copy)
    with pytest.raises(ValueError, match="changed during copy"):
        prepare_judging(**artifacts)
    assert not artifacts["output"].exists()
    assert not list(artifacts["output"].parent.glob(".judging-*"))
    assert (artifacts["generation_roots"][0] / "gdp-one/generation.json").is_file()


def test_judge_recipe_has_no_generation_and_uses_canonical_stages():
    recipe = Path(__file__).resolve().parents[3] / "benchmarks/gdpval/nooa_judge_only.yaml"
    config = yaml.safe_load(recipe.read_text())
    agent = config["gdpval_stirrup_agent"]["responses_api_agents"]["stirrup_agent"]
    resources = config["gdpval_resources_server"]["resources_servers"]["gdpval"]
    assert agent["judge_only"] and not agent["execute_only"]
    assert agent["concurrency"] == 16
    assert agent["count_eval_missing_as_loss"] is False
    assert resources["count_eval_missing_as_loss"] is False
    assert resources["preconvert_max_concurrent"] == 2
    assert config["count_failure_classes_as_zero"] == []
    assert config["model_endpoint_readiness_timeout_seconds"] == 0
    for proxy, limit in (("gpt55", 4), ("gemini31", 2), ("claude48", 4)):
        model = config[f"gdpval_{proxy}_judge_model"]["responses_api_models"]["openai_model"]
        assert model["max_concurrent_requests"] == limit
    assert resources["reward_mode"] == "comparison"
    # The pinned benchmark supplies distinct fixed-model proxies and the native
    # media/effort contract; a panel-only override would silently lose them.
    assert config["config_paths"] == ["benchmarks/gdpval/config.yaml"]
    assert "judge_panel" not in resources
    assert resources["num_comparison_trials"] == 4 and resources["judge_sampling_seed"] == 42
    assert [stage["num_tasks"] for stage in config["multistage"]["stages"]] == [45, 220]
    assert config["multistage"]["stages"][1]["num_models"] == 4
    assert config["policy_model"]["responses_api_models"]["openai_model"]["openai_base_url"] == "http://127.0.0.1:9/v1"


def test_explicit_exclusion_preserves_canonical_rows_without_fabricating_output(artifacts):
    original = artifacts["source"].read_bytes()
    second = json.dumps({"task_id": "two", "prompt": "Another task", "responses_create_params": {"input": []}}) + "\n"
    artifacts["source"].write_bytes(original + second.encode())
    artifacts["expected_tasks"] = 2
    output = prepare_judging(**artifacts, excluded_task_ids={"two"})
    manifest = json.loads((output / "preparation.json").read_text())
    assert manifest["canonical_task_count"] == 2 and manifest["task_count"] == 1
    assert manifest["excluded_task_ids"] == ["two"]
    assert (output / "gdpval_benchmark.jsonl").read_bytes() == original + second.encode()
    assert (output / "deliverables/task_one/repeat_0/finish_params.json").is_file()
    assert not (output / "deliverables/task_two").exists()
    assert not (output / "generation/task_two").exists()


def test_exclusion_must_be_known_and_cannot_hide_a_genuine_export(artifacts):
    with pytest.raises(ValueError, match="belong to the canonical"):
        prepare_judging(**artifacts, excluded_task_ids={"unknown"})
    second = json.dumps({"task_id": "two", "prompt": "Another task", "responses_create_params": {"input": []}}) + "\n"
    with artifacts["source"].open("a") as stream:
        stream.write(second)
    artifacts["expected_tasks"] = 2
    with pytest.raises(ValueError, match="excluded task has a generation receipt"):
        prepare_judging(**artifacts, excluded_task_ids={"one"})
    assert not artifacts["output"].exists()


def test_nested_manifest_preserves_tree_and_authored_zip_bytes(artifacts):
    session = artifacts["generation_roots"][0] / "gdp-one"
    outputs = session / "deliverables-one"
    nested = outputs / "project/src/code.py"
    nested.parent.mkdir(parents=True)
    nested.write_bytes(b"exact model-authored bytes\x00\xff")
    archive = outputs / "project.zip"
    archive.write_bytes(b"opaque original ZIP bytes are not rewritten")
    manifest_path = session / "artifacts.json"
    items = json.loads(manifest_path.read_text())
    for file in (nested, archive):
        items.append(
            {"name": file.relative_to(outputs).as_posix(), "size": file.stat().st_size, "sha256": digest(file)}
        )
    manifest_path.write_text(json.dumps(items))
    result = prepare_judging(**artifacts)
    copied = result / "deliverables/task_one/repeat_0"
    assert (copied / "project/src/code.py").read_bytes() == nested.read_bytes()
    assert (copied / "project.zip").read_bytes() == archive.read_bytes()
    recorded = json.loads((result / "preparation.json").read_text())["tasks"][0]["files_sha256"]
    assert recorded["project/src/code.py"] == digest(nested)
    assert recorded["project.zip"] == digest(archive)


@pytest.mark.parametrize("kind", ["symlink_directory", "hardlink", "unlisted_nested", "traversal", "noncanonical"])
def test_nested_artifacts_fail_closed(artifacts, kind):
    session = artifacts["generation_roots"][0] / "gdp-one"
    outputs = session / "deliverables-one"
    manifest = session / "artifacts.json"
    if kind == "symlink_directory":
        (outputs / "alias").symlink_to(outputs / "reference_files", target_is_directory=True)
    elif kind == "hardlink":
        (outputs / "alias").hardlink_to(outputs / "report.txt")
    elif kind == "unlisted_nested":
        (outputs / "project").mkdir()
        (outputs / "project/file").write_text("unlisted")
    else:
        items = json.loads(manifest.read_text())
        items[0]["name"] = "../report.txt" if kind == "traversal" else "./report.txt"
        manifest.write_text(json.dumps(items))
    with pytest.raises(ValueError):
        prepare_judging(**artifacts)
    assert not artifacts["output"].exists()


@pytest.mark.parametrize("separator", ["\u0085", "\u2028", "\u2029"])
def test_canonical_unicode_separators_preserve_one_task(artifacts, separator):
    prompt = f"Make{separator}a report"
    row = json.loads(artifacts["source"].read_text())
    row["prompt"] = prompt
    original = (json.dumps(row, ensure_ascii=False) + "\n").encode()
    artifacts["source"].write_bytes(original)
    receipt_path = artifacts["generation_roots"][0] / "gdp-one/generation.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["verify_request"]["prompt"] = prompt
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False))
    result = prepare_judging(**artifacts)
    assert (result / "gdpval_benchmark.jsonl").read_bytes() == original
    manifest = json.loads((result / "preparation.json").read_text())
    assert manifest["task_count"] == 1
    assert manifest["tasks"][0]["task_id"]["task_id"] == "one"
