# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.gdpval.hsg.aav2 import completion, snapshot


@pytest.fixture
def prepared(tmp_path):
    source = tmp_path / "repo"
    package = source / snapshot.PACKAGE
    package.mkdir(parents=True)
    for name in snapshot.FILES:
        (package / name).write_text(f"# {name}\n")
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "add", "."], cwd=source, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.com", "commit", "-qm", "fixture"],
        cwd=source,
        check=True,
    )
    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{"task_id":"one"}\n')
    config = tmp_path / "judge.yaml"
    config.write_text("multistage: {enabled: true}\n")
    profile = tmp_path / "profile.env"
    profile.write_text('EXTRA_ARGS=\'--speculative-config {"method":"mtp","num_speculative_tokens":1}\'\n')
    secret = tmp_path / "secret.env"
    secret.write_text("export JUDGE_API_KEY=fixture\n")
    args = SimpleNamespace(
        source=source,
        revision="HEAD",
        run_dir=tmp_path / "run with spaces",
        dataset=dataset,
        smoke_dataset=None,
        judge_config=config,
        profile=profile,
        existing_rollout=None,
        env_file=secret,
        uv_source=Path("/bin/sh"),
        agent_sif=Path("/bin/sh"),
        judge_sif=None,
        apptainer_bin=Path("/bin"),
        concurrency=8,
        agent_max_turns=250,
    )
    snapshot.prepare(args)
    return args


def test_snapshot_freezes_source_and_inputs_without_copying_credentials(prepared):
    root = prepared.run_dir
    snapshot.verify(root)
    assert (root / "serving.env").read_bytes() == prepared.profile.read_bytes()
    assert not any(path.name == "secret.env" for path in root.rglob("*"))
    environment = subprocess.check_output(
        ["bash", "-c", 'source "$1/run.env"; printf "%s" "$RUN_DIR"', "bash", str(root)]
    )
    assert environment.decode() == str(root)
    prepared.dataset.write_text('{"task_id":"changed"}\n')
    assert completion.task_ids(root / "dataset.jsonl") == {"one"}
    (root / "dataset.jsonl").chmod(0o600)
    (root / "dataset.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="prepared input changed"):
        snapshot.verify(root)


def test_snapshot_rejects_existing_run_dirty_source_and_unsafe_path(prepared):
    with pytest.raises(ValueError, match="directory exists"):
        snapshot.prepare(prepared)
    prepared.run_dir = prepared.run_dir.with_name("run=changed")
    with pytest.raises(ValueError, match="run path"):
        snapshot.prepare(prepared)
    prepared.run_dir = prepared.run_dir.with_name("second")
    (prepared.source / snapshot.PACKAGE / "run_aav2.sh").write_text("changed")
    with pytest.raises(ValueError, match="tracked source changes"):
        snapshot.prepare(prepared)
    assert not prepared.run_dir.exists()


def test_finished_empty_submission_is_valid_rollout(tmp_path):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{"task_id":"one"}\n')
    with pytest.raises(ValueError, match="no finish marker"):
        completion.rollout_complete(dataset, tmp_path)
    marker = tmp_path / "task_one/repeat_0/finish_params.json"
    marker.parent.mkdir(parents=True)
    marker.write_text("null\n")
    assert completion.rollout_complete(dataset, tmp_path) == 1


def test_import_copies_only_finished_evidence_and_freezes_limits(prepared, monkeypatch):
    original = prepared.run_dir.parent / "old rollout"
    evidence = original / "deliverables/task_one/repeat_0"
    evidence.mkdir(parents=True)
    (evidence / "finish_params.json").write_text('{"paths":["answer.txt"]}')
    (evidence / "answer.txt").write_text("submitted evidence")
    cache = evidence.parent / "repeat_0_verify_response_0123456789abcdef.json"
    cache.write_text('{"reward":1}')
    (original / "judge_full").mkdir()
    (original / "judge_full/rollouts.jsonl").write_text("old judgments")
    prepared.run_dir = prepared.run_dir.with_name("fresh import")
    prepared.existing_rollout = original
    prepared.profile = None
    monkeypatch.setenv("GDPVAL_MAX_SECTION_RAW_ATTACHMENT_BYTES_FOR_JUDGE", "209715200")
    monkeypatch.setenv("GDPVAL_MAX_SECTION_ENCODED_ATTACHMENT_CHARS_FOR_JUDGE", "293601280")
    snapshot.prepare(prepared)
    snapshot.verify(prepared.run_dir)
    assert not (prepared.run_dir / "judge_full").exists()
    assert not list(prepared.run_dir.rglob("*verify_response*"))
    copied = prepared.run_dir / "deliverables/task_one/repeat_0/answer.txt"
    assert copied.read_bytes() == (evidence / "answer.txt").read_bytes()
    assert cache.read_text() == '{"reward":1}'
    settings = json.loads((prepared.run_dir / "run.json").read_text())
    assert settings["PROFILE"] == ""
    assert settings["GDPVAL_MAX_SECTION_RAW_ATTACHMENT_BYTES_FOR_JUDGE"] == "209715200"
    assert settings["GDPVAL_MAX_SECTION_ENCODED_ATTACHMENT_CHARS_FOR_JUDGE"] == "293601280"
    assert json.loads((prepared.run_dir / "import.json").read_text())["source"] == str(original / "deliverables")
    copied.chmod(0o600)
    copied.write_text("changed")
    with pytest.raises(ValueError, match="prepared input changed"):
        snapshot.verify(prepared.run_dir)


def test_import_rejects_unfinished_tasks_and_smoke_prompt_drift(prepared):
    prepared.run_dir = prepared.run_dir.with_name("incomplete import")
    prepared.existing_rollout = prepared.run_dir.parent / "unfinished"
    (prepared.existing_rollout / "deliverables/task_one/repeat_0").mkdir(parents=True)
    with pytest.raises(ValueError, match="no finish marker"):
        snapshot.prepare(prepared)
    assert not prepared.run_dir.exists()
    prepared.smoke_dataset = prepared.run_dir.parent / "changed_smoke.jsonl"
    prepared.smoke_dataset.write_text('{"task_id":"one","prompt":"different"}\n')
    with pytest.raises(ValueError, match="match the canonical dataset"):
        snapshot.prepare(prepared)
    assert not prepared.run_dir.exists()


@pytest.mark.parametrize("mode,count,trials", [("smoke", 4, 1), ("pilot", 12, 2), ("full", 20, 4)])
def test_judge_completion_requires_profile_coverage_and_valid_votes(tmp_path, mode, count, trials):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("".join(json.dumps({"task_id": str(i)}) + "\n" for i in range(20)))
    output = tmp_path / "rollouts.jsonl"
    metrics = {
        f"comparison/{name}": value
        for name, value in {
            "final_stage_present": 1,
            "final_stage_complete": 1,
            "final_stage_fit": 1,
            "final_stage_degraded": 0,
        }.items()
    }
    aggregate = tmp_path / "rollouts_aggregate_metrics.json"
    aggregate.write_text(json.dumps([{"agent_metrics": metrics}]))
    rows = [
        {
            "task_id": str(i),
            "stage_index": 0 if mode == "smoke" else 1,
            "expected_final_stage_index": 0 if mode == "smoke" else 1,
            "judge_response": {"total_judged": trials, "total_invalid": 0},
        }
        for i in range(count)
    ]

    def write_rows():
        output.write_text("".join(json.dumps(row) + "\n" for row in rows))

    write_rows()
    assert completion.judge_complete(dataset, output, mode) == count
    rows[-1]["task_id"] = "0"
    write_rows()
    with pytest.raises(ValueError, match="duplicate, missing, or unexpected"):
        completion.judge_complete(dataset, output, mode)
    rows[-1]["task_id"] = str(count - 1)
    rows[-1]["judge_response"]["total_invalid"] = 1
    write_rows()
    with pytest.raises(ValueError, match="valid votes"):
        completion.judge_complete(dataset, output, mode)
    rows[-1]["judge_response"]["total_invalid"] = 0
    write_rows()
    metrics["comparison/final_stage_complete"] = 0
    aggregate.write_text(json.dumps([{"agent_metrics": metrics}]))
    with pytest.raises(ValueError, match="native aggregate"):
        completion.judge_complete(dataset, output, mode)
