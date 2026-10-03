# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.gdpval.hsg.aav2 import completion, preconvert, snapshot


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


def test_jsonl_reader_preserves_unicode_separators_inside_strings(tmp_path):
    path = tmp_path / "rollouts.jsonl"
    rows = [{"task_id": "one", "text": "line\u2028paragraph\u2029next\u0085end"}, {"task_id": "two"}]
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    assert completion.read_rows(path) == rows


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


@pytest.mark.parametrize("absent_task", [False, True])
def test_opt_in_import_prepares_missing_tasks_without_fabricating_markers(prepared, monkeypatch, absent_task):
    original = prepared.run_dir.parent / "unfinished"
    prepared.existing_rollout = original
    prepared.run_dir = prepared.run_dir.with_name("opt in import")
    prepared.count_missing_as_loss = True
    rows = [{"task_id": name, "reference_files": ["input.txt"]} for name in ("null", "empty", "missing")]
    prepared.dataset.write_text("".join(json.dumps(row) + "\n" for row in rows))
    for name, marker in (("null", "null"), ("empty", "{}")):
        task = original / "deliverables" / f"task_{name}" / "repeat_0"
        (task / "reference_files").mkdir(parents=True)
        (task / "finish_params.json").write_text(marker)
        (task / "reference_files/input.txt").write_text(f"original {name} input")
    if not absent_task:
        task = original / "deliverables/task_missing/repeat_0"
        task.mkdir(parents=True)
        (task / "history.json").write_text("[]")
        (task / "broken.mp4").write_bytes(b"unreadable partial video" * 500000)
    snapshot.prepare(prepared)
    snapshot.verify(prepared.run_dir)
    settings = json.loads((prepared.run_dir / "run.json").read_text())
    receipt = json.loads((prepared.run_dir / "import.json").read_text())
    assert settings["COUNT_EVAL_MISSING_AS_LOSS"] == "true"
    assert json.loads(settings["MISSING_EVAL_TASK_IDS"]) == receipt["missing_eval_task_ids"] == ["missing"]
    assert receipt["count_eval_missing_as_loss"] is True
    monkeypatch.setattr(preconvert, "preconvert_dir", lambda *_args, **_kwargs: (0, 0, []))
    output = preconvert.prepare(prepared.run_dir)
    assert preconvert.prepare(prepared.run_dir) == output
    assert not (output / "candidate/task_missing/repeat_0/finish_params.json").exists()
    assert not (output / "candidate/task_missing").exists()
    for name in ("null", "empty"):
        path = output / "candidate" / f"task_{name}" / "repeat_0/reference_files/input.txt"
        assert path.read_text() == f"original {name} input"
    marker = prepared.run_dir / "deliverables/task_missing/repeat_0/finish_params.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("{}")
    with pytest.raises(ValueError, match="fresh import"):
        snapshot.verify(prepared.run_dir)
    with pytest.raises(ValueError, match="fresh import"):
        preconvert.prepare(prepared.run_dir)


@pytest.mark.parametrize("marker", ["[]", "false", "not json"])
def test_missing_loss_allowance_does_not_accept_invalid_present_marker(prepared, marker):
    prepared.run_dir = prepared.run_dir.with_name("invalid import")
    prepared.existing_rollout = prepared.run_dir.parent / "invalid source"
    prepared.count_missing_as_loss = True
    task = prepared.existing_rollout / "deliverables/task_one/repeat_0"
    task.mkdir(parents=True)
    (task / "finish_params.json").write_text(marker)
    with pytest.raises(ValueError):
        snapshot.prepare(prepared)
    assert not prepared.run_dir.exists()


def test_snapshot_freezes_seed_aliases_and_explicit_override(prepared, monkeypatch):
    settings = json.loads((prepared.run_dir / "run.json").read_text())
    assert settings["JUDGE_SEED"] == "42" and settings["STAGE0_SEED"] == ""
    monkeypatch.setenv("MULTISTAGE_SEED", "19")
    monkeypatch.setenv("JUDGE_STAGE0_SEED", "882")
    prepared.run_dir = prepared.run_dir.with_name("seeded")
    snapshot.prepare(prepared)
    settings = json.loads((prepared.run_dir / "run.json").read_text())
    assert settings["JUDGE_SEED"] == "19" and settings["STAGE0_SEED"] == "882"
    prepared.run_dir = prepared.run_dir.with_name("explicit seed")
    prepared.judge_seed, prepared.stage0_seed = 0, 123
    snapshot.prepare(prepared)
    settings = json.loads((prepared.run_dir / "run.json").read_text())
    assert settings["JUDGE_SEED"] == "0" and settings["STAGE0_SEED"] == "123"


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
    metrics["comparison/stage_1/eval_elo"] = 1100.0
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
        calibration = []
        if mode == "full":
            calibration = [
                {**row, "stage_index": 0, "judge_response": {"total_judged": trials, "total_invalid": 0}}
                for row in rows
            ]
            plans = [
                {"stage_index": stage, "status": "planned", "task_ids": [str(i) for i in range(count)]}
                for stage in (0, 1)
            ]
            output.with_stem(output.stem + "_multistage_state").write_text(
                "".join(json.dumps(plan) + "\n" for plan in plans)
            )
        output.write_text("".join(json.dumps(row) + "\n" for row in calibration + rows))

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


@pytest.mark.parametrize(
    "case",
    ["complete", "calibration_missing", "calibration_imputed", "final_imputed", "failed", "unrecorded", "unattempted"],
)
def test_full_judge_requires_all_calibration_tasks_and_accounts_for_final_failures(
    tmp_path, monkeypatch, capsys, case
):
    dataset = tmp_path / "dataset.jsonl"
    ids = [str(i) for i in range(50)]
    dataset.write_text("".join(json.dumps({"task_id": task_id}) + "\n" for task_id in ids))
    output = tmp_path / "rollouts.jsonl"
    plans = [
        {
            "status": "planned",
            "stage_index": stage,
            "task_ids": ids[:45] if stage == 0 else ids,
            "task_reference_ids": dict.fromkeys(ids, "reference"),
        }
        for stage in (0, 1)
    ]
    output.with_stem(output.stem + "_multistage_state").write_text("".join(json.dumps(plan) + "\n" for plan in plans))
    rows = [
        {
            "task_id": task_id,
            "stage_index": plan["stage_index"],
            "expected_final_stage_index": 1,
            "judge_response": {"total_judged": 1, "total_invalid": 3},
        }
        for plan in plans
        for task_id in plan["task_ids"]
    ]
    if case == "calibration_missing":
        rows.pop(0)
    if case in ("calibration_imputed", "final_imputed"):
        rows[0 if case == "calibration_imputed" else -1]["judge_response"] = {
            "total_judged": 4,
            "total_invalid": 0,
            "manual_imputation": "eval_missing_as_loss",
        }
    partial = case in ("failed", "unrecorded", "unattempted")
    if partial:
        rows.pop()
    output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    if case in ("failed", "unattempted"):
        failure = {
            "task_id": ids[-1],
            "stage_index": 1,
            "reference_ids": ["reference"],
            "_ng_failure_class": "transport_ineligible",
            "_ng_no_persist": case == "unattempted",
        }
        output.with_stem(output.stem + "_failures").write_text(json.dumps(failure) + "\n")
    metrics = {
        "comparison/final_stage_present": 1,
        "comparison/final_stage_complete": int(not partial),
        "comparison/final_stage_fit": 1,
        "comparison/final_stage_degraded": int(partial),
        "comparison/stage_1/eval_elo": 1100.0,
    }
    metric_path = output.with_stem(output.stem + "_aggregate_metrics").with_suffix(".json")
    metric_path.write_text(json.dumps([{"agent_metrics": metrics}]))
    if case in ("calibration_missing", "calibration_imputed"):
        with pytest.raises(ValueError, match="calibration"):
            completion.judge_complete(dataset, output)
    elif case in ("unrecorded", "unattempted"):
        with pytest.raises(ValueError, match="without recorded failed attempts"):
            completion.judge_complete(dataset, output)
    else:
        before = {path: path.read_bytes() for path in tmp_path.iterdir()}
        assert completion.judge_complete(dataset, output) == (49 if partial else 50)
        monkeypatch.setattr("sys.argv", ["completion.py", "judge", "--dataset", str(dataset), "--output", str(output)])
        completion.main()
        report = capsys.readouterr().out
        if case == "final_imputed":
            assert "49/50 actually judged tasks, 1 imputed-loss tasks" in report
        else:
            assert report.startswith("PARTIAL: judge, 49/50" if partial else "COMPLETE: judge, 50")
        assert all(path.read_bytes() == content for path, content in before.items())


def test_pilot_report_does_not_count_imputed_losses_as_judged_tasks(tmp_path, monkeypatch, capsys):
    dataset, output = tmp_path / "dataset.jsonl", tmp_path / "rollouts.jsonl"
    dataset.write_text('{"task_id":"one"}\n')
    rows = [
        {
            "task_id": "one",
            "stage_index": stage,
            "expected_final_stage_index": 1,
            "judge_response": {"total_judged": 2, "total_invalid": 0},
        }
        for stage in (0, 1)
    ]
    rows[1]["judge_response"]["manual_imputation"] = "eval_missing_as_loss"
    output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output.with_stem(output.stem + "_aggregate_metrics").with_suffix(".json").write_text(
        json.dumps([{"agent_metrics": {"comparison/final_stage_present": 1, "comparison/final_stage_complete": 1}}])
    )
    monkeypatch.setattr(
        "sys.argv", ["completion.py", "judge", "--dataset", str(dataset), "--output", str(output), "--mode", "pilot"]
    )
    completion.main()
    assert "COMPLETE: judge, 0/1 actually judged tasks, 1 imputed-loss tasks" in capsys.readouterr().out
