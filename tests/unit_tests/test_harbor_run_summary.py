# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The per-run summary a Harbor run writes beside its rollouts (nemo_gym.tasks.harbor.run_summary)."""

import json
import os
import sys
from pathlib import Path

import pytest
import yaml

import nemo_gym.cli.main as main_module
from nemo_gym.config import global_config
from nemo_gym.path_utils import failures_path_for, materialized_path_for, run_summary_path_for
from nemo_gym.rollout_health import write_provisional_summary
from nemo_gym.tasks.harbor.cli import prepare_target, resolve_agent
from nemo_gym.tasks.harbor.run_summary import build_run_summary, write_run_summary
from tests.unit_tests.test_harbor_tasks import no_real_registry, write_task  # noqa: F401 (autouse fixture)
from tests.unit_tests.test_harbor_validation import (  # noqa: F401 (fixtures)
    oracle_collection,
    result,
    validation_cli,
    write_jsonl,
)


def _selected(task, index, repeat=0, *, taskset="ds", task_input=None):
    """A collector input row: what `_preprocess_rows_from_config` hands the summary as `expected`.

    Materialized rows carry no `task_source`; collection stamps it on the results rows it writes."""
    return {
        "task_id": {"taskset": taskset, "task_id": task},
        "_ng_task_index": index,
        "_ng_rollout_index": repeat,
        "task_input": task_input
        or {
            "responses_create_params": {"input": [], "metadata": {"agent_timeout_sec": "900.0"}},
            "task_data": {"_ng_digest": f"digest-{task}", "_ng_dataset_digest": "dataset-digest"},
        },
    }


def _logs_dir(root: Path, name: str, *files: str) -> Path:
    folder = root / name
    folder.mkdir(parents=True)
    for file in files:
        (folder / file).parent.mkdir(parents=True, exist_ok=True)
        (folder / file).write_text("x")
    return folder


def health_reports(folder: Path, rollouts_name: str, *, verdicts: bool = True) -> None:
    """What the health check leaves after checking ``folder / rollouts_name``: its quality summary, recording the
    rollouts file it was computed from, and (unless the run was stopped) the verdicts beside it."""
    (folder / "quality_summary.json").write_text(json.dumps({"run": {"rollouts": [rollouts_name]}}))
    if verdicts:
        (folder / "rollout_verdicts.jsonl").write_text("{}\n")


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("a", "b", "c", "d"):
        write_task(tmp_path / "ds" / name)
    # A task folder that does not load is skipped by name (see test_malformed_task_is_skipped_by_name).
    write_task(tmp_path / "ds" / "broken", toml="[task\nname = oops")
    return prepare_target(str(tmp_path / "ds"), output_root=tmp_path / "results")


def test_summary_points_at_every_file_by_relative_path_and_reports_what_is_missing(tmp_path, prepared):
    rollouts = prepared.output_dir / "rollouts.jsonl"
    verifier = prepared.output_dir / "verifier"
    # a: a complete shared-mode verification; b: separate mode with collected artifacts and a trajectory;
    # c: the row names a log folder that is gone; d: never produced a result and failed its attempt, in a
    # sandbox whose entrypoint died, so the failure row still names the folder with its entrypoint output.
    logs_a = _logs_dir(verifier, "session-a", "reward.txt", "test-stdout.txt")
    logs_b = _logs_dir(verifier, "session-b", "reward.json", "trajectory.json", "artifacts/00/app/out.txt")
    logs_d = _logs_dir(verifier, "session-d", "entrypoint-stdout.txt")
    write_jsonl(
        rollouts,
        [
            result(
                "a",
                index=0,
                verifier_logs_dir=str(logs_a),
                verifier_mode="shared",
                verifier_seconds=12.4,
                verifier_network="public",
                verifier_return_code=0,
                verifier_rewards={"reward": 1.0},
                task_source="ds",
            ),
            result(
                "b",
                index=1,
                reward=0.0,
                verifier_logs_dir=str(logs_b),
                verifier_mode="separate",
                task_source="ds",
            ),
            result(
                "c", index=2, mask_sample=True, failure_kind="harbor:lost", verifier_logs_dir=str(verifier / "gone")
            ),
        ],
    )
    write_jsonl(
        failures_path_for(rollouts),
        [
            # Superseded by a's result: a is reported once, as a result.
            result("a", index=0, _ng_attempt_index=0, _ng_failure_class="agent_request_failed"),
            result(
                "d",
                index=3,
                _ng_attempt_index=1,
                _ng_failure_class="environment_server_failed",
                _ng_failure_stage="seed",
                _ng_failure_message="entrypoint exited before the sandbox was ready",
                verifier_logs_dir=str(logs_d),
            ),
        ],
    )
    # What the data processor and the collector leave beside the rows and under `<stem>/preprocessed_datasets/`.
    rows_path = Path(prepared.rows_path)
    rows_path.with_name("tasks_prepare.jsonl").write_text("")
    rows_path.with_name("tasks_metrics.json").write_text("{}")
    _logs_dir(prepared.output_dir, "rollouts/preprocessed_datasets", "validation.jsonl", "validation_metrics.json")
    health_reports(prepared.output_dir, "rollouts.jsonl", verdicts=False)
    expected = [_selected(name, index) for index, name in enumerate("abcd")]
    path = write_run_summary(
        rollouts,
        command="gym dataset validate",
        target=str(tmp_path / "ds"),
        prepared=prepared,
        agent=resolve_agent("harbor_oracle_agent"),
        run_config=prepared.output_dir / "run_config_harbor_oracle_agent.yaml",
        expected=expected,
        split="validation",
    )
    assert path == run_summary_path_for(rollouts) == prepared.output_dir / "rollouts_run_summary.json"
    summary = json.loads(path.read_text())
    base = path.parent

    assert summary["schema_version"] == 1 and summary["summary_dir"] == str(base)
    assert summary["command"] == "gym dataset validate" and summary["taskset"] == "ds"
    assert summary["agent"] == {"instance": "harbor_oracle_agent", "implementation": "harbor_oracle_agent"}
    # The task folders that did not load, by name with the loader's reason, as `skipped.json` has them.
    assert summary["skipped"] == prepared.skipped and list(summary["skipped"]) == ["broken"]
    assert "task.toml" in summary["skipped"]["broken"]
    assert summary["dataset"] == {
        "folder": str((tmp_path / "ds").resolve()),
        "source": "local",
        "digest": prepared.dataset_digest,
        "dataset_toml_digest": prepared.dataset_toml_digest,
    }
    # The run config's agent block for the taskset, and the value the rows agree on (here the same).
    assert summary["dataset"]["digest"] and summary["task_source"] == "ds"
    assert summary["rows_task_source"] == "ds"
    # Every file of the run, relative to the summary's folder; a file the run did not write is null.
    assert summary["files"] == {
        "rollouts": "rollouts.jsonl",
        "failures": "rollouts_failures.jsonl",
        "materialized_inputs": None,
        "aggregate_metrics": None,
        "quality_summary": "quality_summary.json",
        "rollout_verdicts": None,
        "rows": "tasks.jsonl",
        "rows_prepare": "tasks_prepare.jsonl",
        "rows_metrics": "tasks_metrics.json",
        "preprocessed_datasets": "rollouts/preprocessed_datasets",
        "collector_input": "rollouts/preprocessed_datasets/validation.jsonl",
        "collector_input_metrics": "rollouts/preprocessed_datasets/validation_metrics.json",
        "run_config": None,
        "skipped": "skipped.json",
        "images": "images.json",
        "verifier_root": "verifier",
    }
    assert summary["counts"] == {"selected": 4, "results": 3, "masked": 1, "failed": 1, "missing": 0}
    assert summary["stale_files"] == {}

    by_task = {entry["task_id"]: entry for entry in summary["tasks"]}
    assert [entry["task_id"] for entry in summary["tasks"]] == ["a", "b", "c", "d"]
    a = by_task["a"]
    assert a["status"] == "result" and a["reward"] == 1.0 and a["task_index"] == 0 and a["rollout_index"] == 0
    assert a["verifier"] == {
        "mode": "shared",
        "seconds": 12.4,
        "network": "public",
        "return_code": 0,
        "rewards": {"reward": 1.0},
    }
    assert a["paths"] == {
        "verifier_logs": "verifier/session-a",
        "artifacts": None,
        "test_stdout": "verifier/session-a/test-stdout.txt",
        "trajectory": None,
        "entrypoint_stdout": None,
    }
    assert a["missing_paths"] == [] and a["response_metadata"] == {"oracle": "solved"}
    # The input the row carried: the task and dataset digests and the request metadata.
    assert a["input"] == {
        "digest": "digest-a",
        "dataset_digest": "dataset-digest",
        "metadata": {"agent_timeout_sec": "900.0"},
    }
    b = by_task["b"]
    assert b["paths"]["artifacts"] == "verifier/session-b/artifacts"
    assert b["paths"]["trajectory"] == "verifier/session-b/trajectory.json"
    assert b["verifier"]["mode"] == "separate" and b["reward"] == 0.0
    c = by_task["c"]
    assert c["mask_sample"] is True and c["failure_kind"] == "harbor:lost"
    assert c["paths"]["verifier_logs"] == "verifier/gone" and c["missing_paths"] == ["verifier_logs"]
    d = by_task["d"]
    assert d["status"] == "failed" and d["reward"] is None
    assert d["failure"] == {
        "class": "environment_server_failed",
        "stage": "seed",
        "message": "entrypoint exited before the sandbox was ready",
        "attempt": 1,
    }
    # A failure row that names a log folder points at it like a result does: here only the entrypoint output.
    assert d["paths"] == {
        "verifier_logs": "verifier/session-d",
        "artifacts": None,
        "test_stdout": None,
        "trajectory": None,
        "entrypoint_stdout": "verifier/session-d/entrypoint-stdout.txt",
    }
    assert d["missing_paths"] == [] and d["input"]["digest"] == "digest-d"

    # Every path the summary names resolves from the summary's folder, and none is absolute.
    named = [value for entry in summary["tasks"] for value in entry.get("paths", {}).values() if value]
    named += [value for value in summary["files"].values() if value]
    assert named and all(not Path(value).is_absolute() for value in named)
    assert all((base / value).exists() for value in named if value != "verifier/gone")


def test_summary_without_expected_rows_or_rollouts_still_describes_the_run(tmp_path, prepared):
    rollouts = tmp_path / "elsewhere" / "custom.jsonl"
    summary = build_run_summary(
        rollouts,
        command="gym eval run",
        target="harbor:some/dataset",
        prepared=prepared,
        agent=resolve_agent("hermes_agent"),
        run_config=None,
        expected=None,
    )
    assert summary["files"]["rollouts"] is None and summary["tasks"] == []
    assert summary["counts"] == {"selected": 0, "results": 0, "masked": 0, "failed": 0, "missing": 0}
    # `task_source` is known before dispatch (the run config declares it); only the rows' echo needs rows.
    assert summary["task_source"] == "ds" and summary["rows_task_source"] is None
    assert summary["target"] == "harbor:some/dataset"
    # Without a split there is no collated input to name; the folder is reported when it exists.
    assert summary["files"]["preprocessed_datasets"] is None and summary["files"]["collector_input"] is None
    # Files that do exist are still reached from the summary's folder, wherever `--output` put it.
    rows = summary["files"]["rows"]
    assert rows.startswith("../") and (rollouts.parent / rows).resolve() == prepared.rows_path.resolve()

    # A selected task with no row at all is listed as missing, with its input echoed from the rows file.
    summary = build_run_summary(
        rollouts,
        command="gym eval run",
        target="ds",
        prepared=prepared,
        agent=resolve_agent("hermes_agent"),
        run_config=None,
        expected=[{**_selected("b", 1), "task_input": None}],
    )
    (entry,) = summary["tasks"]
    assert entry["status"] == "missing" and entry["task_id"] == "b"
    materialized = next(task for task in prepared.tasks if task.task_id == "b")
    assert entry["input"]["digest"] == materialized.digest
    assert entry["input"]["dataset_digest"] == prepared.dataset_digest
    assert entry["input"]["metadata"]["agent_timeout_sec"]


def test_rows_written_by_another_run_show_up_in_the_task_source_cross_check(tmp_path, prepared):
    rollouts = prepared.output_dir / "rollouts.jsonl"
    write_jsonl(rollouts, [result("a", task_source="harbor_other_agent"), result("b", index=1, task_source="x")])
    summary = build_run_summary(
        rollouts,
        command="gym eval run",
        target="ds",
        prepared=prepared,
        agent=resolve_agent("hermes_agent"),
        run_config=None,
        expected=[_selected("a", 0)],
    )
    # The declared source stands; rows that disagree with each other echo nothing rather than one of them.
    assert summary["task_source"] == "ds" and summary["rows_task_source"] is None
    write_jsonl(rollouts, [result("a", task_source="harbor_other_agent")])
    summary = build_run_summary(
        rollouts,
        command="gym eval run",
        target="ds",
        prepared=prepared,
        agent=resolve_agent("hermes_agent"),
        run_config=None,
        expected=[_selected("a", 0)],
    )
    assert summary["task_source"] == "ds" and summary["rows_task_source"] == "harbor_other_agent"


def test_task_source_is_the_dataset_label_the_rows_were_materialized_with(tmp_path, prepared):
    """Materialization stamps `prepared.task_source` (the dataset folder's name, else the taskset) on every row and
    collection preserves it, so the declared `task_source` is that label, not the run config's agent instance
    (`harbor_<taskset>_agent`): with the instance, the cross-check with `rows_task_source` disagreed with the run's
    own results for every run."""
    rows = [json.loads(line) for line in Path(prepared.rows_path).read_text().splitlines()]
    assert {row["task_source"] for row in rows} == {"ds"}
    rollouts = prepared.output_dir / "rollouts.jsonl"
    write_jsonl(rollouts, [result("a", task_source="ds")])
    summary = build_run_summary(
        rollouts,
        command="gym eval run",
        target="ds",
        prepared=prepared,
        agent=resolve_agent("hermes_agent"),
        run_config=None,
        expected=[_selected("a", 0)],
    )
    assert summary["task_source"] == summary["rows_task_source"] == "ds"
    # A single task folder inside the dataset keeps the dataset's label while its taskset is the folder's own name.
    single = prepare_target(str(tmp_path / "ds" / "a"), output_root=tmp_path / "single")
    assert (single.taskset, single.task_source) == ("a", "ds")
    rollouts = single.output_dir / "rollouts.jsonl"
    write_jsonl(rollouts, [result("a", _ng_task_id={"taskset": "a", "task_id": "a"}, task_source="ds")])
    summary = build_run_summary(
        rollouts,
        command="gym eval run",
        target="ds/a",
        prepared=single,
        agent=resolve_agent("hermes_agent"),
        run_config=None,
        expected=[_selected("a", 0, taskset="a")],
    )
    assert summary["task_source"] == summary["rows_task_source"] == "ds"
    assert summary["taskset"] == "a" and summary["counts"]["results"] == 1


def test_summary_reads_the_rows_files_one_record_at_a_time(tmp_path, prepared):
    """The summary keeps a few fields per row; a long run's responses are never all in memory at once.

    A `gym eval run` over a large taskset leaves a rollouts file of long multi-turn trajectories; the summary
    projects each row as it is read, so its peak memory tracks one record, not the file."""
    import tracemalloc

    rollouts = prepared.output_dir / "rollouts.jsonl"
    trajectory = {"metadata": {"oracle": "solved"}, "output": [{"type": "message", "content": "x" * (256 * 1024)}]}
    write_jsonl(
        rollouts,
        [result("a", repeat=repeat, response=trajectory, task_source="ds") for repeat in range(32)],
    )
    write_jsonl(
        failures_path_for(rollouts), [result("b", index=1, repeat=repeat, response=trajectory) for repeat in range(32)]
    )
    size = rollouts.stat().st_size + failures_path_for(rollouts).stat().st_size
    assert size > 16 * 1024 * 1024
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        summary = build_run_summary(
            rollouts,
            command="gym eval run",
            target="ds",
            prepared=prepared,
            agent=resolve_agent("hermes_agent"),
            run_config=None,
            expected=[_selected("a", 0, repeat) for repeat in range(32)],
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert summary["counts"] == {"selected": 32, "results": 32, "masked": 0, "failed": 32, "missing": 0}
    assert summary["rows_task_source"] == "ds"
    assert all(entry["response_metadata"] == {"oracle": "solved"} for entry in summary["tasks"][:32])
    # Reading every record whole (and the file's bytes beside them) costs more than the files themselves.
    assert peak < size / 2, f"peak {peak} bytes for {size} bytes of rows"


def test_files_written_after_collection_are_listed_only_when_newer_than_the_rollouts(tmp_path, prepared):
    rollouts = prepared.output_dir / "rollouts.jsonl"
    post_collection = {
        "aggregate_metrics": prepared.output_dir / "rollouts_aggregate_metrics.json",
        "quality_summary": prepared.output_dir / "quality_summary.json",
        "rollout_verdicts": prepared.output_dir / "rollout_verdicts.jsonl",
    }
    post_collection["aggregate_metrics"].write_text("{}")
    health_reports(prepared.output_dir, "rollouts.jsonl")

    def files():
        summary = build_run_summary(
            rollouts,
            command="gym dataset validate",
            target="ds",
            prepared=prepared,
            agent=resolve_agent("harbor_oracle_agent"),
            run_config=None,
            expected=[_selected("a", 0)],
        )
        return {key: summary["files"][key] for key in ("rollouts", *post_collection)}, summary["stale_files"]

    left_behind = {
        "aggregate_metrics": "rollouts_aggregate_metrics.json",
        "quality_summary": "quality_summary.json",
        "rollout_verdicts": "rollout_verdicts.jsonl",
    }
    # A run that died before any row: the reports an earlier run left in the folder are not this run's, and
    # are named as such so the null beside a file by that name is explained.
    assert files() == (
        {"rollouts": None, "aggregate_metrics": None, "quality_summary": None, "rollout_verdicts": None},
        left_behind,
    )
    # The rollouts were written after them (a run that stopped before the health check): still not listed.
    write_jsonl(rollouts, [result("a")])
    stale = rollouts.stat().st_mtime - 60
    for path in post_collection.values():
        os.utime(path, (stale, stale))
    assert files() == (
        {"rollouts": "rollouts.jsonl", "aggregate_metrics": None, "quality_summary": None, "rollout_verdicts": None},
        left_behind,
    )
    # Rewritten after the rollouts (what a completed collection does): listed.
    fresh = rollouts.stat().st_mtime + 1
    for path in post_collection.values():
        os.utime(path, (fresh, fresh))
    assert files() == ({"rollouts": "rollouts.jsonl", **left_behind}, {})


def test_health_reports_are_linked_only_when_they_record_this_rollouts_file(tmp_path, prepared):
    """Review (round 38): the health reports keep one name per folder whatever the `--output` stem, so after runs
    of `first.jsonl` and `second.jsonl` in one folder the reports are the later run's, newer than both rollouts
    files. The health check records the rollouts it was computed from in the quality summary (`run.rollouts`), and
    the summary links the reports only when that record names its own rollouts file; the verdicts, written and
    removed with the summary, follow it."""
    folder = prepared.output_dir
    first, second = folder / "first.jsonl", folder / "second.jsonl"
    write_jsonl(first, [result("a")])
    write_jsonl(second, [result("a")])

    def reports(rollouts):
        summary = build_run_summary(
            rollouts,
            command="gym eval run",
            target="ds",
            prepared=prepared,
            agent=resolve_agent("hermes_agent"),
            run_config=None,
            expected=[_selected("a", 0)],
        )
        return {key: summary["files"][key] for key in ("quality_summary", "rollout_verdicts")}, summary["stale_files"]

    # The real writer of a stopped run's summary, for second.jsonl: it names that file and removes the verdicts.
    write_provisional_summary(second, set())
    recorded = json.loads((folder / "quality_summary.json").read_text())["run"]["rollouts"]
    assert recorded == ["second.jsonl"]
    assert reports(second) == ({"quality_summary": "quality_summary.json", "rollout_verdicts": None}, {})
    # Newer than first.jsonl, but not its: null, and named as left behind so the null beside the file is explained.
    assert reports(first) == (
        {"quality_summary": None, "rollout_verdicts": None},
        {"quality_summary": "quality_summary.json"},
    )
    # A report that records first.jsonl is first's, with the verdicts beside it; second's summary says so too.
    health_reports(folder, "first.jsonl")
    assert reports(first) == (
        {"quality_summary": "quality_summary.json", "rollout_verdicts": "rollout_verdicts.jsonl"},
        {},
    )
    assert reports(second) == (
        {"quality_summary": None, "rollout_verdicts": None},
        {"quality_summary": "quality_summary.json", "rollout_verdicts": "rollout_verdicts.jsonl"},
    )
    # A report without the record (an older check's) is nobody's.
    (folder / "quality_summary.json").write_text("{}")
    assert reports(first)[0] == {"quality_summary": None, "rollout_verdicts": None}
    # The check of a file `gym eval aggregate` merged reads and records its shards: the merged file's run owns it.
    merged = folder / "merged.jsonl"
    write_jsonl(merged, [result("a")])
    (folder / "shards").mkdir()
    write_jsonl(folder / "shards" / "rollouts.jsonl", [result("a")])
    (folder / "merged_sources.json").write_text(
        json.dumps({"shards": [["shards/rollouts.jsonl", 1, "x"]], "merged_bytes": 1, "merged_sha256": "x"})
    )
    health_reports(folder, "shards/rollouts.jsonl")
    assert reports(merged)[0] == {
        "quality_summary": "quality_summary.json",
        "rollout_verdicts": "rollout_verdicts.jsonl",
    }
    assert reports(first)[0] == {"quality_summary": None, "rollout_verdicts": None}


def test_multi_turn_rows_report_each_turns_logs(tmp_path, prepared):
    rollouts = prepared.output_dir / "rollouts.jsonl"
    first = _logs_dir(prepared.output_dir / "verifier", "session-1", "reward.txt", "trajectory.json")
    second = prepared.output_dir / "verifier" / "session-2"
    write_jsonl(
        rollouts,
        [
            result(
                "a",
                reward=0.5,
                verifier_logs_dir=None,
                turns=[
                    {
                        "name": "one",
                        "reward": 1.0,
                        "verifier_logs_dir": str(first),
                        "verifier_seconds": 3.0,
                        "verifier_network": "public",
                        "verifier_return_code": 0,
                    },
                    {
                        "name": "two",
                        "reward": 0.0,
                        "mask_sample": True,
                        "failure_kind": "harbor:verifier_lost",
                        "verifier_logs_dir": str(second),
                    },
                ],
                stopped_after_turn="two",
            )
        ],
    )
    task_input = {
        "task_data": {"_ng_digest": "d", "_ng_dataset_digest": "dd"},
        "turns": [
            {"name": "one", "responses_create_params": {"input": [], "metadata": {"harbor_step": "one"}}},
            {"name": "two", "responses_create_params": {"input": [], "metadata": {"harbor_step": "two"}}},
        ],
    }
    summary = build_run_summary(
        rollouts,
        command="gym eval run",
        target="ds",
        prepared=prepared,
        agent=resolve_agent("hermes_agent"),
        run_config=None,
        expected=[_selected("a", 0, task_input=task_input)],
    )
    (entry,) = summary["tasks"]
    # The episode row names no folder of its own: nothing is promised, so nothing is missing.
    assert entry["paths"]["verifier_logs"] is None and entry["missing_paths"] == []
    assert entry["stopped_after_turn"] == "two"
    one, two = entry["turns"]
    assert one["name"] == "one" and one["paths"]["trajectory"] == "verifier/session-1/trajectory.json"
    assert one["verifier"] == {"seconds": 3.0, "network": "public", "return_code": 0}
    assert one["reward"] == 1.0 and one["mask_sample"] is False and one["missing_paths"] == []
    assert two["verifier"] == {"seconds": None, "network": None, "return_code": None}
    assert two["mask_sample"] is True and two["missing_paths"] == ["verifier_logs"]
    assert two["paths"]["verifier_logs"] == "verifier/session-2"
    assert entry["input"]["turns"] == [
        {"name": "one", "metadata": {"harbor_step": "one"}},
        {"name": "two", "metadata": {"harbor_step": "two"}},
    ]


def test_validate_writes_the_summary_beside_the_rollouts_even_when_collection_fails(
    tmp_path, monkeypatch, capsys, validation_cli
):
    write_task(tmp_path / "ds" / "a")
    output_dir = tmp_path / "results" / "harbor" / "validation" / "ds"
    attempts = []

    def collect(target, tokens):
        from nemo_gym.config import get_global_config_dict

        rollouts = Path(get_global_config_dict().output_jsonl_fpath)
        attempts.append(rollouts)
        if len(attempts) == 1:
            raise RuntimeError("None of the 1 dispatched rollouts produced a result")
        logs = _logs_dir(output_dir / "verifier", "session-live", "reward.txt", "test-stdout.txt")
        write_jsonl(rollouts, [result(verifier_logs_dir=str(logs), verifier_mode="shared", task_source="x")])
        # The data processor's files: beside the rows and in the collator's folder, named after the split, which
        # M37 takes from the dataset's [gym] type (benchmark for a dataset that declares none).
        (output_dir / "tasks_harbor_oracle_agent_prepare.jsonl").write_text("")
        _logs_dir(rollouts.with_suffix(""), "preprocessed_datasets", "benchmark.jsonl", "benchmark_metrics.json")

    monkeypatch.setattr(main_module, "dispatch", collect)
    with pytest.raises(RuntimeError, match="produced a result"):
        validation_cli()
    summary_path = output_dir / "rollouts_run_summary.json"
    assert [path.resolve() for path in attempts] == [output_dir / "rollouts.jsonl"] and summary_path.is_file()
    summary = json.loads(summary_path.read_text())
    assert summary["files"]["rollouts"] is None and summary["counts"]["missing"] == 1
    (entry,) = summary["tasks"]
    assert entry["status"] == "missing" and entry["task_id"] == "a" and entry["input"]["digest"]
    assert f"Run summary written to {summary_path}" in capsys.readouterr().out

    validation_cli("+resume_from_cache=false")
    summary = json.loads(summary_path.read_text())
    assert summary["command"] == "gym dataset validate" and summary["target"] == str(tmp_path / "ds")
    assert summary["agent"]["instance"] == "harbor_oracle_agent" and summary["dataset"]["source"] == "local"
    assert summary["files"]["rollouts"] == "rollouts.jsonl"
    assert summary["files"]["run_config"] == "run_config_harbor_oracle_agent.yaml"
    assert summary["files"]["rows"] == "tasks_harbor_oracle_agent.jsonl"
    assert summary["files"]["rows_prepare"] == "tasks_harbor_oracle_agent_prepare.jsonl"
    assert summary["files"]["rows_metrics"] is None
    assert summary["files"]["collector_input"] == "rollouts/preprocessed_datasets/benchmark.jsonl"
    assert summary["files"]["collector_input_metrics"] == "rollouts/preprocessed_datasets/benchmark_metrics.json"
    assert summary["task_source"] == "ds" and summary["rows_task_source"] == "x"
    (entry,) = summary["tasks"]
    assert entry["status"] == "result" and entry["reward"] == 1.0
    assert entry["paths"]["verifier_logs"] == "verifier/session-live" and entry["missing_paths"] == []
    assert (summary_path.parent / entry["paths"]["test_stdout"]).is_file()
    assert entry["input"]["metadata"] == {"agent_timeout_sec": "120.0"}
    assert entry["input"]["dataset_digest"] == summary["dataset"]["digest"]


def test_fresh_start_that_dies_before_the_collector_clears_its_outputs_reports_no_earlier_rows(
    tmp_path, monkeypatch, validation_cli, oracle_collection
):
    """A fresh start (`resume_from_cache=false`, the default) clears the rollouts, failures and materialized inputs
    only once its servers are up. One that dies before that (servers that fail to start) must not pass the files
    an earlier run left under those names off as its own; a resumed run's cached rows are its own; a fresh start
    that reaches the collector rewrites the files and owns them."""
    from nemo_gym.cli.eval import RunHelper

    write_task(tmp_path / "ds" / "a")
    output_dir = tmp_path / "results" / "harbor" / "validation" / "ds"
    rollouts = output_dir / "rollouts.jsonl"
    summary_path = output_dir / "rollouts_run_summary.json"
    # An earlier, complete run through the real collector: rollouts and materialized inputs with the task's real
    # digests, plus a failures file and the health check's reports beside them, all a minute old.
    oracle_collection.invoke()
    assert len(oracle_collection.dispatched) == 1
    write_jsonl(failures_path_for(rollouts), [result(_ng_attempt_index=0, _ng_failure_class="agent_request_failed")])
    health_reports(output_dir, "rollouts.jsonl")
    earlier = {
        path: path.read_bytes()
        for path in (
            rollouts,
            failures_path_for(rollouts),
            materialized_path_for(rollouts),
            output_dir / "quality_summary.json",
            output_dir / "rollout_verdicts.jsonl",
        )
    }
    a_minute_ago = rollouts.stat().st_mtime - 60
    for path in earlier:
        os.utime(path, (a_minute_ago, a_minute_ago))
    summary = json.loads(summary_path.read_text())
    assert summary["files"]["rollouts"] == "rollouts.jsonl" and summary["counts"]["results"] == 1
    assert summary["stale_files"] == {}

    def failing_start(*args):
        raise RuntimeError("environment server failed to start")

    # Resumed (the fixture's default): the servers fail to start, and the cached rows are still this run's.
    monkeypatch.setattr(RunHelper, "start", failing_start)
    with pytest.raises(RuntimeError, match="failed to start"):
        oracle_collection.invoke()
    resumed = json.loads(summary_path.read_text())
    assert resumed["files"]["rollouts"] == "rollouts.jsonl"
    assert resumed["files"]["failures"] == "rollouts_failures.jsonl"
    assert resumed["files"]["materialized_inputs"] == "rollouts_materialized_inputs.jsonl"
    assert resumed["files"]["quality_summary"] == "quality_summary.json"
    assert resumed["counts"] == {"selected": 1, "results": 1, "masked": 0, "failed": 0, "missing": 0}
    assert resumed["task_source"] == "ds" == resumed["rows_task_source"] and resumed["stale_files"] == {}

    # A fresh start that dies the same way: the files under its names are the earlier run's, left untouched.
    monkeypatch.setattr(global_config, "_GLOBAL_CONFIG_DICT", None)
    with pytest.raises(RuntimeError, match="failed to start"):
        validation_cli("+disable_aggregation=true", "+resume_from_cache=false")
    fresh = json.loads(summary_path.read_text())
    assert {name: fresh["files"][name] for name in fresh["stale_files"]} == dict.fromkeys(fresh["stale_files"])
    assert fresh["stale_files"] == {
        "rollouts": "rollouts.jsonl",
        "failures": "rollouts_failures.jsonl",
        "materialized_inputs": "rollouts_materialized_inputs.jsonl",
        "quality_summary": "quality_summary.json",
        "rollout_verdicts": "rollout_verdicts.jsonl",
    }
    assert fresh["counts"] == {"selected": 1, "results": 0, "masked": 0, "failed": 0, "missing": 1}
    (entry,) = fresh["tasks"]
    assert entry["status"] == "missing" and entry["task_id"] == "a" and entry["input"]["digest"]
    assert fresh["rows_task_source"] is None
    # The run config and rows are this run's, and the earlier run's files are as they were, for a resume.
    assert fresh["files"]["run_config"] == "run_config_harbor_oracle_agent.yaml"
    assert {path: path.read_bytes() for path in earlier} == earlier

    # A fresh start that reaches the collector: it clears and rewrites the files, which are then this run's.
    monkeypatch.setattr(RunHelper, "start", lambda *args: None)
    monkeypatch.setattr(global_config, "_GLOBAL_CONFIG_DICT", None)
    validation_cli("+disable_aggregation=true", "+resume_from_cache=false")
    assert len(oracle_collection.dispatched) == 2
    rerun = json.loads(summary_path.read_text())
    assert rerun["files"]["rollouts"] == "rollouts.jsonl"
    assert rerun["files"]["materialized_inputs"] == "rollouts_materialized_inputs.jsonl"
    # The collector unlinks the earlier failures file and opens a new, empty one: rewritten, so this run's.
    assert rerun["files"]["failures"] == "rollouts_failures.jsonl" and failures_path_for(rollouts).stat().st_size == 0
    assert rerun["counts"] == {"selected": 1, "results": 1, "masked": 0, "failed": 0, "missing": 0}
    assert rerun["rows_task_source"] == "ds"
    # The run removed the earlier job's health reports before its servers started (a report left by an earlier job
    # must not stand for this run's), so nothing under those names is left to list as stale.
    assert rerun["stale_files"] == {}
    assert rerun["files"]["quality_summary"] is None and rerun["files"]["rollout_verdicts"] is None
    assert not (output_dir / "quality_summary.json").exists() and not (output_dir / "rollout_verdicts.jsonl").exists()
    assert rollouts.read_bytes() != earlier[rollouts] or rollouts.stat().st_mtime > a_minute_ago


def test_a_resumed_run_that_fails_at_startup_does_not_link_another_stems_newer_reports(
    tmp_path, monkeypatch, validation_cli, oracle_collection
):
    """Review (round 38): reports for `rollouts.jsonl`, then for `second.jsonl` in the same folder; resuming the
    first and failing during server startup (before the collector's `_remove_health_reports`) must not link the
    second run's newer reports as the first's."""
    from nemo_gym.cli.eval import RunHelper

    write_task(tmp_path / "ds" / "a")
    output_dir = tmp_path / "results" / "harbor" / "validation" / "ds"
    first, second = output_dir / "rollouts.jsonl", output_dir / "second.jsonl"
    # Two complete runs through the real collector (the fixture disables aggregation, so each run's health check
    # stands in here through the real writer of a stopped run's summary, which records the file it checked).
    oracle_collection.invoke()
    write_provisional_summary(first, set())
    assert json.loads(run_summary_path_for(first).read_text())["files"]["quality_summary"] is None  # written after
    oracle_collection.invoke(f"+output_jsonl_fpath={second}")
    write_provisional_summary(second, set())
    assert json.loads((output_dir / "quality_summary.json").read_text())["run"]["rollouts"] == ["second.jsonl"]

    def failing_start(*args):
        raise RuntimeError("environment server failed to start")

    monkeypatch.setattr(RunHelper, "start", failing_start)
    with pytest.raises(RuntimeError, match="failed to start"):
        oracle_collection.invoke()
    resumed = json.loads(run_summary_path_for(first).read_text())
    # The cached rows are this run's; the folder's reports are the second stem's, newer or not.
    assert resumed["files"]["rollouts"] == "rollouts.jsonl" and resumed["counts"]["results"] == 1
    assert resumed["files"]["quality_summary"] is None and resumed["files"]["rollout_verdicts"] is None
    assert resumed["stale_files"] == {"quality_summary": "quality_summary.json"}
    # Resuming the second stem the same way: its own reports.
    with pytest.raises(RuntimeError, match="failed to start"):
        oracle_collection.invoke(f"+output_jsonl_fpath={second}")
    second_summary = json.loads(run_summary_path_for(second).read_text())
    assert second_summary["files"]["quality_summary"] == "quality_summary.json" and second_summary["stale_files"] == {}


def test_validate_names_the_collator_files_after_the_resolved_split(tmp_path, monkeypatch, validation_cli):
    """`build_run` keeps a user's `+split=`; the summary names `<split>.jsonl`, not a hard-coded validation file."""
    write_task(tmp_path / "ds" / "a")

    def collect(target, tokens):
        from nemo_gym.config import get_global_config_dict

        config = get_global_config_dict()
        assert config.split == "train"
        rollouts = Path(config.output_jsonl_fpath)
        write_jsonl(rollouts, [result()])
        # The collator's folder with the dispatched split beside a decoy from an earlier validation run.
        collated = rollouts.with_suffix("") / "preprocessed_datasets"
        _logs_dir(collated.parent, collated.name, "train.jsonl", "train_metrics.json", "validation.jsonl")

    monkeypatch.setattr(main_module, "dispatch", collect)
    validation_cli("+split=train")
    summary_path = tmp_path / "results" / "harbor" / "validation" / "ds" / "rollouts_run_summary.json"
    summary = json.loads(summary_path.read_text())
    assert summary["files"]["collector_input"] == "rollouts/preprocessed_datasets/train.jsonl"
    assert summary["files"]["collector_input_metrics"] == "rollouts/preprocessed_datasets/train_metrics.json"
    assert summary["files"]["preprocessed_datasets"] == "rollouts/preprocessed_datasets"


def test_validate_with_no_solvable_task_dispatches_nothing_and_still_writes_the_summary(
    tmp_path, monkeypatch, capsys, validation_cli
):
    write_task(tmp_path / "ds" / "a")
    (tmp_path / "ds/a/solution/solve.sh").unlink()
    monkeypatch.setattr(main_module, "dispatch", lambda *args: pytest.fail("must not dispatch solutionless tasks"))
    # An earlier run's rollouts under the default name: nothing ran, so they are not this run's.
    output_dir = tmp_path / "results" / "harbor" / "validation" / "ds"
    write_jsonl(output_dir / "rollouts.jsonl", [result("a", task_source="ds")])
    with pytest.raises(SystemExit) as exc:
        validation_cli()
    assert exc.value.code == 1
    summary_path = output_dir / "rollouts_run_summary.json"
    summary = json.loads(summary_path.read_text())
    assert summary["command"] == "gym dataset validate" and summary["task_source"] == "ds"
    assert summary["counts"] == {"selected": 0, "results": 0, "masked": 0, "failed": 0, "missing": 0}
    assert summary["tasks"] == [] and summary["files"]["rollouts"] is None
    assert summary["stale_files"] == {"rollouts": "rollouts.jsonl"} and summary["rows_task_source"] is None
    # The selection explains itself: the task was left out for want of a reference solution, not failed.
    assert summary["unvalidated"] == {"a": "no reference solution"}
    # No run config was generated (nothing to run); the rows file (empty) and skipped.json were.
    assert summary["files"]["run_config"] is None and summary["files"]["rows"] == "tasks_harbor_oracle_agent.jsonl"
    assert summary["files"]["skipped"] == "skipped.json"
    assert f"Run summary written to {summary_path}" in capsys.readouterr().out


@pytest.mark.parametrize("source", ["cli", "file", "flag"])
def test_validate_with_no_solvable_task_writes_the_summary_beside_the_requested_output(
    tmp_path, monkeypatch, capsys, validation_cli, source
):
    write_task(tmp_path / "ds" / "a")
    (tmp_path / "ds/a/solution/solve.sh").unlink()
    monkeypatch.setattr(main_module, "dispatch", lambda *args: pytest.fail("must not dispatch solutionless tasks"))
    # The same three ways the dispatching path reads its effective output: a Hydra override, a `--config`
    # overlay and the quoted `--output` flag (a name with parentheses, a comma and Unicode survives the parse).
    requested = tmp_path / "elsewhere" / ("résumé(1),tb.jsonl" if source == "flag" else "tb.jsonl")
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(yaml.safe_dump({"output_jsonl_fpath": str(requested)}))
    flags = {
        "cli": [f"+output_jsonl_fpath='{requested}'"],
        "file": ["--config", str(overlay)],
        "flag": ["--output", str(requested)],
    }[source]
    with pytest.raises(SystemExit) as exc:
        validation_cli(*flags)
    assert exc.value.code == 1
    summary_path = run_summary_path_for(requested)
    assert summary_path.parent == requested.parent and summary_path.is_file()
    assert not (tmp_path / "results" / "harbor" / "validation" / "ds" / "rollouts_run_summary.json").exists()
    summary = json.loads(summary_path.read_text())
    assert summary["summary_dir"] == str(requested.parent) and summary["files"]["rollouts"] is None
    assert summary["tasks"] == [] and summary["files"]["run_config"] is None
    assert summary["unvalidated"] == {"a": "no reference solution"}
    # The rows and skipped.json still live in the results folder, reached from the requested folder.
    rows = summary["files"]["rows"]
    assert rows.startswith("../") and (requested.parent / rows).resolve() == Path(
        tmp_path / "results" / "harbor" / "validation" / "ds" / "tasks_harbor_oracle_agent.jsonl"
    )
    assert f"Run summary written to {summary_path}" in capsys.readouterr().out


def test_validate_lists_the_tasks_it_left_out_for_want_of_a_solution_beside_the_ones_it_ran(
    tmp_path, monkeypatch, validation_cli
):
    """Review (round 38, non-blocking): a task without `solution/` is excluded from reference validation rather than
    scored; the summary names it under `unvalidated` with the reason, so the result folder says why it did not run,
    while the tasks that ran keep their entries and nothing is recorded as failed."""
    write_task(tmp_path / "ds" / "a")
    write_task(tmp_path / "ds" / "b")
    (tmp_path / "ds/b/solution/solve.sh").unlink()

    def collect(target, tokens):
        from nemo_gym.config import get_global_config_dict

        write_jsonl(Path(get_global_config_dict().output_jsonl_fpath), [result("a", task_source="ds")])

    monkeypatch.setattr(main_module, "dispatch", collect)
    validation_cli()
    summary = json.loads((tmp_path / "results/harbor/validation/ds/rollouts_run_summary.json").read_text())
    assert [(entry["task_id"], entry["status"]) for entry in summary["tasks"]] == [("a", "result")]
    assert summary["counts"] == {"selected": 1, "results": 1, "masked": 0, "failed": 0, "missing": 0}
    assert summary["unvalidated"] == {"b": "no reference solution"} and summary["skipped"] == {}


def test_eval_run_writes_the_summary_for_a_model_run(tmp_path, monkeypatch, capsys):
    from nemo_gym.rollout_collection import RolloutCollectionHelper

    write_task(tmp_path / "ds" / "a")
    monkeypatch.chdir(tmp_path)
    order = []
    read_rows = RolloutCollectionHelper._preprocess_rows_from_config

    def reading(self, config):
        order.append("rows")
        return read_rows(self, config)

    monkeypatch.setattr(main_module, "dispatch", lambda *args: order.append("dispatch"))
    monkeypatch.setattr(RolloutCollectionHelper, "_preprocess_rows_from_config", reading)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gym",
            "eval",
            "run",
            str(tmp_path / "ds"),
            "--agent",
            "hermes_agent",
            "--model-type",
            "vllm_model",
            "+policy_model_name=test-model",
            "+policy_base_url=http://localhost:1",
            "+policy_api_key=none",
        ],
    )
    main_module.main()
    summary_path = tmp_path / "results" / "harbor" / "ds" / "rollouts_run_summary.json"
    summary = json.loads(summary_path.read_text())
    assert summary["command"] == "gym eval run"
    assert summary["agent"] == {"instance": "hermes_agent", "implementation": "hermes_agent"}
    assert summary["files"]["run_config"] == "run_config_hermes_agent.yaml"
    assert summary["files"]["rows"] == "tasks_hermes_agent.jsonl"
    assert [entry["status"] for entry in summary["tasks"]] == ["missing"]
    # The label the rows were materialized with, so known although the run wrote no rows; the rows' echo needs rows.
    assert summary["task_source"] == "ds" and summary["rows_task_source"] is None
    # Reference validation alone leaves tasks out for want of a solution; a model run validates none.
    assert summary["unvalidated"] == {}
    assert "Run summary written to" in capsys.readouterr().out
    # A fresh run (no cache to check) reads its selected rows once, for the summary, after the collector ran.
    assert order == ["dispatch", "rows"]


def test_eval_run_reads_the_selected_rows_inside_the_summary_guard(tmp_path, monkeypatch, capfd):
    from nemo_gym.rollout_collection import RolloutCollectionHelper

    write_task(tmp_path / "ds" / "a")
    monkeypatch.chdir(tmp_path)
    dispatched = []
    monkeypatch.setattr(main_module, "dispatch", lambda *args: dispatched.append(args))

    def broken(self, config):
        raise ValueError("rows file rejected")

    monkeypatch.setattr(RolloutCollectionHelper, "_preprocess_rows_from_config", broken)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gym",
            "eval",
            "run",
            str(tmp_path / "ds"),
            "--agent",
            "hermes_agent",
            "--model-type",
            "vllm_model",
            "+policy_model_name=test-model",
            "+policy_base_url=http://localhost:1",
            "+policy_api_key=none",
        ],
    )
    main_module.main()  # the run itself is not failed by the summary's read of the rows
    assert len(dispatched) == 1
    captured = capfd.readouterr()
    assert "Could not write the run summary" in captured.err and "rows file rejected" in captured.err
    assert not (tmp_path / "results" / "harbor" / "ds" / "rollouts_run_summary.json").exists()


def test_eval_run_fresh_start_that_dies_before_the_collector_clears_its_outputs_reports_no_earlier_rows(
    tmp_path, monkeypatch
):
    """`gym eval run` (a model run) takes the same fresh-start snapshot as `gym dataset validate` right before it
    dispatches: an earlier run's rollouts, failures and materialized inputs left under its names by a run that dies
    before the collector's cleanup go under `stale_files`; a resumed run's cached rows are its own; a fresh start
    that reaches the collector rewrites the files and owns them."""
    from omegaconf import OmegaConf

    from nemo_gym.config import TASK_SOURCE_KEY_NAME, get_global_config_dict
    from nemo_gym.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper

    write_task(tmp_path / "ds" / "a")
    monkeypatch.chdir(tmp_path)
    output_dir = tmp_path / "results" / "harbor" / "ds"
    rollouts = output_dir / "rollouts.jsonl"
    summary_path = output_dir / "rollouts_run_summary.json"
    argv = [
        "gym",
        "eval",
        "run",
        str(tmp_path / "ds"),
        "--agent",
        "hermes_agent",
        "--model-type",
        "vllm_model",
        "+policy_model_name=test-model",
        "+policy_base_url=http://localhost:1",
        "+policy_api_key=none",
    ]

    def run(*flags):
        monkeypatch.setattr(global_config, "_GLOBAL_CONFIG_DICT", None)
        monkeypatch.setattr(sys, "argv", [*argv, *flags])
        main_module.main()

    def model_row():
        return result(**{"response": {"output": []}, TASK_SOURCE_KEY_NAME: "ds"})

    def collecting(target, tokens):
        # What the collector does once its servers are up: rewrite the materialized inputs with the rows it
        # selected (the run's real digests), recreate the rollouts and failures files, then write the results.
        resolved = OmegaConf.to_container(get_global_config_dict(), resolve=True)
        config = RolloutCollectionConfig.model_validate(
            {**resolved, "input_jsonl_fpath": str(output_dir / "tasks_hermes_agent.jsonl")}
        )
        write_jsonl(materialized_path_for(rollouts), RolloutCollectionHelper()._preprocess_rows_from_config(config))
        rollouts.unlink(missing_ok=True)
        failures_path_for(rollouts).unlink(missing_ok=True)
        write_jsonl(rollouts, [model_row()])
        write_jsonl(failures_path_for(rollouts), [])

    def failing_dispatch(target, tokens):
        raise RuntimeError("environment server failed to start")

    # An earlier, complete model run, with a retried attempt in its failures file, all a minute old.
    monkeypatch.setattr(main_module, "dispatch", collecting)
    run()
    write_jsonl(failures_path_for(rollouts), [model_row() | {"_ng_attempt_index": 0, "_ng_failure_class": "timeout"}])
    earlier = {
        path: path.read_bytes() for path in (rollouts, failures_path_for(rollouts), materialized_path_for(rollouts))
    }
    a_minute_ago = rollouts.stat().st_mtime - 60
    for path in earlier:
        os.utime(path, (a_minute_ago, a_minute_ago))
    summary = json.loads(summary_path.read_text())
    assert summary["files"]["rollouts"] == "rollouts.jsonl" and summary["counts"]["results"] == 1
    assert summary["stale_files"] == {} and summary["rows_task_source"] == "ds"

    # Resumed: the servers fail to start, and the cached rows are still this run's.
    monkeypatch.setattr(main_module, "dispatch", failing_dispatch)
    with pytest.raises(RuntimeError, match="failed to start"):
        run("+resume_from_cache=true")
    resumed = json.loads(summary_path.read_text())
    assert resumed["command"] == "gym eval run" and resumed["agent"]["instance"] == "hermes_agent"
    assert resumed["files"]["rollouts"] == "rollouts.jsonl"
    assert resumed["files"]["failures"] == "rollouts_failures.jsonl"
    assert resumed["files"]["materialized_inputs"] == "rollouts_materialized_inputs.jsonl"
    assert resumed["counts"] == {"selected": 1, "results": 1, "masked": 0, "failed": 0, "missing": 0}
    assert resumed["rows_task_source"] == "ds" and resumed["stale_files"] == {}

    # A fresh start (the default) that dies the same way: the files under its names are the earlier run's.
    with pytest.raises(RuntimeError, match="failed to start"):
        run()
    fresh = json.loads(summary_path.read_text())
    assert fresh["stale_files"] == {
        "rollouts": "rollouts.jsonl",
        "failures": "rollouts_failures.jsonl",
        "materialized_inputs": "rollouts_materialized_inputs.jsonl",
    }
    assert {name: fresh["files"][name] for name in fresh["stale_files"]} == dict.fromkeys(fresh["stale_files"])
    assert fresh["counts"] == {"selected": 1, "results": 0, "masked": 0, "failed": 0, "missing": 1}
    (entry,) = fresh["tasks"]
    assert entry["status"] == "missing" and entry["task_id"] == "a" and entry["input"]["digest"]
    assert fresh["task_source"] == "ds" and fresh["rows_task_source"] is None
    assert fresh["files"]["run_config"] == "run_config_hermes_agent.yaml"
    assert {path: path.read_bytes() for path in earlier} == earlier

    # A fresh start that reaches the collector: it clears and rewrites the files, which are then this run's.
    monkeypatch.setattr(main_module, "dispatch", collecting)
    run()
    rerun = json.loads(summary_path.read_text())
    assert rerun["files"]["rollouts"] == "rollouts.jsonl"
    assert rerun["files"]["failures"] == "rollouts_failures.jsonl" and failures_path_for(rollouts).stat().st_size == 0
    assert rerun["files"]["materialized_inputs"] == "rollouts_materialized_inputs.jsonl"
    assert rerun["counts"] == {"selected": 1, "results": 1, "masked": 0, "failed": 0, "missing": 0}
    assert rerun["task_source"] == "ds" == rerun["rows_task_source"] and rerun["stale_files"] == {}


def test_a_resumed_run_summarizes_the_saved_schedule_it_resumed_not_the_current_repeats(tmp_path, monkeypatch):
    """Review (round 38): run once with `num_repeats=1`, then resume with `resume_from_cache=true` and
    `num_repeats=2`. The compatibility check accepts it (same tasks and digests) and the collector resumes the
    saved one-attempt schedule without dispatching more work, but the summary used the current two-repeat inputs
    and reported a missing attempt. A resumed run's expected attempts and input echo are the saved materialized
    inputs the collector actually resumed."""
    from omegaconf import OmegaConf

    from nemo_gym.config import TASK_SOURCE_KEY_NAME, get_global_config_dict
    from nemo_gym.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper

    write_task(tmp_path / "ds" / "a")
    monkeypatch.chdir(tmp_path)
    output_dir = tmp_path / "results" / "harbor" / "ds"
    rollouts = output_dir / "rollouts.jsonl"
    argv = [
        "gym",
        "eval",
        "run",
        str(tmp_path / "ds"),
        "--agent",
        "hermes_agent",
        "--model-type",
        "vllm_model",
        "+policy_model_name=test-model",
        "+policy_base_url=http://localhost:1",
        "+policy_api_key=none",
    ]

    def run(*flags):
        monkeypatch.setattr(global_config, "_GLOBAL_CONFIG_DICT", None)
        monkeypatch.setattr(sys, "argv", [*argv, *flags])
        main_module.main()

    def collecting(target, tokens):
        # The collector's first run: it saves the schedule it expanded (one repeat) as the materialized inputs and
        # writes one result. The saved input carries a budget the current rows do not, to tell their echoes apart.
        resolved = OmegaConf.to_container(get_global_config_dict(), resolve=True)
        config = RolloutCollectionConfig.model_validate(
            {**resolved, "input_jsonl_fpath": str(output_dir / "tasks_hermes_agent.jsonl")}
        )
        saved = RolloutCollectionHelper()._preprocess_rows_from_config(config)
        for row in saved:
            row["task_input"]["responses_create_params"]["metadata"]["agent_timeout_sec"] = "7.0"
        write_jsonl(materialized_path_for(rollouts), saved)
        write_jsonl(rollouts, [result(**{"response": {"output": []}, TASK_SOURCE_KEY_NAME: "ds"})])

    monkeypatch.setattr(main_module, "dispatch", collecting)
    run()
    saved = [json.loads(line) for line in materialized_path_for(rollouts).read_text().splitlines()]
    assert len(saved) == 1
    summary = json.loads((output_dir / "rollouts_run_summary.json").read_text())
    assert summary["counts"] == {"selected": 1, "results": 1, "masked": 0, "failed": 0, "missing": 0}

    # Resumed with two repeats: the collector resumes its saved one-attempt schedule and dispatches nothing more.
    monkeypatch.setattr(main_module, "dispatch", lambda *args: None)
    run("+resume_from_cache=true", "+num_repeats=2")
    resumed = json.loads((output_dir / "rollouts_run_summary.json").read_text())
    assert resumed["counts"] == {"selected": 1, "results": 1, "masked": 0, "failed": 0, "missing": 0}
    (entry,) = resumed["tasks"]
    assert (entry["task_id"], entry["rollout_index"], entry["status"]) == ("a", 0, "result")
    # The input echo is the saved row's, not the current rows' (which carry the task's own budget).
    assert entry["input"]["metadata"] == {"agent_timeout_sec": "7.0"}
    assert entry["input"]["digest"] == saved[0]["task_input"]["task_data"]["_ng_digest"]


def test_summary_writer_failure_never_hides_the_run(tmp_path, monkeypatch, capfd, validation_cli):
    from nemo_gym.tasks.harbor import cli as cli_module

    write_task(tmp_path / "ds" / "a")

    def collect(target, tokens):
        from nemo_gym.config import get_global_config_dict

        write_jsonl(Path(get_global_config_dict().output_jsonl_fpath), [result()])

    def broken(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(main_module, "dispatch", collect)
    monkeypatch.setattr(cli_module, "write_run_summary", broken)
    validation_cli()  # the validation itself still passes
    assert "Could not write the run summary" in capfd.readouterr().err
