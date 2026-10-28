# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The data processor's metrics sidecars on the Harbor path are derived files of one run's selection, never a
source of truth: prepare removes the rows' sidecars when it rewrites the rows, and a fresh start removes the
collated split metrics before it dispatches, so a changed selection never fails data preparation with "Found
conflicting aggregate metrics"."""

import json

import nemo_gym.cli.main as main_module
import nemo_gym.config.global_config as global_config
from nemo_gym.data.tasks.harbor.cli import prepare_target
from nemo_gym.data.tasks.harbor.validation_report import _validation_key
from tests.unit_tests.test_harbor_tasks import no_real_registry, write_task  # noqa: F401 (autouse fixture)
from tests.unit_tests.test_harbor_validation import oracle_collection, result, validation_cli  # noqa: F401 (fixtures)


def examples(path):
    return json.loads(path.read_text())["Number of examples"]


def test_prepare_removes_the_rows_metrics_sidecars_it_owns(tmp_path, monkeypatch):
    """The rows are regenerated from the task folder on every command, so the metrics the data processor left
    beside an earlier selection's rows (and the conflict file it wrote on a mismatch) go with the rows."""
    monkeypatch.chdir(tmp_path)
    write_task(tmp_path / "ds" / "a")
    output_dir = tmp_path / "results" / "harbor" / "ds"
    output_dir.mkdir(parents=True)
    metrics = output_dir / "tasks_metrics.json"
    conflict = output_dir / "tasks_metrics_conflict.json"
    metrics.write_text(json.dumps({"Number of examples": 5}))
    conflict.write_text(json.dumps({"Number of examples": 1}))
    # Neighbours by other names are not the rows writer's to remove.
    (output_dir / "rollouts.jsonl").write_text("")
    (output_dir / "other_metrics.json").write_text("{}")

    prepared = prepare_target(str(tmp_path / "ds"))

    assert prepared.rows_path.resolve() == output_dir / "tasks.jsonl" and prepared.rows_path.is_file()
    assert not metrics.exists() and not conflict.exists()
    assert (output_dir / "rollouts.jsonl").is_file() and (output_dir / "other_metrics.json").is_file()


def test_fresh_runs_over_a_changed_selection_never_conflict_on_metrics(
    tmp_path, monkeypatch, validation_cli, oracle_collection
):
    """Real preparation and data processor, twice over one output folder: the selection grows from one task to two
    and shrinks back to one through `--only-tasks`, and every run's sidecars describe its own selection."""
    write_task(tmp_path / "ds" / "a")
    output_dir = tmp_path / "results" / "harbor" / "validation" / "ds"
    rows_metrics = output_dir / "tasks_harbor_oracle_agent_metrics.json"
    collated_metrics = output_dir / "rollouts" / "preprocessed_datasets" / "benchmark_metrics.json"
    conflicts = [path.with_name(f"{path.stem}_conflict.json") for path in (rows_metrics, collated_metrics)]
    # The oracle's result for whichever task was dispatched, so every selection validates.
    oracle_collection.payload = lambda row: result(_validation_key(row)[1], index=row["_ng_task_index"])

    def fresh(*flags):
        monkeypatch.setattr(global_config, "_GLOBAL_CONFIG_DICT", None)
        validation_cli("+disable_aggregation=true", "+resume_from_cache=false", *flags)

    fresh()
    assert examples(rows_metrics) == 1 and examples(collated_metrics) == 1
    assert len(oracle_collection.dispatched) == 1

    write_task(tmp_path / "ds" / "b")
    fresh()
    assert examples(rows_metrics) == 2 and examples(collated_metrics) == 2
    assert not any(path.exists() for path in conflicts)
    assert len(oracle_collection.dispatched) == 3

    fresh("--only-tasks", "a")
    assert examples(rows_metrics) == 1 and examples(collated_metrics) == 1
    assert not any(path.exists() for path in conflicts)
    assert len(oracle_collection.dispatched) == 4


def test_resume_keeps_the_collated_metrics_a_fresh_start_clears_before_dispatch(
    tmp_path, monkeypatch, validation_cli, oracle_collection
):
    """A resumed run's rows are unchanged (the resume guard sees to that), so its collated metrics stay; a run that
    reuses the prepared input (`reuse_existing_data_preparation`) skips the data processor and keeps them too; a
    fresh start that will prepare again clears them right before it dispatches."""
    write_task(tmp_path / "ds" / "a")
    oracle_collection.invoke()
    output_dir = tmp_path / "results" / "harbor" / "validation" / "ds"
    collated = output_dir / "rollouts" / "preprocessed_datasets" / "benchmark_metrics.json"
    conflict = collated.with_name("benchmark_metrics_conflict.json")
    assert collated.is_file() and (collated.with_name("benchmark.jsonl")).is_file()
    conflict.write_text("{}")
    seen = []
    monkeypatch.setattr(main_module, "dispatch", lambda *args: seen.append((collated.exists(), conflict.exists())))

    oracle_collection.invoke()
    assert seen == [(True, True)]

    monkeypatch.setattr(global_config, "_GLOBAL_CONFIG_DICT", None)
    validation_cli("+disable_aggregation=true", "+resume_from_cache=false", "+reuse_existing_data_preparation=true")
    assert seen == [(True, True), (True, True)]

    monkeypatch.setattr(global_config, "_GLOBAL_CONFIG_DICT", None)
    validation_cli("+disable_aggregation=true", "+resume_from_cache=false")
    assert seen == [(True, True), (True, True), (False, False)]
    # Nothing else under the collator's folder was touched.
    assert collated.with_name("benchmark.jsonl").is_file()
