# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The per-run summary of a Harbor run: one JSON file beside the rollouts that says where everything is.

A run of ``gym dataset validate`` or ``gym eval run <task target>`` leaves its evidence in several places:
the rollouts JSONL and its sidecars (failures, materialized inputs, aggregate metrics), the prepared rows with
the data processor's ``_prepare.jsonl`` and ``_metrics.json`` beside them, the collated collector input under
``<rollouts stem>/preprocessed_datasets/``, the run config, the health check's ``quality_summary.json`` and
``rollout_verdicts.jsonl``, and one verifier log folder per verification under the server's ``artifacts_dir``
(with ``test-stdout.txt``, the reward file, the ATIF ``trajectory.json`` and, in separate mode, the collected
``artifacts/``). The rows name the log folders by absolute path, one row at a time.

:func:`write_run_summary` joins them into ``<rollouts stem>_run_summary.json`` (:func:`run_summary_path_for`)
next to the rollouts: the run's identity (target, taskset, agent, dataset source and digests, the dataset label
the rows were materialized with as ``task_source`` and the one the result rows carry), every file of the run
and, per selected task and repeat, its outcome, the paths of its verifier logs, artifacts and trajectory, and
the input the row carried (its digests and the ``responses_create_params`` metadata, per turn for a multi-step
task). Every path is relative to the summary's own folder, so a results folder moved as a whole still resolves,
and a path a row names that is not on disk is listed under ``missing_paths`` rather than silently dropped. The
files that follow a completed collection are listed only when they are this run's: the aggregate metrics when
at least as new as the rollouts file (a run that stops early writes none, and the one left by an earlier run
under the same name is reported as ``null``), and the health check's two reports when the quality summary
records this rollouts file as the one it was computed from (:func:`_health_reports`: the reports' names are not
scoped to the ``--output`` stem, so a folder with several stems' rollouts holds whichever run's completed last,
newer than every one of them).
A fresh start (``resume_from_cache`` off, or nothing to resume) clears the rollouts and failures files and
rewrites the materialized inputs only once its servers are up, so the CLI records what sits under those names
before it dispatches (:func:`output_signatures`), and a file still carrying that signature afterwards was left
by an earlier run the collector never reached: it is ``null`` under ``files``, its rows are not this run's (the
selected tasks are ``missing``), and it is named under ``stale_files`` so the folder still explains itself. A
resumed run's cached rows are its own and are read as before. The two ``_metrics.json`` are never an earlier
run's: prepare removes the one beside the rows when it rewrites them, and a fresh start removes the collated
``<split>_metrics.json`` (with any ``_conflict.json``) before it dispatches, so the data processor rewrites both
for this run's selection instead of refusing it; a resumed run, or one that reuses the prepared input, keeps the
collated one. A run that reuses the prepared input has no rows sidecar of its own (``files.rows_metrics`` is
``null``): prepare removed the earlier one and the processor that would rewrite it did not run. The file is
written whatever the run did: a run
that wrote no rollouts, or a validate that dispatched nothing because no selected task has a ``solution/``,
gets a summary that says so (``files.rollouts`` is ``null`` and every task is ``missing``), beside the rollouts
path the command asked for. The tasks a validate left out for want of a ``solution/`` are not scored and have
no task entry; they are named under ``unvalidated`` with the reason, as the tasks that did not load are under
``skipped``, so an all-unsolvable selection explains why nothing ran.

Nothing here depends on the run having succeeded, and the writer never raises into the run: the CLI logs a
summary it could not write and keeps the run's own outcome.
"""

import json
import os
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nemo_gym.artifacts.run_files import (
    QUALITY_SUMMARY_FILENAME,
    QUALITY_SUMMARY_SOURCES_KEY,
    ROLLOUT_VERDICTS_FILENAME,
)
from nemo_gym.config.global_config import TASK_SOURCE_KEY_NAME
from nemo_gym.data.tasks.harbor.digest import DATASET_DIGEST_KEY, DIGEST_KEY
from nemo_gym.data.tasks.harbor.validation_report import _iter_validation_rows, _validation_key
from nemo_gym.utils.path_utils import (
    aggregate_metrics_path_for,
    aggregation_sources_path_for,
    failures_path_for,
    materialized_path_for,
    run_summary_path_for,
)


if TYPE_CHECKING:
    from nemo_gym.data.tasks.harbor.prepare import AgentSelection, PreparedTaskset

RUN_SUMMARY_SCHEMA_VERSION = 1
# What a verification leaves in its log folder, by the name the summary reports it under.
VERIFIER_LOG_FILES = {
    "test_stdout": "test-stdout.txt",
    "trajectory": "trajectory.json",
    "entrypoint_stdout": "entrypoint-stdout.txt",
}
# Where separate-mode verification stages the collected ``/logs/artifacts`` and ``artifacts`` entries.
ARTIFACTS_SUBDIR = "artifacts"
# Where ``e2e_rollout_collection`` collates the rows it dispatches: ``<split>.jsonl`` and ``<split>_metrics.json``.
PREPROCESSED_SUBDIR = "preprocessed_datasets"
STATUS_RESULT = "result"
STATUS_FAILED = "failed"
STATUS_MISSING = "missing"
# Why reference validation left a selected task out (``unvalidated``): it ships no ``solution/`` to play.
NO_REFERENCE_SOLUTION = "no reference solution"
# What tells one write of a file from another: inode, size and modification time as the filesystem keeps it.
# Equal before and after a run, the file was not touched by it.
FileSignature = tuple[int, int, int]


def _relative(path: Path, base: Path) -> str:
    """``path`` relative to ``base`` (``..`` segments allowed), or absolute when no relative form exists."""
    try:
        return Path(os.path.relpath(path, base)).as_posix()
    except ValueError:  # pragma: no cover - a different drive, which only Windows has
        return str(path)


def _existing(path: Path | None, base: Path) -> str | None:
    """The relative path when it exists on disk, else ``None``."""
    if path is None or not path.exists():
        return None
    return _relative(path, base)


def _written_after(path: Path, rollouts: Path | None, base: Path) -> str | None:
    """The relative path when ``path`` exists and is at least as new as ``rollouts``, else ``None``.

    For the files written once the rollouts are complete: one older than the rollouts file, or present without
    one (``rollouts`` is ``None`` when the run has no rollouts of its own), was left by an earlier run (a run that
    stopped early writes no new one).
    """
    if rollouts is None or not rollouts.is_file() or not path.is_file():
        return None
    if path.stat().st_mtime < rollouts.stat().st_mtime:
        return None
    return _relative(path, base)


def _located(path: Path) -> Path:
    """``path`` with its directories resolved and its own name kept, as the health check locates the files it
    records (a shard that is a link keeps its name)."""
    return path.parent.resolve() / path.name


def _recorded_sources(quality_summary: Path) -> set[Path] | None:
    """The rollouts files the quality summary records as the ones it was computed from (``run.rollouts``, which
    the health check writes relative to the report's folder), located; ``None`` when it records none (a report an
    older check wrote, or one that does not parse)."""
    try:
        with quality_summary.open("rb") as handle:
            recorded = json.load(handle)["run"][QUALITY_SUMMARY_SOURCES_KEY]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not isinstance(recorded, list):
        return None
    return {_located(quality_summary.parent / str(source)) for source in recorded}


def _aggregation_shards(rollouts: Path) -> set[Path]:
    """The shards ``gym eval aggregate`` merged into ``rollouts``, as its ``<stem>_sources.json`` lists them,
    located; empty without that sidecar. The health check of a merged file reads the shards (with their failures
    and inventories) and records them as its sources, so the merged file's run owns such a report."""
    try:
        shards = json.loads(aggregation_sources_path_for(rollouts).read_bytes())["shards"]
    except (OSError, ValueError, KeyError, TypeError):
        return set()
    return {_located(rollouts.parent / str(shard[0])) for shard in shards if isinstance(shard, list) and shard}


def _health_reports(rollouts: Path | None, base: Path) -> dict[str, str | None]:
    """The health check's two reports in the rollouts' folder when they are this run's, else ``None`` each.

    The reports keep one name per folder whatever the ``--output`` stem, so a folder with several stems' rollouts
    holds the reports of whichever run completed last, newer than every rollouts file in it: timestamps alone do
    not tell whose they are (a resumed run that failed before the collector's cleanup would link another stem's).
    The quality summary records the rollouts it was computed from (:func:`_recorded_sources`), so it is this
    run's when that record names this rollouts file (or the shards of a merged one, :func:`_aggregation_shards`)
    and it is at least as new as the rollouts; the verdicts are written with the summary and removed with it
    (the provisional summary of a stopped run writes none), so they follow it. A report without the record is not
    linked; ``rollouts`` is ``None`` when the run has no rollouts of its own.
    """
    quality_summary = base / QUALITY_SUMMARY_FILENAME
    none: dict[str, str | None] = {"quality_summary": None, "rollout_verdicts": None}
    if _written_after(quality_summary, rollouts, base) is None:
        return none
    recorded = _recorded_sources(quality_summary)
    if recorded is None:
        return none
    shards = _aggregation_shards(rollouts)
    if _located(rollouts) not in recorded and not (shards and recorded == shards):
        return none
    return {
        "quality_summary": _relative(quality_summary, base),
        "rollout_verdicts": _existing(base / ROLLOUT_VERDICTS_FILENAME, base),
    }


def _signature(path: Path) -> FileSignature | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_ino, stat.st_size, stat.st_mtime_ns)


def _run_outputs(rollouts: Path) -> dict[str, Path]:
    """The files collection writes under the run's own name, by the key ``files`` lists them under."""
    return {
        "rollouts": rollouts,
        "failures": failures_path_for(rollouts),
        "materialized_inputs": materialized_path_for(rollouts),
    }


def output_signatures(rollouts: Path) -> dict[str, FileSignature | None]:
    """What sits under the run's output names now, taken by a fresh start before it dispatches.

    A fresh start (``resume_from_cache`` off, or nothing to resume) unlinks the rollouts and failures files and
    rewrites the materialized inputs only once its servers are up, so a run that dies before that leaves the
    files an earlier run put under the same names. Handed to :func:`build_run_summary` as ``before_dispatch``,
    these signatures let the summary tell such a file (unchanged since) from one this run wrote. ``None`` marks
    a name with no file; a resumed run passes no signatures, its cached rows are its own.
    """
    rollouts = Path(rollouts).resolve()
    return {name: _signature(path) for name, path in _run_outputs(rollouts).items()}


def _verifier_paths(logs_dir: str | None, base: Path) -> tuple[dict[str, str | None], list[str]]:
    """The relative paths of one verification's log folder and what it holds, plus the ones not on disk.

    ``logs_dir`` is the row's absolute ``verifier_logs_dir``. A row that names none (``test.sh`` never ran)
    reports every path as ``None`` and nothing missing: there was nothing to collect. A row that names a
    folder that is gone lists ``verifier_logs`` as missing; a folder without an optional file (no
    ``artifacts/`` in shared mode, no trajectory when the export was skipped) lists nothing, those are not
    promised.
    """
    paths: dict[str, str | None] = {"verifier_logs": None, ARTIFACTS_SUBDIR: None}
    paths.update(dict.fromkeys(VERIFIER_LOG_FILES))
    if not logs_dir:
        return paths, []
    folder = Path(logs_dir)
    if not folder.is_dir():
        paths["verifier_logs"] = _relative(folder, base)
        return paths, ["verifier_logs"]
    paths["verifier_logs"] = _relative(folder, base)
    paths[ARTIFACTS_SUBDIR] = _existing(folder / ARTIFACTS_SUBDIR, base)
    for key, name in VERIFIER_LOG_FILES.items():
        paths[key] = _existing(folder / name, base)
    return paths, []


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _input_echo(task_input: Any) -> dict[str, Any]:
    """What the materialized row carried for the task: its digests and the request metadata, per turn."""
    task_input = _mapping(task_input)
    task_data = _mapping(task_input.get("task_data"))
    echo: dict[str, Any] = {
        "digest": task_data.get(DIGEST_KEY),
        "dataset_digest": task_data.get(DATASET_DIGEST_KEY),
        "metadata": _mapping(_mapping(task_input.get("responses_create_params")).get("metadata")),
    }
    turns = task_input.get("turns")
    if isinstance(turns, list) and turns:
        echo["turns"] = [
            {
                "name": _mapping(turn).get("name"),
                "metadata": _mapping(_mapping(_mapping(turn).get("responses_create_params")).get("metadata")),
            }
            for turn in turns
        ]
    return echo


def _number(value: Any) -> float | int | None:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _turn_entries(turns: Any, base: Path) -> list[dict[str, Any]]:
    entries = []
    for turn in turns if isinstance(turns, list) else []:
        turn = _mapping(turn)
        paths, missing = _verifier_paths(turn.get("verifier_logs_dir"), base)
        entries.append(
            {
                "name": turn.get("name"),
                "reward": _number(turn.get("reward")),
                "mask_sample": bool(turn.get("mask_sample")),
                "failure_kind": turn.get("failure_kind"),
                "verifier": {
                    "seconds": _number(turn.get("verifier_seconds")),
                    "network": turn.get("verifier_network"),
                    "return_code": turn.get("verifier_return_code"),
                },
                "paths": paths,
                "missing_paths": missing,
            }
        )
    return entries


def _identity(key: tuple[str, str, Any, Any]) -> dict[str, Any]:
    taskset, task_id, task_index, rollout_index = key
    return {"task_id": task_id, "taskset": taskset, "task_index": task_index, "rollout_index": rollout_index}


def _result_entry(key: tuple[str, str, Any, Any], row: Mapping[str, Any], base: Path) -> dict[str, Any]:
    paths, missing = _verifier_paths(row.get("verifier_logs_dir"), base)
    entry: dict[str, Any] = {
        **_identity(key),
        "status": STATUS_RESULT,
        "reward": _number(row.get("reward")),
        "mask_sample": bool(row.get("mask_sample")),
        "failure_kind": row.get("failure_kind"),
        "verifier": {
            "mode": row.get("verifier_mode"),
            "seconds": _number(row.get("verifier_seconds")),
            "network": row.get("verifier_network"),
            "return_code": row.get("verifier_return_code"),
            "rewards": row.get("verifier_rewards") if isinstance(row.get("verifier_rewards"), Mapping) else None,
        },
        "paths": paths,
        "missing_paths": missing,
        "response_metadata": _mapping(_mapping(row.get("response")).get("metadata")),
    }
    if row.get("turns"):
        entry["turns"] = _turn_entries(row.get("turns"), base)
        entry["stopped_after_turn"] = row.get("stopped_after_turn")
    return entry


def _failure_entry(key: tuple[str, str, Any, Any], row: Mapping[str, Any], base: Path) -> dict[str, Any]:
    # A staged failure may still carry the verifier folder of the turn that failed (an entrypoint that died).
    paths, missing = _verifier_paths(row.get("verifier_logs_dir"), base)
    return {
        **_identity(key),
        "status": STATUS_FAILED,
        "reward": None,
        "mask_sample": bool(row.get("mask_sample")),
        "failure_kind": row.get("failure_kind"),
        "failure": {
            "class": row.get("_ng_failure_class"),
            "stage": row.get("_ng_failure_stage"),
            "message": row.get("_ng_failure_message") or row.get("failure_reason"),
            "attempt": row.get("_ng_attempt_index"),
        },
        "paths": paths,
        "missing_paths": missing,
    }


def _missing_entry(key: tuple[str, str, Any, Any]) -> dict[str, Any]:
    return {**_identity(key), "status": STATUS_MISSING, "reward": None, "mask_sample": False, "failure_kind": None}


def _sort_key(key: tuple[str, str, Any, Any]) -> tuple[Any, ...]:
    taskset, task_id, task_index, rollout_index = key
    return (
        task_index if isinstance(task_index, int) else -1,
        rollout_index if isinstance(rollout_index, int) else -1,
        taskset,
        task_id,
    )


def _rows_task_source(sources: Iterable[str]) -> str | None:
    """The ``task_source`` the run's result and failure rows carry, when they agree (``None`` without rows).

    Materialization stamps the dataset label on every row and collection preserves it, so this cross-checks
    ``task_source`` (the label, known before dispatch): a results file another run wrote shows up as a different
    value here.
    """
    sources = set(sources)
    return sources.pop() if len(sources) == 1 else None


def build_run_summary(
    rollouts: Path,
    *,
    command: str,
    target: str,
    prepared: "PreparedTaskset",
    agent: "AgentSelection",
    run_config: Path | None,
    expected: list[dict[str, Any]] | None,
    split: str | None = None,
    before_dispatch: Mapping[str, FileSignature | None] | None = None,
    unvalidated: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """The summary document for ``rollouts`` (see the module docstring); paths are relative to its folder.

    ``expected`` is the collector's selected, expanded input rows (after ``limit`` and repeats), which name
    every task and repeat the run was to produce: for a resumed run, the saved materialized inputs it resumed
    (its schedule is that run's, whatever ``num_repeats`` the current config says). Without it (the run stopped
    before they were known) the selected set is whatever the rollouts and failures files hold. ``split`` is the
    collector's split (``validation`` on this path), which names the collated input under
    ``preprocessed_datasets/``.
    ``before_dispatch`` is :func:`output_signatures` taken before a fresh start dispatched (or before a validate
    that dispatched nothing wrote its summary): an output file unchanged since was left by an earlier run and
    is reported under ``stale_files``, not ``files``, and its rows are not read. ``None`` (a resumed run) reads
    whatever is on disk. ``unvalidated`` names, with the reason, the selected tasks reference validation left out
    without scoring them (a task without a ``solution/``, :data:`NO_REFERENCE_SOLUTION`); they have no row and
    no task entry, so a validate that ran nothing explains itself here. A model run validates none.
    """
    rollouts = Path(rollouts).resolve()
    base = rollouts.parent
    summary_path = run_summary_path_for(rollouts)
    outputs = _run_outputs(rollouts)
    # A fresh start's output file unchanged since before dispatch was left by an earlier run: not this run's.
    stale = {
        name
        for name, path in outputs.items()
        if before_dispatch is not None
        and before_dispatch.get(name) is not None
        and _signature(path) == before_dispatch[name]
    }

    def own(name: str) -> Path | None:
        return None if name in stale else outputs[name]

    # The `task_source` every result and failure row carries, for the cross-check with the declared one.
    sources: list[str] = []

    def entries(name: str, entry: Callable[..., dict[str, Any]]) -> dict[tuple[str, str, Any, Any], dict[str, Any]]:
        """This run's rows of ``name`` as summary entries by key, projected one record at a time: the file of
        a long run (every turn's full ``response``) is never held whole, only the few fields the summary keeps."""
        path = own(name)
        projected: dict[tuple[str, str, Any, Any], dict[str, Any]] = {}
        for row in _iter_validation_rows(path) if path is not None else ():
            key = _validation_key(row)
            projected[key] = entry(key, row, base)
            if row.get(TASK_SOURCE_KEY_NAME) is not None:
                sources.append(str(row[TASK_SOURCE_KEY_NAME]))
        return projected

    results = entries("rollouts", _result_entry)
    failures = entries("failures", _failure_entry)
    selected = (
        {_validation_key(row): row for row in expected}
        if expected is not None
        else dict.fromkeys([*results, *failures])
    )
    # The input each task was materialized with, by task id: the echo of a task that produced no row.
    inputs = {
        _validation_key(row)[1]: _input_echo(row.get("task_input"))
        for row in _iter_validation_rows(Path(prepared.rows_path))
    }
    tasks = []
    for key in sorted(set(selected) | set(results) | set(failures), key=_sort_key):
        entry = results.get(key) or failures.get(key) or _missing_entry(key)
        # The collector's expanded row carries the input it ran; the rows file stands in for a task it never named.
        task_input = (selected.get(key) or {}).get("task_input") if expected is not None else None
        entry["input"] = _input_echo(task_input) if task_input else inputs.get(key[1]) or _input_echo(None)
        tasks.append(entry)
    counts = {
        "selected": len(selected),
        "results": len(results),
        "masked": sum(1 for entry in results.values() if entry["mask_sample"]),
        "failed": sum(1 for entry in tasks if entry["status"] == STATUS_FAILED),
        "missing": sum(1 for entry in tasks if entry["status"] == STATUS_MISSING),
    }
    output_dir = Path(prepared.output_dir).resolve()
    rows_path = Path(prepared.rows_path).resolve()
    # Where `e2e_rollout_collection` collates the input it dispatches (`<rollouts stem>/preprocessed_datasets/`).
    preprocessed = rollouts.with_suffix("") / PREPROCESSED_SUBDIR
    # Written once collection is complete: the aggregate metrics under the run's own name (one older than this
    # run's rollouts, or present without them, is an earlier run's) and the health check's two reports in the
    # rollouts' folder (this run's when the quality summary records this rollouts file, `_health_reports`).
    post_collection = {
        "aggregate_metrics": aggregate_metrics_path_for(rollouts),
        "quality_summary": base / QUALITY_SUMMARY_FILENAME,
        "rollout_verdicts": base / ROLLOUT_VERDICTS_FILENAME,
    }
    files = {
        "rollouts": _existing(own("rollouts"), base),
        "failures": _existing(own("failures"), base),
        "materialized_inputs": _existing(own("materialized_inputs"), base),
        "aggregate_metrics": _written_after(post_collection["aggregate_metrics"], own("rollouts"), base),
        **_health_reports(own("rollouts"), base),
        "rows": _existing(rows_path, base),
        # The data processor's per-declaration copy of the rows and their dataset metrics, beside the rows.
        "rows_prepare": _existing(rows_path.with_name(f"{rows_path.stem}_prepare.jsonl"), base),
        "rows_metrics": _existing(rows_path.with_name(f"{rows_path.stem}_metrics.json"), base),
        "preprocessed_datasets": _existing(preprocessed, base),
        "collector_input": _existing(preprocessed / f"{split}.jsonl" if split else None, base),
        "collector_input_metrics": _existing(preprocessed / f"{split}_metrics.json" if split else None, base),
        "run_config": _existing(Path(run_config).resolve() if run_config is not None else None, base),
        "skipped": _existing(output_dir / "skipped.json", base),
        "images": _existing(output_dir / "images.json", base),
        # Where the run config sends the verifier logs (`artifacts_dir`), the parent of every `verifier_logs`.
        "verifier_root": _existing(output_dir / "verifier", base),
    }
    # The files under this run's names that are on disk but not this run's, so a reader is not left guessing
    # why `files` says null beside a file by that name.
    stale_files = {name: _relative(outputs[name], base) for name in outputs if name in stale}
    stale_files.update(
        {
            name: _relative(path, base)
            for name, path in post_collection.items()
            if files[name] is None and path.is_file()
        }
    )
    return {
        "schema_version": RUN_SUMMARY_SCHEMA_VERSION,
        "command": command,
        "target": target,
        "taskset": prepared.taskset,
        "agent": {"instance": agent.instance_name, "implementation": agent.impl_name},
        "dataset": {
            "folder": str(Path(prepared.folder).resolve()),
            "source": prepared.dataset_source,
            "digest": prepared.dataset_digest,
            "dataset_toml_digest": prepared.dataset_toml_digest,
        },
        # The dataset label materialization stamped on every row (`write_rows`: the prepared label, else the
        # taskset), which collection preserves on the rows it writes.
        TASK_SOURCE_KEY_NAME: prepared.task_source or prepared.taskset,
        f"rows_{TASK_SOURCE_KEY_NAME}": _rows_task_source(sources),
        "summary_dir": str(summary_path.parent),
        "files": files,
        "stale_files": stale_files,
        "counts": counts,
        "tasks": tasks,
        "skipped": dict(prepared.skipped),
        "unvalidated": dict(unvalidated or {}),
    }


def write_run_summary(rollouts: Path, **kwargs: Any) -> Path:
    """Write :func:`build_run_summary`'s document to :func:`run_summary_path_for` ``(rollouts)`` and return it.

    The file is replaced atomically, so a reader never sees a half-written summary; the folder is created
    when the run wrote nothing there.
    """
    rollouts = Path(rollouts).resolve()
    path = run_summary_path_for(rollouts)
    document = build_run_summary(rollouts, **kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".tmp")
    staging.write_text(json.dumps(document, indent=2) + "\n")
    os.replace(staging, path)
    return path
