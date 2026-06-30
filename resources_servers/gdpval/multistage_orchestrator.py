# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run multi-stage adaptive ELO *through* the standard Gym rollout collection.

This is the single supported way to run multi-stage ELO. It drives the
**standard** rollout-collection machinery so a multi-stage run produces the exact
same artifacts a normal ``ng_e2e_collect_rollouts`` run does —
``evaluator_rollouts.jsonl`` plus ``<stem>_aggregate_metrics.json`` carrying
``comparison/eval_elo`` — which nemo-evaluator parses and exports to mlflow. That
makes multi-stage ELO a drop-in mode of the normal flow: enable it with
``++multistage.enabled=true`` (a plain full run is just a single-stage run).

How adaptivity maps onto the single-pass flow:

* Each stage is one pass of the standard rollout collection over an
  adaptively-chosen subset of tasks. The stage's sampled tasks come from the
  task distribution (``task_distribution``); each row is tagged with the stage's
  selected ``reference_ids`` (honored by the GDPVal verifier's per-request
  reference filter) and a ``stage_index``.
* Between stages we fit the stage's anchored Bradley-Terry MLE ELO (the same
  math the server's ``aggregate_metrics`` uses) to pick the next stage's
  references — references whose known ELO is closest to the running estimate.
* A task's deliverable is reference-independent, so it is produced at most once:
  when a ``(task, repeat)`` recurs in a later stage its row is tagged
  ``reuse_cached_deliverable=True`` and the agent judges the cached deliverable
  against that stage's references instead of re-running the policy.
* After the last stage, all stages' rollouts are concatenated and handed to the
  standard ``_call_aggregate_metrics``; the GDPVal ``aggregate_metrics`` is
  stage-aware (it sees the ``stage_index`` tags) and reports the **last** stage's
  ELO as the headline ``comparison/eval_elo`` while exposing every stage's
  estimate as a ``comparison/stage_<k>/*`` extra.

The pure staging logic (task planning, reference selection, ELO fit) is reused
from ``multistage_elo``; this module only adds the wiring to the rollout
collection. The rollout-execution step is injected (``run_rollouts``) so the
orchestration is unit-testable without any servers.
"""

from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import AbstractSet, Any, Awaitable, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import orjson

from nemo_gym.global_config import AGENT_REF_KEY_NAME, ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from resources_servers.gdpval.multistage_elo import (
    PerReferenceTotals,
    StageSpec,
    ensure_distribution,
    fit_stage_elo,
    plan_stage_task_ids,
    pool_per_reference,
    select_references,
)


# A rollout runner: given a list of fully-formed rollout rows, run them and
# return ``(row, result)`` pairs (result == the agent's /run response, i.e. the
# GDPVal verify response). Injected so tests can avoid real servers.
RolloutRunner = Callable[[List[Dict[str, Any]]], Awaitable[List[Tuple[Dict[str, Any], Dict[str, Any]]]]]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class MultiStageRunConfig:
    """Parsed ``multistage`` config block from the e2e rollout-collection config.

    ``stages`` is a list of :class:`StageSpec` (``num_tasks`` + optional
    ``num_models``/``seed``); the remaining fields configure task sampling and
    deliverable reuse for the staged run.
    """

    enabled: bool
    stages: List[StageSpec]
    column: List[str] = field(default_factory=lambda: ["occupation"])
    distribution_path: Optional[str] = None
    dataset_path: Optional[str] = None
    nested_tasks: bool = False
    seed: Optional[int] = None
    # Judge a task's cached deliverable in later stages instead of re-running the
    # policy. Falls back to a fresh rollout when the deliverable is missing.
    reuse_cached_deliverables: bool = True


def parse_multistage_config(raw: Mapping[str, Any]) -> MultiStageRunConfig:
    """Build a :class:`MultiStageRunConfig` from a raw config mapping.

    Accepts stages as a list of mappings (``{num_tasks, num_models?, seed?}``)
    or as a list of ``"num_tasks[:num_models[:seed]]"`` strings (handy for CLI
    overrides). Raises ``ValueError`` on an empty/invalid stage list.
    """
    stages_raw = raw.get("stages") or []
    stages: List[StageSpec] = []
    for entry in stages_raw:
        if isinstance(entry, Mapping):
            num_tasks = int(entry["num_tasks"])
            num_models = entry.get("num_models")
            seed = entry.get("seed")
            stages.append(
                StageSpec(
                    num_tasks=num_tasks,
                    num_models=int(num_models) if num_models is not None else None,
                    seed=int(seed) if seed is not None else None,
                )
            )
        else:
            parts = str(entry).split(":")
            num_tasks = int(parts[0])
            num_models = int(parts[1]) if len(parts) > 1 and parts[1] != "" else None
            seed = int(parts[2]) if len(parts) > 2 and parts[2] != "" else None
            stages.append(StageSpec(num_tasks=num_tasks, num_models=num_models, seed=seed))

    if not stages:
        raise ValueError(
            "multistage.enabled=true but no stages were configured. Set "
            "multistage.stages, e.g. ++multistage.stages='[{num_tasks: 5}, {num_tasks: 88, num_models: 4}]'."
        )

    column = raw.get("column") or raw.get("columns") or ["occupation"]
    if isinstance(column, str):
        column = [column]

    return MultiStageRunConfig(
        enabled=bool(raw.get("enabled", False)),
        stages=stages,
        column=list(column),
        distribution_path=raw.get("distribution_path"),
        dataset_path=raw.get("dataset_path"),
        nested_tasks=bool(raw.get("nested_tasks", False)),
        seed=raw.get("seed"),
        reuse_cached_deliverables=bool(raw.get("reuse_cached_deliverables", True)),
    )


def find_gdpval_reference_elos(global_config_dict: Mapping[str, Any]) -> Dict[str, float]:
    """Extract ``ref_id -> anchor ELO`` from the GDPVal resources server config.

    Scans the global config for any server instance exposing
    ``resources_servers.gdpval.reference_models`` (the layout NEL/Hydra produce)
    and reads each reference's ``elo``. Returns an empty mapping if none is
    found (the caller raises a clearer error then).
    """
    for value in global_config_dict.values():
        if not isinstance(value, Mapping):
            continue
        resources_servers = value.get("resources_servers")
        if not isinstance(resources_servers, Mapping):
            continue
        gdpval_cfg = resources_servers.get("gdpval")
        if not isinstance(gdpval_cfg, Mapping):
            continue
        reference_models = gdpval_cfg.get("reference_models") or {}
        elos: Dict[str, float] = {}
        for ref_id, ref_cfg in reference_models.items():
            if isinstance(ref_cfg, Mapping) and ref_cfg.get("elo") is not None:
                elos[ref_id] = float(ref_cfg["elo"])
        if elos:
            return elos
    return {}


# ---------------------------------------------------------------------------
# Row helpers
# ---------------------------------------------------------------------------


def row_task_id(row: Mapping[str, Any]) -> Optional[str]:
    """Read a row's task id from the top level or ``responses_create_params.metadata``."""
    task_id = row.get("task_id")
    if task_id is None:
        meta = (row.get("responses_create_params") or {}).get("metadata") or {}
        task_id = meta.get("task_id")
    return str(task_id) if task_id is not None else None


def index_rows_by_task(rows: Sequence[Mapping[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group materialized rollout rows by task id (preserving all repeats)."""
    by_task: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        task_id = row_task_id(row)
        if task_id is not None:
            by_task.setdefault(task_id, []).append(dict(row))
    return by_task


def build_stage_rows(
    rows_by_task: Mapping[str, Sequence[Mapping[str, Any]]],
    task_ids: Sequence[str],
    reference_ids: Sequence[str],
    stage_index: int,
    produced: Optional[AbstractSet[Tuple[str, int]]] = None,
) -> List[Dict[str, Any]]:
    """Materialize a stage's rollout rows from the sampled tasks.

    Each row copies a base materialized row for one of ``task_ids`` and adds the
    stage's ``reference_ids`` (the GDPVal verifier judges only against these) and
    ``stage_index``. Task/rollout indices are kept at their original values: the
    same rollout judged in two stages is distinguished by ``stage_index``, and the
    rollout index must match the on-disk deliverable dir (``repeat_<index>/``).

    ``produced`` lists ``(task_id, rollout_index)`` deliverables already created by
    earlier stages; matching rows are tagged ``reuse_cached_deliverable=True`` so
    the agent judges the cached deliverable instead of re-running the policy.
    """
    stage_rows: List[Dict[str, Any]] = []
    for task_id in task_ids:
        for base_row in rows_by_task.get(task_id, []):
            row = deepcopy(dict(base_row))
            row["reference_ids"] = list(reference_ids)
            row["stage_index"] = stage_index
            if produced is not None:
                rollout_index = int(row.get(ROLLOUT_INDEX_KEY_NAME, 0) or 0)
                if (task_id, rollout_index) in produced:
                    row["reuse_cached_deliverable"] = True
            stage_rows.append(row)
    return stage_rows


def tag_results(
    pairs: Sequence[Tuple[Mapping[str, Any], Mapping[str, Any]]],
    stage_index: int,
) -> List[Dict[str, Any]]:
    """Attach rollout identity + ``stage_index`` to each stage result row.

    Mirrors what ``RolloutCollectionHelper.run_from_config`` writes onto each
    result (task/rollout indices, agent ref) so the merged rollouts file and the
    standard ``_call_aggregate_metrics`` see well-formed rows, and stamps
    ``stage_index``/``task_id`` so the stage-aware aggregation can group by stage.
    """
    tagged: List[Dict[str, Any]] = []
    for row, result in pairs:
        out = dict(result)
        out[TASK_INDEX_KEY_NAME] = row[TASK_INDEX_KEY_NAME]
        out[ROLLOUT_INDEX_KEY_NAME] = row[ROLLOUT_INDEX_KEY_NAME]
        out[AGENT_REF_KEY_NAME] = row[AGENT_REF_KEY_NAME]
        out["stage_index"] = stage_index
        if out.get("task_id") is None:
            tid = row_task_id(row)
            if tid is not None:
                out["task_id"] = tid
        tagged.append(out)
    return tagged


# ---------------------------------------------------------------------------
# Core staged loop (server-agnostic; rollout execution injected)
# ---------------------------------------------------------------------------


async def run_multistage_stages(
    ms_config: MultiStageRunConfig,
    reference_elos: Mapping[str, float],
    distribution: Mapping[str, Mapping[str, object]],
    materialized_rows: Sequence[Mapping[str, Any]],
    run_rollouts: RolloutRunner,
    *,
    rng: Optional[random.Random] = None,
    on_event: Optional[Callable[[str, dict], None]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run every stage and return ``(all_result_rows, stage_summaries)``.

    For each stage: select references (closest known ELO to the running
    estimate), build the stage's rollout rows from the sampled tasks, execute
    them via ``run_rollouts``, tag the results, pool the per-reference votes, and
    fit the stage ELO (threaded into the next stage's selection). ``all_result_rows``
    is the concatenation of every stage's tagged results (ready to write as the
    standard rollouts file); ``stage_summaries`` is one dict per stage for logging.
    """
    base_rng = rng or (random.Random(ms_config.seed) if ms_config.seed is not None else random.Random())
    rows_by_task = index_rows_by_task(materialized_rows)

    stage_task_sets = plan_stage_task_ids(
        distribution,
        ms_config.stages,
        rng=base_rng,
        nested=ms_config.nested_tasks,
    )
    total_stages = len(ms_config.stages)

    def _emit(name: str, **data: object) -> None:
        if on_event is not None:
            on_event(name, data)

    _emit("planned", stage_task_counts=[len(s) for s in stage_task_sets], total_stages=total_stages)

    all_results: List[Dict[str, Any]] = []
    stage_summaries: List[Dict[str, Any]] = []
    eval_elo: Optional[float] = None
    # (task_id, rollout_index) deliverables already produced by earlier stages.
    # Later stages reuse these instead of re-running the policy.
    produced: set[Tuple[str, int]] = set()
    for index, stage in enumerate(ms_config.stages):
        reference_ids = select_references(reference_elos, eval_elo, stage.num_models)
        task_ids = stage_task_sets[index]
        stage_rows = build_stage_rows(
            rows_by_task,
            task_ids,
            reference_ids,
            index,
            produced=produced if ms_config.reuse_cached_deliverables else None,
        )
        num_reused = sum(1 for r in stage_rows if r.get("reuse_cached_deliverable"))
        _emit(
            "stage_start",
            index=index,
            total_stages=total_stages,
            reference_ids=list(reference_ids),
            num_tasks=len(task_ids),
            num_rollouts=len(stage_rows),
            num_reused=num_reused,
            prior_elo=eval_elo,
        )

        pairs = await run_rollouts(stage_rows)
        tagged = tag_results(pairs, index)
        all_results.extend(tagged)

        # Record this stage's deliverables so later stages can reuse them.
        for row in stage_rows:
            tid = row_task_id(row)
            if tid is not None:
                produced.add((tid, int(row.get(ROLLOUT_INDEX_KEY_NAME, 0) or 0)))

        per_reference: PerReferenceTotals = pool_per_reference(tagged)
        stage_elo, normalized, num_references = fit_stage_elo(per_reference, reference_elos)
        if stage_elo is not None:
            eval_elo = stage_elo

        _emit(
            "stage_end",
            index=index,
            total_stages=total_stages,
            eval_elo=stage_elo,
            normalized_elo=normalized,
            num_references=num_references,
        )
        stage_summaries.append(
            {
                "stage_index": index,
                "num_tasks": len(task_ids),
                "num_rollouts": len(stage_rows),
                "num_reused": num_reused,
                "reference_ids": list(reference_ids),
                "eval_elo": stage_elo,
                "normalized_elo": normalized,
                "num_references": num_references,
            }
        )

    return all_results, stage_summaries


def write_rollouts(all_results: Sequence[Mapping[str, Any]], output_fpath: str | Path) -> Path:
    """Write the merged stage results to the standard rollouts JSONL, sorted."""
    output_fpath = Path(output_fpath)
    output_fpath.parent.mkdir(parents=True, exist_ok=True)
    # A (task, rollout) recurs once per stage, so stage_index is part of the row
    # identity and the primary sort key.
    ordered = sorted(
        all_results,
        key=lambda r: (r.get("stage_index", 0), r.get(TASK_INDEX_KEY_NAME, 0), r.get(ROLLOUT_INDEX_KEY_NAME, 0)),
    )
    with output_fpath.open("wb") as handle:
        for row in ordered:
            handle.write(orjson.dumps(row) + b"\n")
    return output_fpath


# ---------------------------------------------------------------------------
# Integration entrypoint (wires the standard rollout-collection helper)
# ---------------------------------------------------------------------------


async def run_rollout_collection(
    rollout_collection_config, global_config_dict: Mapping[str, Any]
) -> Optional[Path]:  # pragma: no cover
    """Rollout-collection driver entrypoint (wired via ``rollout_collection_driver``).

    Runs the multi-stage adaptive ELO procedure when ``multistage.enabled=true``;
    otherwise delegates to the standard single-pass collection so rubric and
    non-staged comparison runs behave exactly as they would without a driver.
    """
    if (global_config_dict.get("multistage") or {}).get("enabled"):
        return await run_e2e_multistage(rollout_collection_config, global_config_dict)

    from nemo_gym.rollout_collection import RolloutCollectionHelper

    await RolloutCollectionHelper().run_from_config(rollout_collection_config)
    return None


async def run_e2e_multistage(
    rollout_collection_config, global_config_dict: Mapping[str, Any]
) -> Optional[Path]:  # pragma: no cover
    """Drive a multi-stage ELO run through the standard rollout-collection helper.

    Called by ``ng_e2e_collect_rollouts`` when ``multistage.enabled=true``. Brings
    nothing up itself (the caller's ``RunHelper`` has already started the servers);
    it preprocesses the prepared dataset into materialized rows, samples/judges
    stage-by-stage via the helper's ``run_examples``, writes the merged rollouts,
    and runs the standard stage-aware ``_call_aggregate_metrics``.
    """
    from contextlib import nullcontext

    from nemo_gym.rollout_collection import RolloutCollectionHelper

    ms_config = parse_multistage_config(global_config_dict.get("multistage") or {})

    helper = RolloutCollectionHelper()
    materialized_rows = helper._preprocess_rows_from_config(rollout_collection_config)

    reference_elos = find_gdpval_reference_elos(global_config_dict)
    if not reference_elos:
        raise ValueError(
            "multistage.enabled=true but no GDPVal reference_models with ELOs were found in the config. "
            "Multi-stage ELO requires a comparison-mode GDPVal resources server with reference_models.<id>.elo set."
        )

    input_jsonl_fpath = getattr(rollout_collection_config, "input_jsonl_fpath", None)
    distribution, _ = ensure_distribution(
        ms_config.distribution_path,
        dataset_path=ms_config.dataset_path or input_jsonl_fpath,
        columns=ms_config.column,
    )

    semaphore_size = getattr(rollout_collection_config, "num_samples_in_parallel", None)

    async def run_rollouts(rows: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        semaphore = None
        if semaphore_size:
            from asyncio import Semaphore

            semaphore = Semaphore(semaphore_size)
        results: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        for future in helper.run_examples(rows, semaphore=semaphore or nullcontext()):
            row, result = await future
            results.append((row, result))
        return results

    all_results, stage_summaries = await run_multistage_stages(
        ms_config,
        reference_elos,
        distribution,
        materialized_rows,
        run_rollouts,
        on_event=_log_event,
    )

    output_fpath = Path(rollout_collection_config.output_jsonl_fpath)
    write_rollouts(all_results, output_fpath)

    print("[multistage-elo] computing stage-aware aggregate metrics")
    aggregate_metrics_fpath = await helper._call_aggregate_metrics(all_results, all_results, output_fpath)
    print(
        f"""[multistage-elo] finished multi-stage rollout collection!
Rollouts: {output_fpath}
Aggregate metrics: {aggregate_metrics_fpath}
Stages: {orjson.dumps(stage_summaries, option=orjson.OPT_INDENT_2).decode()}"""
    )
    return aggregate_metrics_fpath


def _log_event(name: str, data: dict) -> None:  # pragma: no cover
    """Human-readable stderr progress for the integration entrypoint."""
    import sys

    if name == "planned":
        print(
            f"[multistage-elo] planned {data['total_stages']} stage(s); tasks per stage: {data['stage_task_counts']}",
            file=sys.stderr,
            flush=True,
        )
    elif name == "stage_start":
        prior = data.get("prior_elo")
        prior_str = f"{prior:.1f}" if isinstance(prior, (int, float)) else "n/a"
        num_reused = data.get("num_reused", 0)
        reused_str = f", {num_reused} reused from cache" if num_reused else ""
        print(
            f"[multistage-elo] stage {data['index'] + 1}/{data['total_stages']}: "
            f"{data['num_tasks']} task(s) ({data['num_rollouts']} rollout(s){reused_str}) vs "
            f"{len(data['reference_ids'])} ref(s) {data['reference_ids']} (prior ELO: {prior_str})",
            file=sys.stderr,
            flush=True,
        )
    elif name == "stage_end":
        elo = data.get("eval_elo")
        elo_str = f"{elo:.1f}" if isinstance(elo, (int, float)) else "unset (no games)"
        print(
            f"[multistage-elo] stage {data['index'] + 1}/{data['total_stages']} done: "
            f"eval ELO = {elo_str} (fit over {data.get('num_references')} ref(s))",
            file=sys.stderr,
            flush=True,
        )
