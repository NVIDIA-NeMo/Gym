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

* Each stage is one pass of the standard rollout collection over the stage's
  sampled ``T`` tasks (``T`` is configurable per stage and defaults to the full
  task distribution). The stage includes a set of reference models (chosen
  adaptively — see below) and assigns **each task a single reference** from that
  set; the default is an independent uniform sample, while partial-completion
  stages use a seeded balanced assignment. The task's row is tagged with that
  one ``reference_ids=[ref]`` (honored by the GDPVal verifier's per-request
  reference filter) and a ``stage_index``.
* Between stages we fit the stage's anchored Bradley-Terry MLE ELO (the same
  math the server's ``aggregate_metrics`` uses) — pooling each reference's
  win/loss/tie counts over the tasks assigned to it — to pick the next stage's
  references — those whose known ELO is closest to the running estimate.
* A task's deliverable is reference-independent, so it is produced at most once:
  when a ``(task, repeat)`` recurs in a later stage its row is tagged
  ``reuse_cached_deliverable=True`` and the agent judges the cached deliverable
  against that stage's freshly-assigned reference instead of re-running the
  policy.
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

import hashlib
import math
import os
import random
import secrets
import tempfile
import time
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import AbstractSet, Any, Awaitable, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import orjson

from nemo_gym.global_config import (
    AGENT_REF_KEY_NAME,
    ATTEMPT_INDEX_KEY_NAME,
    ROLLOUT_INDEX_KEY_NAME,
    TASK_INDEX_KEY_NAME,
)
from nemo_gym.path_utils import failures_path_for
from nemo_gym.rollout_collection import (
    NG_FAILURE_CLASS_KEY,
    NG_NO_PERSIST_KEY,
    DispatchLatencyTracker,
    _get_max_rollout_attempts,
    _is_terminal_failure,
    _migrate_invalid_judge_main_rows,
    _observed_elapsed,
)
from resources_servers.gdpval.multistage_elo import (
    PartialStagePolicy,
    PerReferenceTotals,
    StageSpec,
    assign_task_references,
    ensure_distribution,
    fit_stage_elo,
    plan_stage_task_ids,
    pool_per_reference,
    select_references,
    stage_assignment_rng,
)


# A rollout runner: given a list of fully-formed rollout rows, run them and
# return ``(row, result)`` pairs (result == the agent's /run response, i.e. the
# GDPVal verify response). Injected so tests can avoid real servers.
RolloutRunner = Callable[[List[Dict[str, Any]]], Awaitable[List[Tuple[Dict[str, Any], Dict[str, Any]]]]]
AssignmentRepair = Callable[
    [int, Sequence[str], Mapping[str, str]],
    Tuple[Dict[str, str], Dict[str, Any]],
]


@dataclass
class StageResume:
    """Resume seam injected into :func:`run_multistage_stages`.

    ``plans`` holds each stage's recorded references/tasks; ``outcomes`` presence
    means the stage completed. ``rows_by_stage`` holds persisted success rows
    (feed pooling + aggregate); ``gated_keys`` holds per-stage
    ``(task_index, rollout_index)`` not to re-dispatch (success, terminal, or
    max-attempt). ``on_plan``/``on_outcome``/``on_rows`` persist newly produced
    state (``on_rows`` routes success/failure/kill_shaped like ``run_from_config``).
    ``restart_from(i)`` invokes ``on_restart`` and preserves stage ``i``'s plan
    and evidence while invalidating its outcome and every dependent later stage.
    """

    plans: Mapping[int, dict]
    outcomes: Mapping[int, dict]
    rows_by_stage: Mapping[int, List[Dict[str, Any]]]
    gated_keys: Mapping[int, AbstractSet[Tuple[Any, Any]]]
    on_plan: Callable[[int, dict], None]
    on_outcome: Callable[[int, dict], None]
    on_rows: Callable[[int, List[Dict[str, Any]]], None]
    on_restart: Callable[[int], None] = field(default=lambda _index: None, repr=False)
    # Longest observed failed-attempt duration, keyed by stage then
    # (task_index, rollout_index). This is the stage-aware equivalent of the
    # single-pass dispatcher's resume timing map.
    elapsed_by_stage: Mapping[int, Mapping[Tuple[Any, Any], float]] = field(default_factory=dict)
    # Failed judging attempts whose persisted policy artifact should be reused
    # on retry, keyed by stage then (task_index, rollout_index).
    reuse_cached_keys: Mapping[int, AbstractSet[Tuple[Any, Any]]] = field(default_factory=dict)
    # Number of sidecar attempts already consumed, keyed stage/task/rollout.
    attempts_by_stage: Mapping[int, Mapping[Tuple[Any, Any], int]] = field(default_factory=dict)
    # Latest persisted failure per stage/task/rollout.  An explicit partial-stage
    # policy may accept an already-recorded timeout without dispatching it again.
    latest_failures_by_stage: Mapping[int, Mapping[Tuple[Any, Any], Dict[str, Any]]] = field(default_factory=dict)
    # Latest attempt disposition written by this orchestrator, including
    # drained/no-persist attempts that deliberately never enter the sidecar.
    latest_attempt_dispositions_by_stage: Mapping[int, Mapping[Tuple[Any, Any], Dict[str, Any]]] = field(
        default_factory=dict
    )

    def restart_from(self, index: int) -> None:
        """Durably invalidate a stale outcome and every dependent later stage."""
        self.on_restart(index)
        self.plans = {stage: value for stage, value in self.plans.items() if stage <= index}
        self.outcomes = {stage: value for stage, value in self.outcomes.items() if stage < index}
        for name in (
            "rows_by_stage",
            "gated_keys",
            "elapsed_by_stage",
            "reuse_cached_keys",
            "attempts_by_stage",
            "latest_failures_by_stage",
            "latest_attempt_dispositions_by_stage",
        ):
            values = getattr(self, name)
            setattr(self, name, {stage: value for stage, value in values.items() if stage <= index})


def _is_success_row(row: Mapping[str, Any]) -> bool:
    """A row is a success iff it carries neither a failure class nor no-persist."""
    return (
        row.get(NG_FAILURE_CLASS_KEY) is None
        and not row.get(NG_NO_PERSIST_KEY)
        and not row.get("invalid_judge_response")
    )


def _stage_key(row: Mapping[str, Any]) -> Tuple[Any, Any]:
    return (row.get(TASK_INDEX_KEY_NAME), row.get(ROLLOUT_INDEX_KEY_NAME))


def _fit_eligible_stage_keys(
    stage_rows: Sequence[Mapping[str, Any]], successful_rows: Sequence[Mapping[str, Any]]
) -> set[Tuple[Any, Any]]:
    """Keys whose persisted success contains a usable assigned-reference battle."""
    planned_by_key = {_stage_key(row): row for row in stage_rows}
    successful_by_key = {_stage_key(row): row for row in successful_rows if _stage_key(row) in planned_by_key}
    fit_eligible: set[Tuple[Any, Any]] = set()
    for key, success in successful_by_key.items():
        refs = list(planned_by_key[key].get("reference_ids") or [])
        if not refs:
            continue
        counts = (success.get("per_reference") or {}).get(str(refs[0])) or {}
        if sum(float(counts.get(name, 0) or 0) for name in ("wins", "losses", "ties")) > 0:
            fit_eligible.add(key)
    return fit_eligible


def _recorded_key_set(record: Mapping[str, Any], field_name: str) -> set[Tuple[Any, Any]]:
    keys: set[Tuple[Any, Any]] = set()
    for value in record.get(field_name, []) or []:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            keys.add((value[0], value[1]))
    return keys


def _partial_policy_record(policy: PartialStagePolicy) -> Dict[str, Any]:
    return {
        "min_success_fraction": policy.min_success_fraction,
        "min_per_reference_success_fraction": policy.min_per_reference_success_fraction,
        "min_successful_rows_per_reference": policy.min_successful_rows_per_reference,
        "newly_waivable_failure_classes": sorted(policy.waivable_failure_classes),
    }


def _elo_evidence_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    """Hash exactly the pooled battle evidence that determines a stage ELO."""
    pooled = pool_per_reference(rows)
    normalized = {
        str(reference_id): {
            "wins": int(counts.get("wins", 0) or 0),
            "losses": int(counts.get("losses", 0) or 0),
            "ties": int(counts.get("ties", 0) or 0),
            "reference_elo": counts.get("reference_elo"),
        }
        for reference_id, counts in pooled.items()
    }
    return hashlib.sha256(orjson.dumps(normalized, option=orjson.OPT_SORT_KEYS)).hexdigest()


def _partial_stage_outcome(
    policy: PartialStagePolicy,
    stage_rows: Sequence[Mapping[str, Any]],
    successful_rows: Sequence[Mapping[str, Any]],
    new_results: Sequence[Mapping[str, Any]],
    unresolved_keys: AbstractSet[Tuple[Any, Any]],
    reference_ids: Sequence[str],
    stage_elo: Optional[float],
    num_references: int,
) -> Optional[Dict[str, Any]]:
    """Return a durable partial outcome when every configured gate passes."""
    if (
        stage_elo is None
        or not math.isfinite(stage_elo)
        or not stage_rows
        or not reference_ids
        or num_references != len(reference_ids)
    ):
        return None

    planned_by_key = {_stage_key(row): row for row in stage_rows}
    if len(planned_by_key) != len(stage_rows):
        return None
    successful_by_key = {_stage_key(row): row for row in successful_rows if _stage_key(row) in planned_by_key}

    planned_per_reference: Counter[str] = Counter()
    successful_per_reference: Counter[str] = Counter()
    judged_per_reference: Counter[str] = Counter()
    fit_eligible_keys: set[Tuple[Any, Any]] = set()
    for key, row in planned_by_key.items():
        refs = list(row.get("reference_ids") or [])
        if refs:
            planned_per_reference[str(refs[0])] += 1
        success = successful_by_key.get(key)
        if success is None or not refs:
            continue
        reference_id = str(refs[0])
        successful_per_reference[reference_id] += 1
        counts = (success.get("per_reference") or {}).get(reference_id) or {}
        if sum(float(counts.get(name, 0) or 0) for name in ("wins", "losses", "ties")) > 0:
            judged_per_reference[reference_id] += 1
            fit_eligible_keys.add(key)

    # A main-file row without a usable battle is not calibration evidence.  Do
    # not let it masquerade as a success or become part of the frozen snapshot.
    if set(successful_by_key) - fit_eligible_keys:
        return None

    success_fraction = len(fit_eligible_keys) / len(planned_by_key)
    if success_fraction < policy.min_success_fraction:
        return None

    per_reference_success_fractions: Dict[str, float] = {}
    for reference_id in reference_ids:
        planned = planned_per_reference[reference_id]
        if planned <= 0:
            # Balanced assignment gives a reference zero tasks whenever
            # num_tasks < num_models. A reference with nothing planned has no
            # coverage floor to violate; it contributes no evidence.
            per_reference_success_fractions[reference_id] = 1.0
            continue
        success_fraction_for_reference = judged_per_reference[reference_id] / planned
        per_reference_success_fractions[reference_id] = success_fraction_for_reference
        if (
            judged_per_reference[reference_id] < policy.min_successful_rows_per_reference
            or success_fraction_for_reference < policy.min_per_reference_success_fraction
        ):
            return None

    planned_keys = list(planned_by_key)
    included_keys = [key for key in planned_keys if key in fit_eligible_keys]
    omitted_keys = [key for key in planned_keys if key not in fit_eligible_keys]
    included_rows = [successful_by_key[key] for key in included_keys]
    already_resolved_omitted_keys = set(omitted_keys) - set(unresolved_keys)
    result_by_key = {_stage_key(result): result for result in new_results}
    for key in unresolved_keys:
        result = result_by_key.get(key)
        if (
            result is None
            or result.get(NG_NO_PERSIST_KEY)
            or result.get(NG_FAILURE_CLASS_KEY) not in policy.waivable_failure_classes
        ):
            return None

    per_reference = {
        reference_id: {
            "planned": planned_per_reference[reference_id],
            "successful": successful_per_reference[reference_id],
            "judged": judged_per_reference[reference_id],
            "success_fraction": per_reference_success_fractions[reference_id],
        }
        for reference_id in reference_ids
    }
    return {
        "status": "partial_complete",
        "included_keys": [list(key) for key in included_keys],
        "omitted_keys": [list(key) for key in omitted_keys],
        "accepted_unresolved_keys": [list(key) for key in planned_keys if key in unresolved_keys],
        "already_resolved_omitted_keys": [list(key) for key in planned_keys if key in already_resolved_omitted_keys],
        "evidence_sha256": _elo_evidence_sha256(included_rows),
        "success_fraction": success_fraction,
        "persisted_success_fraction": len(successful_by_key) / len(planned_by_key),
        "per_reference": per_reference,
        "policy": _partial_policy_record(policy),
    }


def _cached_partial_snapshot_is_valid(
    outcome: Mapping[str, Any],
    expected_policy: Optional[PartialStagePolicy],
    stage_rows: Sequence[Mapping[str, Any]],
    persisted_rows: Sequence[Mapping[str, Any]],
    reference_ids: Sequence[str],
    reference_elos: Mapping[str, float],
) -> bool:
    """Revalidate a frozen partial outcome from its included evidence."""
    omitted_keys = _recorded_key_set(outcome, "omitted_keys")
    accepted_unresolved_keys = _recorded_key_set(outcome, "accepted_unresolved_keys")
    already_resolved_omitted_keys = _recorded_key_set(outcome, "already_resolved_omitted_keys")
    if (
        accepted_unresolved_keys & already_resolved_omitted_keys
        or accepted_unresolved_keys | already_resolved_omitted_keys != omitted_keys
    ):
        return False

    policy_record = outcome.get("policy")
    required_policy_fields = {
        "min_success_fraction",
        "min_per_reference_success_fraction",
        "min_successful_rows_per_reference",
        "newly_waivable_failure_classes",
    }
    if (
        expected_policy is None
        or not isinstance(policy_record, Mapping)
        or not required_policy_fields <= set(policy_record)
        or {field: policy_record[field] for field in required_policy_fields} != _partial_policy_record(expected_policy)
    ):
        return False
    # The durable record spells the waiver set `newly_waivable_failure_classes`;
    # the config field is `waivable_failure_classes`. Map it back before parsing.
    replayed_policy_fields = {
        field: policy_record[field] for field in required_policy_fields if field != "newly_waivable_failure_classes"
    }
    replayed_policy_fields["waivable_failure_classes"] = policy_record["newly_waivable_failure_classes"]
    try:
        policy = _parse_partial_stage_policy(replayed_policy_fields)
    except (TypeError, ValueError):
        return False
    if policy is None:
        return False

    included_keys = _recorded_key_set(outcome, "included_keys")
    frozen_rows = [row for row in persisted_rows if _stage_key(row) in included_keys and _is_success_row(row)]
    stage_elo, _, num_references = fit_stage_elo(pool_per_reference(frozen_rows), reference_elos)
    candidate = _partial_stage_outcome(
        policy,
        stage_rows,
        frozen_rows,
        [],
        set(),
        reference_ids,
        stage_elo,
        num_references,
    )
    return bool(
        candidate is not None
        and _recorded_key_set(candidate, "included_keys") == included_keys
        and _recorded_key_set(candidate, "omitted_keys") == omitted_keys
        and candidate.get("evidence_sha256") == outcome.get("evidence_sha256")
    )


def compute_fingerprint(
    multistage_config: MultiStageRunConfig,
    reference_elos: Mapping[str, float],
    distribution: Mapping[str, Mapping[str, object]],
    *,
    materialized_rows: Optional[Sequence[Mapping[str, Any]]] = None,
    rollout_collection_config: Optional[Any] = None,
    resolved_global_config: Optional[Mapping[str, Any]] = None,
) -> str:
    """Stable hash of everything that affects stage planning.

    A mismatch between the current run's fingerprint and a journal's marks the
    journal stale: the plans/outcomes it records were produced under a different
    configuration or task distribution and cannot be safely replayed.
    """

    def jsonable(value: Any) -> Any:
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        if isinstance(value, Mapping):
            return {str(key): jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [jsonable(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        return repr(value)

    component_types = {"responses_api_agents", "responses_api_models", "resources_servers"}

    # Connection endpoints and credentials change across allocations (head-node
    # IPs, rotated keys) without affecting any judgment, and secrets should not
    # be hashed into persisted state at all. Only fields at the top level of a
    # server config are stripped; nested config remains fingerprint-relevant.
    connection_fields = {"host", "port"}

    def is_connection_field(name: str) -> bool:
        # ``endpoint_*`` covers the vLLM proxy's endpoint-file liveness knobs
        # (stale grace, connection retries, poll interval): how a replica is
        # reached, never what it generates.
        return (
            name in connection_fields
            or name.endswith("_api_key")
            or name.endswith("_base_url")
            or name.startswith("endpoint_")
        )

    def jsonable_runtime_block(block: Mapping[str, Any]) -> Dict[str, Any]:
        """Serialize a component block without endpoints or credentials."""
        result: Dict[str, Any] = {}
        for key, value in block.items():
            key = str(key)
            if key not in component_types:
                result[key] = jsonable(value)
                continue

            servers = jsonable(value)
            if not isinstance(servers, Mapping):
                result[key] = servers
                continue

            result[key] = {
                str(server_name): (
                    {
                        str(field): field_value
                        for field, field_value in server_config.items()
                        if not is_connection_field(str(field))
                    }
                    if isinstance(server_config, Mapping)
                    else server_config
                )
                for server_name, server_config in servers.items()
            }
        return result

    payload = {
        "stages": [(s.num_tasks, s.num_models, s.seed) for s in multistage_config.stages],
        "seed": multistage_config.seed,
        "nested_tasks": multistage_config.nested_tasks,
        "reuse_cached_deliverables": multistage_config.reuse_cached_deliverables,
        "column": list(multistage_config.column),
        "reference_elos": {k: reference_elos[k] for k in sorted(reference_elos)},
        "distribution": {
            grp: {
                "task_ids": sorted((distribution[grp] or {}).get("task_ids", []) or []),
                "percentage": (distribution[grp] or {}).get("percentage"),
            }
            for grp in sorted(distribution)
        },
    }
    # Keep partial completion out of the rollout-cache fingerprint so an
    # interrupted strict run can opt in without discarding successful rows. Its
    # recorded plan remains authoritative; only a fresh policy-enabled plan uses
    # balanced assignment. A persisted partial outcome records the exact policy
    # and is revalidated on every resume.
    if materialized_rows is not None:
        materialized_hasher = hashlib.sha256()
        for row in materialized_rows:
            materialized_hasher.update(orjson.dumps(row, option=orjson.OPT_SORT_KEYS))
            materialized_hasher.update(b"\n")
        payload["materialized_rows"] = {
            "count": len(materialized_rows),
            "sha256": materialized_hasher.hexdigest(),
        }
    if rollout_collection_config is not None:
        # ``input_jsonl_fpath`` is deliberately absent: the materialized rows
        # above hash the input by content, and the CLI's generated preprocessed
        # path moves between releases without a byte of input changing.
        result_affecting_fields = (
            "agent_name",
            "limit",
            "num_repeats",
            "num_repeats_add_seed",
            "responses_create_params",
            "prompt_config",
            "skills",
        )

        def config_value(name: str) -> Any:
            if isinstance(rollout_collection_config, Mapping):
                return rollout_collection_config.get(name)
            return getattr(rollout_collection_config, name, None)

        payload["rollout_collection_config"] = {name: jsonable(config_value(name)) for name in result_affecting_fields}
    if resolved_global_config is not None:
        payload["runtime_components"] = {
            str(name): jsonable_runtime_block(block)
            for name, block in resolved_global_config.items()
            if isinstance(block, Mapping) and component_types.intersection(block)
        }
        # Transport-assignment repair rewrites which reference each task is
        # judged against, so its settings must invalidate journals like any
        # other planner input. Absent config hashes exactly as before.
        transport_repair = (resolved_global_config.get("multistage") or {}).get("transport_assignment_repair")
        if transport_repair is not None:
            payload["transport_assignment_repair"] = jsonable(transport_repair)
    encoded = orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)
    return hashlib.sha256(encoded).hexdigest()


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


# Failure classes a partial-calibration policy may newly waive. A waived row is
# omitted from the fit and still counts against every coverage floor, so this is
# a bound on which *causes* are acceptable, not on how much may be missing.
# `skipped` is excluded: an unusable sample is not evidence of anything.
_WAIVABLE_FAILURE_CLASSES = frozenset({"timeout_exceeded", "transient"})


def _validate_partial_stage_policy(policy: PartialStagePolicy) -> None:
    """Validate parsed and directly constructed partial policies."""
    for name in ("min_success_fraction", "min_per_reference_success_fraction"):
        value = getattr(policy, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 1:
            raise ValueError(f"partial_completion.{name} must be in (0, 1]")
    minimum_rows = policy.min_successful_rows_per_reference
    if isinstance(minimum_rows, bool) or not isinstance(minimum_rows, int) or minimum_rows <= 0:
        raise ValueError("partial_completion.min_successful_rows_per_reference must be a positive integer")
    waivable = policy.waivable_failure_classes
    if isinstance(waivable, (str, bytes)) or not isinstance(waivable, Sequence) or not waivable:
        raise ValueError("partial_completion.waivable_failure_classes must be a non-empty sequence of strings")
    unknown_classes = sorted(set(map(str, waivable)) - _WAIVABLE_FAILURE_CLASSES)
    if unknown_classes:
        raise ValueError(
            "partial_completion.waivable_failure_classes may only contain "
            f"{sorted(_WAIVABLE_FAILURE_CLASSES)}; got {unknown_classes}"
        )


def _parse_partial_stage_policy(raw: Any) -> Optional[PartialStagePolicy]:
    """Parse and validate an opt-in partial-completion mapping."""
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("stage partial_completion must be a mapping")

    allowed_fields = {
        "min_success_fraction",
        "min_per_reference_success_fraction",
        "min_successful_rows_per_reference",
        "waivable_failure_classes",
    }
    unknown_fields = sorted(set(raw) - allowed_fields)
    if unknown_fields:
        raise ValueError(f"unknown partial_completion field(s): {', '.join(map(str, unknown_fields))}")

    numeric_fields = allowed_fields - {"waivable_failure_classes"}
    if any(isinstance(raw.get(field), bool) for field in numeric_fields if field in raw):
        raise ValueError("partial_completion thresholds must be numeric, not boolean")

    raw_waivable = raw.get("waivable_failure_classes", ("timeout_exceeded",))
    if isinstance(raw_waivable, (str, bytes)) or not isinstance(raw_waivable, Sequence):
        raise ValueError("partial_completion.waivable_failure_classes must be a sequence of strings")
    waivable = tuple(str(value) for value in raw_waivable)

    try:
        raw_minimum_rows = raw.get("min_successful_rows_per_reference", 1)
        minimum_rows = int(raw_minimum_rows)
        policy = PartialStagePolicy(
            min_success_fraction=float(raw.get("min_success_fraction", 1.0)),
            min_per_reference_success_fraction=float(raw.get("min_per_reference_success_fraction", 1.0)),
            min_successful_rows_per_reference=minimum_rows,
            waivable_failure_classes=waivable,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("partial_completion thresholds must be numeric") from exc
    if minimum_rows != raw_minimum_rows:
        raise ValueError("partial_completion.min_successful_rows_per_reference must be a positive integer")
    _validate_partial_stage_policy(policy)
    return policy


def parse_multistage_config(raw: Mapping[str, Any]) -> MultiStageRunConfig:
    """Build a :class:`MultiStageRunConfig` from a raw config mapping.

    Accepts stages as a list of mappings (``{num_tasks?, num_models?, seed?,
    partial_completion?}``)
    or as a list of ``"[num_tasks][:num_models[:seed]]"`` strings (handy for CLI
    overrides). ``num_tasks`` is the per-stage task count and defaults to the full
    task set when omitted (empty leading field in the string form). Raises
    ``ValueError`` on an empty/invalid stage list.
    """
    stages_raw = raw.get("stages") or []
    stages: List[StageSpec] = []
    for entry in stages_raw:
        if isinstance(entry, Mapping):
            num_tasks = entry.get("num_tasks")
            num_models = entry.get("num_models")
            seed = entry.get("seed")
            stages.append(
                StageSpec(
                    num_models=int(num_models) if num_models is not None else None,
                    seed=int(seed) if seed is not None else None,
                    num_tasks=int(num_tasks) if num_tasks is not None else None,
                    partial_completion=_parse_partial_stage_policy(entry.get("partial_completion")),
                )
            )
        else:
            parts = str(entry).split(":")
            num_tasks = int(parts[0]) if parts[0] != "" else None
            num_models = int(parts[1]) if len(parts) > 1 and parts[1] != "" else None
            seed = int(parts[2]) if len(parts) > 2 and parts[2] != "" else None
            stages.append(StageSpec(num_models=num_models, seed=seed, num_tasks=num_tasks))

    if not stages:
        raise ValueError(
            "multistage.enabled=true but no stages were configured. Set "
            "multistage.stages, e.g. ++multistage.stages='[{num_tasks: 110, num_models: 12}, {num_models: 4}]'."
        )
    if stages[-1].partial_completion is not None:
        raise ValueError("partial_completion is allowed only on non-final calibration stages")

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
    task_reference_ids: Mapping[str, str],
    stage_index: int,
    produced: Optional[AbstractSet[Tuple[str, int]]] = None,
) -> List[Dict[str, Any]]:
    """Materialize a stage's rollout rows from the per-task reference assignment.

    ``task_reference_ids`` maps each task id to the **single** reference model it
    is judged against this stage (see ``assign_task_references``). Each row copies
    a base materialized row for that task and sets ``reference_ids=[ref]`` (the
    GDPVal verifier judges only against this one reference) plus ``stage_index``.
    Task/rollout indices are kept at their original values: the same rollout
    judged in two stages is distinguished by ``stage_index``, and the rollout
    index must match the on-disk deliverable dir (``repeat_<index>/``).

    ``produced`` lists ``(task_id, rollout_index)`` deliverables already created by
    earlier stages; matching rows are tagged ``reuse_cached_deliverable=True`` so
    the agent judges the cached deliverable instead of re-running the policy.
    """
    stage_rows: List[Dict[str, Any]] = []
    for task_id, reference_id in task_reference_ids.items():
        for base_row in rows_by_task.get(task_id, []):
            row = deepcopy(dict(base_row))
            row["reference_ids"] = [reference_id]
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
    expected_final_stage_index: Optional[int] = None,
    expected_stage_row_count: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Attach rollout identity + ``stage_index`` to each stage result row.

    Mirrors what ``RolloutCollectionHelper.run_from_config`` writes onto each
    result (task/rollout indices, agent ref) so the merged rollouts file and the
    standard ``_call_aggregate_metrics`` see well-formed rows, and stamps
    ``stage_index``/``task_id`` so the stage-aware aggregation can group by stage.
    Multi-stage callers also provide ``expected_final_stage_index`` and
    ``expected_stage_row_count`` so aggregation can distinguish a complete run
    from a missing or partially drained final stage.
    """
    tagged: List[Dict[str, Any]] = []
    for row, result in pairs:
        out = dict(result)
        out[TASK_INDEX_KEY_NAME] = row[TASK_INDEX_KEY_NAME]
        out[ROLLOUT_INDEX_KEY_NAME] = row[ROLLOUT_INDEX_KEY_NAME]
        out[AGENT_REF_KEY_NAME] = row[AGENT_REF_KEY_NAME]
        if ATTEMPT_INDEX_KEY_NAME in row:
            out[ATTEMPT_INDEX_KEY_NAME] = row[ATTEMPT_INDEX_KEY_NAME]
        if "verify_cache_namespace" in row:
            out["verify_cache_namespace"] = row["verify_cache_namespace"]
        out["stage_index"] = stage_index
        if expected_final_stage_index is not None:
            out["expected_final_stage_index"] = expected_final_stage_index
        if expected_stage_row_count is not None:
            out["expected_stage_row_count"] = expected_stage_row_count
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
    multistage_config: MultiStageRunConfig,
    reference_elos: Mapping[str, float],
    distribution: Mapping[str, Mapping[str, object]],
    materialized_rows: Sequence[Mapping[str, Any]],
    run_rollouts: RolloutRunner,
    *,
    rng: Optional[random.Random] = None,
    on_event: Optional[Callable[[str, dict], None]] = None,
    resume: Optional[StageResume] = None,
    dispatch_longest_first: bool = False,
    assignment_repair: Optional[AssignmentRepair] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run every stage and return ``(all_result_rows, stage_summaries)``.

    For each stage: sample the stage's ``T`` tasks (``T`` defaults to the full
    task set), select the included references (closest known ELO to the running
    estimate), assign each sampled task one reference, build the stage's rollout
    rows, execute them via ``run_rollouts``, tag the results, pool the
    per-reference votes, and fit the stage ELO (threaded into the next stage's
    selection). Fresh partial-completion stages balance assignments across the
    selected references; other fresh stages use independent uniform draws.
    ``all_result_rows`` is the concatenation of every stage's tagged results
    (ready to write as the standard rollouts file); ``stage_summaries`` is one
    dict per stage for logging.

    ``rng`` seeds task sampling (defaults to ``multistage_config.seed``); per-task
    reference assignment is seeded independently per stage via
    ``stage_assignment_rng``.

    When ``resume`` is provided the loop reuses persisted state: complete stages
    are not re-dispatched (their cached rows are re-fitted for ELO threading),
    interrupted stages re-dispatch only the ``(task, rollout)`` rows without a
    persisted success, and recorded plans (including each stage's per-task
    reference assignment) are replayed so selection is identical even when
    ``multistage.seed`` is ``None``. ``resume=None`` preserves the same execution
    semantics without file-backed persistence.
    """
    base_rng = rng or (
        random.Random(multistage_config.seed) if multistage_config.seed is not None else random.Random()
    )
    rows_by_task = index_rows_by_task(materialized_rows)

    # Sample each stage's ``T`` tasks up front (``num_tasks=None`` ⇒ the full task
    # set). Within a stage each task is then assigned one reference (see
    # ``_plan_stage``); stages differ in ``T`` and in which references are included.
    stage_task_sets = plan_stage_task_ids(
        distribution,
        multistage_config.stages,
        rng=base_rng,
        nested=multistage_config.nested_tasks,
    )
    total_stages = len(multistage_config.stages)
    for stage in multistage_config.stages:
        if stage.partial_completion is not None:
            _validate_partial_stage_policy(stage.partial_completion)
    if total_stages and multistage_config.stages[-1].partial_completion is not None:
        raise ValueError("partial_completion is allowed only on non-final calibration stages")

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
    sidecar_produced_by_stage: Dict[int, set[Tuple[str, int]]] = {}
    if resume is not None and resume.reuse_cached_keys:
        task_id_by_key: Dict[Tuple[Any, Any], str] = {}
        for row in materialized_rows:
            task_id = row_task_id(row)
            if task_id is not None:
                task_id_by_key[(row.get(TASK_INDEX_KEY_NAME), row.get(ROLLOUT_INDEX_KEY_NAME))] = task_id
        for stage_index, keys in resume.reuse_cached_keys.items():
            for key in keys:
                task_id = task_id_by_key.get(key)
                if task_id is not None:
                    sidecar_produced_by_stage.setdefault(stage_index, set()).add((task_id, int(key[1] or 0)))

    max_attempts = _get_max_rollout_attempts()
    for index, stage in enumerate(multistage_config.stages):
        # A sidecar reuse flag is emitted only after the policy artifact exists.
        # Treat it as produced from this stage onward even when the judging row
        # is terminal/max-attempt gated and therefore never enters the main file.
        produced.update(sidecar_produced_by_stage.get(index, set()))
        if resume is not None and index in resume.outcomes:
            outcome = resume.outcomes[index]
            plan = resume.plans.get(index, {})
            expected_rows = [row for task_id in plan.get("task_ids", []) for row in rows_by_task.get(str(task_id), [])]
            expected_keys = {(row.get(TASK_INDEX_KEY_NAME), row.get(ROLLOUT_INDEX_KEY_NAME)) for row in expected_rows}
            resolved_keys = set(resume.gated_keys.get(index, set()))
            completion_is_covered = len(expected_keys) == len(expected_rows) and expected_keys <= resolved_keys
            task_reference_ids = {
                str(key): str(value) for key, value in (plan.get("task_reference_ids") or {}).items()
            }
            if not task_reference_ids and plan.get("reference_ids"):
                assignment_rng = stage_assignment_rng(multistage_config.seed, plan.get("seed"), index)
                task_reference_ids = assign_task_references(
                    plan.get("task_ids", []),
                    plan.get("reference_ids", []),
                    rng=assignment_rng,
                )
            planned_stage_rows = build_stage_rows(rows_by_task, task_reference_ids, index)
            if outcome.get("status") == "partial_complete":
                included_keys = _recorded_key_set(outcome, "included_keys")
                omitted_keys = _recorded_key_set(outcome, "omitted_keys")
                persisted_success_keys = {_stage_key(row) for row in resume.rows_by_stage.get(index, [])}
                completion_is_covered = (
                    completion_is_covered
                    and not (included_keys & omitted_keys)
                    and included_keys | omitted_keys == expected_keys
                    and included_keys <= persisted_success_keys
                    and _cached_partial_snapshot_is_valid(
                        outcome,
                        stage.partial_completion,
                        planned_stage_rows,
                        resume.rows_by_stage.get(index, []),
                        plan.get("reference_ids", []),
                        reference_elos,
                    )
                )
                if not completion_is_covered:
                    raise RuntimeError(
                        f"cached partial stage {index} snapshot is invalid; refusing to recompute "
                        "adaptive downstream stages from changed evidence"
                    )
            elif index < total_stages - 1:
                # Older runs used "complete" to mean terminal/max-attempt
                # resolved.  Such an outcome is not safe to reuse for adaptive
                # reference selection unless every planned row has usable
                # persisted battle evidence.
                persisted_rows = resume.rows_by_stage.get(index, [])
                completion_is_covered = completion_is_covered and (
                    _fit_eligible_stage_keys(planned_stage_rows, persisted_rows) == expected_keys
                    and len(planned_stage_rows) == len(expected_rows)
                )
            if completion_is_covered:
                resumed_elo = _resume_complete_stage(
                    index,
                    total_stages,
                    len(expected_rows),
                    resume,
                    reference_elos,
                    produced,
                    all_results,
                    stage_summaries,
                    _emit,
                )
                # Match uninterrupted execution: a completed stage with no
                # usable battles does not erase the last fitted ELO. Later
                # adaptive stages continue from the most recent valid fit.
                if resumed_elo is not None:
                    eval_elo = resumed_elo
                continue
            _emit(
                "stage_completion_stale",
                index=index,
                total_stages=total_stages,
                expected_rows=len(expected_rows),
                resolved_rows=len(expected_keys & resolved_keys),
            )
            resume.restart_from(index)
            sidecar_produced_by_stage = {
                stage_index: keys for stage_index, keys in sidecar_produced_by_stage.items() if stage_index <= index
            }

        reference_ids, task_ids, task_reference_ids, replayed = _plan_stage(
            index,
            stage,
            reference_elos,
            eval_elo,
            stage_task_sets,
            multistage_config,
            resume,
            assignment_repair,
        )

        stage_rows = build_stage_rows(
            rows_by_task,
            task_reference_ids,
            index,
            produced=produced if multistage_config.reuse_cached_deliverables else None,
        )

        cached_rows = resume.rows_by_stage.get(index, []) if resume is not None else []
        # Gated == not re-dispatched: successes on disk plus terminal / max-attempt
        # failures from the sidecar (mirrors ``_load_from_cache``, stage-keyed).
        gated_keys = set(resume.gated_keys.get(index, set())) if resume is not None else set()
        pending_rows = [r for r in stage_rows if (r[TASK_INDEX_KEY_NAME], r[ROLLOUT_INDEX_KEY_NAME]) not in gated_keys]
        if resume is not None:
            reuse_cached_keys = resume.reuse_cached_keys.get(index, set())
            attempts_by_key = resume.attempts_by_stage.get(index, {})
            for row in pending_rows:
                key = (row[TASK_INDEX_KEY_NAME], row[ROLLOUT_INDEX_KEY_NAME])
                if key in reuse_cached_keys:
                    row["reuse_cached_deliverable"] = True
                prior_attempts = attempts_by_key.get(key, 0)
                if prior_attempts > 0:
                    row[ATTEMPT_INDEX_KEY_NAME] = prior_attempts

        if dispatch_longest_first and resume is not None:
            elapsed_by_key = resume.elapsed_by_stage.get(index, {})
            if elapsed_by_key:
                known = [
                    row
                    for row in pending_rows
                    if (row[TASK_INDEX_KEY_NAME], row[ROLLOUT_INDEX_KEY_NAME]) in elapsed_by_key
                ]
                unknown = [
                    row
                    for row in pending_rows
                    if (row[TASK_INDEX_KEY_NAME], row[ROLLOUT_INDEX_KEY_NAME]) not in elapsed_by_key
                ]
                known.sort(
                    key=lambda row: elapsed_by_key[(row[TASK_INDEX_KEY_NAME], row[ROLLOUT_INDEX_KEY_NAME])],
                    reverse=True,
                )
                pending_rows = known + unknown

        # A run may opt into partial calibration after an allocation ended with
        # persisted timeouts.  Evaluate that frozen evidence before dispatch so
        # the already-timed-out rows are not forced through another long attempt.
        pre_dispatch_partial_outcome: Optional[Dict[str, Any]] = None
        if resume is not None and index < total_stages - 1 and stage.partial_completion is not None and pending_rows:
            pending_keys_for_policy = {(row[TASK_INDEX_KEY_NAME], row[ROLLOUT_INDEX_KEY_NAME]) for row in pending_rows}
            latest_failures = resume.latest_failures_by_stage.get(index, {})
            latest_dispositions = resume.latest_attempt_dispositions_by_stage.get(index, {})
            # Persisted failures are fsynced before their journal disposition,
            # while no-persist tombstones are journaled first. A tombstone must
            # therefore override an older sidecar; otherwise the sidecar is at
            # least as new as the journal. If a later persisted attempt crashed
            # before journaling, preferring its sidecar fails closed correctly.
            latest_attempts = {
                key: latest_dispositions[key]
                if latest_dispositions.get(key, {}).get(NG_NO_PERSIST_KEY)
                else latest_failures[key]
                for key in pending_keys_for_policy
                if latest_dispositions.get(key, {}).get(NG_NO_PERSIST_KEY) or key in latest_failures
            }
            persisted_failures = [latest_attempts[key] for key in pending_keys_for_policy if key in latest_attempts]
            cached_elo, _, cached_num_references = fit_stage_elo(pool_per_reference(cached_rows), reference_elos)
            candidate = _partial_stage_outcome(
                stage.partial_completion,
                stage_rows,
                cached_rows,
                persisted_failures,
                pending_keys_for_policy,
                reference_ids,
                cached_elo,
                cached_num_references,
            )
            if candidate is not None:
                pre_dispatch_partial_outcome = {"stage_index": index, **candidate}

        num_reused = sum(1 for r in stage_rows if r.get("reuse_cached_deliverable"))
        _emit(
            "stage_start",
            index=index,
            total_stages=total_stages,
            reference_ids=list(reference_ids),
            num_tasks=len(task_ids),
            num_rollouts=len(stage_rows),
            num_reused=num_reused,
            num_cached=len(cached_rows),
            prior_elo=eval_elo,
            replayed=replayed,
        )

        pairs = [] if pre_dispatch_partial_outcome is not None else await run_rollouts(pending_rows)
        expected_stage_row_count = len(stage_rows)
        new_tagged = tag_results(
            pairs,
            index,
            expected_final_stage_index=total_stages - 1,
            expected_stage_row_count=expected_stage_row_count,
        )
        if resume is not None:
            resume.on_rows(index, new_tagged)
        # A verify-side failure can still carry a complete, reusable policy
        # artifact. Make it available immediately to later stages in this same
        # process; startup-only sidecar loading covers only resumed runs.
        for row in new_tagged:
            if not row.get("reuse_cached_deliverable"):
                continue
            tid = row_task_id(row)
            if tid is not None:
                produced.add((tid, int(row.get(ROLLOUT_INDEX_KEY_NAME, 0) or 0)))
        # Only successful rows feed pooling / aggregate.
        new_successes = [r for r in new_tagged if _is_success_row(r)]
        tagged = [
            dict(
                r,
                expected_final_stage_index=total_stages - 1,
                expected_stage_row_count=expected_stage_row_count,
            )
            for r in cached_rows
        ] + new_successes
        all_results.extend(tagged)

        # Record this stage's deliverables so later stages can reuse them.
        for row in tagged:
            tid = row_task_id(row)
            if tid is not None:
                produced.add((tid, int(row.get(ROLLOUT_INDEX_KEY_NAME, 0) or 0)))

        per_reference: PerReferenceTotals = pool_per_reference(tagged)
        stage_elo, normalized, num_references = fit_stage_elo(per_reference, reference_elos)
        if stage_elo is not None:
            eval_elo = stage_elo

        # Outcomes contain no authoritative ELO; it is re-fit from rows on
        # resume. A retryable failure or drained row leaves the stage open unless
        # an explicit non-final partial-completion policy accepts its evidence.
        returned_keys: set[Tuple[Any, Any]] = set()
        prior_attempts = resume.attempts_by_stage.get(index, {}) if resume is not None else {}
        for result in new_tagged:
            key = (result[TASK_INDEX_KEY_NAME], result[ROLLOUT_INDEX_KEY_NAME])
            resolved = _is_success_row(result) or _is_terminal_failure(result)
            if (
                not resolved
                and result.get(NG_FAILURE_CLASS_KEY) is not None
                and not result.get(NG_NO_PERSIST_KEY)
                and prior_attempts.get(key, 0) + 1 >= max_attempts
            ):
                resolved = True
            if resolved:
                returned_keys.add(key)
        pending_keys = {(r[TASK_INDEX_KEY_NAME], r[ROLLOUT_INDEX_KEY_NAME]) for r in pending_rows}
        unresolved_keys = pending_keys - returned_keys
        stage_complete = not unresolved_keys
        partial_outcome = pre_dispatch_partial_outcome
        coverage_rejected = False
        successful_keys = {_stage_key(row) for row in tagged}
        planned_keys = {_stage_key(row) for row in stage_rows}
        missing_success_keys = planned_keys - successful_keys
        if index < total_stages - 1:
            if stage.partial_completion is not None:
                coverage_outcome = partial_outcome or _partial_stage_outcome(
                    stage.partial_completion,
                    stage_rows,
                    tagged,
                    new_tagged,
                    unresolved_keys,
                    reference_ids,
                    stage_elo,
                    num_references,
                )
                if missing_success_keys:
                    partial_outcome = coverage_outcome
                    if partial_outcome is not None:
                        if "stage_index" not in partial_outcome:
                            partial_outcome = {"stage_index": index, **partial_outcome}
                        stage_complete = True
                    else:
                        stage_complete = False
                        coverage_rejected = not unresolved_keys
                elif coverage_outcome is None:
                    # Even a persisted "success" must contain usable battle
                    # evidence for every configured coverage gate.
                    stage_complete = False
                    coverage_rejected = True
            else:
                # Terminal/max-attempt means "do not retry", not "safe adaptive
                # calibration".  Without an explicit partial policy every
                # planned non-final row must contribute usable battle evidence.
                if (
                    stage_elo is None
                    or not math.isfinite(stage_elo)
                    or _fit_eligible_stage_keys(stage_rows, tagged) != planned_keys
                ):
                    stage_complete = False
                    coverage_rejected = not unresolved_keys
        if resume is not None and stage_complete:
            resume.on_outcome(
                index,
                partial_outcome or {"stage_index": index, "status": "complete"},
            )

        _emit(
            "stage_end",
            index=index,
            total_stages=total_stages,
            eval_elo=stage_elo,
            normalized_elo=normalized,
            num_references=num_references,
        )
        summary = {
            "stage_index": index,
            "num_tasks": len(task_ids),
            "num_rollouts": len(stage_rows),
            "num_reused": num_reused,
            "reference_ids": list(reference_ids),
            "eval_elo": stage_elo,
            "normalized_elo": normalized,
            "num_references": num_references,
        }
        if partial_outcome is not None:
            summary.update(
                partial=True,
                success_fraction=partial_outcome["success_fraction"],
                num_successful=len(partial_outcome["included_keys"]),
                num_omitted=len(partial_outcome["omitted_keys"]),
            )
            _emit(
                "stage_partial_complete",
                index=index,
                total_stages=total_stages,
                success_fraction=partial_outcome["success_fraction"],
                num_omitted=len(partial_outcome["omitted_keys"]),
            )
        stage_summaries.append(summary)
        if not stage_complete:
            _emit(
                "stage_incomplete",
                index=index,
                total_stages=total_stages,
                num_pending=len(unresolved_keys),
                num_omitted=len(missing_success_keys),
                coverage_blocked=coverage_rejected,
            )
            # Later reference selection depends on an accepted ELO. Stop before
            # creating downstream plans; retryable rows can resume, while a
            # coverage-only rejection requires an explicit policy/data change.
            break

    return all_results, stage_summaries


def _plan_stage(
    index: int,
    stage: StageSpec,
    reference_elos: Mapping[str, float],
    eval_elo: Optional[float],
    stage_task_sets: Sequence[Sequence[str]],
    multistage_config: MultiStageRunConfig,
    resume: Optional[StageResume],
    assignment_repair: Optional[AssignmentRepair] = None,
) -> Tuple[List[str], List[str], Dict[str, str], bool]:
    """Return ``(reference_ids, task_ids, task_reference_ids, replayed)`` for a stage.

    ``reference_ids`` is the stage's included reference set (all references, or
    the ``num_models`` closest to the running ELO estimate). ``task_ids`` is the
    stage's sampled ``T`` tasks (the full task set when ``num_tasks`` is unset).
    ``task_reference_ids`` maps each task to the single reference it is judged
    against. Fresh partial-completion stages use a seeded balanced assignment;
    other fresh stages use independent uniform draws.

    ``replayed`` is True when the recorded plan was returned from
    ``resume.plans[index]`` (deterministic replay, even with ``seed=None``), False
    when a fresh plan was computed and persisted via ``resume.on_plan``. When a
    recorded plan predates per-task assignments (no ``task_reference_ids``), the
    assignment is recomputed deterministically from the recorded reference/task
    sets so it matches the rows that were originally dispatched.
    """
    if resume is not None and index in resume.plans:
        recorded = resume.plans[index]
        reference_ids = list(recorded["reference_ids"])
        task_ids = list(recorded["task_ids"])
        task_reference_ids = {str(k): v for k, v in (recorded.get("task_reference_ids") or {}).items()}
        if not task_reference_ids and reference_ids:
            rng = stage_assignment_rng(multistage_config.seed, recorded.get("seed"), index)
            task_reference_ids = assign_task_references(task_ids, reference_ids, rng=rng)
        return reference_ids, task_ids, task_reference_ids, True

    reference_ids = select_references(reference_elos, eval_elo, stage.num_models)
    task_ids = list(stage_task_sets[index])
    rng = stage_assignment_rng(multistage_config.seed, stage.seed, index)
    task_reference_ids = assign_task_references(
        task_ids,
        reference_ids,
        rng=rng,
        balanced=stage.partial_completion is not None,
    )
    repair_receipt: Optional[Dict[str, Any]] = None
    if assignment_repair is not None:
        task_reference_ids, repair_receipt = assignment_repair(
            index,
            reference_ids,
            task_reference_ids,
        )
        if set(task_reference_ids) != set(task_ids):
            raise ValueError("transport assignment repair changed the stage task set")
        if any(reference_id not in reference_ids for reference_id in task_reference_ids.values()):
            raise ValueError("transport assignment repair selected a reference outside the stage")
    if resume is not None:
        plan = {
            "stage_index": index,
            "status": "planned",
            "reference_ids": list(reference_ids),
            "task_ids": list(task_ids),
            "task_reference_ids": task_reference_ids,
            "seed": stage.seed,
            "prior_eval_elo": eval_elo,
        }
        if repair_receipt is not None:
            plan["transport_assignment_repair"] = repair_receipt
        resume.on_plan(index, plan)
    return list(reference_ids), task_ids, task_reference_ids, False


def _resume_complete_stage(
    index: int,
    total_stages: int,
    expected_stage_row_count: int,
    resume: StageResume,
    reference_elos: Mapping[str, float],
    produced: set[Tuple[str, int]],
    all_results: List[Dict[str, Any]],
    stage_summaries: List[Dict[str, Any]],
    emit: Callable[..., None],
) -> Optional[float]:
    """Reuse a completed stage's cached rows without dispatch; return threaded ELO.

    ELO is re-fit from the cached tagged rows (authoritative single source of
    truth) rather than trusting the recorded ``eval_elo`` field, so the value
    threaded to later stages is always consistent with the persisted rows.
    """
    outcome = resume.outcomes.get(index, {})
    partial = outcome.get("status") == "partial_complete"
    included_keys = _recorded_key_set(outcome, "included_keys") if partial else None
    cached_rows = [
        dict(
            r,
            expected_final_stage_index=total_stages - 1,
            expected_stage_row_count=expected_stage_row_count,
        )
        for r in resume.rows_by_stage.get(index, [])
        if included_keys is None or _stage_key(r) in included_keys
    ]
    plan = resume.plans.get(index, {})
    reference_ids = list(plan.get("reference_ids", []))
    task_ids = list(plan.get("task_ids", []))

    for row in cached_rows:
        tid = row_task_id(row)
        if tid is not None:
            produced.add((tid, int(row.get(ROLLOUT_INDEX_KEY_NAME, 0) or 0)))

    all_results.extend(cached_rows)
    per_reference: PerReferenceTotals = pool_per_reference(cached_rows)
    stage_elo, normalized, num_references = fit_stage_elo(per_reference, reference_elos)

    emit(
        "stage_cached",
        index=index,
        total_stages=total_stages,
        eval_elo=stage_elo,
        normalized_elo=normalized,
        num_references=num_references,
        num_rollouts=len(cached_rows),
        partial=partial,
    )
    summary = {
        "stage_index": index,
        "num_tasks": len(task_ids),
        "num_rollouts": len(cached_rows),
        "num_reused": 0,
        "reference_ids": reference_ids,
        "eval_elo": stage_elo,
        "normalized_elo": normalized,
        "num_references": num_references,
        "cached": True,
    }
    if partial:
        summary.update(
            partial=True,
            num_rollouts=expected_stage_row_count,
            num_successful=len(cached_rows),
            success_fraction=outcome.get("success_fraction"),
            num_omitted=len(_recorded_key_set(outcome, "omitted_keys")),
        )
    stage_summaries.append(summary)
    return stage_elo


def write_rollouts(all_results: Sequence[Mapping[str, Any]], output_fpath: str | Path) -> Path:
    """Write the merged stage results to the standard rollouts JSONL, sorted.

    Dedupes by ``(stage_index, task_index, rollout_index)`` (last write wins), so
    concatenating incrementally-persisted stage rows with in-memory ones stays
    idempotent across resume.
    """
    output_fpath = Path(output_fpath)
    output_fpath.parent.mkdir(parents=True, exist_ok=True)
    # stage_index is part of row identity (a (task, rollout) recurs per stage).
    deduped: Dict[Tuple[Any, Any, Any], Mapping[str, Any]] = {}
    for row in all_results:
        key = (row.get("stage_index", 0), row.get(TASK_INDEX_KEY_NAME, 0), row.get(ROLLOUT_INDEX_KEY_NAME, 0))
        deduped[key] = row
    ordered = sorted(
        deduped.values(),
        key=lambda r: (r.get("stage_index", 0), r.get(TASK_INDEX_KEY_NAME, 0), r.get(ROLLOUT_INDEX_KEY_NAME, 0)),
    )
    prior_mode = output_fpath.stat().st_mode & 0o7777 if output_fpath.exists() else None
    temp_path: Optional[Path] = None
    try:
        # NamedTemporaryFile forces 0600, which would silently make a fresh
        # shared rollout file owner-only. Create the staging file with the same
        # 0666+umask semantics as a normal ``open(..., 'wb')`` instead.
        for _ in range(100):
            candidate = output_fpath.parent / f".{output_fpath.name}.merge-{secrets.token_hex(8)}"
            try:
                fd = os.open(candidate, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
            except FileExistsError:
                continue
            temp_path = candidate
            break
        else:  # pragma: no cover - cryptographically-random collisions
            raise FileExistsError(f"could not allocate staging file beside {output_fpath}")

        with os.fdopen(fd, "wb") as handle:
            for row in ordered:
                handle.write(orjson.dumps(row) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        if prior_mode is not None:
            os.chmod(temp_path, prior_mode)
        os.replace(temp_path, output_fpath)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return output_fpath


# ---------------------------------------------------------------------------
# Stage journal + file-backed resume seam
# ---------------------------------------------------------------------------


def journal_path_for(output_fpath: str | Path) -> Path:
    """``<output_stem>_multistage_state.jsonl`` sibling of the rollouts file."""
    output_fpath = Path(output_fpath)
    return output_fpath.with_name(f"{output_fpath.stem}_multistage_state.jsonl")


def aggregate_metrics_path_for(output_fpath: str | Path) -> Path:
    """Standard aggregate-metrics path associated with a rollout JSONL."""
    output_fpath = Path(output_fpath)
    return output_fpath.with_stem(output_fpath.stem + "_aggregate_metrics").with_suffix(".json")


def read_journal(journal_fpath: str | Path) -> Tuple[Dict[int, dict], Dict[int, dict], Optional[str]]:
    """Read the append-only journal; latest record per ``stage_index`` wins.

    Returns ``(plans, outcomes, fingerprint)``. ``fingerprint`` is taken from the
    last record carrying one (all records share it within a run).
    """
    plans: Dict[int, dict] = {}
    outcomes: Dict[int, dict] = {}
    fingerprint: Optional[str] = None
    path = Path(journal_fpath)
    if not path.exists():
        return plans, outcomes, fingerprint
    with path.open("rb") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = orjson.loads(line)
            fingerprint = record.get("fingerprint", fingerprint)
            index = record.get("stage_index")
            if index is None:
                continue
            status = record.get("status")
            if status == "restart_from_stage":
                restart_stage = int(index)
                # The marker is durable before downstream files are pruned. It
                # therefore invalidates old journal state immediately even if
                # the process dies during cleanup; newer records later in the
                # journal can repopulate the recomputed stages.
                plans = {i: plan for i, plan in plans.items() if i <= restart_stage}
                outcomes = {i: outcome for i, outcome in outcomes.items() if i < restart_stage}
            elif status == "planned":
                plans[int(index)] = record
            elif status in {"complete", "partial_complete"}:
                outcomes[int(index)] = record
    return plans, outcomes, fingerprint


def load_latest_attempt_dispositions(
    journal_fpath: str | Path,
) -> Dict[int, Dict[Tuple[Any, Any], Dict[str, Any]]]:
    """Load latest failure/no-persist attempt state recorded in the journal."""
    latest: Dict[int, Dict[Tuple[Any, Any], Dict[str, Any]]] = {}
    path = Path(journal_fpath)
    if not path.exists():
        return latest
    with path.open("rb") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = orjson.loads(line)
            status = record.get("status")
            if status == "restart_from_stage":
                restart_stage = int(record["stage_index"])
                latest = {index: rows for index, rows in latest.items() if index <= restart_stage}
                continue
            if status != "attempt_dispositions":
                continue
            stage_index = int(record["stage_index"])
            stage_latest = latest.setdefault(stage_index, {})
            for disposition in record.get("attempts", []) or []:
                if TASK_INDEX_KEY_NAME not in disposition or ROLLOUT_INDEX_KEY_NAME not in disposition:
                    continue
                key = (disposition[TASK_INDEX_KEY_NAME], disposition[ROLLOUT_INDEX_KEY_NAME])
                stage_latest[key] = disposition
    return latest


def _pending_restart_stage(journal_fpath: str | Path) -> Optional[int]:
    """Return a restart marker whose downstream file cleanup was not acknowledged."""
    pending: Optional[int] = None
    path = Path(journal_fpath)
    if not path.exists():
        return None
    with path.open("rb") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = orjson.loads(line)
            status = record.get("status")
            if status == "restart_from_stage":
                pending = int(record["stage_index"])
            elif status == "restart_cleanup_complete" and pending == int(record["stage_index"]):
                pending = None
    return pending


def _atomic_filter_jsonl(path: Path, keep: Callable[[Mapping[str, Any]], bool]) -> None:
    """Atomically rewrite ``path`` with only records accepted by ``keep``."""
    if not path.exists():
        return
    mode = path.stat().st_mode & 0o7777
    temp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.prune-", delete=False
        ) as out:
            temp_path = Path(out.name)
            with path.open("rb") as source:
                for line in source:
                    stripped = line.strip()
                    if stripped and keep(orjson.loads(stripped)):
                        out.write(stripped + b"\n")
            out.flush()
            os.fsync(out.fileno())
        os.chmod(temp_path, mode)
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _prune_downstream_files(output_fpath: str | Path, restart_stage: int) -> None:
    """Move persisted main/sidecar rows after ``restart_stage`` to a quarantine file.

    Pruned rows are completed provider evidence; a restart decision must never
    destroy them. Each prune appends the dropped rows to a timestamped
    ``.pruned.<ns>`` sibling before atomically rewriting the source.
    """
    output_fpath = Path(output_fpath)

    def keep(record: Mapping[str, Any]) -> bool:
        return int(record.get("stage_index", 0) or 0) <= restart_stage

    stale_suffix = f".pruned.{time.time_ns()}"
    for path in (output_fpath, failures_path_for(output_fpath)):
        if not path.exists():
            continue
        dropped: list[bytes] = []
        with path.open("rb") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped and not keep(orjson.loads(stripped)):
                    dropped.append(stripped)
        if dropped:
            quarantine = path.with_name(path.name + stale_suffix)
            with quarantine.open("wb") as out:
                out.write(b"\n".join(dropped) + b"\n")
                out.flush()
                os.fsync(out.fileno())
        _atomic_filter_jsonl(path, keep)


def load_persisted_rows(output_fpath: str | Path) -> Dict[int, List[Dict[str, Any]]]:
    """Group the main-jsonl success rows by ``stage_index``.

    The main jsonl holds successes only; these feed pooling / aggregate. Within a
    stage, the last row for a ``(task_index, rollout_index)`` key wins.
    """
    path = Path(output_fpath)
    by_stage: Dict[int, Dict[Tuple[Any, Any], Dict[str, Any]]] = {}
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = orjson.loads(line)
            if not _is_success_row(row):
                continue
            index = int(row.get("stage_index", 0) or 0)
            key = (row.get(TASK_INDEX_KEY_NAME), row.get(ROLLOUT_INDEX_KEY_NAME))
            by_stage.setdefault(index, {})[key] = row
    return {index: list(rows.values()) for index, rows in by_stage.items()}


def load_gated_keys(
    output_fpath: str | Path, rows_by_stage: Mapping[int, List[Dict[str, Any]]]
) -> Dict[int, set[Tuple[Any, Any]]]:
    """Per-stage ``(task_index, rollout_index)`` keys that must not be re-dispatched.

    Mirrors ``_load_from_cache`` with the stage dimension added: a stage-row is
    gated if it is a success (main jsonl), a terminal sidecar failure
    (``_ng_failure_terminal``), or has hit ``_get_max_rollout_attempts`` attempts
    in the sidecar. Everything else is re-dispatched.
    """
    gated: Dict[int, set[Tuple[Any, Any]]] = {
        index: {(r.get(TASK_INDEX_KEY_NAME), r.get(ROLLOUT_INDEX_KEY_NAME)) for r in rows}
        for index, rows in rows_by_stage.items()
    }

    failures_fpath = failures_path_for(Path(output_fpath))
    if not failures_fpath.exists():
        return gated

    max_attempts = _get_max_rollout_attempts()
    attempts: Dict[Tuple[int, Any, Any], int] = {}
    terminal: set[Tuple[int, Any, Any]] = set()
    with failures_fpath.open("rb") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            fr = orjson.loads(line)
            if TASK_INDEX_KEY_NAME not in fr or ROLLOUT_INDEX_KEY_NAME not in fr:
                continue
            key = (int(fr.get("stage_index", 0) or 0), fr[TASK_INDEX_KEY_NAME], fr[ROLLOUT_INDEX_KEY_NAME])
            attempts[key] = attempts.get(key, 0) + 1
            if _is_terminal_failure(fr):
                terminal.add(key)

    for key in attempts:
        stage_index, task_index, rollout_index = key
        if key in terminal or attempts[key] >= max_attempts:
            gated.setdefault(stage_index, set()).add((task_index, rollout_index))
    return gated


def load_failure_timings(output_fpath: str | Path) -> Dict[int, Dict[Tuple[Any, Any], float]]:
    """Load longest failed-attempt duration per stage/task/rollout key."""
    elapsed_by_stage: Dict[int, Dict[Tuple[Any, Any], float]] = {}
    failures_fpath = failures_path_for(Path(output_fpath))
    if not failures_fpath.exists():
        return elapsed_by_stage

    with failures_fpath.open("rb") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = orjson.loads(line)
            if TASK_INDEX_KEY_NAME not in row or ROLLOUT_INDEX_KEY_NAME not in row:
                continue
            elapsed = _observed_elapsed(row)
            if elapsed is None:
                continue
            stage_index = int(row.get("stage_index", 0) or 0)
            key = (row[TASK_INDEX_KEY_NAME], row[ROLLOUT_INDEX_KEY_NAME])
            stage_timings = elapsed_by_stage.setdefault(stage_index, {})
            stage_timings[key] = max(stage_timings.get(key, 0.0), elapsed)
    return elapsed_by_stage


def load_failure_attempts(output_fpath: str | Path) -> Dict[int, Dict[Tuple[Any, Any], int]]:
    """Load prior sidecar attempt counts per stage/task/rollout key."""
    attempts_by_stage: Dict[int, Dict[Tuple[Any, Any], int]] = {}
    failures_fpath = failures_path_for(Path(output_fpath))
    if not failures_fpath.exists():
        return attempts_by_stage

    with failures_fpath.open("rb") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = orjson.loads(line)
            if TASK_INDEX_KEY_NAME not in row or ROLLOUT_INDEX_KEY_NAME not in row:
                continue
            stage_index = int(row.get("stage_index", 0) or 0)
            key = (row[TASK_INDEX_KEY_NAME], row[ROLLOUT_INDEX_KEY_NAME])
            stage_attempts = attempts_by_stage.setdefault(stage_index, {})
            stage_attempts[key] = stage_attempts.get(key, 0) + 1
    return attempts_by_stage


def load_latest_failures(output_fpath: str | Path) -> Dict[int, Dict[Tuple[Any, Any], Dict[str, Any]]]:
    """Load the newest persisted sidecar failure for each stage-row key."""
    latest_by_stage: Dict[int, Dict[Tuple[Any, Any], Dict[str, Any]]] = {}
    failures_fpath = failures_path_for(Path(output_fpath))
    if not failures_fpath.exists():
        return latest_by_stage

    with failures_fpath.open("rb") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = orjson.loads(line)
            if TASK_INDEX_KEY_NAME not in row or ROLLOUT_INDEX_KEY_NAME not in row:
                continue
            stage_index = int(row.get("stage_index", 0) or 0)
            key = (row[TASK_INDEX_KEY_NAME], row[ROLLOUT_INDEX_KEY_NAME])
            latest_by_stage.setdefault(stage_index, {})[key] = row
    return latest_by_stage


def load_retryable_failure_stages(
    output_fpath: str | Path,
    gated_keys: Mapping[int, AbstractSet[Tuple[Any, Any]]],
    rows_by_stage: Mapping[int, Sequence[Mapping[str, Any]]],
) -> set[int]:
    """Stages with a persisted failure that still needs another attempt."""
    retryable: set[int] = set()
    failures_fpath = failures_path_for(Path(output_fpath))
    if not failures_fpath.exists():
        return retryable

    completed_keys = {
        stage_index: {(row.get(TASK_INDEX_KEY_NAME), row.get(ROLLOUT_INDEX_KEY_NAME)) for row in rows}
        for stage_index, rows in rows_by_stage.items()
    }

    with failures_fpath.open("rb") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = orjson.loads(line)
            if TASK_INDEX_KEY_NAME not in row or ROLLOUT_INDEX_KEY_NAME not in row:
                continue
            stage_index = int(row.get("stage_index", 0) or 0)
            key = (row[TASK_INDEX_KEY_NAME], row[ROLLOUT_INDEX_KEY_NAME])
            # Historical failures remain in the append-only sidecar after a
            # later attempt succeeds. Such a key is resolved, not retryable.
            if key not in completed_keys.get(stage_index, set()) and key not in gated_keys.get(stage_index, set()):
                retryable.add(stage_index)
    return retryable


def load_reuse_cached_keys(output_fpath: str | Path) -> Dict[int, set[Tuple[Any, Any]]]:
    """Load stage-aware retry keys that should reuse a persisted deliverable."""
    reuse_cached_keys: Dict[int, set[Tuple[Any, Any]]] = {}
    failures_fpath = failures_path_for(Path(output_fpath))
    if not failures_fpath.exists():
        return reuse_cached_keys

    with failures_fpath.open("rb") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = orjson.loads(line)
            if not row.get("reuse_cached_deliverable"):
                continue
            if TASK_INDEX_KEY_NAME not in row or ROLLOUT_INDEX_KEY_NAME not in row:
                continue
            stage_index = int(row.get("stage_index", 0) or 0)
            key = (row[TASK_INDEX_KEY_NAME], row[ROLLOUT_INDEX_KEY_NAME])
            reuse_cached_keys.setdefault(stage_index, set()).add(key)
    return reuse_cached_keys


def append_journal_record(journal_fpath: str | Path, record: Mapping[str, Any], fingerprint: str) -> None:
    """Append a single journal record stamped with the run fingerprint."""
    path = Path(journal_fpath)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = dict(record)
    out["fingerprint"] = fingerprint
    with path.open("ab") as handle:
        handle.write(orjson.dumps(out) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def route_stage_rows(output_fpath: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Route freshly dispatched tagged rows the way ``run_from_config`` does.

    Success -> main rollouts jsonl; non-kill failure (``_ng_failure_class`` set)
    -> failures sidecar (one row per attempt, still carrying ``stage_index``);
    kill_shaped (``_ng_no_persist``) -> written nowhere.
    """
    if not rows:
        return
    output_fpath = Path(output_fpath)
    output_fpath.parent.mkdir(parents=True, exist_ok=True)
    failures_fpath = failures_path_for(output_fpath)
    with output_fpath.open("ab") as main_handle, failures_fpath.open("ab") as fail_handle:
        for row in rows:
            if row.get(NG_NO_PERSIST_KEY):
                continue
            if row.get(NG_FAILURE_CLASS_KEY) is not None:
                fail_handle.write(orjson.dumps(row) + b"\n")
            else:
                main_handle.write(orjson.dumps(row) + b"\n")
        main_handle.flush()
        fail_handle.flush()
        os.fsync(main_handle.fileno())
        os.fsync(fail_handle.fileno())


def _gate_partial_outcome_omissions(
    outcomes: Mapping[int, Mapping[str, Any]],
    gated_keys: Dict[int, set[Tuple[Any, Any]]],
) -> None:
    """Keep journaled partial omissions closed on deterministic resume."""
    for stage_index, outcome in outcomes.items():
        if outcome.get("status") == "partial_complete":
            gated_keys.setdefault(stage_index, set()).update(_recorded_key_set(outcome, "omitted_keys"))


def build_file_resume(output_fpath: str | Path, journal_fpath: str | Path, fingerprint: str) -> StageResume:
    """Build a file-backed :class:`StageResume` from the journal + rollout files."""
    output_fpath = Path(output_fpath)
    journal_fpath = Path(journal_fpath)
    _migrate_invalid_judge_main_rows(output_fpath)

    # Finish an interrupted cleanup before reading any reusable stage state.
    recovered_restart = _pending_restart_stage(journal_fpath)
    if recovered_restart is not None:
        aggregate_metrics_path_for(output_fpath).unlink(missing_ok=True)
        _prune_downstream_files(output_fpath, recovered_restart)
        append_journal_record(
            journal_fpath,
            {"stage_index": recovered_restart, "status": "restart_cleanup_complete"},
            fingerprint,
        )

    plans, outcomes, _ = read_journal(journal_fpath)
    rows_by_stage = load_persisted_rows(output_fpath)
    gated_keys = load_gated_keys(output_fpath, rows_by_stage)
    _gate_partial_outcome_omissions(outcomes, gated_keys)
    elapsed_by_stage = load_failure_timings(output_fpath)
    reuse_cached_keys = load_reuse_cached_keys(output_fpath)
    attempts_by_stage = load_failure_attempts(output_fpath)
    latest_failures_by_stage = load_latest_failures(output_fpath)
    latest_attempt_dispositions_by_stage = load_latest_attempt_dispositions(journal_fpath)
    # Older multistage runs wrote a completion marker even when a retryable
    # sidecar failure remained. The sidecar is authoritative for retry state;
    # discard only those stale outcomes so the recorded plan is replayed.
    retryable_failure_stages = load_retryable_failure_stages(output_fpath, gated_keys, rows_by_stage)
    if retryable_failure_stages and recovered_restart is None:
        restart_stage = min(retryable_failure_stages)
        # Persist the logical invalidation first. If cleanup is interrupted, the
        # next process sees this marker, refuses stale downstream journal state,
        # and idempotently finishes pruning the data files.
        append_journal_record(
            journal_fpath,
            {"stage_index": restart_stage, "status": "restart_from_stage"},
            fingerprint,
        )
        aggregate_metrics_path_for(output_fpath).unlink(missing_ok=True)
        _prune_downstream_files(output_fpath, restart_stage)
        append_journal_record(
            journal_fpath,
            {"stage_index": restart_stage, "status": "restart_cleanup_complete"},
            fingerprint,
        )
        plans, outcomes, _ = read_journal(journal_fpath)
        rows_by_stage = load_persisted_rows(output_fpath)
        gated_keys = load_gated_keys(output_fpath, rows_by_stage)
        _gate_partial_outcome_omissions(outcomes, gated_keys)
        elapsed_by_stage = load_failure_timings(output_fpath)
        reuse_cached_keys = load_reuse_cached_keys(output_fpath)
        attempts_by_stage = load_failure_attempts(output_fpath)
        latest_failures_by_stage = load_latest_failures(output_fpath)
        latest_attempt_dispositions_by_stage = load_latest_attempt_dispositions(journal_fpath)

    def on_plan(index: int, plan: dict) -> None:
        append_journal_record(journal_fpath, plan, fingerprint)

    def on_outcome(index: int, outcome: dict) -> None:
        append_journal_record(journal_fpath, outcome, fingerprint)

    def on_rows(index: int, rows: List[Dict[str, Any]]) -> None:
        dispositions = [
            {
                TASK_INDEX_KEY_NAME: row[TASK_INDEX_KEY_NAME],
                ROLLOUT_INDEX_KEY_NAME: row[ROLLOUT_INDEX_KEY_NAME],
                NG_FAILURE_CLASS_KEY: row.get(NG_FAILURE_CLASS_KEY),
                NG_NO_PERSIST_KEY: bool(row.get(NG_NO_PERSIST_KEY)),
            }
            for row in rows
            if not _is_success_row(row) and TASK_INDEX_KEY_NAME in row and ROLLOUT_INDEX_KEY_NAME in row
        ]
        no_persist_dispositions = [row for row in dispositions if row[NG_NO_PERSIST_KEY]]
        persisted_dispositions = [row for row in dispositions if not row[NG_NO_PERSIST_KEY]]
        # A no-persist attempt is a tombstone for any older timeout sidecar.
        # Journal it before routing so a crash can only cause a conservative
        # retry, never acceptance based on the stale timeout.
        if no_persist_dispositions:
            append_journal_record(
                journal_fpath,
                {
                    "stage_index": index,
                    "status": "attempt_dispositions",
                    "attempts": no_persist_dispositions,
                },
                fingerprint,
            )
        route_stage_rows(output_fpath, rows)
        if persisted_dispositions:
            append_journal_record(
                journal_fpath,
                {
                    "stage_index": index,
                    "status": "attempt_dispositions",
                    "attempts": persisted_dispositions,
                },
                fingerprint,
            )

    def on_restart(index: int) -> None:
        append_journal_record(
            journal_fpath,
            {"stage_index": index, "status": "restart_from_stage"},
            fingerprint,
        )
        aggregate_metrics_path_for(output_fpath).unlink(missing_ok=True)
        _prune_downstream_files(output_fpath, index)
        append_journal_record(
            journal_fpath,
            {"stage_index": index, "status": "restart_cleanup_complete"},
            fingerprint,
        )

    return StageResume(
        plans=plans,
        outcomes=outcomes,
        rows_by_stage=rows_by_stage,
        gated_keys=gated_keys,
        on_plan=on_plan,
        on_outcome=on_outcome,
        on_rows=on_rows,
        on_restart=on_restart,
        elapsed_by_stage=elapsed_by_stage,
        reuse_cached_keys=reuse_cached_keys,
        attempts_by_stage=attempts_by_stage,
        latest_failures_by_stage=latest_failures_by_stage,
        latest_attempt_dispositions_by_stage=latest_attempt_dispositions_by_stage,
    )


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

    from nemo_gym.base_responses_api_model import observability_enabled_from_config
    from nemo_gym.rollout_collection import RolloutCollectionHelper, _attach_ng_perf

    multistage_config = parse_multistage_config(global_config_dict.get("multistage") or {})
    observability_enabled = observability_enabled_from_config(global_config_dict)

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
        multistage_config.distribution_path,
        dataset_path=multistage_config.dataset_path or input_jsonl_fpath,
        columns=multistage_config.column,
    )

    semaphore_size = getattr(rollout_collection_config, "num_samples_in_parallel", None)
    dispatch_budget_s = getattr(rollout_collection_config, "dispatch_budget_s", None)
    drain_margin_s = getattr(rollout_collection_config, "drain_margin_s", None)
    # The multistage driver bypasses run_from_config, so the failure-routing
    # knob must be forwarded explicitly or a failed /run silently reverts to
    # raising and killing every in-flight rollout of the stage.
    route_failures_to_sidecar = bool(getattr(rollout_collection_config, "route_failures_to_sidecar", False))
    if dispatch_budget_s is not None and (semaphore_size is None or semaphore_size <= 0):
        raise ValueError(
            "dispatch_budget_s requires a finite positive num_samples_in_parallel; "
            "unbounded dispatch can POST the entire queue before the budget is re-checked"
        )

    latency_tracker = DispatchLatencyTracker()
    dispatch_started_at = time.monotonic()

    async def run_rollouts(rows: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        semaphore = None
        if semaphore_size:
            from asyncio import Semaphore

            semaphore = Semaphore(semaphore_size)
        results: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        remaining_budget_s = None
        if dispatch_budget_s is not None:
            remaining_budget_s = max(0.0, dispatch_budget_s - (time.monotonic() - dispatch_started_at))
        for future in helper.run_examples(
            rows,
            semaphore=semaphore or nullcontext(),
            dispatch_budget_s=remaining_budget_s,
            drain_margin_s=drain_margin_s,
            latency_tracker=latency_tracker,
            route_failures_to_sidecar=route_failures_to_sidecar,
        ):
            row, result = await future
            _attach_ng_perf(result, observability_enabled=observability_enabled)
            results.append((row, result))
        return results

    output_fpath = Path(rollout_collection_config.output_jsonl_fpath)
    journal_fpath = journal_path_for(output_fpath)
    fingerprint = compute_fingerprint(
        multistage_config,
        reference_elos,
        distribution,
        materialized_rows=materialized_rows,
        rollout_collection_config=rollout_collection_config,
        resolved_global_config=global_config_dict,
    )
    for row in materialized_rows:
        row["verify_cache_namespace"] = fingerprint
    resume = _prepare_resume(rollout_collection_config, output_fpath, journal_fpath, fingerprint)

    assignment_repair = None
    transport_config = (global_config_dict.get("multistage") or {}).get("transport_assignment_repair")
    if isinstance(transport_config, Mapping) and transport_config.get("enabled"):
        from resources_servers.gdpval.transport_assignment import make_assignment_repair

        assignment_repair = make_assignment_repair(global_config_dict, transport_config)

    all_results, stage_summaries = await run_multistage_stages(
        multistage_config,
        reference_elos,
        distribution,
        materialized_rows,
        run_rollouts,
        on_event=_log_event,
        resume=resume,
        dispatch_longest_first=bool(getattr(rollout_collection_config, "dispatch_longest_first", False)),
        assignment_repair=assignment_repair,
    )

    print(latency_tracker.summary())

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


def _prepare_resume(
    rollout_collection_config, output_fpath: Path, journal_fpath: Path, fingerprint: str
) -> StageResume:
    """Build the file-backed :class:`StageResume` for the run.

    Always returns a writing StageResume so even a fresh run persists the journal
    and rows incrementally, giving a later resume state to read. Prior state is
    reused only when ``resume_from_cache`` is set and both the rollouts file and a
    fingerprint-matching journal exist; every other case clears stale files and
    starts fresh (with the reason logged).
    """
    import sys

    resume_requested = bool(getattr(rollout_collection_config, "resume_from_cache", False))
    if not resume_requested:
        reason = "resume_from_cache not set"
    elif not output_fpath.exists() or not journal_fpath.exists():
        reason = f"no prior cache (rollouts exist={output_fpath.exists()}, journal exists={journal_fpath.exists()})"
    elif read_journal(journal_fpath)[2] != fingerprint:
        reason = f"journal STALE (fingerprint {read_journal(journal_fpath)[2]} != {fingerprint})"
    else:
        reason = None

    if reason is not None:
        print(f"[multistage-elo] starting fresh: {reason}", file=sys.stderr, flush=True)
        # Completed judged rollouts are expensive provider evidence. A stale
        # fingerprint must never destroy them: set them aside recoverably.
        stale_suffix = f".stale.{time.time_ns()}"
        for stale in (
            output_fpath,
            failures_path_for(output_fpath),
            journal_fpath,
            aggregate_metrics_path_for(output_fpath),
        ):
            if stale.exists():
                quarantined = stale.with_name(stale.name + stale_suffix)
                stale.replace(quarantined)
                print(f"[multistage-elo] quarantined stale state: {quarantined}", file=sys.stderr, flush=True)
    else:
        print("[multistage-elo] resuming multi-stage run from cache (fingerprint match)", file=sys.stderr, flush=True)
    return build_file_resume(output_fpath, journal_fpath, fingerprint)


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
        plan_str = "replayed from journal" if data.get("replayed") else "planned fresh"
        print(
            f"[multistage-elo] stage {data['index'] + 1}/{data['total_stages']} ({plan_str}): "
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
    elif name == "stage_cached":
        elo = data.get("eval_elo")
        elo_str = f"{elo:.1f}" if isinstance(elo, (int, float)) else "unset (no games)"
        cache_kind = "partial result" if data.get("partial") else "complete result"
        print(
            f"[multistage-elo] stage {data['index'] + 1}/{data['total_stages']} reused {cache_kind} from cache: "
            f"eval ELO = {elo_str} ({data.get('num_rollouts')} cached rollout(s))",
            file=sys.stderr,
            flush=True,
        )
    elif name == "stage_partial_complete":
        print(
            f"[multistage-elo] stage {data['index'] + 1}/{data['total_stages']} accepted partial calibration: "
            f"success coverage {data.get('success_fraction', 0):.1%}, omitted "
            f"{data.get('num_omitted')} rollout(s)",
            file=sys.stderr,
            flush=True,
        )
    elif name == "stage_incomplete":
        detail = (
            f"coverage policy rejected {data.get('num_omitted')} resolved omission(s)"
            if data.get("coverage_blocked")
            else f"{data.get('num_pending')} rollout(s) retryable or undispatched"
        )
        print(
            f"[multistage-elo] stage {data['index'] + 1}/{data['total_stages']} remains incomplete "
            f"({detail}); stopping before downstream stages",
            file=sys.stderr,
            flush=True,
        )
