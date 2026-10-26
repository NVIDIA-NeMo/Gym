# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read Gym's canonical trajectories and health results into compact analysis cohorts."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

import orjson
from pydantic import ValidationError

from nemo_gym.global_config import (
    RESPONSE_KEY_NAME,
    RESPONSES_CREATE_PARAMS_KEY_NAME,
    REWARD_KEY_NAME,
    ROLLOUT_INDEX_KEY_NAME,
    TASK_INDEX_KEY_NAME,
    TASK_SOURCE_KEY_NAME,
    rollout_agent_label,
    rollout_run_key,
)
from nemo_gym.health.types import HealthCheckResult, RolloutDigest
from nemo_gym.rollout_observability import NG_TRAJECTORY_KEY, TrajectoryRecord

from .models import Call, Rollout


def _digest(value: object) -> str:
    return hashlib.sha256(orjson.dumps(value, option=orjson.OPT_SORT_KEYS)).hexdigest()[:12]


def from_gym(row: dict, *, run_id: str, health: RolloutDigest | None = None) -> tuple[Rollout, str | None, set[str]]:
    """Project one persisted trajectory; payloads and harness join details stay out of reports.

    The returned model evidence is resolved across the collection by ``read_records``.
    Invalid trajectory evidence leaves this rollout's measurements unavailable.
    """
    raw_trajectory = row.get(NG_TRAJECTORY_KEY)
    trajectory = None
    gaps = []
    if raw_trajectory is not None:
        try:
            trajectory = TrajectoryRecord.model_validate(raw_trajectory)
        except ValidationError:
            gaps.append("trajectory_invalid")
    calls = []
    models = set()
    if trajectory is not None:
        gaps.extend(g.code for g in trajectory.gaps)
        for index, call in enumerate(trajectory.model_calls):
            if call.response_metadata.model:
                models.add(call.response_metadata.model)
            try:
                calls.append(
                    Call(
                        id=call.model_call_id or f"call:{index}",
                        index=index,
                        output_tokens=call.token_stats.completion_tokens,
                        reasoning_tokens=call.token_stats.reasoning_tokens,
                        input_tokens=call.token_stats.prompt_tokens,
                        duration_s=call.duration_ms / 1000 if call.duration_ms is not None else None,
                    )
                )
            except ValidationError:
                # Preserve the call in denominators, with invalid measurements unavailable.
                gaps.append("model_call_measurements_invalid")
                calls.append(Call(id=call.model_call_id or f"call:{index}", index=index))
    capture = row.get("ng_model_call_capture")
    complete = None
    if isinstance(capture, dict):
        gaps.extend(g.get("code", "unknown") for g in capture.get("gaps", []))
        captured_ids = {c.get("model_call_id") for c in capture.get("calls", [])}
        complete = (
            bool(calls)
            and captured_ids == {c.id for c in calls}
            and not any(
                g.startswith("model_call_capture_") or g in ("trajectory_invalid", "model_call_measurements_invalid")
                for g in gaps
            )
        )
    task = row.get(TASK_INDEX_KEY_NAME)
    repeat = row.get(ROLLOUT_INDEX_KEY_NAME)
    if task is None or repeat is None:
        raise ValueError("Gym analyses require materialized task and rollout indices")
    if health is not None and (str(health.task_index), str(health.rollout_index), health.stage_index) != (
        str(task),
        str(repeat),
        row.get("stage_index"),
    ):
        raise ValueError("Health result does not match the analyzed rollout")
    verdict = health.verdict if health is not None else "unobserved"
    task_id = row.get("task_id")
    taskset = task_id.get("taskset") if isinstance(task_id, dict) else None
    benchmark = str(taskset or row.get(TASK_SOURCE_KEY_NAME) or rollout_run_key(row) or "collection")
    sampling = {
        k: v
        for k, v in (row.get(RESPONSES_CREATE_PARAMS_KEY_NAME) or {}).items()
        if k not in ("input", "metadata", "instructions", "tools")
    }
    primary_model = row.get("model") or (row.get(RESPONSE_KEY_NAME) or {}).get("model") or sampling.get("model")
    reward = row.get(REWARD_KEY_NAME)
    return (
        Rollout(
            run_id=run_id,
            task_id=str(task),
            repeat=repeat,
            stage=row.get("stage_index"),
            agent=rollout_agent_label(row) or "unknown",
            model=primary_model or "unknown",
            benchmark=benchmark,
            config_id=_digest({"sampling": sampling, "environment": rollout_run_key(row)}),
            run_key=rollout_run_key(row),
            success=bool(reward) if reward in (0, 1) else None,
            health="quarantined" if row.get("mask_sample") else ("unknown" if verdict == "unobserved" else verdict),
            capture_complete=complete,
            calls=tuple(calls),
            gaps=tuple(sorted(set(gaps))),
        ),
        primary_model,
        models,
    )


def read_records(
    paths: list[Path], *, run_id: str, health_result: HealthCheckResult | None = None
) -> dict[tuple[str, ...], list[Rollout]]:
    """Read Gym JSONL once and resolve cohorts consistently, including call-less rollouts.

    Health digests use the same file and row order as ``run_health_checks``. Passing
    the result directly avoids stale sidecars and collisions across shard identities.
    """
    contexts = defaultdict(list)
    digests = iter(health_result.rollouts) if health_result is not None else None
    for path in paths:
        with path.open("rb") as stream:
            for line in stream:
                if not line.strip():
                    continue
                health = next(digests, None) if digests is not None else None
                if digests is not None and health is None:
                    raise ValueError("Health result has fewer rows than the analyzed collection")
                record, primary, models = from_gym(orjson.loads(line), run_id=run_id, health=health)
                contexts[(record.benchmark, record.agent, record.config_id, record.run_key)].append(
                    (record, primary, models)
                )
    if digests is not None and next(digests, None) is not None:
        raise ValueError("Health result has more rows than the analyzed collection")
    groups = defaultdict(list)
    for entries in contexts.values():
        primary_models = {primary for _, primary, _ in entries if primary}
        observed_models = set().union(*(models for _, _, models in entries))
        if len(primary_models) == 1:
            fallback = next(iter(primary_models))
        elif primary_models:
            fallback = "unknown"
        else:
            # A multi-model policy is one configuration, including auxiliary calls.
            fallback = ", ".join(sorted(observed_models)) or "unknown"
        for record, primary, _ in entries:
            record = record.model_copy(update={"model": primary or fallback})
            groups[record.cohort].append(record)
    return dict(groups)
