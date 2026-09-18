# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Validated Eval Factory batch manifests and compact, atomic Gym status artifacts."""

import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence

import orjson
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from nemo_gym.config_types import ConfigError, ConfigPathNotFoundError
from nemo_gym.global_config import (
    AGENT_REF_KEY_NAME,
    ATTEMPT_INDEX_KEY_NAME,
    RESPONSES_CREATE_PARAMS_KEY_NAME,
    ROLLOUT_ID_KEY_NAME,
    ROLLOUT_INDEX_KEY_NAME,
    SKILLS_REF_KEY_NAME,
    TASK_INDEX_KEY_NAME,
    TASK_SOURCE_KEY_NAME,
)


AGGREGATION_ERROR_KEY = "aggregation_error"
BATCH_STATUS_FNAME = "batch_status.json"
BATCH_STATUS_WRITE_INTERVAL_SECONDS = 5.0
_DATASET_DIGEST_DOMAIN = b"nemo-gym-batch-dataset-v1\0"
_FAILURE_CLASS_KEY = "_ng_failure_class"
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MISSING = object()


class BatchRepeatPolicy(BaseModel):
    """The repeat shape Gym must observe for every task belonging to one member."""

    model_config = ConfigDict(extra="forbid")

    num_repeats: int = Field(ge=1)
    seeded: bool


class BatchManifestMember(BaseModel):
    """Eval Factory's expectations for one logical benchmark in a shared Gym run."""

    model_config = ConfigDict(extra="forbid")

    agent_name: str
    task_sources: List[str] = Field(min_length=1)
    dataset_sha256: str
    expected_task_count: int = Field(ge=1)
    expected_rollout_count: int = Field(ge=1)
    repeat_policy: BatchRepeatPolicy
    resolved_recipe_sha256: str
    metric_keys: List[str]

    @field_validator("agent_name")
    @classmethod
    def _validate_agent_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("agent_name must not be blank")
        return value

    @field_validator("task_sources", "metric_keys")
    @classmethod
    def _validate_unique_names(cls, values: List[str]) -> List[str]:
        blank = [value for value in values if not value.strip()]
        if blank:
            raise ValueError("values must not be blank")
        duplicates = sorted(name for name, count in Counter(values).items() if count > 1)
        if duplicates:
            raise ValueError(f"values must be unique; duplicates: {duplicates}")
        return sorted(values)

    @field_validator("dataset_sha256", "resolved_recipe_sha256")
    @classmethod
    def _validate_sha256(cls, value: str) -> str:
        normalized = value.lower()
        if not _HEX_SHA256.fullmatch(normalized):
            raise ValueError("must be a 64-character SHA-256 hex digest")
        return normalized

    @model_validator(mode="after")
    def _validate_rollout_count(self) -> "BatchManifestMember":
        derived = self.expected_task_count * self.repeat_policy.num_repeats
        if self.expected_rollout_count != derived:
            raise ValueError(
                "expected_rollout_count must equal expected_task_count * repeat_policy.num_repeats "
                f"({self.expected_rollout_count} != {derived})"
            )
        return self


class BatchManifest(BaseModel):
    """Versioned, secret-free handoff from Eval Factory to Gym."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    members: Dict[str, BatchManifestMember] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_members(self) -> "BatchManifest":
        blank_slugs = [slug for slug in self.members if not slug.strip()]
        if blank_slugs:
            raise ValueError("member benchmark slugs must not be blank")
        agents = [member.agent_name for member in self.members.values()]
        duplicates = sorted(agent for agent, count in Counter(agents).items() if count > 1)
        if duplicates:
            raise ValueError(
                "members must use unique agent names because batch_status.json is keyed by agent; "
                f"duplicates: {duplicates}"
            )
        return self


@dataclass(frozen=True)
class ObservedBatchMember:
    """Dataset and repeat facts derived from materialized Gym input rows."""

    task_sources: List[str]
    dataset_sha256: str
    task_count: int
    rollout_count: int
    repeat_policy: BatchRepeatPolicy


def load_batch_manifest(manifest_fpath: Path) -> tuple[BatchManifest, str]:
    """Load a manifest and return it with the SHA-256 of the exact input bytes."""
    try:
        raw = manifest_fpath.read_bytes()
    except OSError as exc:
        raise ConfigPathNotFoundError(f"Cannot read batch manifest at '{manifest_fpath}': {exc}") from exc
    try:
        payload = orjson.loads(raw)
    except orjson.JSONDecodeError as exc:
        raise ConfigError(f"Batch manifest '{manifest_fpath}' is not valid JSON: {exc}") from exc
    try:
        manifest = BatchManifest.model_validate(payload)
    except ValidationError as exc:
        details = []
        for error in exc.errors(include_url=False, include_context=False, include_input=False):
            location = ".".join(str(part) for part in error["loc"]) or "<root>"
            details.append(f"- {location}: {error['msg']} ({error['type']})")
        raise ConfigError(f"Batch manifest '{manifest_fpath}' is invalid:\n" + "\n".join(details)) from exc
    return manifest, hashlib.sha256(raw).hexdigest()


def _seed_from_row(row: Mapping[str, Any]) -> Any:
    response_params = row.get(RESPONSES_CREATE_PARAMS_KEY_NAME) or {}
    metadata = response_params.get("metadata") or {}
    extra_body = metadata.get("extra_body")
    if isinstance(extra_body, str):
        try:
            extra_body = json.loads(extra_body)
        except (TypeError, json.JSONDecodeError):
            return _MISSING
    if not isinstance(extra_body, Mapping) or "seed" not in extra_body:
        return _MISSING
    return extra_body["seed"]


def _canonical_task_bytes(row: Mapping[str, Any]) -> bytes:
    """Canonicalize task content independently of routing, repeats, retries, and skills."""
    task = deepcopy(dict(row))
    for key in (
        AGENT_REF_KEY_NAME,
        TASK_INDEX_KEY_NAME,
        ROLLOUT_INDEX_KEY_NAME,
        ROLLOUT_ID_KEY_NAME,
        ATTEMPT_INDEX_KEY_NAME,
        SKILLS_REF_KEY_NAME,
    ):
        task.pop(key, None)

    response_params = task.get(RESPONSES_CREATE_PARAMS_KEY_NAME)
    if isinstance(response_params, dict):
        metadata = response_params.get("metadata")
        if isinstance(metadata, dict) and "extra_body" in metadata:
            original_extra_body = metadata["extra_body"]
            parsed_extra_body = original_extra_body
            if isinstance(original_extra_body, str):
                try:
                    parsed_extra_body = json.loads(original_extra_body)
                except (TypeError, json.JSONDecodeError):
                    parsed_extra_body = original_extra_body
            if isinstance(parsed_extra_body, dict) and "seed" in parsed_extra_body:
                parsed_extra_body.pop("seed")
                if parsed_extra_body:
                    metadata["extra_body"] = (
                        json.dumps(parsed_extra_body, sort_keys=True, separators=(",", ":"))
                        if isinstance(original_extra_body, str)
                        else parsed_extra_body
                    )
                else:
                    metadata.pop("extra_body")
            if not metadata:
                response_params.pop("metadata")

    return orjson.dumps(task, option=orjson.OPT_SORT_KEYS)


def observe_materialized_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, ObservedBatchMember]:
    """Derive per-agent dataset fingerprints and repeat policies from full materialized inputs.

    The dataset digest hashes one canonical row per task in its materialized order. Runtime
    routing and identity fields, repeat seeds, retry attempts, and run-level skills are excluded;
    prompt-expanded inputs and ``task_source`` remain part of the dataset identity.
    """
    rows_by_agent: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    errors: List[str] = []
    seen_rollouts = set()
    for position, row in enumerate(rows):
        agent_name = (row.get(AGENT_REF_KEY_NAME) or {}).get("name")
        task_index = row.get(TASK_INDEX_KEY_NAME)
        rollout_index = row.get(ROLLOUT_INDEX_KEY_NAME)
        task_source = row.get(TASK_SOURCE_KEY_NAME)
        if not isinstance(agent_name, str) or not agent_name.strip():
            errors.append(f"row {position} has no {AGENT_REF_KEY_NAME}.name")
            continue
        if not isinstance(task_index, int) or isinstance(task_index, bool) or task_index < 0:
            errors.append(f"row {position} for agent '{agent_name}' has an invalid task index {task_index!r}")
            continue
        if not isinstance(rollout_index, int) or isinstance(rollout_index, bool) or rollout_index < 0:
            errors.append(f"row {position} for agent '{agent_name}' has no stable task/rollout identity")
            continue
        if not isinstance(task_source, str) or not task_source.strip():
            errors.append(f"row {position} for agent '{agent_name}' has no {TASK_SOURCE_KEY_NAME}")
            continue
        identity = (agent_name, task_index, rollout_index)
        if identity in seen_rollouts:
            errors.append(f"duplicate materialized rollout identity {identity!r}")
            continue
        seen_rollouts.add(identity)
        rows_by_agent[str(agent_name)].append(row)

    observations: Dict[str, ObservedBatchMember] = {}
    for agent_name, agent_rows in rows_by_agent.items():
        canonical_by_task: Dict[Any, bytes] = {}
        task_order: List[Any] = []
        repeats_by_task: Counter = Counter()
        seeds_by_task: Dict[Any, List[Any]] = defaultdict(list)
        seed_presence: List[bool] = []
        task_sources = set()
        for row in agent_rows:
            task_index = row[TASK_INDEX_KEY_NAME]
            canonical = _canonical_task_bytes(row)
            if task_index not in canonical_by_task:
                canonical_by_task[task_index] = canonical
                task_order.append(task_index)
            elif canonical_by_task[task_index] != canonical:
                errors.append(f"agent '{agent_name}' has different task content under task index {task_index!r}")
            repeats_by_task[task_index] += 1
            task_sources.add(row[TASK_SOURCE_KEY_NAME])
            seed = _seed_from_row(row)
            seed_presence.append(seed is not _MISSING)
            if seed is not _MISSING:
                seeds_by_task[task_index].append(seed)

        repeat_counts = set(repeats_by_task.values())
        if len(repeat_counts) != 1:
            errors.append(
                f"agent '{agent_name}' has a non-uniform repeat policy: {dict(sorted(repeats_by_task.items()))}"
            )
            continue
        num_repeats = next(iter(repeat_counts))
        if any(seed_presence) and not all(seed_presence):
            errors.append(f"agent '{agent_name}' has seeds on only some materialized rollouts")
            continue
        seeded = bool(seed_presence and all(seed_presence))
        if seeded:
            for task_index, seeds in seeds_by_task.items():
                serialized_seeds = {orjson.dumps(seed, option=orjson.OPT_SORT_KEYS) for seed in seeds}
                if len(serialized_seeds) != len(seeds):
                    errors.append(f"agent '{agent_name}' repeats task {task_index!r} with duplicate seeds")

        digest = hashlib.sha256(_DATASET_DIGEST_DOMAIN)
        for task_index in task_order:
            canonical = canonical_by_task[task_index]
            digest.update(len(canonical).to_bytes(8, "big"))
            digest.update(canonical)
        observations[agent_name] = ObservedBatchMember(
            task_sources=sorted(task_sources),
            dataset_sha256=digest.hexdigest(),
            task_count=len(canonical_by_task),
            rollout_count=len(agent_rows),
            repeat_policy=BatchRepeatPolicy(num_repeats=num_repeats, seeded=seeded),
        )

    if errors:
        raise ConfigError("Materialized inputs do not satisfy the batch contract:\n- " + "\n- ".join(errors))
    if not observations:
        raise ConfigError("Materialized inputs contain no batch members")
    return observations


def validate_batch_manifest(manifest: BatchManifest, observations: Mapping[str, ObservedBatchMember]) -> None:
    """Reject any manifest expectation that disagrees with Gym's materialized inputs."""
    expected_agents = {member.agent_name for member in manifest.members.values()}
    observed_agents = set(observations)
    errors: List[str] = []
    missing = sorted(expected_agents - observed_agents)
    unexpected = sorted(observed_agents - expected_agents)
    if missing:
        errors.append(f"manifest agents missing from materialized inputs: {missing}")
    if unexpected:
        errors.append(f"materialized inputs contain agents absent from the manifest: {unexpected}")

    for benchmark_slug, member in manifest.members.items():
        observed = observations.get(member.agent_name)
        if observed is None:
            continue
        comparisons = (
            ("task_sources", member.task_sources, observed.task_sources),
            ("dataset_sha256", member.dataset_sha256, observed.dataset_sha256),
            ("expected_task_count", member.expected_task_count, observed.task_count),
            ("expected_rollout_count", member.expected_rollout_count, observed.rollout_count),
            ("repeat_policy", member.repeat_policy.model_dump(), observed.repeat_policy.model_dump()),
        )
        for field, expected, actual in comparisons:
            if expected != actual:
                errors.append(
                    f"member '{benchmark_slug}' ({member.agent_name}) {field} mismatch: "
                    f"expected {expected!r}, observed {actual!r}"
                )
    if errors:
        raise ConfigError("Batch manifest does not match materialized inputs:\n- " + "\n- ".join(errors))


def _read_aggregate_entries(metrics_fpath: Optional[Path]) -> tuple[Dict[str, Dict[str, Any]], Optional[str]]:
    if metrics_fpath is None:
        return {}, None
    try:
        payload = orjson.loads(metrics_fpath.read_bytes())
    except (OSError, orjson.JSONDecodeError):
        return {}, "InvalidAggregateMetricsArtifact"
    if not isinstance(payload, list):
        return {}, "InvalidAggregateMetricsArtifact"
    entries = {}
    for entry in payload:
        if not isinstance(entry, dict):
            return {}, "InvalidAggregateMetricsArtifact"
        agent_name = (entry.get(AGENT_REF_KEY_NAME) or {}).get("name")
        if not agent_name or agent_name in entries:
            return {}, "InvalidAggregateMetricsArtifact"
        entries[agent_name] = entry
    return entries, None


def _safe_aggregation_error(error: Any) -> Dict[str, Any]:
    """Copy only non-secret diagnostics into batch_status.json."""
    if not isinstance(error, Mapping):
        return {"type": "AggregationError"}
    safe = {"type": str(error.get("type") or "AggregationError")}
    status = error.get("http_status")
    if isinstance(status, int):
        safe["http_status"] = status
    return safe


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as output:
            output.write(orjson.dumps(payload, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS) + b"\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class BatchStatusTracker:
    """Validate one batch manifest and materialize its current per-agent status."""

    def __init__(
        self,
        manifest_fpath: Path,
        materialized_rows: Sequence[Mapping[str, Any]],
        *,
        write_interval_seconds: Optional[float] = None,
    ) -> None:
        self.manifest_fpath = manifest_fpath
        self.status_fpath = manifest_fpath.with_name(BATCH_STATUS_FNAME)
        self.manifest, self.manifest_sha256 = load_batch_manifest(manifest_fpath)
        self.observations = observe_materialized_rows(materialized_rows)
        validate_batch_manifest(self.manifest, self.observations)
        self.write_interval_seconds = (
            BATCH_STATUS_WRITE_INTERVAL_SECONDS if write_interval_seconds is None else write_interval_seconds
        )
        self._last_write = 0.0
        self._last_progress_count = 0

    def build_status(
        self,
        completed_results: Sequence[Mapping[str, Any]],
        failure_rows: Sequence[Mapping[str, Any]],
        *,
        aggregate_metrics_fpath: Optional[Path] = None,
        aggregation_deferred: bool = False,
    ) -> Dict[str, Any]:
        completed_by_agent: Dict[str, set] = defaultdict(set)
        for result in completed_results:
            agent_name = (result.get(AGENT_REF_KEY_NAME) or {}).get("name")
            task_index = result.get(TASK_INDEX_KEY_NAME)
            rollout_index = result.get(ROLLOUT_INDEX_KEY_NAME)
            if agent_name is not None and task_index is not None and rollout_index is not None:
                completed_by_agent[str(agent_name)].add((task_index, rollout_index))

        latest_failures = {}
        for failure in failure_rows:
            agent_name = (failure.get(AGENT_REF_KEY_NAME) or {}).get("name")
            task_index = failure.get(TASK_INDEX_KEY_NAME)
            rollout_index = failure.get(ROLLOUT_INDEX_KEY_NAME)
            if agent_name is not None and task_index is not None and rollout_index is not None:
                latest_failures[(str(agent_name), task_index, rollout_index)] = failure

        failures_by_agent: Dict[str, Counter] = defaultdict(Counter)
        for (agent_name, task_index, rollout_index), failure in latest_failures.items():
            if (task_index, rollout_index) in completed_by_agent[agent_name]:
                continue
            failures_by_agent[agent_name][failure.get(_FAILURE_CLASS_KEY) or "unknown"] += 1

        aggregate_entries, aggregate_artifact_error = _read_aggregate_entries(aggregate_metrics_fpath)
        members = {}
        for benchmark_slug, expected in sorted(self.manifest.members.items()):
            agent_name = expected.agent_name
            observed = self.observations[agent_name]
            aggregate_entry = aggregate_entries.get(agent_name)
            observed_metric_keys = sorted((aggregate_entry or {}).get("key_metrics") or {})
            aggregation_error = None
            if aggregation_deferred:
                aggregation_status = "deferred"
            elif aggregate_metrics_fpath is None:
                aggregation_status = "pending"
            elif aggregate_artifact_error is not None:
                aggregation_status = "error"
                aggregation_error = {"type": aggregate_artifact_error}
            elif aggregate_entry is None:
                aggregation_status = "error"
                aggregation_error = {"type": "MissingAggregateMetricsEntry"}
            elif AGGREGATION_ERROR_KEY in aggregate_entry:
                aggregation_status = "error"
                aggregation_error = _safe_aggregation_error(aggregate_entry[AGGREGATION_ERROR_KEY])
            else:
                missing_metric_keys = sorted(set(expected.metric_keys) - set(observed_metric_keys))
                if missing_metric_keys:
                    aggregation_status = "error"
                    aggregation_error = {
                        "type": "MissingExpectedMetricKeys",
                        "missing_metric_keys": missing_metric_keys,
                    }
                else:
                    aggregation_status = "complete"

            completed_count = len(completed_by_agent[agent_name])
            members[agent_name] = {
                "benchmark_slug": benchmark_slug,
                "resolved_recipe_sha256": expected.resolved_recipe_sha256,
                "observed_task_sources": observed.task_sources,
                "observed_dataset_sha256": observed.dataset_sha256,
                "observed_task_count": observed.task_count,
                "expected_rollout_count": expected.expected_rollout_count,
                "completed_rollout_count": completed_count,
                "remaining_rollout_count": max(expected.expected_rollout_count - completed_count, 0),
                "failures_by_class": dict(sorted(failures_by_agent[agent_name].items())),
                "repeat_policy": observed.repeat_policy.model_dump(),
                "expected_metric_keys": sorted(expected.metric_keys),
                "observed_metric_keys": observed_metric_keys,
                "aggregation_status": aggregation_status,
                **({"aggregation_error": aggregation_error} if aggregation_error is not None else {}),
            }

        return {
            "schema_version": "1",
            "manifest_sha256": self.manifest_sha256,
            "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "members": members,
        }

    def write_status(
        self,
        completed_results: Sequence[Mapping[str, Any]],
        failure_rows: Sequence[Mapping[str, Any]],
        *,
        aggregate_metrics_fpath: Optional[Path] = None,
        aggregation_deferred: bool = False,
        force: bool = False,
    ) -> bool:
        """Atomically refresh status, throttling progress writes unless ``force`` is true."""
        now = monotonic()
        progress_count = len(completed_results) + len(failure_rows)
        first_durable_progress = progress_count > 0 and self._last_progress_count == 0
        if not force and not first_durable_progress and now - self._last_write < self.write_interval_seconds:
            return False
        status = self.build_status(
            completed_results,
            failure_rows,
            aggregate_metrics_fpath=aggregate_metrics_fpath,
            aggregation_deferred=aggregation_deferred,
        )
        _atomic_write_json(self.status_fpath, status)
        self._last_write = now
        self._last_progress_count = progress_count
        return True
