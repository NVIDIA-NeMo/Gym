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
import asyncio
import bisect
import functools
import glob as glob_module
import hashlib
import json
import logging
import math
import os
import re
import secrets
import signal
import time
import warnings
from asyncio import Future, Semaphore
from collections import Counter, defaultdict, deque
from collections.abc import Collection, Iterable, Mapping, Sequence
from contextlib import AsyncExitStack, contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import timedelta
from difflib import get_close_matches
from itertools import repeat
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Literal, Optional, Tuple, Union
from uuid import uuid4

import aiohttp
import orjson
from aiohttp import ClientConnectorError, ClientError
from omegaconf import DictConfig, ListConfig, OmegaConf
from pydantic import BaseModel, Field, PrivateAttr, ValidationError, field_validator, model_validator
from tqdm.asyncio import tqdm

from nemo_gym import _resolve_under_cwd_or_install
from nemo_gym.artifacts.exporters import export_metrics, export_rollouts, get_exporters
from nemo_gym.artifacts.exporters.metrics import flatten_aggregate_metrics
from nemo_gym.artifacts.rollout_rows import (
    NG_DISPATCH_DRAINED_KEY,
    NG_FAILURE_CLASS_KEY,
    NG_NO_PERSIST_KEY,
    _rollout_for_export,
    _strip_capture_payloads,
)
from nemo_gym.artifacts.run_files import AGGREGATION_ERROR_KEY
from nemo_gym.batch_status import BatchStatusTracker
from nemo_gym.config.config_types import (
    AgentWithoutEnvironmentServerError,
    AmbiguousEnvironmentServerError,
    BaseNeMoGymCLIConfig,
    BaseServerConfig,
    ConfigError,
    ConfigPathNotFoundError,
    RemovedRolloutOptionsMixin,
    UploadRolloutsConfigMixin,
)
from nemo_gym.config.global_config import (
    AGENT_REF_KEY_NAME,
    AGENT_SERVER_TYPE_KEY_NAME,
    ALLOW_UNSUPPORTED_PAIRING_ENV_VAR_NAME,
    ALLOWED_AGENTS_KEY_NAME,
    ATTEMPT_INDEX_KEY_NAME,
    ENVIRONMENT_SERVER_STAMP_KEY_NAME,
    ENVIRONMENT_SERVER_TYPE_KEY_NAME,
    RESPONSES_CREATE_PARAMS_KEY_NAME,
    ROLLOUT_ID_KEY_NAME,
    ROLLOUT_INDEX_KEY_NAME,
    SKILLS_REF_KEY_NAME,
    TASK_INDEX_KEY_NAME,
    TASK_SOURCE_KEY_NAME,
    allowed_agents_for,
    dataset_agent_pins,
    environment_server_agent_names,
    environment_server_attributed_agent,
    get_first_server_config_dict,
    get_global_config_dict,
    label_runs,
    pairing_override_enabled,
    resolve_dataset_agent,
)
from nemo_gym.core.episode import EpisodeId, TaskId, maybe_rollout_id_from_run_body
from nemo_gym.core.failure_kinds import (
    CANCELLED,
    COLLECTION_DISPATCH_FAILED,
    ENVIRONMENT_PROTOCOL_VIOLATION,
    ENVIRONMENT_SERVER_FAILED,
    JUDGE_FAILED,
    is_namespaced,
    is_registered,
)
from nemo_gym.failure_transport import FailureHeader, FailureReportRow
from nemo_gym.health.types import RunHealthConfig
from nemo_gym.observability.model_calls import (
    await_model_call_capture_quiescence,
    clear_model_call_captures_for_rollouts,
    merge_model_call_capture_into_record,
    model_call_capture_dirs_from_config,
    observability_enabled_from_config,
)
from nemo_gym.omitted_rollouts import omitted_rollout_record
from nemo_gym.prompt import apply_prompt_to_row, load_prompt_config, validate_prompt_compatibility
from nemo_gym.rollout_outcomes import InvalidRolloutResult, RolloutFailureRecord, StaleRolloutResult
from nemo_gym.rollout_records import (
    RUN_ID_KEY,
    coverage_path_for,
    journal_path_for,
    logical_rollout_id,
    resolve_rollout_path,
)
from nemo_gym.rollout_recovery import (
    _DEFAULT_MAX_ROLLOUT_ATTEMPTS as _DEFAULT_MAX_ROLLOUT_ATTEMPTS,
)
from nemo_gym.rollout_recovery import (
    IncompleteEvaluationError,
    RunManifest,
    atomic_write_json,
    has_only_intentional_gaps,
    is_terminal_failure,
    load_resubmit_offset,
    manifest_path_for,
    matches_failure_class,
    observed_elapsed,
    run_lock,
)
from nemo_gym.rollout_recovery import (
    _get_max_rollout_attempts as _get_max_rollout_attempts,
)
from nemo_gym.rollout_recovery import (
    record_resubmit_offset as record_resubmit_offset,
)
from nemo_gym.rollout_store import RolloutStore
from nemo_gym.server_runtime.base_resources_server import AggregateMetrics, AggregateMetricsRequest
from nemo_gym.server_runtime.episode_types import (
    EpisodeFailure,
    FailureBoundary,
    FailureReport,
    is_materialized_task_row,
)
from nemo_gym.telemetry._fallbacks import is_span_group_enabled, managed_span
from nemo_gym.telemetry.endpoints import ROLLOUT_ID_ATTRIBUTE
from nemo_gym.telemetry.gym_metrics import (
    DISPATCH_NAME_ATTRIBUTE,
    HEALTH_CHECKS_ATTRIBUTE,
    HEALTH_VERDICT_ATTRIBUTE,
    HTTP_DESTINATION_SERVER_NAME_ATTRIBUTE,
    UNREGISTERED_FAILURE_KIND,
    record_rollout_completed,
    record_rollout_health,
    record_rollout_reward,
)
from nemo_gym.telemetry.observer import MeteredAdmission, count_resident, driver_metrics, observe_event_loop
from nemo_gym.telemetry.setup import is_metrics_exporter_active, is_trace_exporter_active
from nemo_gym.telemetry.span_groups import GymSpanGroup
from nemo_gym.trajectory.trajectory_record import (
    NG_AGENT_OBSERVATIONS_KEY,
    NG_TRAJECTORY_KEY,
    NG_VERIFIER_INVOCATIONS_KEY,
    AgentInvocation,
    AgentObservationBundle,
    ModelCallRef,
    ObservationGap,
    ToolCallObservation,
    TrajectoryModelCall,
    TrajectoryRecord,
    TrajectoryTokenStats,
    TrajectoryToolCall,
    TrajectoryTurn,
    VerifierInvocation,
)
from nemo_gym.typed_exceptions import NgFailureEvent, failure_report_for
from nemo_gym.utils.path_utils import (
    aggregate_metrics_path_for,
    aggregation_sources_path_for,
    failures_path_for,
    materialized_path_for,
    resubmit_offset_path_for,
)


if TYPE_CHECKING:
    from opentelemetry.trace import SpanContext


_failures_path_for = failures_path_for  # Backwards-compatible alias
from nemo_gym.client.http_client import (
    get_response_json,
    is_global_aiohttp_client_request_debug_enabled,
    raise_for_status,
    request,
)
from nemo_gym.client.server_client import ServerClient
from nemo_gym.client.server_client import (
    setup_server_client as setup_server_client_utils,
)
from nemo_gym.health.timeouts import ELAPSED_S_KEY, classify_timeout, export_run_health_timeouts
from nemo_gym.skills import SkillsConfig, load_skill_directory
from nemo_gym.task_materialization import TASK_ID_FIELDS
from nemo_gym.token_id_capture import (
    TokenCaptureStore,
    TokenIdCaptureConfig,
    clear_token_captures_for_rollouts,
    installed_token_source,
    mask_incomplete_when_attributed_from_config,
    token_id_capture_dirs_from_config,
)
from nemo_gym.token_id_capture.config import token_id_capture_enabled_for_agent
from nemo_gym.token_id_capture.delivery import (
    MASK_SAMPLE_KEY,
    capture_build_can_retire,
    finalize_rollout_token_capture,
    retire_rollout_token_capture,
)


logger = logging.getLogger(__name__)


def _masking_step_metrics(agent_name: str, scored: Counter, dropped: Counter) -> Dict[str, float]:
    """In-progress view of what a run is losing to its environment rather than its policy.

    ``scored`` covers persisted rollouts only, split into the unmasked ones (``count``,
    ``reward``) and the masked ones; ``dropped`` counts what never reached the main output
    at all. ``reward_unmasked`` averages over the unmasked rollouts alone, so the gap
    against the existing ``reward`` series is the score lost to infrastructure. Failed and
    omitted attempts are reported as counts, never folded into a quality average.

    Empty until something is actually masked or dropped, so a healthy run exports exactly
    what it exported before. The final numbers come from ``/aggregate_metrics``; this is
    the progress view while the run is still going.
    """
    masked, unmasked = int(scored["masked"]), int(scored["count"])
    failed, omitted = int(dropped["failed"]), int(dropped["omitted"])
    if not (masked or failed or omitted):
        return {}

    metrics: Dict[str, float] = {}
    persisted = masked + unmasked
    if masked and persisted:
        metrics[f"progress/{agent_name}/masked_pct"] = round(100 * masked / persisted, 2)
    if unmasked:
        metrics[f"progress/{agent_name}/reward_unmasked"] = round(100 * scored["reward"] / unmasked, 2)
    if failed:
        metrics[f"progress/{agent_name}/failed"] = failed
    if omitted:
        metrics[f"progress/{agent_name}/omitted"] = omitted
    return metrics


# ---------------------------------------------------------------------------
# Failure-routing sentinels (set by environment servers, read by the dispatcher).
#
# Background:
#   The historical contract was "every dispatched task produces one row in
#   the main rollouts jsonl, succeeded or failed." That contract broke
#   resume-after-walltime: synthetic ``-failed`` rows written during a
#   SIGTERM grace window look identical to real successes to the dedup in
#   ``_load_from_cache`` (which keys only on (task_index, rollout_index)),
#   so chain-hop 2 thinks failed tasks are done and never retries them.
#
# New contract:
#   - Successes go to the main jsonl (``output_jsonl_fpath``).
#   - Failures go to a sidecar (``<output_stem>_failures.jsonl``), one row
#     per attempt, with ``_ng_failure_class`` set. An ``agent_run_error`` or
#     ``agent_request_failed`` row holds no reward and no response: there was
#     no rollout. The two differ in whether the agent answered at all.
#   - Reported failures and intentional omissions are saved in the sidecar.
#   - The run manifest reserves execution identities before dispatch. Its newest
#     reservation selects the outcome, including an interruption with no payload.
#   - Only nonterminal delivered/possibly-delivered failures consume the retry cap.
#     Interrupted and budget-drained work can resume without spending a retry.
#   - The controller holds a run lock across preparation, collection and reporting.
#   - ``record_omitted_rollouts`` adds a sidecar row, marked ``_ng_omitted_rollout``, for a row
#     the dispatch budget never started. It is evidence only: the run history skips it, and resume
#     still runs the row. An interrupted attempt needs no such row: it is saved as a ``kill_shaped`` failure.
# ---------------------------------------------------------------------------

NG_FAILURE_LEGACY_CLASS_KEY = "_ng_failure_legacy_class"
# The failure report judge_failsafe attaches to its result (#5072): an event map, the judge failure's event id, and
# whether events were left out. Kept on the row as diagnostics, outside the versioned failure record.
NG_FAILURE_REPORT_KEYS = ("_ng_failure_events", "_ng_failure_id", "_ng_failure_events_omitted")
# Set only on the failures-sidecar row ``record_omitted_rollouts`` writes for a rollout that produced no result.
# Readers key on it, not on ``_ng_no_persist``, which other sidecar rows (ATIF reverification) carry and count.
NG_OMITTED_ROLLOUT_KEY = "_ng_omitted_rollout"


def is_omitted_rollout_row(row: Mapping[str, Any]) -> bool:
    """Whether a failures-sidecar row is a ``record_omitted_rollouts`` record of a rollout that produced no result."""
    return bool(row.get(NG_OMITTED_ROLLOUT_KEY))


def is_unstarted_rollout_row(row: Mapping[str, Any]) -> bool:
    """Whether a sidecar row records a rollout the dispatch budget never started: no execution, so no history."""
    return is_omitted_rollout_row(row) and bool(row.get(NG_DISPATCH_DRAINED_KEY))


# Set only on the omitted-rollout row that keeps the evidence of a failed /run with route_failures_to_sidecar off.
NG_UNROUTED_RUN_ERROR_KEY = "_ng_unrouted_run_error"


def is_non_superseding_rollout_row(row: Mapping[str, Any]) -> bool:
    """Whether readers skip a sidecar row before they pick a rollout's latest attempt.

    A row never started is no execution. A failed /run kept as evidence with routing off is no new outcome: the
    attempt it records stands no more than before it was kept, so an older failure still stands and a later attempt
    still replaces it. An interrupted ``record_omitted_rollouts`` row is neither and still hides older failures.
    """
    return is_unstarted_rollout_row(row) or (
        is_omitted_rollout_row(row) and row.get(NG_UNROUTED_RUN_ERROR_KEY) is True
    )


NG_TERMINAL_KEY = "_ng_failure_terminal"
# Set by the producer of a final failure row that must be counted as a zero in the metrics, whatever
# count_failure_classes_as_zero names (a rollout the evaluated model failed, for example).
NG_FAILURE_COUNTS_AS_ZERO_KEY = "_ng_failure_counts_as_zero"
# The attempt cap the collecting process ran with, stamped on each sidecar row: offline aggregation reads it
# instead of its own environment. The `_ng_failure_` prefix keeps it out of the metrics of a counted row.
NG_FAILURE_MAX_ATTEMPTS_KEY = "_ng_failure_max_attempts"
NG_FAILURE_JUDGE_ERROR_KEY = "_ng_failure_judge_error"
AGENT_REQUEST_FAILED_FAILURE_CLASS = "agent_request_failed"
AGENT_RUN_ERROR_FAILURE_CLASS = "agent_run_error"
ENVIRONMENT_SERVER_FAILURE_CLASS = ENVIRONMENT_SERVER_FAILED
NG_ENVIRONMENT_SERVER_KEY = ENVIRONMENT_SERVER_STAMP_KEY_NAME
# Implementation name under `environment_servers:`, so readers know which result type a record holds.
NG_RESULT_TYPE_KEY = "_ng_result_type"
NG_TASK_ID_KEY = "_ng_task_id"
# Provenance stamps preparation writes into a materialized row's task_data (the same strings as
# nemo_gym.tasks.harbor.digest's DIGEST_KEY and DATASET_DIGEST_KEY, which this module does not import).
# The collector copies only these from a row onto its record: task_data is dataset-controlled, and any
# other `_ng_*` key there (`_ng_no_persist`, `_ng_failure_class`, ...) would steer the collector.
NG_DIGEST_KEY = "_ng_digest"
NG_DATASET_DIGEST_KEY = "_ng_dataset_digest"
PROVENANCE_KEYS = (NG_DIGEST_KEY, NG_DATASET_DIGEST_KEY)
_NO_RESULT_FAILURE_CLASSES = frozenset(
    {
        AGENT_REQUEST_FAILED_FAILURE_CLASS,
        AGENT_RUN_ERROR_FAILURE_CLASS,
        ENVIRONMENT_SERVER_FAILURE_CLASS,
    }
)
NG_PERF_KEY = "ng_perf"


def _environment_servers_by_agent(global_config_dict: DictConfig) -> dict[str, list[str]]:
    """Map each agent name to the environment servers that front it."""
    servers_by_agent: dict[str, list[str]] = {}
    for name, instance in global_config_dict.items():
        if not isinstance(instance, DictConfig):
            continue
        servers = instance.get(ENVIRONMENT_SERVER_TYPE_KEY_NAME)
        if not isinstance(servers, DictConfig):
            continue
        for server in servers.values():
            if not isinstance(server, DictConfig):
                continue
            for agent_name in environment_server_agent_names(server):
                servers_by_agent.setdefault(agent_name, []).append(str(name))
    return servers_by_agent


def _environment_server_for_agent(agent_name: str, servers_by_agent: Mapping[str, list[str]]) -> str:
    """Return the one environment server that fronts an agent.

    A row routed by its agent cannot choose between several environment servers.
    Several servers may still front one agent when every row names its server directly.
    Materialized tasksets name their environment server through ``environment_server_routes``.
    """
    servers = servers_by_agent.get(agent_name, [])
    if len(servers) == 1:
        return servers[0]
    if not servers:
        raise AgentWithoutEnvironmentServerError(
            f"Agent '{agent_name}' has no environment server, so collection cannot reach it. "
            "Config validation should have caught this before any server started."
        )
    raise AmbiguousEnvironmentServerError(
        f"Agent '{agent_name}' is fronted by several environment servers: {sorted(servers)}. "
        "Rows that route by agent cannot choose between them. "
        "Set environment_routing_mode=legacy with environment_server_name, or route these rows by taskset."
    )


def _materialized_taskset(row: Mapping[str, Any]) -> str | None:
    if not is_materialized_task_row(row):
        return None
    task_id = row.get("task_id")
    assert isinstance(task_id, Mapping)
    taskset = task_id.get("taskset")
    return taskset if isinstance(taskset, str) and taskset else None


def _declared_datasets(agent_config: DictConfig) -> list[DictConfig]:
    """The ``datasets`` entries one agent block declares; a list that is missing, a placeholder or not a list
    declares nothing."""
    datasets = OmegaConf.select(agent_config, "datasets", throw_on_resolution_failure=False)
    if not isinstance(datasets, ListConfig):
        return []
    return [entry for entry in datasets if isinstance(entry, DictConfig)]


def _datasets_declaring_taskset(global_config_dict: DictConfig, taskset: str) -> list[DictConfig]:
    """The entries that declare the rows of ``taskset``, on whichever agent block declares them: an entry whose
    ``taskset`` is that name, or whose ``name`` is when it sets none (a Harbor dataset's generated entry, named after
    its taskset and routed by it)."""
    declaring: list[DictConfig] = []
    for _, block in global_config_dict.items_ex(resolve=False):
        if not isinstance(block, DictConfig):
            continue
        agents = OmegaConf.select(block, AGENT_SERVER_TYPE_KEY_NAME, throw_on_resolution_failure=False)
        if not isinstance(agents, DictConfig):
            continue
        for agent_config in agents.values():
            if not isinstance(agent_config, DictConfig):
                continue
            for entry in _declared_datasets(agent_config):
                declared = entry.get("taskset")
                if declared == taskset or (declared is None and entry.get("name") == taskset):
                    declaring.append(entry)
    return declaring


def _dataset_admitted_agents(
    global_config_dict: DictConfig, agent_config: DictConfig, taskset: str | None
) -> list[tuple[str, list[str]]]:
    """``(dataset name, allowed_agents)`` for each dataset entry that declares the dispatched rows with a non-empty
    ``allowed_agents``.

    A materialized row names its dataset through its taskset, so only the entry that declares that taskset counts
    (:func:`_datasets_declaring_taskset`), not every dataset the executing agent happens to declare: a second,
    unrelated entry on the agent cannot refuse the rows. The lookup covers every agent block, not the executing
    agent's alone: a Harbor dataset's entry sits on the agent block ``gym eval run`` generated for it, and a run
    config edited to route the taskset's environment server to another agent block afterwards still has to meet
    the dataset's list. A flat row routed to an agent directly names no dataset, so every dataset that agent
    declares counts. So do the executing agent's entries that set no ``taskset`` when no entry anywhere declares
    the rows' taskset: a hand-written entry may list pre-materialized rows under a name of its own
    (``collate_samples`` accepts such rows in any entry), and those rows are still that agent's dataset, so they are
    checked rather than none; an entry that sets a ``taskset`` materializes and routes its rows as that taskset, so
    one naming another taskset does not hold these rows.

    A bare string is one entry, as ``allowed_agents_for`` reads a resources server's.
    """
    entries = _declared_datasets(agent_config)
    if taskset is not None:
        entries = _datasets_declaring_taskset(global_config_dict, taskset) or [
            entry for entry in entries if entry.get("taskset") is None
        ]
    admitted: list[tuple[str, list[str]]] = []
    for entry in entries:
        declared = entry.get(ALLOWED_AGENTS_KEY_NAME)
        if not declared:
            continue
        names = [declared] if isinstance(declared, str) else [str(name) for name in declared]
        admitted.append((str(entry.get("name")), names))
    return admitted


def _environment_server_for_config_row(row: Mapping[str, Any], config: Any) -> str | None:
    """Pick the environment server a row is dispatched to, or None for today's agent path.

    A materialized task (``task_id.taskset`` plus ``task_input``) always routes by its taskset:
    it is already an episode request and no agent-server ``/run`` accepts it. A flat row follows
    ``environment_routing_mode``: ``agent`` keeps today's routing (its agent's environment server
    is resolved at dispatch), ``legacy`` sends every flat row to ``environment_server_name``, and
    ``taskset`` refuses flat rows so a run of materialized rows cannot silently pick up legacy input.

    One batch may therefore hold both kinds of rows in ``agent`` and ``legacy`` mode. The chosen
    server is stamped on the row as ``_ng_environment_server`` and travels with it through the
    materialized input file, retries, and results.
    """
    taskset = _materialized_taskset(row)
    if taskset is not None:
        try:
            return config.environment_server_routes[taskset]
        except KeyError as error:
            raise ValueError(f"No environment server route is configured for taskset {taskset!r}") from error
    if config.environment_routing_mode == "agent":
        return None
    if config.environment_routing_mode == "legacy":
        return config.environment_server_name
    raise ValueError(
        "taskset routing requires rows containing task_id.taskset and task_input; "
        "use environment_routing_mode=agent to mix materialized and flat rows in one batch"
    )


def _episode_request_body(row: Mapping[str, Any]) -> dict[str, Any]:
    attempt = row.get(ATTEMPT_INDEX_KEY_NAME, 0)
    if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 0:
        raise ValueError(f"Invalid episode attempt: {attempt!r}")
    # Decode the identity the way the agent does.
    # An explicit id may carry the attempt suffix itself, as a restored attempt does.
    capture_key = maybe_rollout_id_from_run_body(row) or f"{row[TASK_INDEX_KEY_NAME]}-{row[ROLLOUT_INDEX_KEY_NAME]}"
    return {
        "episode_id": EpisodeId.from_capture_key(capture_key).model_dump(mode="json"),
        "task": {
            "task_id": row["task_id"],
            "task_input": row["task_input"],
        },
    }


def _reject_attempt_qualified_rollout_ids(rows: Iterable[Mapping[str, Any]], fpath: Path) -> None:
    """Refuse dataset rows whose explicit rollout ID names an attempt.

    The suffix ``-a<N>`` is reserved: ``"<id>-a<N>"`` always means attempt N of rollout ``<id>``,
    which is how a training controller names a restored attempt.
    A resume retry stamps the next attempt on a row, and an ID that already names one would then fail mid-run,
    so the row is refused when the input loads.
    """
    for line_no, row in enumerate(rows, 1):
        explicit = row.get(ROLLOUT_ID_KEY_NAME) if isinstance(row, Mapping) else None
        if not isinstance(explicit, str):
            continue
        episode_id = EpisodeId.from_capture_key(explicit)
        if episode_id.attempt:
            raise ConfigError(
                f"Row at line {line_no} of '{fpath}' has {ROLLOUT_ID_KEY_NAME} {explicit!r}, "
                f"which ends with the reserved suffix '-a<N>': it names attempt {episode_id.attempt} "
                f"of rollout {episode_id.rollout_id!r}. Give the row a rollout ID without that suffix."
            )


def _is_episode_response(result: Any) -> bool:
    """True for a ``BaseEpisodeResponse``-shaped reply: object identities plus a ``result`` or ``failure`` key.

    The collector only applies this to a row it dispatched as an episode request
    (``_materialized_taskset(row)``), so an agent's verify response that echoes identity fields is left alone.
    """
    return (
        isinstance(result, Mapping)
        and isinstance(result.get("episode_id"), Mapping)
        and isinstance(result.get("task_id"), Mapping)
        and ("result" in result or "failure" in result)
    )


def _is_collector_key(key: str) -> bool:
    """Keys rollout collection writes itself; an Environment Server result must not use them.

    ``ng_trajectory`` is not one of them: it is the producer's slot, where an Environment Server forwards
    the harness's own turns from its session close (as an agent's ``/run`` result does), and the collector
    replaces it with the canonical projection built from those turns.
    """
    return key.startswith("_ng_") or key in ("ng_model_call_capture", NG_PERF_KEY)


# Legacy harnesses intentionally return these failure and observation fields.
# Other collector keys (including identities and persisted records) may be stale
# echoes of an input and must not control the current execution.
_LEGACY_RESULT_FIELDS = frozenset(
    {
        NG_FAILURE_CLASS_KEY,
        NG_NO_PERSIST_KEY,
        NG_TERMINAL_KEY,
        "_ng_failure_type",
        "_ng_failure_kind",  # The legacy Environment Server adapter preserves an explicitly absent kind.
        "_ng_failure_message",
        "_ng_failure_http_status",
        "_ng_failure_response_body",
        "_ng_failure_response",
        "_ng_failure_partial_response",
        "_ng_failure_stage",
        "_ng_failure_judge_error",
        NG_FAILURE_COUNTS_AS_ZERO_KEY,
        *NG_FAILURE_REPORT_KEYS,
        "_ng_agent_observations",
        "_ng_terminal_model_call_id",
        "_ng_token_capture",
        "_ng_timeout_inflight",
        "_ng_timeout_concurrency_limit",
        "_ng_group_id",
        "_ng_group_attempt",
        NG_TRAJECTORY_KEY,
        NG_PERF_KEY,
        "ng_model_call_capture",
    }
)


def _sanitize_legacy_result(result: Dict) -> None:
    """Discard collector bookkeeping while preserving declared legacy evidence."""
    for key in list(result):
        if _is_collector_key(key) and key not in _LEGACY_RESULT_FIELDS:
            del result[key]


def _task_data_provenance(row: Mapping[str, Any]) -> Dict[str, Any]:
    """The :data:`PROVENANCE_KEYS` of a materialized row's ``task_data``: provenance stamps for the record.

    Preparation writes them (a Harbor row carries the task folder digest as ``_ng_digest`` and the
    dataset digest as ``_ng_dataset_digest``) so that every results row names the inputs it ran, without
    the materialized-inputs sidecar. An Environment Server never returns them (``_ng_*`` is the
    collector's namespace), so the collector copies them from the row onto the record. Only the
    allowlisted keys are copied: ``task_data`` comes from the dataset, and the collector reads other
    ``_ng_*`` keys (``_ng_no_persist``, ``_ng_failure_class``, ``_ng_result_type``, ...) that a success
    record does not carry, so a dataset could otherwise drop, reroute or relabel its own results.
    """
    task_input = row.get("task_input")
    task_data = task_input.get("task_data") if isinstance(task_input, Mapping) else None
    if not isinstance(task_data, Mapping):
        return {}
    return {key: task_data[key] for key in PROVENANCE_KEYS if key in task_data}


def _reply_failure_report(response: Mapping[str, Any]) -> tuple[FailureReport, ...]:
    """The ``FailureReport`` an environment reply carries beside its outcome; an invalid one is omitted."""
    report = response.get("failure_report")
    if report is None:
        return ()
    try:
        return (FailureReport.model_validate(report),)
    except ValidationError:
        logger.warning("Invalid failure report in an environment reply; its failure events are omitted")
        return (FailureReport(omitted=True),)


def _episode_record(response: Dict[str, Any]) -> Dict[str, Any]:
    """Turn a ``BaseEpisodeResponse`` into the rollout record the collector stores.

    A handled ``failure`` becomes a failures-sidecar row, so resume retries a non-terminal one and
    never a terminal one. A ``result`` is stored as the Environment Server returned it; the collector
    adds only its own ``_ng_*`` keys, so any Environment Server type can be collected without the
    collector knowing its result fields.

    Every Environment Server type is scored the same way: through the result's top-level ``reward``,
    with optional top-level ``reward_components``. A reward nested elsewhere in the result is stored as
    data, and a result without a top-level ``reward`` is unscored.

    A failure's ``failure_kind`` is kept under that name, next to the ``_ng_failure_*`` diagnostics: it is
    the key ``_record_rollout_outcome`` reads for the metric label and ``_counted_failure_row`` drops.
    """
    task_id = response.get("task_id")
    failure = response.get("failure")
    if failure is not None:
        if not isinstance(failure, Mapping):
            raise ValueError(
                f"environment server reply for task {task_id!r} carries a non-object failure: {failure!r}"
            )
        failure_kind = failure.get("failure_kind")
        record: Dict[str, Any] = {
            NG_TASK_ID_KEY: task_id,
            NG_FAILURE_CLASS_KEY: failure_kind or ENVIRONMENT_SERVER_FAILURE_CLASS,
            "_ng_failure_kind": failure_kind,
            NG_TERMINAL_KEY: bool(failure.get("terminal", False)),
            "_ng_failure_message": failure.get("failure_reason"),
        }
        if failure_kind:
            # Every Environment Server failure was routed as this class before its kind became the class.
            # Keep it as the legacy class, so existing selectors such as count_failure_classes_as_zero still match.
            record[NG_FAILURE_LEGACY_CLASS_KEY] = ENVIRONMENT_SERVER_FAILURE_CLASS
        if failure.get("stage") is not None:
            record["_ng_failure_stage"] = failure["stage"]
        # Keep the diagnostic kind separate from the routing class above.
        if failure.get("failure_kind") is not None:
            record["failure_kind"] = failure["failure_kind"]
            record["_ng_failure_kind"] = failure["failure_kind"]
        record = FailureReportRow.merged_into(record, *_reply_failure_report(response))
        if failure.get("partial_response") is not None:
            record["_ng_failure_partial_response"] = failure["partial_response"]
        if failure.get("agent_observations") is not None:
            record[NG_AGENT_OBSERVATIONS_KEY] = failure["agent_observations"]
        if failure.get("trajectory") is not None:
            # The harness's own turns before the failure, in the producer's slot the projection reads first.
            record[NG_TRAJECTORY_KEY] = failure["trajectory"]
        if failure.get("verifier_invocations") is not None:
            record[NG_VERIFIER_INVOCATIONS_KEY] = failure["verifier_invocations"]
        return record
    result = response.get("result")
    if not isinstance(result, Mapping):
        raise ValueError(f"environment server reply for task {task_id!r} carries a non-object result: {result!r}")
    # judge_failsafe is a built-in producer of these legacy markers and of a diagnostic failure report.
    # Accept them only on its declared failure form: the judge_failed class with a judge kind.
    # Other collector-owned keys remain forbidden.
    kind = result.get("failure_kind")
    judge_markers = (
        {NG_FAILURE_CLASS_KEY, NG_FAILURE_JUDGE_ERROR_KEY, *NG_FAILURE_REPORT_KEYS}
        if result.get(NG_FAILURE_CLASS_KEY) == JUDGE_FAILED and isinstance(kind, str) and kind.startswith("judge_")
        else set()
    )
    # A producer that settles its rollout as a final zero marks it explicitly: a registered failure class, final,
    # and the marker. Only that complete form may carry these collector keys; the marker alone reopens nothing.
    final_zero_markers = (
        {NG_FAILURE_CLASS_KEY, NG_TERMINAL_KEY, NG_FAILURE_COUNTS_AS_ZERO_KEY, *NG_FAILURE_REPORT_KEYS}
        if result.get(NG_FAILURE_COUNTS_AS_ZERO_KEY) is True
        and result.get(NG_TERMINAL_KEY) is True
        and isinstance(result.get(NG_FAILURE_CLASS_KEY), str)
        else set()
    )
    reserved = sorted(
        key for key in result if _is_collector_key(key) and key not in judge_markers | final_zero_markers
    )
    if reserved:
        raise ValueError(
            f"environment server result for task {task_id!r} uses keys reserved for rollout collection: {reserved}"
        )
    record = dict(result)
    record[NG_TASK_ID_KEY] = task_id
    reply_report = _reply_failure_report(response)
    if reply_report:
        record = FailureReportRow.merged_into(record, *reply_report)
    return record


def _native_episode_record(row: Dict, response: Any) -> Dict:
    """Validate the dispatched episode identity before translating its outcome."""
    try:
        if not _is_episode_response(response):
            raise ValueError("Environment server must return an episode result or failure.")
        request = _episode_request_body(row)
        if EpisodeId.model_validate(response["episode_id"]) != EpisodeId.model_validate(request["episode_id"]):
            raise StaleRolloutResult("Episode response does not match the dispatched rollout attempt.")
        if TaskId.model_validate(response["task_id"]) != TaskId.model_validate(row["task_id"]):
            raise ValueError("Episode response does not match the dispatched task.")
        if (response.get("result") is None) == (response.get("failure") is None):
            raise ValueError("Episode response requires exactly one result or failure.")
        return _episode_record(response)
    except StaleRolloutResult:
        raise
    except (ValueError, TypeError) as error:
        raise InvalidRolloutResult(str(error)) from error


@dataclass(frozen=True)
class _CompletedRollout:
    """A finished ``/run`` dispatch, with timing carried alongside (not inside) the raw result."""

    row: Dict[str, Any]
    result: Dict[str, Any] | RolloutFailureRecord
    rollout_latency_ms: Optional[float]
    # Actual generation retained for judge-only recovery, outside RolloutFailureRecord.
    verification_response: Optional[Dict[str, Any]] = None
    diagnostics: Optional[Dict[str, Any]] = None
    environment_server: Optional[str] = None
    environment_server_type: Optional[str] = None
    # The `/run` wall clock in seconds, success or failure; None for a row that never started.
    elapsed_s: Optional[float] = None
    dispatch_span_context: Optional["SpanContext"] = None
    # The failure report the `/run` reply forwarded, or Collection's own for a failed dispatch. Only the writer
    # saves it, so run_examples yields the same result as before.
    failure_report: Optional[FailureReport] = None
    # A failed `/run` with route_failures_to_sidecar off: the writer keeps evidence of it, then raises this error.
    run_error: Optional[BaseException] = None


def _collection_failure_report(exc: BaseException, failure_class: str, headers: Any) -> FailureReport:
    """Collection's own failure for a failed `/run` dispatch, caused by the dispatch error, with the report the
    reply forwarded. Routing classes remain outcome metadata; this diagnostic has a registered collection kind."""
    failure = NgFailureEvent(
        boundary=FailureBoundary.COLLECTION,
        failure_kind=COLLECTION_DISPATCH_FAILED,
        failure_reason=f"Rollout dispatch failed: {type(exc).__name__}",
        operation="collection.run",
    )
    failure.__cause__ = exc
    report = failure_report_for(failure)
    received = FailureHeader.decode(headers)
    return report.merged(received) if received is not None else report


def _round_robin_by_agent(examples: List[Dict]) -> List[Dict]:
    """Interleave resolved rows by environment server or agent, preserving each input order.

    Agent order is the order in which each name first appears. Rows are returned unchanged, so
    task/rollout indexes, repeat seeds, and the identities used by resume remain intact.
    """
    queues: Dict[str, deque[Dict]] = {}
    for row in examples:
        dispatch_name = RolloutCollectionHelper._dispatch_name(row)
        queues.setdefault(dispatch_name, deque()).append(row)

    interleaved: List[Dict] = []
    while queues:
        for agent_name in list(queues):
            interleaved.append(queues[agent_name].popleft())
            if not queues[agent_name]:
                del queues[agent_name]
    return interleaved


def _batch_status_row(result: Dict, agent_refs_by_key: Dict[Tuple[int, int], Dict]) -> Dict:
    """Keep status identities and failure classes without retaining rollout payloads."""
    status_row = {
        key: result[key]
        for key in (AGENT_REF_KEY_NAME, TASK_INDEX_KEY_NAME, ROLLOUT_INDEX_KEY_NAME, NG_FAILURE_CLASS_KEY)
        if key in result
    }
    if AGENT_REF_KEY_NAME not in status_row:
        key = (result.get(TASK_INDEX_KEY_NAME), result.get(ROLLOUT_INDEX_KEY_NAME))
        if key in agent_refs_by_key:
            status_row[AGENT_REF_KEY_NAME] = agent_refs_by_key[key]
    return status_row


def _nonnegative_int(value: Any) -> Optional[int]:
    return value if type(value) is int and value >= 0 else None


def _has_observation_gap(result: dict[str, Any], code: str) -> bool:
    for key in (NG_TRAJECTORY_KEY, NG_AGENT_OBSERVATIONS_KEY):
        observations = result.get(key)
        gaps = observations.get("gaps") if isinstance(observations, dict) else None
        if isinstance(gaps, list) and any(isinstance(gap, dict) and gap.get("code") == code for gap in gaps):
            return True
    return False


def _trajectory_identity(row: dict[str, Any]) -> tuple[str, str]:
    identity = row.get("trajectory_identity")
    if isinstance(identity, dict):
        task_id = identity.get("task_id")
        rollout_id = identity.get("rollout_id")
        if isinstance(task_id, str) and task_id and isinstance(rollout_id, str) and rollout_id:
            return task_id, rollout_id
    task_id = next(
        (str(row[key]) for key in TASK_ID_FIELDS if row.get(key) is not None),
        str(row[TASK_INDEX_KEY_NAME]),
    )
    rollout_id = maybe_rollout_id_from_run_body(row) or f"{row[TASK_INDEX_KEY_NAME]}-{row[ROLLOUT_INDEX_KEY_NAME]}"
    return task_id, rollout_id


def _turn_content(request: Any, response: Any) -> tuple[Any, Any, Any, int]:
    """Split one captured model call into the turn's question, answer, reasoning, and tool-call count.

    Handles Responses API output items and chat-completions messages, the two dialects the Model
    Server captures.
    """
    question = request.get("input", request.get("messages")) if isinstance(request, dict) else request
    if isinstance(request, dict) and isinstance(request.get("input"), list):
        # Responses API input messages may omit `type`, which defaults to "message"; turns state it.
        question = [
            {"type": "message", **item} if isinstance(item, dict) and "role" in item and "type" not in item else item
            for item in request["input"]
        ]
    if isinstance(response, dict) and isinstance(response.get("output"), list):
        output = [item for item in response["output"] if isinstance(item, dict)]
        reasoning = [item for item in output if item.get("type") == "reasoning"] or None
        answer = [item for item in output if item.get("type") != "reasoning"]
        tool_calls = sum(1 for item in answer if item.get("type") == "function_call")
        return question, answer, reasoning, tool_calls
    choices = response.get("choices") if isinstance(response, dict) else None
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message") or {}
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        answer = {key: value for key, value in message.items() if key not in ("reasoning_content", "reasoning")}
        return question, answer, reasoning, len(message.get("tool_calls") or [])
    return question, None, None, 0


def _turns_from_model_calls(
    task_id: str,
    rollout_id: str,
    invocations: list[AgentInvocation],
    model_calls: list[TrajectoryModelCall],
    resolved: Any,
    *,
    verifier_invocations: Iterable[VerifierInvocation] = (),
) -> list[TrajectoryTurn]:
    """Build one turn per captured model call that returned a response, for an agent that sent no trajectory.

    A call belongs to the invocation whose ``model_calls`` reference it. Unreferenced calls stay
    in the raw evidence but do not become turns: even a single-agent rollout may capture judge
    or auxiliary calls that do not belong to the agent. Explicit verifier ownership also excludes
    a call from turns when an agent incorrectly references it; retain both ownership claims for validation.
    """
    verifier_call_ids = {
        ref.model_call_id for invocation in verifier_invocations for ref in invocation.model_calls if ref.model_call_id
    }
    invocation_by_call_id: dict[str, str] = {}
    invocation_by_response_id: dict[str, str] = {}
    for invocation in invocations:
        for ref in invocation.model_calls:
            if ref.model_call_id:
                invocation_by_call_id[ref.model_call_id] = invocation.invocation_id
            if ref.response_id:
                invocation_by_response_id[ref.response_id] = invocation.invocation_id

    turns: list[TrajectoryTurn] = []
    turn_counts: Counter = Counter()
    tool_counts: Counter = Counter()
    for call in model_calls:
        metadata = call.response_metadata
        if call.model_call_id in verifier_call_ids:
            continue
        if call.response is None or (metadata.status_code is not None and metadata.status_code >= 400):
            # HTTP error payloads are retained attempts, not model decisions.
            continue
        invocation_id = invocation_by_call_id.get(call.model_call_id or "") or invocation_by_response_id.get(
            metadata.response_id or ""
        )
        if invocation_id is None:
            continue
        question, answer, reasoning, tool_calls = _turn_content(call.request, call.response)
        turn_counts[invocation_id] += 1
        turns.append(
            TrajectoryTurn(
                invocation_id=invocation_id,
                task_id=task_id,
                rollout_id=rollout_id,
                turn_no=turn_counts[invocation_id],
                timestamp=call.started_at or call.completed_at or 0.0,
                question=question,
                answer=answer,
                reasoning_content=reasoning,
                step_count=tool_counts[invocation_id],
                model_calls=[
                    ModelCallRef(
                        model_call_id=call.model_call_id,
                        model_ref=metadata.model_ref,
                        response_id=metadata.response_id,
                    )
                ],
            )
        )
        tool_counts[invocation_id] += tool_calls
    if turns and isinstance(resolved, bool):
        turns[-1] = turns[-1].model_copy(update={"resolved": resolved})
    return turns


def _build_trajectory_record(row: dict[str, Any], result: dict[str, Any]) -> TrajectoryRecord:
    task_id, rollout_id = _trajectory_identity(row)
    gaps: list[ObservationGap] = []
    invocations: list[AgentInvocation] = []
    verifier_invocations: list[VerifierInvocation] = []
    turns: list[TrajectoryTurn] = []
    tools: list[TrajectoryToolCall] = []
    model_calls: list[TrajectoryModelCall] = []

    producer_turns_observed = False
    raw_trajectory = result.get(NG_TRAJECTORY_KEY)
    if isinstance(raw_trajectory, dict):
        try:
            trajectory = TrajectoryRecord.model_validate(raw_trajectory)
            producer_turns_observed = True
            mismatches = [
                field
                for field, producer, canonical in (
                    ("task_id", trajectory.task_id, task_id),
                    ("rollout_id", trajectory.rollout_id, rollout_id),
                )
                if producer != canonical
            ]
            if mismatches:
                gaps.append(ObservationGap(code="producer_trajectory_identity_mismatch", detail=",".join(mismatches)))
                turns = [
                    turn.model_copy(update={"task_id": task_id, "rollout_id": rollout_id}) for turn in trajectory.turns
                ]
            else:
                turns = trajectory.turns
            gaps.extend(trajectory.gaps)
            invocations = trajectory.invocations
            verifier_invocations = trajectory.verifier_invocations
            tools = trajectory.tool_calls
            model_calls = trajectory.model_calls
        except Exception as exc:
            gaps.append(ObservationGap(code="producer_trajectory_invalid", detail=type(exc).__name__))

    observations_read = False
    raw_observations = result.get(NG_AGENT_OBSERVATIONS_KEY)
    if raw_observations is not None:
        try:
            observations = AgentObservationBundle.model_validate(raw_observations)
            gaps.extend(observations.gaps)
            observed_invocations = [record for record in observations.records if isinstance(record, AgentInvocation)]
            producer_invocation_ids = {record.invocation_id for record in invocations}
            invocations.extend(
                record for record in observed_invocations if record.invocation_id not in producer_invocation_ids
            )
            observed_tools = [record for record in observations.records if isinstance(record, ToolCallObservation)]
            if observed_tools:
                outputs = {
                    (invocation.invocation_id, item.call_id): item.output
                    for invocation in invocations
                    for item in invocation.conversation
                    if getattr(item, "type", None) in {"function_call_output", "custom_tool_call_output"}
                }
                positions = {(tool.invocation_id, tool.tool_call_id): index for index, tool in enumerate(tools)}
                for observed in observed_tools:
                    key = (observed.invocation_id, observed.tool_call_id)
                    position = positions.get(key)
                    existing = tools[position] if position is not None else None
                    merged = existing.model_dump(mode="json") if existing is not None else {}
                    update = observed.model_dump(mode="json", exclude_none=True)
                    if existing is not None and observed.status == "unknown" and existing.status != "unknown":
                        update.pop("status", None)
                    merged.update(update)
                    if key in outputs:
                        merged["output"] = outputs[key]
                    projected = TrajectoryToolCall.model_validate(merged)
                    if position is None:
                        positions[key] = len(tools)
                        tools.append(projected)
                    else:
                        tools[position] = projected
            observations_read = True
        except Exception as exc:
            gaps.append(ObservationGap(code="agent_observations_invalid", detail=type(exc).__name__))

    # Without a valid producer trajectory or valid observations, the empty tool list below is the collector's
    # default, not a report that no tool ran.
    if not producer_turns_observed and not observations_read:
        gaps.append(ObservationGap(code="tool_timing_unavailable"))

    turns.sort(key=lambda turn: (turn.timestamp, turn.invocation_id, turn.turn_no))

    capture = result.get("ng_model_call_capture")
    capture = capture if isinstance(capture, dict) else {}
    raw_calls = capture.get("calls") or []
    model_call_positions = {
        call.model_call_id: index for index, call in enumerate(model_calls) if call.model_call_id is not None
    }
    for raw_call in raw_calls:
        if not isinstance(raw_call, dict):
            continue
        model_call_id = raw_call.get("model_call_id")
        metadata = {
            key: raw_call[key]
            for key in (
                "response_id",
                "model_ref",
                "model",
                "dialect",
                "status_code",
                "response_status",
                "finish_reason",
                "upstream_attempted",
                "response_source",
                "upstream_status_code",
                "local_response_reason",
                "error_category",
                "latency_ttft_ms",
            )
            if raw_call.get(key) is not None
        }
        response = raw_call.get("response")
        if isinstance(response, dict) and isinstance(response.get("status"), str):
            metadata.setdefault("response_status", response["status"])
        projected = TrajectoryModelCall(
            model_call_id=model_call_id,
            client_session_id=raw_call.get("client_session_id"),
            client_assistant_message_id=raw_call.get("client_assistant_message_id"),
            started_at=raw_call.get("started_at"),
            completed_at=raw_call.get("completed_at"),
            duration_ms=raw_call.get("latency_total_ms"),
            request=raw_call.get("request") if raw_call.get("request") is not None else raw_call.get("request_raw"),
            response=raw_call.get("response")
            if raw_call.get("response") is not None
            else raw_call.get("response_raw"),
            response_metadata=metadata,
            token_stats=TrajectoryTokenStats(
                prompt_tokens=_nonnegative_int(raw_call.get("tokens_in")),
                completion_tokens=_nonnegative_int(raw_call.get("tokens_out")),
                reasoning_tokens=_nonnegative_int(raw_call.get("tokens_reasoning")),
                total_tokens=_nonnegative_int(raw_call.get("tokens_total")),
                cached_tokens=_nonnegative_int(raw_call.get("cached_tokens")),
            ),
        )
        position = model_call_positions.pop(model_call_id, None) if model_call_id is not None else None
        if position is None:
            model_calls.append(projected)
        else:
            merged = model_calls[position].model_dump(mode="json")
            update = projected.model_dump(mode="json", exclude_none=True)
            for key in ("response_metadata", "token_stats"):
                merged[key].update(update.pop(key))
            merged.update(update)
            projected = TrajectoryModelCall.model_validate(merged)
            model_calls[position] = projected

    # Join returned verifier outcomes with model-call references reconstructed from capture.
    for raw_invocation in [
        *(capture.get("verifier_invocations") or []),
        *(result.get(NG_VERIFIER_INVOCATIONS_KEY) or []),
    ]:
        try:
            invocation = VerifierInvocation.model_validate(raw_invocation)
            existing = next((i for i in verifier_invocations if i.invocation_id == invocation.invocation_id), None)
            if existing is None:
                verifier_invocations.append(invocation)
            else:
                existing.model_calls.extend(ref for ref in invocation.model_calls if ref not in existing.model_calls)
                for field in ("grading_operations", "sandbox_observations"):
                    returned = getattr(invocation, field)
                    if returned is None:
                        continue
                    observed = getattr(existing, field)
                    if observed is None:
                        setattr(existing, field, returned)
                    elif observed != returned:
                        # Contradictory returned facts cannot establish authoritative verifier coverage.
                        gaps.append(ObservationGap(code=f"verifier_{field}_conflict"))
                        setattr(existing, field, [])
                if invocation.status != "unknown":
                    existing.status = invocation.status
                    existing.failure_kind = invocation.failure_kind
        except Exception:
            gaps.append(ObservationGap(code="verifier_invocation_invalid"))

    for raw_gap in capture.get("gaps") or []:
        if isinstance(raw_gap, dict):
            try:
                gaps.append(ObservationGap.model_validate(raw_gap))
            except Exception:
                gaps.append(ObservationGap(code="model_call_capture_gap_invalid"))
    if not isinstance(raw_trajectory, dict):
        # An agent that sends its own trajectory owns its turns, and an empty list there means no turn
        # completed. Otherwise, build turns only from calls explicitly referenced by invocations.
        turns = _turns_from_model_calls(
            task_id,
            rollout_id,
            invocations,
            model_calls,
            result.get("resolved"),
            verifier_invocations=verifier_invocations,
        )
    if not model_calls:
        gaps.append(ObservationGap(code="model_calls_unavailable"))
    if not turns and not producer_turns_observed:
        # A producer that published a trajectory and reported no turns has told
        # us something; only the absence of a producer leaves turns unavailable.
        # Without this, `rollout_missing_agent_turns` is skipped in exactly the
        # case it exists to catch.
        gaps.append(ObservationGap(code="turns_unavailable"))
    if not any(invocation.conversation for invocation in invocations):
        gaps.append(ObservationGap(code="conversation_unavailable"))

    return TrajectoryRecord(
        task_id=task_id,
        rollout_id=rollout_id,
        invocations=invocations,
        verifier_invocations=verifier_invocations,
        turns=turns,
        model_calls=model_calls,
        tool_calls=tools,
        gaps=list({(gap.code, gap.invocation_id, gap.detail): gap for gap in gaps}.values()),
    )


NG_POLICY_MODEL_SERVER_KEY = "ng_policy_model_server"


def _record_policy_model_server(result: dict[str, Any], global_config: Any, agent_name: str | None) -> None:
    """Name the agent's configured policy server on the record (landed or failed), so offline timing needs no
    live config and a capture joined later from the capture dir is still partitioned by role."""
    if not agent_name:
        return
    try:
        model_server = get_first_server_config_dict(global_config, agent_name).get("model_server")
    except (KeyError, IndexError, TypeError, AttributeError):
        return
    name = model_server.get("name") if hasattr(model_server, "get") else None
    if isinstance(name, str) and name:
        result[NG_POLICY_MODEL_SERVER_KEY] = name


def _attach_trajectory_record(row: dict[str, Any], result: dict[str, Any]) -> None:
    try:
        result[NG_TRAJECTORY_KEY] = _build_trajectory_record(row, result).model_dump(mode="json")
    except Exception as exc:
        result.pop(NG_TRAJECTORY_KEY, None)
        logger.warning("Could not project standardized trajectory evidence.", exc_info=True)
        gap = ObservationGap(code="trajectory_projection_failed", detail=type(exc).__name__).model_dump(
            mode="json", exclude_none=True
        )
        target = result.get("ng_model_call_capture")
        if not isinstance(target, dict):
            target = result.get(NG_AGENT_OBSERVATIONS_KEY)
        gap_attached = False
        if isinstance(target, dict):
            gaps = target.setdefault("gaps", [])
            if isinstance(gaps, list):
                gaps.append(gap)
                gap_attached = True
        if not gap_attached:
            try:
                task_id, rollout_id = _trajectory_identity(row)
                result[NG_TRAJECTORY_KEY] = TrajectoryRecord(
                    task_id=task_id,
                    rollout_id=rollout_id,
                    gaps=[ObservationGap.model_validate(gap)],
                ).model_dump(mode="json")
            except Exception:
                logger.warning("Could not retain the trajectory projection failure gap.", exc_info=True)
    else:
        # Raw capture payloads remain as a fallback on failure. After success,
        # ng_trajectory owns them, so remove only the duplicate copies.
        _strip_capture_payloads(result)


def _build_ng_perf(result: dict[str, Any], *, rollout_latency_ms: Optional[float]) -> Optional[dict[str, Any]]:
    """Assemble the per-rollout ``ng_perf`` summary from ``ng_trajectory``.

    Returns ``None`` (``ng_perf`` stays absent) unless at least one reasoning turn was
    observed: per-turn evidence is needed rather than just raw model-call capture,
    so a rollout collected with observability disabled produces no ``ng_perf`` at all.

    Token fields are summed over every model call referenced by a reasoning-turn
    ``AgentInvocation``. This includes compaction calls whenever the harness also lists
    them in ``AgentInvocation.model_calls``.

    ``num_turns`` counts reasoning turns summed across all invocations (an ``AgentInvocation``
    is one root-agent or subagent conversation that may span many turns). Each invocation
    contributes its explicit ``TrajectoryTurn`` count when the harness emits turn records,
    falling back to its owned model-call count (one assistant response per turn), then to 1
    (an invocation that ran had at least one turn) -- so hybrid trajectories where only some
    invocations report turns still count every conversation.

    ``token_observability_coverage`` reports what fraction of those turns actually resolved to a
    captured call: a turn whose ``ModelCallRef`` was unmatched or ambiguous silently loses its
    tokens from the sums below, and this is the only signal that it happened.
    """
    raw_trajectory = result.get(NG_TRAJECTORY_KEY)
    if not isinstance(raw_trajectory, dict):
        return None
    try:
        trajectory = TrajectoryRecord.model_validate(raw_trajectory)
    except Exception:
        return None
    if not trajectory.invocations:
        return None

    # Index captured calls by both identities ModelCallRef supports, mirroring
    # join_agent_model_calls: a ref may carry model_call_id, or the exact
    # (model_ref, response_id) pair.
    calls_by_id: dict[str, list[int]] = {}
    calls_by_response: dict[tuple[str, str, str], list[int]] = {}
    for index, call in enumerate(trajectory.model_calls):
        if call.model_call_id:
            calls_by_id.setdefault(call.model_call_id, []).append(index)
        call_model_ref = call.response_metadata.model_ref
        call_response_id = call.response_metadata.response_id
        if call_model_ref is not None and call_response_id:
            calls_by_response.setdefault((call_model_ref.type, call_model_ref.name, call_response_id), []).append(
                index
            )

    def _match_call_index(ref: ModelCallRef) -> Optional[int]:
        if ref.model_call_id:
            candidates = [
                index
                for index in calls_by_id.get(ref.model_call_id, [])
                if (
                    ref.model_ref is None or ref.model_ref == trajectory.model_calls[index].response_metadata.model_ref
                )
                and (
                    ref.response_id is None
                    or ref.response_id == trajectory.model_calls[index].response_metadata.response_id
                )
            ]
        elif ref.model_ref is not None and ref.response_id:
            candidates = calls_by_response.get((ref.model_ref.type, ref.model_ref.name, ref.response_id), [])
        else:
            candidates = []
        return candidates[0] if len(candidates) == 1 else None

    # De-duplicate by matched call *position* rather than model_call_id; the id is
    # optional because a (model_ref, response_id)-only match has none. One physical call
    # can't be claimed twice even if two invocations' refs both resolve to it (e.g. a
    # join_agent_model_calls conflict that left a ref attached to its losing
    # invocation, unremoved, just gap-flagged).
    seen_positions: set[int] = set()
    owned_calls = []
    owned_calls_by_invocation: Counter = Counter()
    for invocation in trajectory.invocations:
        for ref in invocation.model_calls:
            index = _match_call_index(ref)
            if index is None or index in seen_positions:
                continue
            seen_positions.add(index)
            owned_calls.append(trajectory.model_calls[index])
            owned_calls_by_invocation[invocation.invocation_id] += 1
    tool_calls_by_invocation = Counter(tool.invocation_id for tool in trajectory.tool_calls)
    num_tool_calls = sum(
        tool_calls_by_invocation.get(invocation.invocation_id, 0) for invocation in trajectory.invocations
    )

    def _sum_tokens(attr: str) -> Optional[int]:
        values = [
            getattr(call.token_stats, attr) for call in owned_calls if getattr(call.token_stats, attr) is not None
        ]
        return sum(values) if values else None

    turns_by_invocation = Counter(turn.invocation_id for turn in trajectory.turns)
    num_turns = sum(
        turns_by_invocation.get(invocation.invocation_id, 0)
        or owned_calls_by_invocation.get(invocation.invocation_id, 0)
        or 1
        for invocation in trajectory.invocations
    )
    ng_perf: dict[str, Any] = {
        "num_turns": num_turns,
        "num_tool_calls": num_tool_calls,
        "token_observability_coverage": min(1.0, len(owned_calls) / num_turns),
    }
    for ng_perf_key, token_stats_attr in (
        ("prompt_tokens", "prompt_tokens"),
        ("cached_prompt_tokens", "cached_tokens"),
        ("completion_tokens", "completion_tokens"),
        ("reasoning_tokens", "reasoning_tokens"),
    ):
        value = _sum_tokens(token_stats_attr)
        if value is not None:
            ng_perf[ng_perf_key] = value

    if isinstance(rollout_latency_ms, (int, float)):
        ng_perf["total_latency_ms"] = rollout_latency_ms

    return ng_perf


def _attach_ng_perf(
    result: dict[str, Any], *, observability_enabled: bool, rollout_latency_ms: Optional[float] = None
) -> None:
    if not observability_enabled:
        # ng_perf stays absent entirely when observability is off (OQ4): a caller who
        # disabled it only wants the final score, not partial/best-effort perf evidence.
        return
    try:
        ng_perf = _build_ng_perf(result, rollout_latency_ms=rollout_latency_ms)
    except Exception:
        logger.warning("Could not assemble ng_perf for a rollout.", exc_info=True)
        return
    if ng_perf is not None:
        result[NG_PERF_KEY] = ng_perf


get_max_rollout_attempts = _get_max_rollout_attempts  # Public compatibility alias


def _normalize_health_check_ignored_checks(value) -> List[str]:
    from nemo_gym.rollout_health import normalize_ignored_checks

    return list(normalize_ignored_checks(value))


class SharedRolloutCollectionConfig(
    RemovedRolloutOptionsMixin, UploadRolloutsConfigMixin, RunHealthConfig, BaseNeMoGymCLIConfig
):
    output_jsonl_fpath: str = Field(description="The output data jsonl file path.")
    batch_manifest_fpath: Optional[str] = Field(
        default=None,
        description=(
            "Optional Eval Factory batch_manifest.json. Gym validates it against materialized inputs and writes "
            "batch_status.json alongside it."
        ),
    )
    require_complete: bool = Field(
        default=False, description="Fail on missing rollouts; enabled by default by eval submit."
    )
    num_samples_in_parallel: Optional[int] = Field(
        default=None,
        gt=0,
        description=(
            "Limit concurrent requests. Must be positive when set. "
            "If max_resident_rollout_tasks is set, active requests cannot exceed either limit."
        ),
    )
    max_resident_rollout_tasks: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Maximum number of rollout tasks resident in the driver at once. "
            "When unset, no driver-side task admission limit is applied."
        ),
    )
    retain_results_in_memory: bool = Field(
        default=True,
        description=(
            "Retain completed rollout rows and results in driver memory and return the full ordered result list. "
            "When false, completed results are not retained during collection, are persisted incrementally, and "
            "run_from_config returns an empty list. Aggregation and upload may still load all applicable results."
        ),
    )
    responses_create_params: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Overrides for responses_create_params, e.g. temperature or max_output_tokens. "
            "Metadata mappings merge by key with overrides winning; metadata=null clears row metadata. "
            "Other fields and individual metadata values are replaced."
        ),
    )
    disable_aggregation: bool = Field(
        default=False,
        description=(
            "Skip the post-rollout aggregate-metrics computation and file write. "
            "Used when sharding rollouts across multiple jobs that will be aggregated together "
            "afterward by `gym eval aggregate`."
        ),
    )
    disable_health_check: bool = Field(
        default=False,
        description="Skip post-aggregation rollout quality verification and report writing.",
    )
    health_check_workers: Optional[int] = Field(
        default=None,
        ge=1,
        description="Number of rollout-health worker processes (defaults to min(cpus, 8)).",
    )
    health_check_ignored_checks: List[str] = Field(
        default_factory=list,
        description="Health-check IDs to exclude from execution and verdict derivation.",
    )

    @field_validator("health_check_ignored_checks", mode="before")
    @classmethod
    def _validate_health_check_ignored_checks(cls, value):
        return _normalize_health_check_ignored_checks(value)

    count_failure_classes_as_zero: List[str] = Field(
        default_factory=list,
        description=(
            "Failure classes from the failures sidecar to count in aggregate metrics, e.g. "
            "['agent_run_error'], so a failed rollout lands in the denominator. A row that carries "
            "no reward is scored zero for the metrics only; no artifact is changed. "
            "Matches the canonical class or the saved original legacy producer class."
        ),
    )

    count_missing_rollouts_as_zero: bool = Field(
        default=False,
        description=(
            "Count a materialized rollout that produced no row at all as a zero in aggregate "
            "metrics. Covers what the failure classes cannot: a rollout killed mid-flight leaves "
            "nothing in either file, so without this it leaves the denominator too and the score "
            "reads higher than the run earned. A row dispatch_budget_s never started leaves nothing "
            "either and is counted the same way: resume_from_cache still dispatches it, and the "
            "resumed run scores it. Scores the metrics only; no artifact is changed. With "
            "disable_aggregation this run scores nothing, so a run with no row still fails. "
            "Off by default. Needs the run's materialized inputs; a shard without them is warned "
            "about and skipped. A rollout whose latest attempt has a known failure is not counted here, "
            "whatever its class, so this does not override count_failure_classes_as_zero. The "
            "count is reported separately as coverage/imputed. Known limitation, shared with "
            "count_failure_classes_as_zero: the added row carries its task's dataset fields, but no "
            "response and no field its verifier would have computed, so a benchmark metric hook "
            "that needs those cannot score it, and a few score it as a measurement instead of "
            "skipping it."
        ),
    )

    route_failures_to_sidecar: bool = Field(
        default=False,
        description=(
            "Record a failed agent /run as a failures-sidecar row and keep going, instead of ending "
            "the run. The failed rollouts are then absent from the rollouts jsonl and from the "
            "score, which is reported over fewer rollouts than were dispatched. Off, the first failed "
            "/run ends the run, after an omitted-rollout row records it (not scored, not an attempt)."
        ),
    )
    rollout_collection_driver: Optional[str] = Field(
        default=None,
        description=(
            "Optional dotted ``module.path:function`` to run rollout collection instead of the "
            "built-in helper. Lets a benchmark plug in a custom procedure (e.g. an adaptive, "
            "multi-pass run) while still producing the standard rollout + aggregate-metrics "
            "artifacts. The function is awaited with (rollout_collection_config, global_config_dict). "
            "When unset, the standard single-pass collection runs. A driver must honour "
            "``resume_from_cache`` itself; one that ignores it restarts from zero on an auto-resumed job."
        ),
    )
    environment_routing_mode: Literal["agent", "legacy", "taskset"] = Field(
        default="agent",
        description=(
            "How flat (non-materialized) rows are routed. `agent`: today's routing, each row through its "
            "agent's environment server. `legacy`: every flat row to `environment_server_name`. `taskset`: "
            "flat rows are rejected, so the run takes only materialized rows. Materialized rows (`task_id.taskset` plus "
            "`task_input`) always route by `environment_server_routes`, in every mode, so one batch may mix "
            "materialized and compatibility-routed tasksets."
        ),
    )
    environment_server_name: str | None = Field(
        default=None,
        description="Compatibility environment server used for every flat row when environment_routing_mode=legacy.",
    )
    environment_server_routes: dict[str, str] = Field(
        default_factory=dict,
        description="Environment server deployments keyed by materialized TaskId.taskset.",
    )

    @model_validator(mode="after")
    def validate_environment_routing(self) -> "SharedRolloutCollectionConfig":
        if self.environment_routing_mode == "legacy" and self.environment_server_name is None:
            raise ValueError("environment_server_name is required when environment_routing_mode=legacy")
        if self.environment_routing_mode == "taskset" and not self.environment_server_routes:
            raise ValueError("environment_server_routes are required when environment_routing_mode=taskset")
        return self

    def check_completion(
        self,
        *,
        expected: int,
        results: List[Dict[str, Any]],
        completed: int | None = None,
        intentionally_omitted: int = 0,
        retryable: bool,
        made_progress: bool,
        final_zero_rows: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        """Reject unfinished work; allow omissions alongside completed tasks.

        Drivers must supply whether any tasks can still be retried and whether
        this invocation saved outcomes that made progress (not just interruptions). Assuming either can cause endless
        automatic restarts when a driver has no work left or makes no progress.

        ``final_zero_rows`` are failure rows that settle their rollout as a scored zero (see
        ``_final_zero_failure_rows``): resume does not run them again, so they are completed samples. A caller that
        passes ``completed`` has already counted them.
        """
        if not self.require_complete and not intentionally_omitted:
            return
        if completed is None:
            completed = len(
                {
                    (r.get("stage_index"), r[TASK_INDEX_KEY_NAME], r[ROLLOUT_INDEX_KEY_NAME])
                    for r in results
                    if r.get(NG_FAILURE_CLASS_KEY) is None and not r.get(NG_NO_PERSIST_KEY)
                }
                | {
                    (r.get("stage_index"), r[TASK_INDEX_KEY_NAME], r[ROLLOUT_INDEX_KEY_NAME])
                    for r in final_zero_rows
                    if r.get(NG_FAILURE_COUNTS_AS_ZERO_KEY) is True
                }
            )
        if has_only_intentional_gaps(
            expected=expected, completed=completed, intentionally_omitted=intentionally_omitted
        ):
            logger.warning(
                "Evaluation accepted with intentional omissions: %s/%s samples completed, %s intentionally omitted. "
                "Omitted tasks have no measured score.",
                completed,
                expected,
                intentionally_omitted,
            )
            return
        if not self.require_complete:
            return
        if completed < expected:
            raise IncompleteEvaluationError(
                f"EVAL FAILED: {completed}/{expected} samples completed. "
                f"Partial artifacts retained at {self.output_jsonl_fpath}."
                + (" No tasks are eligible for retry." if not retryable else ""),
                retryable=retryable,
                made_progress=made_progress,
            )


class E2ERolloutCollectionConfig(SharedRolloutCollectionConfig):
    """
    Spin up all necessary servers and perform a batch of rollout collection using each dataset inside the provided configs.

    Examples:

    ```bash
    gym eval run \
        +output_jsonl_fpath=weather_rollouts.jsonl \
        +num_samples_in_parallel=10
    ```
    """

    split: Union[Literal["train"], Literal["validation"], Literal["benchmark"]]
    reuse_existing_data_preparation: bool = False

    @model_validator(mode="before")
    @classmethod
    def _reject_input_jsonl_fpath(cls, data):
        # This config has no input_jsonl_fpath field, so pydantic would silently drop it and
        # e2e collection would overwrite it with the prepared split path — the user's file
        # would be ignored without any indication. Match on Mapping, not dict: the CLI passes
        # an OmegaConf DictConfig, which is a Mapping but not a dict.
        if isinstance(data, Mapping) and "input_jsonl_fpath" in data:
            raise ConfigError(
                "`input_jsonl_fpath` (-i/--input) is not supported when serving end-to-end: the input is "
                "always the prepared dataset for the requested split. Either add --no-serve to collect "
                "rollouts from your own input file against already-running servers, or drop -i/--input "
                "to use the prepared data."
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def _reject_example_split(cls, data):
        # `example` is a real dataset type but deliberately not a runnable split: example
        # datasets are the committed smoke-test samples the PR data gate validates, and they
        # are excluded from prepared splits so they never leak into training or eval data.
        # Catch it before the Literal check so the user gets the documented recipe instead of
        # a bare "Input should be 'train'".
        if isinstance(data, Mapping) and data.get("split") == "example":
            raise ConfigError(
                "`--split example` is not runnable end-to-end: example datasets are committed "
                "smoke-test samples, not prepared train/validation/benchmark splits. To run one, "
                "start the servers and point at the example file directly:\n"
                "  gym env start --resources-server <server> ...\n"
                "  gym eval run --no-serve --agent <agent> --input <server_dir>/data/example.jsonl --output <out>.jsonl\n"
                "See the Quickstart: https://docs.nvidia.com/nemo/gym/latest/get-started/quickstart"
            )
        return data


NG_ELAPSED_KEY = "elapsed_seconds"


class RunTimingReport:
    """Health table for a collection whose rollouts have a per-rollout wall-clock limit.

    A per-rollout limit (e.g. BrowseComp's ``rollout_timeout_s``) makes the score depend on the
    deployment as well as the model: if the servers are too few or too slow, rollouts start late,
    queue for the GPU, or run into the limit, and are scored 0 for reasons unrelated to the model.
    This table shows whether a run stayed clear of that, so a score from a starved run is not
    mistaken for a model result. Each row compares this run with a healthy bound:

    - start coverage: rollouts started within ``start_within_s`` and all started by
      ``all_started_within_s`` (late starts eat into a fixed job allocation);
    - rollouts ended by the wall-clock limit (above the bound, the limit is shaping the score);
    - rollouts longer than ``long_rollout_s`` (an early sign that the limit is too tight);
    - summed waiting / running requests on the model servers, when their URLs are given
      (requests queue instead of being served).

    Diagnostic only: it prints a warning per unhealthy row and never stops the run.
    """

    def __init__(
        self,
        *,
        total: int,
        start_within_s: float,
        start_min_fraction: float,
        all_started_within_s: Optional[float],
        long_rollout_s: Optional[float],
        long_rollout_max_fraction: float,
        timed_out_max_fraction: float,
        max_waiting_ratio: Optional[float] = None,
    ) -> None:
        self._total = total
        self._max_waiting_ratio = max_waiting_ratio
        self._waiting_sum = 0.0
        self._running_sum = 0.0
        self._peak_ratio = 0.0
        self._queue_samples = 0
        self._start_within_s = start_within_s
        self._start_min_fraction = start_min_fraction
        self._all_started_within_s = all_started_within_s if all_started_within_s is not None else start_within_s
        self._long_rollout_s = long_rollout_s
        self._long_rollout_max_fraction = long_rollout_max_fraction
        self._timed_out_max_fraction = timed_out_max_fraction
        self._t0 = time.monotonic()
        self._start_offsets: List[float] = []
        self._durations: List[float] = []
        self._timed_out = 0

    def record_start(self, now: float) -> None:
        self._start_offsets.append(now - self._t0)

    def record_queue(self, waiting: float, running: float) -> None:
        """One sample of requests waiting and running, summed over every server."""
        self._queue_samples += 1
        self._waiting_sum += waiting
        self._running_sum += running
        if running > 0:
            self._peak_ratio = max(self._peak_ratio, waiting / running)

    def record_result(self, result: Mapping[str, Any], seconds: float) -> None:
        self._durations.append(seconds)
        if classify_timeout(result):
            self._timed_out += 1

    def table(self) -> str:
        elapsed = time.monotonic() - self._t0
        started = len(self._start_offsets)
        all_started = started >= self._total
        window_min, all_by_min = self._start_within_s / 60, self._all_started_within_s / 60
        rows: List[Tuple[str, str, str, bool]] = []
        if all_started:
            started_by = max(self._start_offsets) / 60
            rows.append(
                (
                    "all rollouts started by",
                    f"{started_by:.1f} min",
                    f"<= {all_by_min:.0f} min",
                    started_by <= all_by_min,
                )
            )
        else:
            # Judged only once the deadline has passed; until then the count is progress, not a failure.
            rows.append(
                (
                    "rollouts started so far",
                    f"{started}/{self._total}",
                    f"all by {all_by_min:.0f} min",
                    elapsed <= self._all_started_within_s,
                )
            )
        within = sum(1 for o in self._start_offsets if o <= self._start_within_s) / self._total if self._total else 0.0
        window_open = not all_started and elapsed <= self._start_within_s
        rows.append(
            (
                f"started within {window_min:.0f} min" + (" (window open)" if window_open else ""),
                f"{within:.1%}",
                f">= {self._start_min_fraction:.0%}",
                window_open or within >= self._start_min_fraction,
            )
        )
        n = len(self._durations)
        if n:
            timed_out = self._timed_out / n
            rows.append(
                (
                    "hit wall-clock limit",
                    f"{timed_out:.1%} ({self._timed_out})",
                    f"<= {self._timed_out_max_fraction:.0%}",
                    timed_out <= self._timed_out_max_fraction,
                )
            )
            if self._long_rollout_s is not None:
                long = sum(1 for d in self._durations if d > self._long_rollout_s) / n
                rows.append(
                    (
                        f"longer than {self._long_rollout_s / 60:.0f} min",
                        f"{long:.1%}",
                        f"<= {self._long_rollout_max_fraction:.0%}",
                        long <= self._long_rollout_max_fraction,
                    )
                )
        if self._queue_samples and self._max_waiting_ratio is not None:
            ratio = self._waiting_sum / self._running_sum if self._running_sum else 0.0
            rows.append(
                (
                    "waiting / running requests",
                    f"{ratio:.2f} (peak {self._peak_ratio:.2f})",
                    f"<= {self._max_waiting_ratio:.2f}",
                    ratio <= self._max_waiting_ratio,
                )
            )
        width = max(len(r[0]) for r in rows)
        lines = ["Run timing summary (this run | healthy):"]
        for label, value, healthy, ok in rows:
            lines.append(f"  {label:<{width}} : {value:>12} | {healthy}" + ("" if ok else "   <-- WARNING"))
        return "\n".join(lines)


_NO_SIGTERM_HOOK = object()


def _sum_vllm_queue(metrics_text: str) -> Tuple[float, float]:
    waiting = running = 0.0
    for line in metrics_text.splitlines():
        if line.startswith("vllm:num_requests_waiting"):
            waiting += float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:num_requests_running"):
            running += float(line.rsplit(" ", 1)[1])
    return waiting, running


async def _sample_server_queues(report: "RunTimingReport", base_urls: List[str], interval_s: float) -> None:
    """Poll every vLLM server's /metrics and record requests waiting and running, summed over servers."""
    metrics_urls = [re.sub(r"/v1/?$", "", u.rstrip("/")) + "/metrics" for u in base_urls]
    while True:
        waiting = running = 0.0
        reached = 0
        for url in metrics_urls:
            try:
                resp = await request("GET", url, _max_num_tries=1, timeout=aiohttp.ClientTimeout(total=5))
                w, r = _sum_vllm_queue(await resp.text())
                waiting, running, reached = waiting + w, running + r, reached + 1
            except Exception:
                continue
        if reached:
            report.record_queue(waiting, running)
        await asyncio.sleep(interval_s)


def _print_on_sigterm(render: Callable[[], str]) -> Any:
    """Print render() when Slurm's time limit sends SIGTERM, then terminate as before.

    Python's default SIGTERM action exits without unwinding, so nothing printed in a ``finally``
    survives a time-limit kill. Returns the previous handler for ``_restore_sigterm``, or
    ``_NO_SIGTERM_HOOK`` when no hook could be installed.
    """
    try:
        loop = asyncio.get_running_loop()
        previous = signal.getsignal(signal.SIGTERM)

        def _on_sigterm() -> None:
            try:
                print(render(), flush=True)
            finally:
                loop.remove_signal_handler(signal.SIGTERM)
                signal.signal(signal.SIGTERM, previous if previous is not None else signal.SIG_DFL)
                os.kill(os.getpid(), signal.SIGTERM)

        loop.add_signal_handler(signal.SIGTERM, _on_sigterm)
        return previous
    except (NotImplementedError, RuntimeError, ValueError):
        return _NO_SIGTERM_HOOK


def _restore_sigterm(previous: Any) -> None:
    if previous is _NO_SIGTERM_HOOK:
        return
    try:
        asyncio.get_running_loop().remove_signal_handler(signal.SIGTERM)
        signal.signal(signal.SIGTERM, previous if previous is not None else signal.SIG_DFL)
    except (NotImplementedError, RuntimeError, ValueError):
        pass


class DispatchLatencyTracker:
    """Per-task latency observed by the dispatcher, and the drain margin from it.

    Rollouts/hr is the number operators watch, and on its own it is misleading:
    raising concurrency raises aggregate throughput while making every individual
    task slower, so the run looks healthier right up until tasks start breaching
    their per-task timeout. Reporting the latency percentiles next to the rate
    makes that trade visible while there is still time to react to it.
    """

    def __init__(self) -> None:
        # Kept sorted: the adaptive drain margin reads a quantile before every dispatch.
        self._durations: List[float] = []
        self._drained = 0

    def record(self, seconds: float) -> None:
        if seconds > 0:
            bisect.insort(self._durations, seconds)

    def record_failure(self, seconds: float) -> None:
        """Record a failed attempt only when it ran at least as long as the median so far.

        A task that timed out or died late shows how long tasks can run, so it can raise the
        drain margin. One that failed at once says nothing about duration and would drag the
        p75 toward zero, so it is ignored, as is any failure before a real completion exists.
        """
        median = self.quantile(0.5)
        if median is not None and seconds >= median:
            self.record(seconds)

    def record_drained(self) -> None:
        self._drained += 1

    @property
    def drained(self) -> int:
        return self._drained

    def quantile(self, q: float) -> Optional[float]:
        if not self._durations:
            return None
        ordered = self._durations
        pos = (len(ordered) - 1) * q
        low = int(pos)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] * (1 - (pos - low)) + ordered[high] * (pos - low)

    def drain_margin(self, configured: Optional[float]) -> Optional[float]:
        """Seconds of headroom a task needs before it is worth starting.

        An explicit value wins. Otherwise adapt to this run's own p75, once
        enough tasks have finished for that to mean anything.
        """
        if configured is not None:
            return configured
        if len(self._durations) < 5:
            return None
        return self.quantile(0.75)

    def summary(self) -> str:
        if not self._durations:
            lines = ["No completed rollouts to report latency for."]
        else:
            total = sum(self._durations)
            lines = [
                f"Per-task latency over {len(self._durations)} finished attempt(s): "
                f"median {self.quantile(0.5) / 60:.1f} min, "
                f"p90 {self.quantile(0.9) / 60:.1f} min, "
                f"p99 {self.quantile(0.99) / 60:.1f} min, "
                f"max {max(self._durations) / 60:.1f} min",
                f"Task-time observed: {total / 3600:.1f} task-hours",
            ]
        if self._drained:
            lines.append(
                f"Drained (not dispatched, no time left in the budget): {self._drained}. "
                "They are not attempts and will be re-dispatched on resume."
            )
        return "\n".join(lines)


def _record_rollout_shape(row: Mapping[str, Any], result: Mapping[str, Any]) -> None:
    """Record the size of one completed rollout, while metrics are exporting.

    Turns and tool calls come from the rollout's ``ng_perf`` when rollout observability produced it.
    Without it, tool calls are counted from the final response's ``function_call`` items, and turns are not recorded.
    """
    observer = driver_metrics()
    if observer is None:
        return
    try:
        input_items = (row.get("responses_create_params") or {}).get("input")
        if isinstance(input_items, list):
            observer.input_items.record(len(input_items))
        output_items = ((result.get("response") or {}).get("output")) or []
        observer.output_items.record(len(output_items))
        ng_perf = result.get(NG_PERF_KEY) or {}
        if "num_turns" in ng_perf:
            observer.turns.record(ng_perf["num_turns"])
        tool_calls = ng_perf.get("num_tool_calls")
        if tool_calls is None:
            tool_calls = sum(
                1 for item in output_items if isinstance(item, dict) and item.get("type") == "function_call"
            )
        observer.tool_calls.record(tool_calls)
    except Exception:
        logger.debug("Could not record rollout shape metrics", exc_info=True)


def _dispatch_drained_result(remaining_s: float, margin_s: Optional[float]) -> Dict[str, Any]:
    """The result Gym builds for a row the dispatch budget never started.

    No row is written anywhere: absence is the resume signal, so the task is
    re-dispatched intact next allocation instead of being started and killed
    part-way through. ``_ng_dispatch_drained`` keeps it out of capture, token
    finalization, progress metrics and the rollouts upload, since nothing ran.
    """
    return {
        NG_FAILURE_CLASS_KEY: CANCELLED,
        NG_NO_PERSIST_KEY: True,
        NG_DISPATCH_DRAINED_KEY: True,
        "error_message": (
            f"not dispatched: {remaining_s:.0f}s left in dispatch budget, below the {margin_s:.0f}s drain margin"
            if margin_s is not None
            else "not dispatched: dispatch budget exhausted"
        ),
    }


class RolloutCollectionConfig(SharedRolloutCollectionConfig):
    """
    Perform a batch of rollout collection.

    Examples:

    ```bash
    gym eval run --no-serve \
        +agent_name=example_single_tool_call_simple_agent \
        +input_jsonl_fpath=weather_query.jsonl \
        +output_jsonl_fpath=weather_rollouts.jsonl \
        +limit=100 \
        +num_repeats=4 \
        +num_samples_in_parallel=10
    ```
    """

    agent_name: Optional[str] = Field(
        default=None,
        description=(
            "The agent to collect rollouts from. Routes every row to this agent, overriding any "
            "agent_ref already present in the data (a warning lists overridden values). "
            "Shorthand for agent_map={_default: <agent_name>}."
        ),
    )
    agent_map: Optional[Dict[str, str]] = Field(
        default=None,
        description=(
            "Explicit per-agent re-routing, keyed by the agent_ref.name or task_source found in "
            "the data (e.g. {old_agent: new_agent}). The special key '_default' applies to every "
            "row with no specific entry, including rows with no agent_ref at all. "
            "Precedence: agent_map[<row value>] > agent_map._default > row agent_ref > task_source resolution."
        ),
    )
    fan_out: Optional[Dict[str, List[str]]] = Field(
        default=None,
        description=(
            "Run each matching row once per listed agent (cross-product), keyed by the row's "
            "agent_ref.name or task_source (e.g. {genrm_compare_resources_server: [agent_a, agent_b]}). "
            "Each copy gets its own rollout index; outputs are tagged with the agent that produced them, "
            "so per-agent metrics separate naturally. Composes with num_repeats (repeats apply per agent)."
        ),
    )
    input_jsonl_fpath: str = Field(
        description="The input data source to use to collect rollouts, in the form of a file path to a jsonl file."
    )
    limit: Optional[int] = Field(
        default=None, description="Maximum number of examples to load and take from the input dataset."
    )
    num_repeats: Union[int, Dict[str, int]] = Field(
        default=1,
        description=(
            "How many times to repeat each example. Either an int (applied to every row) or a "
            "dict keyed by the dispatched agent name or by the row's own routing key (its "
            "agent_ref.name or task_source as written in the data, before any agent_map/fan_out "
            "re-route); the dispatched agent wins when both have entries. In dict form, every row "
            "must match an entry, unless a special '_default' key is provided as a fallback. "
            "Useful for mean@k."
        ),
    )
    num_repeats_add_seed: Union[bool, Dict[str, bool]] = Field(
        default=False,
        description=(
            'Whether to pass a per-rollout "seed" via metadata.extra_body (honored by vLLM model servers). '
            "Either a bool applied to every row or a dict with the same dispatched-agent/original-routing-key "
            "lookup and _default fallback semantics as num_repeats."
        ),
    )
    interleave_repeats: bool = Field(
        default=False,
        description=(
            "Start the repeats round by round (abcabc) rather than each task's back to back (aabbcc), to spread "
            "a task's repeats over the run. Useful when a task is heavy on the machine running it, e.g. its "
            "sandbox loads large data. Best effort: repeats still overlap when the concurrency is high or "
            "rollouts are long."
        ),
    )
    resume_from_cache: bool = Field(
        default=False,
        description="If the same command is run multiple times, check the materialized inputs and current outputs and remove the inputs that have already been run",
    )
    resubmit: bool = Field(
        default=False,
        description=(
            "Set by a job queued with `gym eval resubmit`: resume from cache whatever resume_from_cache says, "
            "and give every failed rollout that is not final a fresh attempt budget."
        ),
    )
    allow_unsafe_resume: bool = Field(
        default=False,
        description=(
            "Explicitly reuse legacy or identity-mismatched saved rollouts. Warns and reuses the saved "
            "materialized inputs. Unsupported or corrupt manifests are still rejected."
        ),
    )
    prompt_config: Optional[str] = Field(
        default=None,
        description="Path to a prompt YAML file. Builds responses_create_params.input from the template at rollout time. Mutually exclusive with pre-populated responses_create_params.input in the JSONL data.",
    )
    skills: Optional[SkillsConfig] = Field(
        default=None,
        description="Run-level skills config (skills.path). Makes a directory of Agent Skills standard skills available to the agent at rollout time and stamps each result with a skills_ref. Applied to a skill-agnostic dataset; not a dataset-row field.",
    )

    @field_validator("num_repeats", mode="before")
    @classmethod
    def _coerce_null_num_repeats(cls, v):
        # default to 1 if num_repeats is None
        # for backwards compatibility
        return 1 if v is None else v

    @model_validator(mode="after")
    def _resubmit_resumes(self) -> "RolloutCollectionConfig":
        # A resubmit continues its run: a fresh start would delete the rollouts it was queued to finish.
        if self.resubmit:
            self.resume_from_cache = True
        return self

    @model_validator(mode="after")
    def _fold_agent_name_into_agent_map(self) -> "RolloutCollectionConfig":
        # agent_name is sugar for agent_map._default; fold it so downstream code has one knob.
        if self.agent_name is None:
            return self
        existing_default = (self.agent_map or {}).get("_default")
        if existing_default is not None and existing_default != self.agent_name:
            raise ValueError(
                f"agent_name={self.agent_name!r} conflicts with agent_map._default={existing_default!r}. "
                "Set only one of them."
            )
        self.agent_map = {**(self.agent_map or {}), "_default": self.agent_name}
        return self

    @model_validator(mode="after")
    def _validate_num_repeats(self) -> "RolloutCollectionConfig":
        nr = self.num_repeats
        if isinstance(nr, int):
            if nr < 1:
                raise ValueError(f"num_repeats must be >= 1, got {nr}")
        else:
            bad = {name: n for name, n in nr.items() if n < 1}
            if bad:
                raise ValueError(f"num_repeats dict values must be >= 1, got {bad}")
        return self

    @model_validator(mode="after")
    def _validate_dispatch_concurrency(self) -> "RolloutCollectionConfig":
        if (
            self.dispatch_budget_s is not None
            and self.num_samples_in_parallel is None
            and self.max_resident_rollout_tasks is None
        ):
            raise ValueError(
                "dispatch_budget_s requires num_samples_in_parallel or max_resident_rollout_tasks; "
                "unbounded dispatch can POST the entire queue before the budget is re-checked"
            )
        return self

    @model_validator(mode="after")
    def _validate_fan_out(self) -> "RolloutCollectionConfig":
        for key, agents in (self.fan_out or {}).items():
            if not agents:
                raise ValueError(
                    f"fan_out[{key!r}] is an empty list, which would silently drop every matching row "
                    "(zero rollouts collected). Remove the key, or list at least one agent."
                )
            duplicates = sorted({a for a in agents if list(agents).count(a) > 1})
            if duplicates:
                raise ValueError(
                    f"fan_out[{key!r}] lists the same agent more than once: {duplicates}. Each listed "
                    "agent already runs every matching row; use num_repeats for repetition."
                )
        return self

    @property
    def materialized_jsonl_fpath(self) -> Path:
        return materialized_path_for(Path(self.output_jsonl_fpath))

    dispatch_budget_s: Optional[float] = Field(
        default=None,
        ge=0,
        description=(
            "Seconds from collection start during which new rollouts may be dispatched. Set it to "
            "the usable walltime of the allocation (Slurm walltime minus startup and teardown). "
            "Once the budget is spent, queued tasks are left undispatched rather than started and "
            "killed mid-flight: they are not attempts, so a `resume_from_cache` run re-dispatches them "
            "with a full allocation ahead of them. Unset (default) preserves the old behaviour of "
            "dispatching everything regardless of how little time is left."
        ),
    )

    record_omitted_rollouts: bool = Field(
        default=False,
        description=(
            "Write a failures-sidecar row for each row dispatch_budget_s never started. An attempt the agent "
            "reports as kill_shaped needs no such row: the run history always saves it as an interruption. The "
            "row carries _ng_no_persist, _ng_omitted_rollout, _ng_dispatch_drained, the class cancelled and a "
            "reason in _ng_failure_message. It is evidence only: the run history skips it, it is not scored or "
            "counted as an attempt or a failure, and resume_from_cache still runs the rollout. Off by default, "
            "which writes nothing for these rows, until readers of the sidecar outside Gym treat such rows as "
            "not run rather than as failed."
        ),
    )

    drain_margin_s: Optional[float] = Field(
        default=None,
        ge=0,
        description=(
            "Refuse to dispatch a new rollout when fewer than this many seconds remain in "
            "``dispatch_budget_s`` — a task that cannot finish is pure waste. When unset, the "
            "margin adapts to the run's own observed p75 task duration (after 5 completions), "
            "which needs no per-benchmark tuning. Ignored unless dispatch_budget_s is set."
        ),
    )

    dispatch_longest_first: bool = Field(
        default=False,
        description=(
            "Dispatch tasks with the longest previously-observed runtime first, so long tasks start "
            "early in an allocation instead of straddling its boundary. Only affects runs resuming "
            "from a cache that recorded timings; tasks never seen before keep their input order and "
            "are dispatched after the known-long ones. No runtime estimate is made for unseen tasks."
        ),
    )

    dispatch_start_report_within_s: Optional[float] = Field(
        default=None,
        ge=0,
        description=(
            "Enable the run timing summary: the share of rollouts started within this many seconds of "
            "collection start, the last start, the share stopped by a per-rollout wall-clock limit and "
            "the share of long rollouts, printed every timing_summary_interval_s, at the end and on "
            "SIGTERM. Diagnostic only, nothing is aborted."
        ),
    )
    dispatch_start_report_min_fraction: float = Field(
        default=0.99,
        ge=0,
        le=1,
        description="Share of rollouts expected to have started within dispatch_start_report_within_s; below it the report is a WARNING.",
    )
    dispatch_all_started_within_s: Optional[float] = Field(
        default=None,
        ge=0,
        description=(
            "Seconds into collection by which every rollout is expected to have started; the timing "
            "summary warns when the last start is later. Defaults to dispatch_start_report_within_s."
        ),
    )
    server_queue_metrics_urls: Optional[List[str]] = Field(
        default=None,
        description=(
            "Only used when dispatch_start_report_within_s is set. "
            "vLLM base URLs (the policy model's base_url list) whose /metrics the run timing summary polls "
            "for requests waiting and running, summed over servers."
        ),
    )
    server_queue_max_waiting_ratio: float = Field(
        default=0.1,
        description="Healthy ceiling for summed waiting / summed running requests over the run.",
    )
    server_queue_sample_interval_s: float = Field(default=30.0, gt=0, description="Seconds between /metrics polls.")
    long_rollout_s: Optional[float] = Field(
        default=None,
        ge=0,
        description="Rollouts running longer than this many seconds are counted in the end-of-run timing summary.",
    )
    long_rollout_max_fraction: float = Field(
        default=0.05,
        ge=0,
        le=1,
        description="Share of rollouts longer than long_rollout_s above which the timing summary warns.",
    )
    timing_summary_interval_s: Optional[float] = Field(
        default=600,
        ge=0,
        description=(
            "Print the run timing summary every this many seconds while rollouts are collected. "
            "Requires dispatch_start_report_within_s; 0 or null disables the periodic print."
        ),
    )
    timed_out_max_fraction: float = Field(
        default=0.01,
        ge=0,
        le=1,
        description=(
            "Share of rollouts ended by a per-rollout wall-clock limit (result field `timed_out`, or "
            "failure class `timeout_exceeded` or `agent_timeout`) above which the timing summary warns."
        ),
    )
    retry_terminal_timeouts: bool = Field(
        default=False,
        description=(
            "On resume, retry failures-sidecar rows of class `timeout_exceeded` even when they are stamped "
            "`_ng_failure_terminal`, up to NEMO_GYM_MAX_ROLLOUT_ATTEMPTS. For sidecars "
            "whose per-attempt timeouts were stamped terminal but should be retried. Off (default): a sidecar row "
            "is terminal iff it is stamped `_ng_failure_terminal`, so an agent that marks its timeouts terminal "
            "on purpose keeps them gated. Managed runs save this policy; omitting the option on resume "
            "retains the saved value."
        ),
    )


def _rollout_request_debug_summary(row: Dict[str, Any]) -> Dict[str, Any]:
    agent_ref = row.get(AGENT_REF_KEY_NAME) or {}
    summary = {
        TASK_INDEX_KEY_NAME: row.get(TASK_INDEX_KEY_NAME),
        ROLLOUT_INDEX_KEY_NAME: row.get(ROLLOUT_INDEX_KEY_NAME),
        ROLLOUT_ID_KEY_NAME: row.get(ROLLOUT_ID_KEY_NAME),
        "agent_name": agent_ref.get("name") if isinstance(agent_ref, dict) else None,
        "environment_server": row.get(NG_ENVIRONMENT_SERVER_KEY),
        "taskset": _materialized_taskset(row),
    }
    return {k: v for k, v in summary.items() if v is not None}


# Request failures that are data, not bugs. Anything else still propagates.
_RUN_FAILURE_ERRORS = (ClientError, orjson.JSONDecodeError, TimeoutError)
# Statuses something in front of the environment server answers with; it returns 500 itself.
_SERVER_DID_NOT_RUN_STATUSES = frozenset({429, 502, 503, 504})
_MAX_FAILURE_BODY_CHARS = 2000


#: ``SpanContext`` of the ``gym.job`` span the driver opened as a standalone trace root, for the
#: duration of that run. None under an embedder's trace or when the ``job`` group is off. With
#: traces off it holds the *invalid* context of a non-recording span rather than None;
#: ``linked_root_span`` checks validity, so that case nests like the others.
_OWNED_JOB_SPAN_CONTEXT: ContextVar[Optional["SpanContext"]] = ContextVar(
    "nemo_gym_owned_job_span_context", default=None
)


def _current_span_is_valid() -> bool:
    try:
        from opentelemetry import trace as otel_trace
    except ImportError:
        return False
    return otel_trace.get_current_span().get_span_context().is_valid


TASK_ID_ATTRIBUTE = "nemo.gym.task.id"
ROLLOUT_REPEAT_ATTRIBUTE = "nemo.gym.rollout.repeat"


def _span_task_id(row: Mapping[str, Any]) -> Optional[str]:
    identity = row.get("task_id")
    if isinstance(identity, Mapping):
        return identity.get("task_id")
    return next((str(row[key]) for key in (*TASK_ID_FIELDS, "task_name") if row.get(key) is not None), None)


@contextmanager
def _dispatch_span(row: Mapping[str, Any], dispatch_name: str, *, server_name: str) -> Iterator["SpanContext"]:
    """One rollout's dispatch as the root of its own trace, linked to the run's `job` span.

    The `/run` call made inside carries this span as `traceparent`, so the agent's
    `gym.rollout` span and everything below it join the new trace instead of nesting under
    `gym.job`. The link, not a parent edge, ties the trace back to the job; both carry the
    run id as a resource attribute. The identity comes from the collector *row*, not from
    the posted body: a materialized episode body carries it only as `episode_id`.
    `dispatch_name` is the value the rollout counters record under the same key, so spans
    and metrics join on it; `server_name` is the server the call goes to, which differs for
    a row routed by agent. Only a job the driver opened as its own root is split; under an
    embedder's trace the dispatch keeps nesting (see `linked_root_span`). Gate on the
    `rollout` group and the trace exporter before calling.

    Yields the span's context, which the driver keeps to attach `gym.rollout.health` later.
    """
    from nemo_gym.telemetry.spans import linked_root_span

    attributes: Dict[str, Any] = {
        DISPATCH_NAME_ATTRIBUTE: dispatch_name,
        HTTP_DESTINATION_SERVER_NAME_ATTRIBUTE: server_name,
    }
    try:
        rollout_id = maybe_rollout_id_from_run_body(row)
    except (TypeError, ValueError):
        # Telemetry never decides whether a rollout runs; a malformed row just gets no id.
        rollout_id = None
    if rollout_id:
        attributes[ROLLOUT_ID_ATTRIBUTE] = rollout_id
    task_id = _span_task_id(row)
    if task_id is not None:
        attributes[TASK_ID_ATTRIBUTE] = task_id
    repeat = row.get(ROLLOUT_INDEX_KEY_NAME)
    if isinstance(repeat, int) and not isinstance(repeat, bool):
        attributes[ROLLOUT_REPEAT_ATTRIBUTE] = repeat
    with linked_root_span("gym.rollout.dispatch", split_from=_OWNED_JOB_SPAN_CONTEXT.get(), **attributes) as span:
        yield span.get_span_context()


def _record_rollout_outcome(result: Dict[str, Any], outcome: str, dispatch_name: str) -> None:
    """Count a persisted attempt, including its numeric reward when scored."""
    if not is_span_group_enabled(GymSpanGroup.ROLLOUT):
        return
    # These are the dispatch's original routing fields, captured before the
    # canonical persisted record maps kinds to sidecar classes. Keep main's
    # dashboard contract: a routing class is not an inferred failure kind.
    failure_class = result.get(NG_FAILURE_CLASS_KEY) if outcome != "scored" else None
    failure_kind = failure_type = None
    if failure_class is not None:
        if "_ng_failure_kind" in result:
            failure_kind = result["_ng_failure_kind"]
            # Native failures and their legacy adapter route by the shared kind.
            # A separate legacy routing class must survive an echoed kind marker.
            if failure_class == failure_kind:
                failure_class = _metric_failure_kind(failure_class)
        else:
            failure_kind = result.get("failure_kind")
        failure_type = result.get("_ng_failure_type")
    record_rollout_completed(
        outcome,
        dispatch_name=dispatch_name,
        failure_class=failure_class,
        failure_kind=_metric_failure_kind(failure_kind),
        failure_type=failure_type if isinstance(failure_type, str) else None,
    )
    reward = result.get("reward")
    # A result without a numeric reward is unscored, not a zero.
    if (
        outcome == "scored"
        and isinstance(reward, (int, float))
        and not isinstance(reward, bool)
        and math.isfinite(reward)
    ):
        record_rollout_reward(float(reward), dispatch_name=dispatch_name, masked=bool(result.get(MASK_SAMPLE_KEY)))


def _metric_failure_kind(failure_kind: Any) -> Optional[str]:
    """The kind as a bounded label: registered or namespaced names pass, anything else is one bucket."""
    if not isinstance(failure_kind, str):
        return None
    if is_registered(failure_kind) or is_namespaced(failure_kind):
        return failure_kind
    return UNREGISTERED_FAILURE_KIND


def _record_rollout_health(
    result: Dict[str, Any],
    outcome: str,
    ignored_checks: frozenset[str],
    dispatch_span_context: Optional["SpanContext"] = None,
) -> None:
    """Evaluate the trajectory health checks on a finished rollout and count the verdict and flagged checks.

    With a `dispatch_span_context`, the verdict and the flagged checks also go on a `gym.rollout.health` span
    under that dispatch span. Gated on telemetry only, not on the post-run report switch. Never raises:
    telemetry must not fail a rollout.
    """
    from nemo_gym.rollout_health import evaluate_record

    try:
        digest = evaluate_record(result, ignored_checks)
        checks = sorted({finding.check for finding in digest.findings})
        record_rollout_health(digest.verdict, outcome=outcome, flagged_checks=checks)
        if dispatch_span_context is not None:
            from nemo_gym.telemetry.spans import record_child_span

            attributes = {HEALTH_VERDICT_ATTRIBUTE: digest.verdict}
            if checks:
                attributes[HEALTH_CHECKS_ATTRIBUTE] = checks
            rollout_id = maybe_rollout_id_from_run_body(result)
            if rollout_id:
                attributes[ROLLOUT_ID_ATTRIBUTE] = rollout_id
            record_child_span("gym.rollout.health", dispatch_span_context, **attributes)
    except Exception:
        logger.debug("Could not record rollout health telemetry.", exc_info=True)


def _agent_request_failure_row(exc: BaseException, status: Optional[int]) -> Dict[str, Any]:
    """One sidecar row for a `/run` call that came back without a result.

    No reward and no response: an infrastructure failure is not a verifier score of zero, and a
    placeholder would read as real generation data to token capture, aggregation and trainers.
    The class says whether the rollout ran. A NeMo Gym server answers 500 when its own handler
    raises, so any status it answered with means the rollout ran and broke, which is also how a
    model server rejecting the model's own output arrives here. A gateway status, or no reply to
    take a status from, says nothing about the rollout. Neither class carries a reward; an evaluation that wants the
    first counted names it in ``count_failure_classes_as_zero``.
    """
    rollout_ran = status is not None and status not in _SERVER_DID_NOT_RUN_STATUSES
    body = getattr(exc, "response_content", None)
    return {
        NG_FAILURE_CLASS_KEY: (AGENT_RUN_ERROR_FAILURE_CLASS if rollout_ran else AGENT_REQUEST_FAILED_FAILURE_CLASS),
        "_ng_failure_type": type(exc).__name__,
        "_ng_failure_message": str(exc) or repr(exc),
        "_ng_failure_http_status": status,
        "_ng_failure_response_body": _truncated_body(body),
    }


def _truncated_body(body: Optional[bytes]) -> Optional[str]:
    """Decode at most the kept prefix, so a huge error page is never decoded in full."""
    if not body:
        return None
    text = body[: _MAX_FAILURE_BODY_CHARS * 4].decode("utf-8", "replace")
    return text[:_MAX_FAILURE_BODY_CHARS] + ("…" if len(body) > _MAX_FAILURE_BODY_CHARS else "")


def _failure_outcome(row: Dict, failure: Dict, stage: str) -> RolloutFailureRecord:
    observed_response = stage in {"agent", "verifier"}
    # Transport phases are not episode execution stages. A timeout does not
    # establish whether an earlier transport attempt reached the server.
    episode_stage = failure.get("_ng_failure_stage")
    if episode_stage is None and failure.get(NG_FAILURE_CLASS_KEY) in ("judge_failed", "judge_invalid"):
        episode_stage = "verification"
    message = str(
        failure.get("failure_reason") or failure.get("_ng_failure_message") or "Agent reported a no-result failure"
    )
    return RolloutFailureRecord(
        episode_id=EpisodeId(rollout_id=logical_rollout_id(row), attempt=row.get(ATTEMPT_INDEX_KEY_NAME, 0)),
        run_id=row[RUN_ID_KEY],
        source="environment" if observed_response else "collector",
        delivery=(
            "delivered"
            if observed_response
            or (
                failure.get("_ng_failure_http_status")
                and failure["_ng_failure_http_status"] not in _SERVER_DID_NOT_RUN_STATUSES
            )
            else "possibly_delivered"
        ),
        failure=EpisodeFailure(
            failure_reason=message[:2000],
            terminal=bool(failure.get(NG_TERMINAL_KEY)),
            # A native reply may deliberately leave its kind unspecified. Legacy
            # response models also emit a null shared field beside a routing class.
            failure_kind=(
                failure["_ng_failure_kind"]
                if "_ng_failure_kind" in failure
                else failure.get("failure_kind") or failure[NG_FAILURE_CLASS_KEY]
            ),
            stage=episode_stage,
        ),
        exception_type=(str(failure["_ng_failure_type"])[:2000] if failure.get("_ng_failure_type") else None),
        http_status=failure.get("_ng_failure_http_status"),
    )


def _normalize_rollout_outcome(row: Dict, result: Any) -> Dict | RolloutFailureRecord:
    if not isinstance(result, dict):
        raise InvalidRolloutResult("Agent /run must return a JSON object.")
    native = _materialized_taskset(row) is not None
    if not native:
        _sanitize_legacy_result(result)
    try:
        if "failure" in result and "episode_id" in result and not native:
            raise InvalidRolloutResult(
                "Agents cannot return a collector-owned RolloutFailureRecord; use the shared failure markers."
            )
        if result.get(NG_FAILURE_CLASS_KEY) is not None:
            return _failure_outcome(row, result, "agent")
        if result.get(NG_NO_PERSIST_KEY):
            return result
        if native:
            # Native result schemas belong to the Environment Server. A completed
            # episode may have no score or Responses API generation at all.
            if "reward" in result and (type(result["reward"]) not in (int, float) or not isfinite(result["reward"])):
                raise InvalidRolloutResult("An episode reward, when present, must be finite.")
            return result
        # Older producers' execute_only completion intentionally skips verification.
        # Missing scores are allowed only for this explicit legacy contract;
        # a present score still has to be valid, and empty/malformed bodies fail.
        unscored = result.get("execute_only") is True and "reward" not in result
        if not isinstance(result.get("response"), dict) or (
            not unscored and (type(result.get("reward")) not in (int, float) or not isfinite(result["reward"]))
        ):
            raise InvalidRolloutResult(
                "Completed results require a response object and a finite reward, unless execute_only is true."
            )
    except (ValidationError, OverflowError) as error:
        raise InvalidRolloutResult(str(error)) from error
    return result


def _judge_failure_response(result: Any) -> Optional[Dict]:
    """Legacy judge_failsafe carries real generation that can be judged again."""
    if isinstance(result, dict) and result.get(NG_FAILURE_CLASS_KEY) in ("judge_failed", "judge_invalid"):
        response = result.get("response")
        if isinstance(response, dict):
            return response
    return None


def _failure_diagnostics(result: Any) -> Dict:
    """Preserve producer diagnostics outside the bounded canonical failure record."""
    if not isinstance(result, dict) or (result.get(NG_FAILURE_CLASS_KEY) is None and result.get("type") != "failure"):
        return {}
    # Legacy producers use different names for errors, trace paths, and other
    # evidence. Preserve their fields, but never turn placeholder scores or
    # generation into a completed result. Actual judge input is restored separately.
    excluded = {"reward", "response", NG_NO_PERSIST_KEY}
    if result.get("type") == "failure":
        # The envelope is a sidecar row; only _ng_failure_record is the typed model.
        excluded.add("type")
    diagnostics = {key: value for key, value in result.items() if key not in excluded}
    response = result.get("response")
    if (
        isinstance(response, dict)
        and response
        and result.get(NG_FAILURE_CLASS_KEY) not in ("judge_failed", "judge_invalid")
    ):
        # This is producer evidence, possibly including synthetic failure text.
        # It is never a scored response or a judge-only recovery candidate.
        diagnostics.setdefault("_ng_failure_response", response)
    return diagnostics


def _failure_compatibility_row(failure: RolloutFailureRecord, verification_response: Optional[Dict] = None) -> Dict:
    """Keep existing sidecar routing keys and diagnostics alongside the canonical failure.

    Remove these aliases when row-oriented consumers and #3180 producers migrate.
    The canonical record never carries generation. The sidecar envelope retains
    actual judge input for existing --judge-failed-only readers, without a reward.
    Other legacy response placeholders are discarded.
    """
    row = {
        "_ng_failure_record": failure.model_dump(mode="json"),
        NG_FAILURE_CLASS_KEY: failure.sidecar_failure_class,
        "_ng_failure_type": failure.exception_type,
        "_ng_failure_message": failure.failure.failure_reason,
        "error": failure.failure.failure_reason,
        "_ng_failure_http_status": failure.http_status,
    }
    if failure.failure.terminal:
        row[NG_TERMINAL_KEY] = True
    if failure.failure.failure_kind in ("judge_failed", "judge_invalid") and verification_response is not None:
        row["response"] = verification_response
    return row


def _aggregation_error_entry(
    agent_name: str, exc: BaseException, response_status: Optional[int] = None
) -> Dict[str, Any]:
    """Build the metrics-file placeholder for one agent whose aggregation failed."""
    message = str(exc) or repr(exc)
    if len(message) > _MAX_FAILURE_BODY_CHARS:
        message = message[:_MAX_FAILURE_BODY_CHARS] + "…"
    return {
        AGENT_REF_KEY_NAME: {"name": agent_name},
        "agent_metrics": {},
        "key_metrics": {},
        "group_level_metrics": [],
        "repeat_level_metrics": [],
        AGGREGATION_ERROR_KEY: {
            "type": type(exc).__name__,
            "message": message,
            "http_status": getattr(exc, "status", None) or response_status,
        },
    }


def _raise_for_aggregation_errors(metrics_fpath: Optional[Path]) -> None:
    """Signal failure after partial metrics and any batch status have been saved."""
    if metrics_fpath is None:
        return
    failed_agents = sorted(
        entry[AGENT_REF_KEY_NAME]["name"]
        for entry in orjson.loads(metrics_fpath.read_bytes())
        if AGGREGATION_ERROR_KEY in entry
    )
    if failed_agents:
        raise RuntimeError(
            f"Aggregation failed for agents: {', '.join(failed_agents)}. "
            f"Metrics and error details were saved to {metrics_fpath}. "
            "Retry with `gym eval aggregate` after resolving the aggregation errors."
        )


def _latest_failure_rows(failures_fpaths: List[Path]) -> Dict[Tuple[Any, Any], Dict[str, Any]]:
    """The last failed attempt recorded for each rollout across the failures sidecars.

    An interrupted attempt that ``record_omitted_rollouts`` recorded is the rollout's latest execution but no
    failure: it hides any older failure, so the rollout is left out here and is missing. A row the dispatch budget
    never started, and the evidence of a failed /run with routing off, are skipped (``is_non_superseding_rollout_row``).
    """
    latest_by_key: Dict[Tuple[Any, Any], Dict[str, Any]] = {}
    for fpath in failures_fpaths:
        if not fpath.exists():
            continue
        with fpath.open("rb") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                row = loads_jsonl_line(line, fpath, line_no)
                if is_non_superseding_rollout_row(row):
                    continue
                key = row.get(TASK_INDEX_KEY_NAME), row.get(ROLLOUT_INDEX_KEY_NAME)
                previous = latest_by_key.get(key)
                if previous is None or row.get(ATTEMPT_INDEX_KEY_NAME, 0) >= previous.get(ATTEMPT_INDEX_KEY_NAME, 0):
                    latest_by_key[key] = row
    return {key: row for key, row in latest_by_key.items() if not is_omitted_rollout_row(row)}


def _matches_failure_class(row: Mapping[str, Any], failure_classes: Collection[str]) -> bool:
    """Match the canonical class or the original legacy producer's routing class."""
    return matches_failure_class(row, failure_classes)


def _counted_failure_rows(rows: List[Dict], failure_classes: List[str]) -> List[Dict]:
    wanted = set(failure_classes)
    return [_counted_failure_row(row) for row in rows if _counts_as_zero(row, wanted)]


def _failure_rows_counted_as_zero(
    failures_fpaths: List[Path], failure_classes: List[str], scored_keys: set
) -> List[Dict[str, Any]]:
    """Sidecar rows the caller opted to count in the metrics denominator.

    The last attempt of a rollout is the one that stands, so it is selected across every failure
    class before the wanted classes are picked out. Selecting the other way round would let a
    stale attempt be counted after a later one landed in a class the caller did not ask for.

    A row that already carries a ``reward`` is counted as it stands. A row that carries none
    records that no rollout happened, so it is counted as a zero here and only here: the score
    enters the metric input, never the sidecar or the rollouts jsonl, which keeps the artifacts
    free of a verdict no verifier gave. A rollout that also succeeded is never counted. A row its
    producer marked with ``_ng_failure_counts_as_zero`` is counted whatever its class.
    """
    latest_by_key = _latest_failure_rows(failures_fpaths)
    wanted = set(failure_classes)
    return [
        _counted_failure_row(row)
        for key, row in latest_by_key.items()
        if key not in scored_keys and _counts_as_zero(row, wanted)
    ]


def _final_zero_failure_rows(
    failures_fpaths: List[Path], scored_keys: set, *, retry_terminal_timeouts: bool
) -> List[Dict[str, Any]]:
    """The latest failure rows that settle their rollout as a scored zero: final, and marked by their producer.

    Resume does not run such a rollout again and the metrics count it as 0, so its work is done. A row counted only
    because ``count_failure_classes_as_zero`` names its class is not settled: that option scores the metrics, and
    resume may still run the rollout.
    """
    return [
        row
        for key, row in _latest_failure_rows(failures_fpaths).items()
        if key not in scored_keys
        and row.get(NG_FAILURE_COUNTS_AS_ZERO_KEY) is True
        and is_terminal_failure(row, retry_terminal_timeouts=retry_terminal_timeouts)
    ]


def _counts_as_zero(row: Mapping[str, Any], wanted: set) -> bool:
    """Whether a rollout's latest sidecar row enters the metrics as a zero: its class is wanted, or it is marked."""
    return _matches_failure_class(row, wanted) or row.get(NG_FAILURE_COUNTS_AS_ZERO_KEY) is True


def _counted_failure_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """The metric-input copy of a sidecar row counted as zero."""
    # Diagnostics stay in the sidecar: an HTTP status is a number, and the aggregator
    # averages every number it is handed.
    scored = {
        k: v
        for k, v in row.items()
        if not k.startswith("_ng_failure_") and k not in ("failure_kind", "failure_reason")
    }
    # This metrics-only copy honors the explicit denominator policy. The original
    # answer, failure diagnostics, and training mask remain untouched in the sidecar.
    scored["mask_sample"] = False
    scored.setdefault("reward", 0.0)
    return scored


def _rollout_order_key(row: Dict[str, Any]) -> tuple:
    """Task then repeat: metrics that read a task's repeats positionally need that order."""
    return (row.get(TASK_INDEX_KEY_NAME) or 0, row.get(ROLLOUT_INDEX_KEY_NAME) or 0)


def _routing_identity(row: Mapping[str, Any]) -> Optional[str]:
    """The agent a rollout ran on, or for a materialized taskset row, which names none, its environment server.

    A materialized row, its result and its sidecar row all name the same one.
    """
    return (row.get(AGENT_REF_KEY_NAME) or {}).get("name") or row.get(NG_ENVIRONMENT_SERVER_KEY)


def _metrics_group(row: Mapping[str, Any], servers_by_agent: Callable[[], Mapping[str, list[str]]]) -> Optional[str]:
    """The environment server a rollout is scored under, as `_call_aggregate_metrics` groups it.

    The row's stamp, else the one server that fronts its agent, looked up only for an unstamped
    row: a materialized row routed by its agent has no stamp while its result has one, and both
    must land in the same group. Runs of one agent behind two servers stay apart this way. An agent
    behind no server or several stays its own group; `_call_aggregate_metrics` rejects such a row.
    """
    stamp = row.get(NG_ENVIRONMENT_SERVER_KEY)
    if isinstance(stamp, str):
        return stamp
    agent_name = (row.get(AGENT_REF_KEY_NAME) or {}).get("name")
    servers = servers_by_agent().get(agent_name, []) if agent_name else []
    return servers[0] if len(servers) == 1 else agent_name


def _identity_key(row: Mapping[str, Any], group: Callable[[Mapping[str, Any]], Optional[str]]) -> tuple:
    """A rollout's key across runs, which each number their tasks from 0."""
    return (group(row), row.get(TASK_INDEX_KEY_NAME), row.get(ROLLOUT_INDEX_KEY_NAME))


def _fill_task_fields(
    added_rows: List[Dict[str, Any]],
    real_rows: List[Dict[str, Any]],
    materialized_fpaths: List[Path],
    group: Callable[[Mapping[str, Any]], Optional[str]],
) -> None:
    """Give the rows the metric input adds, counted failures and imputed zeros, their task's dataset fields.

    Rows are ordered by repeat, so an added row can come first in its task, and metric hooks read
    task-level fields such as a subset label or a weight from a task's first rollout. The fields come
    from the task's materialized row, and only those that real rows of the same group carry: the
    aggregator averages every number it is handed, so a field no real row reports would become a
    metric of its own. A field the verifier computes is not in the materialized row, so the added row
    lacks it.
    """
    wanted = {_identity_key(row, group): row for row in added_rows}
    if not wanted:
        return
    carried: Dict[Optional[str], set] = defaultdict(set)
    for row in real_rows:
        carried[group(row)].update(row)
    for materialized_fpath in materialized_fpaths:
        if not materialized_fpath.exists():
            continue
        with open(materialized_fpath, "rb") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                source = loads_jsonl_line(line, materialized_fpath, line_no)
                target = wanted.get(_identity_key(source, group))
                if target is None:
                    continue
                for key in carried[group(source)] & source.keys():
                    if not key.startswith("_") and key not in (RESPONSES_CREATE_PARAMS_KEY_NAME, "response"):
                        target.setdefault(key, source[key])


def _missing_rollout_rows_counted_as_zero(
    materialized_fpaths: List[Path],
    failures_fpaths: List[Path],
    scored_keys: set,
    *,
    history: Optional[RolloutStore] = None,
) -> List[Dict[str, Any]]:
    """Materialized rollouts that produced no row anywhere, counted as zeros.

    The failure classes reach a rollout that failed and said so. This reaches the one that never
    got that far -- killed mid-flight, or dispatched and lost -- which leaves nothing in the
    rollouts jsonl and nothing in the sidecar. Without it such a rollout leaves the denominator as
    well as the numerator, so the score reads higher the more of the run went missing.

    The sidecar is read here rather than trusted from the caller. A failure whose class the caller
    left out of ``count_failure_classes_as_zero`` is absent from the scored keys, and counting it
    here would score the very rollouts that selection excluded -- silently turning the selection
    into a no-op.

    The zero carries the rollout's identity: its agent, and its environment server stamp when the
    row has one, which a materialized taskset row needs because it names no agent. `_fill_task_fields`
    adds the task's dataset fields. A row that names neither cannot reach any server's metrics, so
    it is warned about rather than counted. The score enters the metric input and nothing else, the
    same way a counted failure row does.
    """
    recorded_failures = set(_latest_failure_rows(failures_fpaths)) if history is None else set()
    counted = []
    unroutable = 0
    for materialized_fpath in materialized_fpaths:
        if not materialized_fpath.exists():
            # Without the inventory there is nothing to compare the rollouts against, so the
            # option silently does nothing for this shard -- which is the very shape of run it
            # exists to expose.
            print(f"[WARNING] {materialized_fpath} is missing; rollouts owed by that shard cannot be counted")
            continue
        with open(materialized_fpath, "rb") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                row = loads_jsonl_line(line, materialized_fpath, line_no)
                key = (row.get(TASK_INDEX_KEY_NAME), row.get(ROLLOUT_INDEX_KEY_NAME))
                if history is not None:
                    # A newer dispatch supersedes old payloads. Only the selected
                    # attempt's unknown outcome is missing; known failures keep
                    # their separate, explicitly requested scoring policy.
                    if history.disposition(row) != "unknown":
                        continue
                elif key in scored_keys or key in recorded_failures:
                    continue
                if _routing_identity(row) is None:
                    unroutable += 1
                    continue
                scored_keys.add(key)
                zero = {
                    TASK_INDEX_KEY_NAME: row.get(TASK_INDEX_KEY_NAME),
                    ROLLOUT_INDEX_KEY_NAME: row.get(ROLLOUT_INDEX_KEY_NAME),
                    AGENT_REF_KEY_NAME: row.get(AGENT_REF_KEY_NAME),
                    "reward": 0.0,
                }
                if NG_ENVIRONMENT_SERVER_KEY in row:
                    zero[NG_ENVIRONMENT_SERVER_KEY] = row[NG_ENVIRONMENT_SERVER_KEY]
                counted.append(zero)
    if unroutable:
        print(
            f"[WARNING] {unroutable} materialized rollout(s) name no agent or environment server and are not counted"
        )
    return counted


_READ_JSONL_YIELD_INTERVAL_S = 0.01


async def _read_jsonl(path: Path) -> List[Dict]:
    """Parse a whole JSONL artifact, letting other tasks on the event loop run between rows.

    Parsing stays on the event loop thread instead of a worker thread.
    The loop thread then reuses the memory it freed during collection for these rows.
    A worker thread allocates from its own malloc arena, so the process would hold both.
    """
    rows = []
    yielded_at = time.monotonic()
    with path.open("rb") as f:
        for line in f:
            if line.strip():
                rows.append(orjson.loads(line))
            if time.monotonic() - yielded_at >= _READ_JSONL_YIELD_INTERVAL_S:
                await asyncio.sleep(0)
                yielded_at = time.monotonic()
    return rows


def _failure_export_metrics(
    failures_fpaths: List[Path],
    identity: Callable[[Mapping[str, Any]], Any],
    landed: set,
    *,
    histories: Sequence[RolloutStore] = (),
) -> Dict[str, int]:
    """Failure and retry counts for exporters, read from the failures sidecars.

    The sidecars hold every failed attempt of the run, across resumes, so these are run totals rather
    than one job's. ``identity`` keys a sidecar row the way ``landed`` keys the rollouts that have a row.
    ``coverage/failed/<class>`` counts the rollouts with no row, by the class of their last attempt.
    ``retries/attempts`` counts failed attempts, ``retries/rollouts_retried`` the rollouts dispatched
    more than once, and ``retries/at_cap`` the rollouts with no row that used all their attempts. Each rollout
    is held to the cap its collection recorded on its last attempt (``_ng_failure_max_attempts``), or the
    current cap for a row written before the cap was recorded. ``retries/max_attempts`` is that cap, and is
    left out when the rollouts were collected with different caps. A ``record_omitted_rollouts`` row is no failed
    attempt; an interrupted one is still the rollout's latest execution, so an older failure's class is not counted.
    """
    if histories:
        parts = [history.failure_export_metrics() for history in histories]
        if failures_fpaths:
            parts.append(_failure_export_metrics(failures_fpaths, identity, landed))
        merged: Counter[str] = Counter()
        for part in parts:
            merged.update({key: value for key, value in part.items() if key != "retries/max_attempts"})
        caps = {part.get("retries/max_attempts") for part in parts}
        if len(caps) == 1 and None not in caps:
            merged["retries/max_attempts"] = caps.pop()
        return dict(merged)

    attempts: Counter = Counter()
    latest: Dict[Any, Dict[str, Any]] = {}
    latest_failure: Dict[Any, Dict[str, Any]] = {}
    # Every class the sidecars name starts at 0: a reopened tracking run keeps a class's last value until it is sent.
    failed: Counter = Counter()
    for fpath in failures_fpaths:
        if not fpath.exists():
            continue
        with fpath.open("rb") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                row = loads_jsonl_line(line, fpath, line_no)
                if is_non_superseding_rollout_row(row):
                    continue
                latest[identity(row)] = row
                if is_omitted_rollout_row(row):
                    continue
                failed.setdefault(row.get(NG_FAILURE_CLASS_KEY) or "unknown", 0)
                attempts[identity(row)] += 1
                latest_failure[identity(row)] = row
    current_cap = get_max_rollout_attempts()
    caps = {key: row.get(NG_FAILURE_MAX_ATTEMPTS_KEY) or current_cap for key, row in latest_failure.items()}
    failed.update(
        row.get(NG_FAILURE_CLASS_KEY) or "unknown"
        for key, row in latest.items()
        if key not in landed and not is_omitted_rollout_row(row)
    )
    recorded_caps = set(caps.values()) or {current_cap}
    return {
        "retries/attempts": sum(attempts.values()),
        # A rollout that landed after n failed attempts was dispatched n + 1 times.
        "retries/rollouts_retried": sum(1 for key, n in attempts.items() if n + (key in landed) > 1),
        "retries/at_cap": sum(1 for key, n in attempts.items() if key not in landed and n >= caps[key]),
        **({"retries/max_attempts": recorded_caps.pop()} if len(recorded_caps) == 1 else {}),
        **{f"coverage/failed/{name}": count for name, count in sorted(failed.items())},
    }


def _count_lines(path: Path) -> int:
    with path.open("rb") as f:
        return sum(1 for _ in f)


def _coverage_report(
    expected: int, scored: int, failure_counts: Counter, failures_hint: Any, *, imputed: int = 0
) -> str:
    """State how much of the input the score covers, for the runs where it is not all of it.

    Silence here is what makes a partial run look complete, so this reports against the
    materialized input rather than the rollouts one hop happened to dispatch, and names the
    rollouts that are in the score only as imputed zeros.
    """
    report = ""
    missing = expected - scored
    if missing > 0:
        routed = ", ".join(f"{count} {name}" for name, count in sorted(failure_counts.items()))
        routed = f"{routed} routed this run; " if routed else ""
        report = (
            f"\nRollouts missing from the score: {missing} of {expected} materialized ({routed}see {failures_hint})"
            f"\nMetrics cover: {scored} of {expected} rollouts"
        )
    if imputed:
        report += f"\nScored as zero because they left no row: {imputed} of {expected} rollouts"
    return report


class _EagerCompletionQueue:
    """Own undelivered outcomes independently of the iterator and its handed-out awaitables."""

    def __init__(self, progress):
        self.ready: asyncio.Queue[Future] = asyncio.Queue()
        self.drained = False
        self.progress = progress

    def put(self, task: Future) -> None:
        if not self.drained:
            self.ready.put_nowait(task)  # Wake one consumer for each available outcome.

    async def get(self):
        try:
            task = await self.ready.get()
        except asyncio.QueueShutDown:
            raise asyncio.CancelledError from None
        self.progress.update(1)
        if self.progress.n == self.progress.total:
            self.progress.close()
        return await task

    def drain(self) -> list[Future]:
        self.drained = True
        tasks = []
        while not self.ready.empty():
            tasks.append(self.ready.get_nowait())
        # Only drain wakes every consumer, including issued awaitables started later.
        self.ready.shutdown()
        return tasks


class _EagerCompletionIterator:
    """Start every rollout in input order on first use, yielding completion-order awaitables."""

    def __init__(self, awaitables: Iterator, *, total: int):
        self._awaitables = awaitables
        self._total = total
        self._remaining = total
        self._tasks: set[Future] = set()
        self._queue: Optional[_EagerCompletionQueue] = None
        self._closed = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed or self._remaining <= 0:
            raise StopIteration
        if self._queue is None:
            progress = tqdm(desc="Collecting rollouts", miniters=10, total=self._total, maxinterval=60)
            self._queue = _EagerCompletionQueue(progress)
            # Creating tasks explicitly preserves input order (as_completed uses a set).
            for awaitable in self._awaitables:
                task = asyncio.ensure_future(awaitable)
                self._tasks.add(task)
                task.add_done_callback(functools.partial(self._on_done, self._queue))
        self._remaining -= 1
        return self._queue.get()

    def _on_done(self, queue: _EagerCompletionQueue, task: Future) -> None:
        self._tasks.discard(task)
        if not task.cancelled():
            # Retrieve failures even if no caller consumes this completion. Awaiting it still raises.
            task.exception()
        queue.put(task)

    async def aclose(self) -> None:
        await self._close(drain=False)

    async def aclose_and_drain(self) -> list[Future]:
        return await self._close(drain=True)

    async def _close(self, *, drain: bool) -> list[Future]:
        if self._closed:
            return []
        self._closed = True
        queue, self._queue = self._queue, None
        tasks = list(self._tasks)
        self._tasks.clear()
        completed = []
        if queue is not None:
            if drain:
                completed = queue.drain()
                # Some tasks may have finished before their done callbacks ran.
                completed.extend(task for task in tasks if task.done())
            queue.progress.close()
        # Plain aclose leaves the detached queue available to existing awaitables.
        # With no such awaitables, the queue and its payloads are released after callbacks run.
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        return completed


class _RowResultIterator:
    """Map completion metadata to ``(row, result)`` for the direct collection APIs.

    A thin map over the internal completion iterator that keeps its ``aclose()`` reachable, so a
    direct caller can stop collecting and cancel the rollouts still in flight.
    """

    def __init__(self, completions: Iterator[Future]):
        self._completions = completions
        self._closed = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed:
            raise StopIteration
        return self._without_metadata(next(self._completions))

    @staticmethod
    async def _without_metadata(future: Future) -> Tuple[Dict, Dict | RolloutFailureRecord]:
        completed = await future
        return completed.row, completed.result

    async def aclose(self) -> None:
        """Cancel unfinished rollouts and await the end of their client tasks in this process.

        Cancelling an in-flight `/run` request closes its connection. A server using
        ``ClientDisconnectCancellationMiddleware`` subsequently cancels its handler. This method
        does not wait for or acknowledge server/agent cleanup, which depends on the agent and may
        continue after return; it does not guarantee sandbox release or retire an attempt.

        A handed-out awaitable may return an already completed result or raise its exception,
        including ``CancelledError`` for cancelled work. In bounded mode, an awaitable first
        awaited after closure raises ``CancelledError`` even if a result was ready. This error
        propagates in the awaiting caller's task without a call to that task's ``cancel()``;
        ``asyncio.gather`` and ``TaskGroup`` treat it as cancellation. Iteration stops after closure.
        Safe to call again after closure completes, or before first use (no rollouts start).
        """
        self._closed = True
        await self._completions.aclose()

    async def aclose_and_drain(self) -> list[Future[Tuple[Dict, Dict | RolloutFailureRecord]]]:
        """Stop admission, cancel unfinished client tasks, and preserve completed outcomes.

        Before its first suspension, this operation stops admitting rollouts and snapshots
        completed outcomes not yet claimed by a completion awaitable. It then cancels and
        awaits unfinished client tasks, and returns settled futures yielding ``(row, result)``
        or raising the original exception. No new rollouts start while draining, including
        in bounded mode. Outcomes are transferred once; repeated calls return an empty list.

        A handed-out awaitable that already claimed an outcome retains it. Other issued
        awaitables raise ``CancelledError``. Callers with an outstanding waiter must also
        await that waiter and consume its outcome if it was not cancelled. The returned
        futures have their exceptions retrieved even if abandoned; awaiting them still raises.
        An outcome already cancelled at the snapshot is returned as a cancelled future and
        raises ``CancelledError`` when awaited. Tasks cancelled by this operation are excluded.

        Calling this before first use starts nothing. Calling it after ``aclose()`` returns
        nothing. Like ``aclose()``, it waits only for local client tasks, not server/agent
        cleanup, and does not guarantee sandbox release.
        """
        self._closed = True
        completed = await self._completions.aclose_and_drain()
        outcomes = []
        loop = asyncio.get_running_loop()
        for task in completed:
            outcome = loop.create_future()
            if task.cancelled():
                outcome.cancel()
            elif (error := task.exception()) is not None:
                outcome.set_exception(error)
                outcome.exception()
            else:
                result = task.result()
                outcome.set_result((result.row, result.result))
            outcomes.append(outcome)
        return outcomes


class _BoundedCompletionIterator:
    """Completion-order iterator with a bounded set of resident asyncio tasks."""

    def __init__(self, awaitables: Iterator, *, max_resident_tasks: int, total: int):
        if max_resident_tasks < 1:
            raise ValueError("max_resident_tasks must be >= 1")

        self._awaitables = iter(awaitables)
        self._max_resident_tasks = max_resident_tasks
        self._remaining = total
        self._pending: set[asyncio.Task] = set()
        self._ready: list[asyncio.Task] = []
        self._lock = asyncio.Lock()
        self._started = False
        self._closed = False
        self._progress = tqdm(
            desc="Collecting rollouts",
            miniters=10,
            total=total,
            maxinterval=60,
        )

    @property
    def _resident_task_count(self) -> int:
        return len(self._pending) + len(self._ready)

    def __iter__(self):
        return self

    def __next__(self):
        if self._remaining <= 0:
            self._progress.close()
            raise StopIteration

        self._remaining -= 1
        return self._next_completed()

    def _admit(self) -> bool:
        try:
            awaitable = next(self._awaitables)
        except StopIteration:
            return False

        self._pending.add(asyncio.create_task(awaitable))
        return True

    def _fill(self) -> None:
        while not self._closed and len(self._pending) + len(self._ready) < self._max_resident_tasks and self._admit():
            pass

    async def _next_completed(self):
        async with self._lock:
            if self._closed:
                raise asyncio.CancelledError

            if not self._started:
                self._started = True
                self._fill()

            if not self._ready:
                if not self._pending:
                    raise RuntimeError("rollout completion iterator exhausted unexpectedly")

                done, pending = await asyncio.wait(
                    self._pending,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if self._closed:
                    # Close owns the snapshot; do not put drained outcomes back or refill.
                    # A cancelled waiter's traceback must not retain the transferred payloads.
                    done.clear()
                    pending.clear()
                    raise asyncio.CancelledError
                self._pending = pending
                self._ready.extend(done)

            task = self._ready.pop()
            self._fill()
            self._progress.update(1)

        # The collection owner closes the iterator and cancels resident work.
        # Direct callers retain asyncio.as_completed semantics: one failed
        # completion does not implicitly cancel unrelated completions.
        return await task

    async def aclose(self) -> None:
        await self._close(drain=False)

    async def aclose_and_drain(self) -> list[Future]:
        return await self._close(drain=True)

    async def _close(self, *, drain: bool) -> list[Future]:
        # A concurrent consumer may hold the lock while waiting for a task.
        # Freeze admission and snapshot outcomes before yielding, without taking that lock.
        if self._closed:
            return []
        self._closed = True
        resident = [*self._ready, *self._pending]
        completed = [task for task in resident if task.done()] if drain else []
        self._pending.clear()
        self._ready.clear()
        for task in resident:
            if not task.done():
                task.cancel()
        if resident:
            await asyncio.gather(*resident, return_exceptions=True)
        self._progress.close()
        return completed


class RolloutCollectionHelper(BaseModel):
    """Collect rollouts by sending tasks to a running Gym deployment.

    Each rollout batch fetches the deployment's configuration from the head server.
    A caller that runs many batches against one deployment can set ``reuse_server_client=True``.
    The helper then fetches the configuration once.
    Build a new helper after that deployment's configuration changes.
    """

    reuse_server_client: bool = Field(
        default=False,
        description="Reuse one resolved client and configuration per helper. "
        "Build a new helper after the deployment's configuration changes.",
    )
    _cached_server_client: Optional[ServerClient] = PrivateAttr(default=None)
    _cached_head_server_config: Optional[BaseServerConfig] = PrivateAttr(default=None)

    # (task index, rollout index) of every record persisted to the rollouts jsonl, kept current as rows land,
    # so a run stopped by SIGTERM can be accounted for without re-reading that file.
    _landed_keys: set = PrivateAttr(default_factory=set)
    # The CLI supplies its signal state; library callers need no CLI import or signal handler.
    _termination_requested: Callable[[], bool] | None = PrivateAttr(default=None)
    _collection_started: bool = PrivateAttr(default=False)
    _store: RolloutStore | None = PrivateAttr(default=None)

    def _preprocess_rows_from_config(self, config: RolloutCollectionConfig) -> List[Dict]:
        range_iterator = repeat(0)
        if config.limit:
            range_iterator = range(config.limit)
            print(f"Limiting the number of rows to {config.limit}")

        # Load prompt config if specified
        prompt_cfg = None
        if config.prompt_config:
            prompt_cfg = load_prompt_config(config.prompt_config)
            print(f"Using prompt config: {config.prompt_config}")

        # Search NEMO_GYM_EXTRA_ROOTS, cwd, then the install root.
        _input_path = _resolve_under_cwd_or_install(config.input_jsonl_fpath)
        if not _input_path.exists():
            raise ConfigPathNotFoundError(
                f"Input file not found: '{config.input_jsonl_fpath}' (--input). Check the path is spelled correctly."
            )
        with open(_input_path) as input_file:
            rows_iterator: Iterator[str] = tqdm(input_file, desc="Reading rows")
            rows_iterator: Iterator[tuple[int, str]] = zip(range_iterator, rows_iterator)
            raw_rows = [
                (row_idx, row_str, loads_jsonl_line(row_str, _input_path, line_no))
                for line_no, (row_idx, row_str) in enumerate(rows_iterator, 1)
            ]
        _reject_attempt_qualified_rollout_ids([row for _, _, row in raw_rows], _input_path)

        # Validate and apply prompt config before per-row processing
        if prompt_cfg is not None:
            validate_prompt_compatibility([row for _, _, row in raw_rows], prompt_cfg)
            raw_rows = [(idx, s, apply_prompt_to_row(row, prompt_cfg)) for idx, s, row in raw_rows]

        return RolloutCollectionHelper._preprocess_raw_rows(raw_rows, config)

    def preprocess_examples(
        self,
        examples: List[Dict],
        *,
        agent_map: Optional[Dict[str, str]] = None,
        fan_out: Optional[Dict[str, List[str]]] = None,
        num_repeats: Union[int, Dict[str, int]] = 1,
        num_repeats_add_seed: Union[bool, Dict[str, bool]] = False,
        global_config_dict: Optional[DictConfig] = None,
    ) -> List[Dict]:
        """Apply run-level routing and repetition to caller-held rows.

        Public entry point for direct ``run_examples`` callers (e.g. trainer integrations that
        drive dispatch themselves): ``run_examples`` resolves task_sources and validates agent
        names, but ``agent_map``, ``fan_out`` and ``num_repeats`` are applied only during
        preprocessing. Call this first, then pass the returned rows to ``run_examples``.

        Pass ``global_config_dict`` (the merged config) to also resolve task_source-only rows to
        their agents here; leave it None to defer that to ``run_examples``, which does it against
        the head server's config. Input rows are not mutated; the expanded, stamped copies are
        returned.
        """
        config = RolloutCollectionConfig(
            input_jsonl_fpath="<in-memory>",
            output_jsonl_fpath="<in-memory>",
            agent_map=agent_map,
            fan_out=fan_out,
            num_repeats=num_repeats,
            num_repeats_add_seed=num_repeats_add_seed,
        )
        raw_rows = [
            (idx, orjson.dumps(row, option=orjson.OPT_SORT_KEYS).decode(), row.copy())
            for idx, row in enumerate(examples)
        ]
        rows = self._preprocess_raw_rows(raw_rows, config)
        if global_config_dict is not None:
            self.resolve_task_sources(rows, global_config_dict)
        return rows

    @staticmethod
    def _preprocess_raw_rows(raw_rows: List[Tuple[int, str, Dict]], config: RolloutCollectionConfig) -> List[Dict]:
        if isinstance(config.num_repeats_add_seed, bool):
            fixed_add_seed: Optional[bool] = config.num_repeats_add_seed
            per_agent_add_seed: Dict[str, bool] = {}
            default_add_seed: Optional[bool] = None
        else:
            fixed_add_seed = None
            per_agent_add_seed = {k: v for k, v in config.num_repeats_add_seed.items() if k != "_default"}
            default_add_seed = config.num_repeats_add_seed.get("_default")
            print(f"Per-agent num_repeats_add_seed: {dict(config.num_repeats_add_seed)}")

        if fixed_add_seed:
            print(
                "Adding unique `seed` values to each input via metadata.extra_body (only honored by vLLM model servers)"
            )

        if config.agent_map:
            print(f"Routing rows via agent_map {config.agent_map}")

        if config.responses_create_params:
            print(f"Overriding responses_create_params fields with {config.responses_create_params}")
            responses_create_params_overrides = OmegaConf.to_container(
                OmegaConf.create(config.responses_create_params), resolve=True
            )
        else:
            responses_create_params_overrides = dict()

        if isinstance(config.num_repeats, int):
            fixed_num_repeats: Optional[int] = config.num_repeats
            per_agent_repeats: Dict[str, int] = {}
            default_repeats: Optional[int] = None
            print(f"Repeating rows {fixed_num_repeats} times (in a pattern of abc to aabbcc)!")
        else:
            fixed_num_repeats = None
            per_agent_repeats = {k: v for k, v in config.num_repeats.items() if k != "_default"}
            default_repeats = config.num_repeats.get("_default")
            print(f"Per-agent num_repeats: {dict(config.num_repeats)}")
        agents_seen: set[str] = set()

        # Resolve skills once for the whole run (hash is content-derived, computed at startup).
        skills_ref_dict = None
        if config.skills:
            skills_ref = load_skill_directory(config.skills.path)
            skills_ref_dict = skills_ref.model_dump()
            print(
                f"Using skills from {config.skills.path} "
                f"(hash={skills_ref.hash}, {len(skills_ref.skills)} skill(s): "
                f"{', '.join(s.name for s in skills_ref.skills)})"
            )

        # For gym eval profile to match rollouts to tasks
        row_to_task_idx: Dict[str, int] = dict()
        task_idx_to_rollout_idx: Dict[int, int] = Counter()
        row_idxs_missing_agent_ref: List[int] = []
        agents_missing_from_num_repeats: set[str] = set()
        agents_missing_from_seed_policy: set[str] = set()
        rows: List[Dict] = []
        overridden_agents: set[Tuple[str, str]] = set()
        for row_idx, row_str, row in tqdm(raw_rows, desc="Preprocessing and repeating rows"):
            task_source = row.get(TASK_SOURCE_KEY_NAME)
            taskset = _materialized_taskset(row)
            environment_server = _environment_server_for_config_row(row, config)
            # Routing basis: the name this row routes by — its agent_ref.name when present, else
            # its task_source (resolved to an agent by resolve_task_sources once the merged config
            # is in hand). agent_map[<basis>] > agent_map._default > row agent_ref > task_source.
            agent_name = (row.get(AGENT_REF_KEY_NAME) or {}).get("name")
            basis = agent_name if agent_name is not None else task_source
            if environment_server is not None:
                basis = taskset or task_source or environment_server
            elif config.agent_map:
                # A row may carry both an agent_ref and a task_source (derived artifacts do);
                # a map entry for either re-routes it, the agent name taking precedence.
                mapped = next(
                    (
                        config.agent_map[key]
                        for key in (agent_name, task_source)
                        if key is not None and key in config.agent_map
                    ),
                    None,
                )
                if mapped is None:
                    mapped = config.agent_map.get("_default")
                if mapped is not None and mapped != agent_name:
                    # Restamp only on a real re-route; a row already on `mapped` keeps its
                    # agent_ref dict byte-for-byte (it may carry extra fields like type).
                    if agent_name is not None:
                        overridden_agents.add((agent_name, mapped))
                    agent_name = mapped
                    row[AGENT_REF_KEY_NAME] = {"name": agent_name}

            # Fan-out: run this row once per listed agent (cross-product). Otherwise a single
            # target — the row's agent when known, else deferred to task_source resolution.
            targets: List[Optional[str]]
            if environment_server is not None:
                targets = [None]
            elif config.fan_out and basis is not None and basis in config.fan_out:
                targets = list(config.fan_out[basis])
            elif agent_name is not None:
                targets = [agent_name]
            elif task_source is not None:
                targets = [None]
            else:
                row_idxs_missing_agent_ref.append(row_idx)
                continue

            if taskset is not None:
                if responses_create_params_overrides:
                    raise ValueError("responses_create_params overrides are not supported for materialized task rows")
                if skills_ref_dict is not None:
                    raise ValueError("run-level skills are not supported for materialized task rows")
            else:
                # Preserve row-specific media references and template controls when run-level
                # sampling settings supply metadata. Values (including JSON strings) still replace
                # by key, and an explicit metadata=null still clears the row's metadata.
                row_params = row[RESPONSES_CREATE_PARAMS_KEY_NAME]
                merged_params = row_params | responses_create_params_overrides
                row_metadata = row_params.get("metadata")
                override_metadata = responses_create_params_overrides.get("metadata")
                if isinstance(row_metadata, Mapping) and isinstance(override_metadata, Mapping):
                    merged_params["metadata"] = {**row_metadata, **override_metadata}
                row[RESPONSES_CREATE_PARAMS_KEY_NAME] = merged_params

            # Stamp the run-level skills_ref onto the row so it is sent to the agent in the
            # /run request body and propagated to results. The source dataset stays untouched.
            if skills_ref_dict is not None and taskset is None:
                row[SKILLS_REF_KEY_NAME] = skills_ref_dict

            # Resolve task index. Honor a caller-provided value when present (e.g. when an
            # upstream slicer has stamped a globally-stable index across chunks so that
            # subsequent /aggregate_metrics groupby unions chunks correctly); otherwise dedupe
            # identical input rows to the same task index as before.
            if TASK_INDEX_KEY_NAME not in row:
                row[TASK_INDEX_KEY_NAME] = row_to_task_idx.setdefault(row_str, len(row_to_task_idx))
            if environment_server is not None:
                row[NG_ENVIRONMENT_SERVER_KEY] = environment_server

            base_row = row
            for target in targets:
                # num_repeats keys match either side of a re-route: the dispatched agent (the
                # fan-out/agent_map target) or the row's original routing key (its agent_ref.name
                # or task_source as written in the data). The dispatched agent wins when both have
                # entries, so `agent_map={source: agent}` composes with `num_repeats={source: k}`.
                # Dict-form misses batch into one consolidated raise after the loop.
                repeat_keys = [k for k in dict.fromkeys((target, basis)) if k is not None]
                agents_seen.update(repeat_keys)
                if fixed_num_repeats is not None:
                    row_num_repeats = fixed_num_repeats
                elif (matched := next((k for k in repeat_keys if k in per_agent_repeats), None)) is not None:
                    row_num_repeats = per_agent_repeats[matched]
                elif default_repeats is not None:
                    row_num_repeats = default_repeats
                else:
                    agents_missing_from_num_repeats.add(" / ".join(repeat_keys))
                    continue

                if fixed_add_seed is not None:
                    row_add_seed = fixed_add_seed
                elif (seed_matched := next((k for k in repeat_keys if k in per_agent_add_seed), None)) is not None:
                    row_add_seed = per_agent_add_seed[seed_matched]
                elif default_add_seed is not None:
                    row_add_seed = default_add_seed
                else:
                    agents_missing_from_seed_policy.add(" / ".join(repeat_keys))
                    continue

                if taskset is not None and row_add_seed:
                    # The seed is written into the top-level responses_create_params, which a
                    # materialized row keeps under task_input; the planner does not modify task_input.
                    raise ValueError("num_repeats_add_seed is not supported for materialized task rows")

                for _ in range(row_num_repeats):
                    row = base_row.copy()
                    # Restamp only when fan-out routes this copy somewhere else; otherwise keep
                    # the row's agent_ref dict byte-for-byte (it may carry extra fields like type).
                    if target is not None and (row.get(AGENT_REF_KEY_NAME) or {}).get("name") != target:
                        row[AGENT_REF_KEY_NAME] = {"name": target}

                    # Resolve rollout index
                    row[ROLLOUT_INDEX_KEY_NAME] = task_idx_to_rollout_idx[row[TASK_INDEX_KEY_NAME]]
                    task_idx_to_rollout_idx[row[TASK_INDEX_KEY_NAME]] += 1
                    if row.get(ROLLOUT_ID_KEY_NAME) is not None and (row_num_repeats > 1 or len(targets) > 1):
                        row[ROLLOUT_ID_KEY_NAME] = f"{logical_rollout_id(row)}-r{row[ROLLOUT_INDEX_KEY_NAME]}"

                    if row_add_seed:
                        row[RESPONSES_CREATE_PARAMS_KEY_NAME] = row[RESPONSES_CREATE_PARAMS_KEY_NAME].copy()
                        metadata = (row[RESPONSES_CREATE_PARAMS_KEY_NAME].get("metadata") or {}).copy()
                        row[RESPONSES_CREATE_PARAMS_KEY_NAME]["metadata"] = metadata
                        extra_body = json.loads(metadata.get("extra_body", "{}"))
                        extra_body["seed"] = row[ROLLOUT_INDEX_KEY_NAME]
                        metadata["extra_body"] = json.dumps(extra_body)

                    rows.append(row)

        if overridden_agents:
            warnings.warn(
                "agent_map overrode agent_ref values already present in the data: "
                f"{sorted(overridden_agents)}. Prior to this release, +agent_name only filled rows "
                "missing an agent_ref; it now re-routes every row.",
                stacklevel=2,
            )

        if row_idxs_missing_agent_ref:
            raise ValueError(
                f"No agent specified for rows {row_idxs_missing_agent_ref}. Provide +agent_name (or "
                "+agent_map with a _default entry), or include agent_ref or task_source in the data."
            )

        if agents_missing_from_num_repeats:
            raise ValueError(
                f"num_repeats dict has no entry for routing keys {sorted(agents_missing_from_num_repeats)} "
                f"and no '_default' fallback. Listed keys: {sorted(per_agent_repeats)}"
            )

        if agents_missing_from_seed_policy:
            raise ValueError(
                "num_repeats_add_seed dict has no entry for routing keys "
                f"{sorted(agents_missing_from_seed_policy)} and no '_default' fallback. "
                f"Listed keys: {sorted(per_agent_add_seed)}"
            )

        unknown_agents = set(per_agent_repeats) - agents_seen
        if unknown_agents:
            warnings.warn(
                f"num_repeats dict contains agent names that never appeared in input rows "
                f"(possible typo?): {sorted(unknown_agents)}",
                stacklevel=2,
            )

        unknown_seed_agents = set(per_agent_add_seed) - agents_seen
        if unknown_seed_agents:
            warnings.warn(
                "num_repeats_add_seed dict contains agent names that never appeared in input rows "
                f"(possible typo?): {sorted(unknown_seed_agents)}",
                stacklevel=2,
            )

        if config.interleave_repeats:
            print("Interleaving repeats (in a pattern of aabbcc to abcabc)")
            # Stable, so each round keeps the input order.
            rows.sort(key=lambda row: row[ROLLOUT_INDEX_KEY_NAME])

        return rows

    def _load_from_cache(
        self,
        config: RolloutCollectionConfig,
        *,
        retain_results_in_memory: bool = True,
        success_keys: Optional[set] = None,
    ) -> Tuple[List[Dict], List[Dict], List[Dict], List[List[bytes]]]:
        output = Path(config.output_jsonl_fpath)
        if manifest_path_for(output).exists():
            store = RolloutStore.read(output)
            results = store.selected("success") if retain_results_in_memory else []
            if success_keys is not None:
                for record in store.selected_records("success").values():
                    result = record.read()
                    success_keys.add((result[TASK_INDEX_KEY_NAME], result[ROLLOUT_INDEX_KEY_NAME]))
            rows = store.inputs_for(results)
            return (
                store.pending(
                    _get_max_rollout_attempts(),
                    retry_terminal_timeouts=(
                        config.retry_terminal_timeouts
                        if "retry_terminal_timeouts" in config.model_fields_set
                        else None
                    ),
                    dispatch_longest_first=config.dispatch_longest_first,
                ),
                rows,
                results,
                [[orjson.dumps(r)] for r in results],
            )
        with config.materialized_jsonl_fpath.open() as f:
            original_input_rows = list(map(orjson.loads, tqdm(f, desc="Reading materialized input rows")))
        _reject_attempt_qualified_rollout_ids(original_input_rows, config.materialized_jsonl_fpath)

        get_key = lambda r: (r[TASK_INDEX_KEY_NAME], r[ROLLOUT_INDEX_KEY_NAME])

        results: List[Dict] = []
        result_strs: List[List[bytes]] = []
        successes_seen: set = success_keys if success_keys is not None else set()

        with Path(config.output_jsonl_fpath).open("rb") as f:
            for line in tqdm(f, desc="Reading existing output rows"):
                stripped = line.strip()
                if not stripped:
                    continue

                result = orjson.loads(stripped)
                successes_seen.add(get_key(result))

                if retain_results_in_memory:
                    results.append(result)
                    result_strs.append([stripped])

        # Sidecar: one row per non-kill_shaped failure attempt. Count attempts
        # per key + flag terminal rows so chain-hop 2 retries the right ones.
        failures_fpath = failures_path_for(Path(config.output_jsonl_fpath))
        attempts_by_key: Counter = Counter()
        # Attempts since the last `gym eval resubmit`: a resubmit gives every non-terminal failure a fresh budget.
        # If the sidecar is later rewritten shorter (a GDPval stage restart), the offset only gives extra attempts.
        resubmit_offset = load_resubmit_offset(config.output_jsonl_fpath)
        budget_used: Counter = Counter()
        # A rollout is sidecar-terminal when its latest row is a terminal failure, picked by the rule scoring uses
        # (`_latest_failure_rows`): a later interrupted attempt supersedes an older terminal failure, the evidence of
        # an unrouted /run and a never-started row do not.
        latest_by_key: Dict[Tuple, Dict[str, Any]] = {}
        elapsed_by_key: Dict[Tuple, float] = {}
        if failures_fpath.exists():
            with failures_fpath.open("rb") as f:
                for line_index, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    fr = orjson.loads(line)
                    if TASK_INDEX_KEY_NAME not in fr or ROLLOUT_INDEX_KEY_NAME not in fr:
                        continue
                    k = (fr[TASK_INDEX_KEY_NAME], fr[ROLLOUT_INDEX_KEY_NAME])
                    if not is_non_superseding_rollout_row(fr):
                        latest_by_key[k] = fr
                    # A rollout that produced no result used no attempt; resume runs it again.
                    if is_omitted_rollout_row(fr):
                        continue
                    attempts_by_key[k] += 1
                    if line_index >= resubmit_offset:
                        budget_used[k] += 1
                    # A failed attempt still tells us how long this task runs -
                    # the only duration signal available for a task that has not
                    # succeeded yet. Keep the longest attempt seen.
                    observed = observed_elapsed(fr)
                    if observed is not None:
                        elapsed_by_key[k] = max(elapsed_by_key.get(k, 0.0), observed)

        terminal_keys = {
            k
            for k, fr in latest_by_key.items()
            if not is_omitted_rollout_row(fr)
            and is_terminal_failure(fr, retry_terminal_timeouts=config.retry_terminal_timeouts)
        }
        max_attempts = _get_max_rollout_attempts()
        maxed_out = {k for k, n in budget_used.items() if n >= max_attempts}
        from nemo_gym.rollout_recovery import has_saved_generation, has_unclassified_execution_failure

        unclassified = {
            k
            for k, row in latest_by_key.items()
            if has_unclassified_execution_failure(row) or has_saved_generation(row)
        }
        gated = successes_seen | terminal_keys | maxed_out | unclassified

        input_rows = [row for row in original_input_rows if get_key(row) not in gated]

        # Stamp the resume attempt (count of prior failures for this key) on actual retries so their
        # captured model calls are keyed separately from the prior attempt's (see
        # maybe_rollout_id_from_run_body). The first attempt (0) is left unstamped -> bare rollout id.
        for row in input_rows:
            key = get_key(row)
            attempt = attempts_by_key.get(key, 0)
            if attempt > 0:
                row[ATTEMPT_INDEX_KEY_NAME] = attempt

        # Longest-first: known-long tasks go to the front so they get a whole
        # allocation to finish in rather than being cut off at its boundary.
        # Tasks with no recorded timing keep their relative input order behind
        # them - guessing from row content is worse than not guessing.
        if config.dispatch_longest_first and elapsed_by_key:
            known = [r for r in input_rows if get_key(r) in elapsed_by_key]
            unknown = [r for r in input_rows if get_key(r) not in elapsed_by_key]
            known.sort(key=lambda r: elapsed_by_key[get_key(r)], reverse=True)
            input_rows = known + unknown
            if known:
                print(
                    f"Dispatching longest-first: {len(known)} row(s) with known timings "
                    f"(longest {elapsed_by_key[get_key(known[0])] / 60:.0f} min) ahead of "
                    f"{len(unknown)} unseen row(s)"
                )

        if retain_results_in_memory:
            key_to_row = dict(zip(map(get_key, original_input_rows), original_input_rows))
            rows = [key_to_row[get_key(result)] for result in results]
        else:
            rows = []

        print(
            f"""Resumed from cache. Found:
- {len(original_input_rows)} original input rows
- {len(successes_seen)} rows already done (in main jsonl)
- {sum(attempts_by_key.values())} prior failure attempts ({len(attempts_by_key)} unique tasks) in sidecar
- {len(terminal_keys)} sidecar-terminal → not retried
- {len(maxed_out)} hit max_attempts={max_attempts} → not retried
- {len(input_rows)} rows that still need to be run"""
        )

        return input_rows, rows, results, result_strs

    async def run_from_config(self, config: RolloutCollectionConfig) -> Tuple[List[Dict]]:
        """Collect rollouts for a whole config. Wrapped in the run-scoped `job` span.

        This is the driver side of an evaluation run and the outermost span Gym produces.
        Rollouts are not its descendants: each dispatch opens its own trace, linked to this
        span (see `_dispatch_span`), so a run's traces stay bounded by one rollout each while
        `job` keeps the run's own lifecycle. `job` is in the `default` preset and not in
        `per_rollout`.
        """
        async with observe_event_loop("driver"):
            if not is_span_group_enabled(GymSpanGroup.JOB):
                return await self._run_from_config(config)
            # Decide ownership before the job span exists: a valid span here means a caller
            # owns the trace, and rollouts keep nesting under it.
            embedded = _current_span_is_valid()
            with managed_span(GymSpanGroup.JOB, "gym.job") as job:
                owned = job.get_span_context() if job is not None and not embedded else None
                token = _OWNED_JOB_SPAN_CONTEXT.set(owned)
                try:
                    return await self._run_from_config(config)
                finally:
                    _OWNED_JOB_SPAN_CONTEXT.reset(token)

    async def _run_from_config(self, config: RolloutCollectionConfig) -> Tuple[List[Dict]]:
        # Choose the target before locking, including fresh runs. Replacing an
        # output alias would otherwise change the lock identity during collection.
        config = config.model_copy(
            update={"output_jsonl_fpath": str(resolve_rollout_path(Path(config.output_jsonl_fpath)))}
        )
        # Hold ownership from identity validation through final aggregation/reporting.
        self._collection_started = False
        self._store = None
        self._health_capture_dirs: List[Path] = []
        with run_lock(Path(config.output_jsonl_fpath)):
            try:
                return await self._run_locked_from_config(config)
            finally:
                if self._collection_started and not config.disable_aggregation and not config.disable_health_check:
                    from nemo_gym.rollout_health import _health_check_collected_rollouts

                    terminated = self._termination_requested() if self._termination_requested is not None else False

                    report = functools.partial(
                        _health_check_collected_rollouts,
                        Path(config.output_jsonl_fpath),
                        config,
                        self._landed_keys,
                        coverage=self._store.coverage() if self._store is not None else None,
                        model_call_capture_dirs=self._health_capture_dirs,
                        terminated=terminated,
                    )
                    if terminated:
                        report()  # The small provisional summary must finish within Slurm's KillWait.
                    else:
                        await asyncio.to_thread(report)

    async def _run_locked_from_config(self, config: RolloutCollectionConfig) -> Tuple[List[Dict]]:
        output_fpath = Path(config.output_jsonl_fpath)
        failures_fpath = failures_path_for(output_fpath)
        environment_server_client = (
            self.setup_server_client()
            if config.environment_routing_mode != "agent" or config.environment_server_routes
            else None
        )
        # With --no-serve, local config may contain only collector options.
        # Fingerprint the running servers, as dispatch does, in every route mode.
        identity_client = environment_server_client or self.setup_server_client()
        global_config = (
            environment_server_client.global_config_dict
            if environment_server_client is not None
            else get_global_config_dict()
        )

        batch_tracker = None
        batch_agent_refs: Dict[Tuple[int, int], Dict] = {}

        def validate_batch(rows: list[dict]) -> None:
            nonlocal batch_tracker, batch_agent_refs
            if config.batch_manifest_fpath:
                batch_tracker = BatchStatusTracker(Path(config.batch_manifest_fpath), rows)
                batch_agent_refs = {
                    (row[TASK_INDEX_KEY_NAME], row[ROLLOUT_INDEX_KEY_NAME]): row[AGENT_REF_KEY_NAME] for row in rows
                }

        if config.resume_from_cache and config.batch_manifest_fpath:
            if config.materialized_jsonl_fpath.exists() and output_fpath.exists():
                # Legacy imports can skip preparation. Validate their full saved
                # inventory before the store repairs or migrates any artifacts.
                with config.materialized_jsonl_fpath.open("rb") as materialized_file:
                    validate_batch([orjson.loads(line) for line in materialized_file if line.strip()])

        def prepare_inputs() -> tuple[list[dict], RunManifest]:
            input_rows = self._preprocess_rows_from_config(config)
            # Materialized inputs must contain resolved agent routing for custom
            # drivers and for reproducible identity checks on resume.
            direct_source_rows = [row for row in input_rows if NG_ENVIRONMENT_SERVER_KEY not in row]
            if any(
                (row.get(AGENT_REF_KEY_NAME) or {}).get("name") is None and row.get(TASK_SOURCE_KEY_NAME) is not None
                for row in direct_source_rows
            ):
                self.resolve_task_sources(direct_source_rows, identity_client.global_config_dict)
            if environment_server_client is not None:
                self._stamp_environment_server_agent_refs(input_rows, environment_server_client.global_config_dict)
            validate_batch(input_rows)
            manifest = RunManifest.create(
                _resolve_under_cwd_or_install(config.input_jsonl_fpath),
                input_rows,
                config.model_dump(),
                identity_client.global_config_dict,
            )
            return input_rows, manifest

        store = RolloutStore.start_or_resume(
            output_fpath,
            prepare_inputs,
            resume=config.resume_from_cache,
            allow_unsafe=config.allow_unsafe_resume,
            resubmit=config.resubmit,
        )
        if config.resume_from_cache and store.started_fresh:
            # The exporters reopened the run recorded next to this output, which starts anew: so does that run.
            for exporter in get_exporters():
                exporter.start_over()
        from nemo_gym.rollout_health import _remove_health_reports

        _remove_health_reports(output_fpath)
        self._store = store
        self._collection_started = True
        persisted_success_keys = self._landed_keys
        persisted_success_keys.clear()
        persisted_success_keys.update(
            (row[TASK_INDEX_KEY_NAME], row[ROLLOUT_INDEX_KEY_NAME])
            for row in store.inputs_for(
                [{ROLLOUT_ID_KEY_NAME: identity} for identity, _ in store.selected_records("success")]
            )
        )
        input_rows = store.pending(
            retry_terminal_timeouts=(
                config.retry_terminal_timeouts if "retry_terminal_timeouts" in config.model_fields_set else None
            ),
            dispatch_longest_first=config.dispatch_longest_first,
        )
        results = store.selected("success") if config.retain_results_in_memory else []
        rows = store.inputs_for(results)

        batch_completed_rows: List[Dict] = []
        batch_failure_rows: List[Dict] = []
        if batch_tracker is not None:
            # The manifest selects the current attempt. Reading raw JSONL here
            # could turn an older success into completed work after a newer retry.
            batch_completed_rows = [
                _batch_status_row(record.read(), batch_agent_refs)
                for record in store.selected_records("success").values()
            ]
            batch_failure_rows = [_batch_status_row(row, batch_agent_refs) for row in store.failures()]
            batch_tracker.write_status(batch_completed_rows, batch_failure_rows, force=True)
            print(f"Validated batch manifest and initialized status at {batch_tracker.status_fpath}")
        semaphore = nullcontext()
        if config.num_samples_in_parallel:
            print(f"Querying with {config.num_samples_in_parallel} concurrent requests")
            semaphore = Semaphore(config.num_samples_in_parallel)

        # Resolve capture dirs once so each rollout's captured model calls can be folded
        # into its record below (uniform across agents; no-op when capture is off / dirs absent).
        capture_dirs = model_call_capture_dirs_from_config(global_config)
        # The post-run health check times failed rollouts from the same captures.
        self._health_capture_dirs = capture_dirs
        observability_enabled = observability_enabled_from_config(global_config)
        # Live health metrics follow the metrics exporter alone: `disable_health_check` only gates the post-run
        # report. A traced dispatch also gets a `gym.rollout.health` span, whatever the metrics exporter does.
        health_metrics_enabled = is_metrics_exporter_active()
        health_ignored_checks = frozenset(config.health_check_ignored_checks)
        # Read the incomplete-capture policy once, then apply it to each rebuild.
        mask_incomplete_when_attributed = mask_incomplete_when_attributed_from_config(global_config)
        # Resolve the training-token store directory once.
        # Training capture is independent of evaluation capture.
        # An empty result disables training-token capture.
        token_capture_dirs = token_id_capture_dirs_from_config(global_config)
        # The finalizer reads and freezes records through this source.
        # The source is absent when capture or response rebuilding is disabled.
        # A framework-owned transport may rebuild through its own source.
        # The sink still records captures when Gym does not rebuild.
        # Reruns still clear deterministic rollout ids before dispatch.
        token_source = None
        owned_token_source = None
        token_capture_config = TokenIdCaptureConfig.model_validate(global_config)
        if token_capture_config.enabled and token_capture_config.token_id_capture.rebuild_response:
            token_source = installed_token_source()

        # Clear only the attempts about to be dispatched. Recovery allocates a new
        # attempt index even when an earlier attempt died without a result.
        if capture_dirs:
            print("Clearing existing model-call captures for rollouts being dispatched")
            clear_model_call_captures_for_rollouts(input_rows, capture_dirs)
        # Participation depends only on the agent name and the static global config.
        # Decide it once per agent, so the per-row and per-rollout checks below do not revalidate the global config.
        token_capture_enabled_by_agent: dict[str | None, bool] = {}

        def token_capture_enabled_for_row(row: dict[str, Any]) -> bool:
            agent_name = self._agent_name_for_row(row, global_config)
            if agent_name not in token_capture_enabled_by_agent:
                token_capture_enabled_by_agent[agent_name] = token_id_capture_enabled_for_agent(
                    global_config, agent_name, config=token_capture_config
                )
            return token_capture_enabled_by_agent[agent_name]

        token_capture_rows = [row for row in input_rows if token_capture_enabled_for_row(row)]
        if (
            token_capture_config.token_id_capture.rebuild_response
            and token_capture_rows
            and token_source is None
            and not token_capture_dirs
        ):
            raise ValueError(
                "Token capture response rebuilding requires a TokenSource in the rollout-collector process. "
                "Call install_token_source before starting collection or configure token_id_capture.dir."
            )
        if token_capture_dirs and token_capture_rows:
            # Token stores append under deterministic rollout ids.
            # Clear stale records to avoid merging different attempts.
            print("Clearing existing token captures for rollouts being dispatched")
            clear_token_captures_for_rollouts(token_capture_rows, token_capture_dirs)

        # Stop a run that produces mostly masked captures.
        finalized_count = 0
        masked_count = 0
        mask_reasons: Counter = Counter()
        warned_malformed_rollout_id = False

        # Intermediate status printing
        pcts_to_print = list(range(1, 100)) + [99.5, 100]
        agent_name_to_metrics = defaultdict(Counter)
        agent_name_to_counts = defaultdict(int)
        # How many results reported each metric, so a result without a metric (such as an unscored
        # result without a reward) does not dilute that metric's average.
        agent_name_to_metric_counts = defaultdict(Counter)
        # Quality accounting restricted to persisted rollouts: `count`/`reward` over the
        # unmasked ones, `masked` over the rest. Token capture already reports its own
        # masking; this is the same accounting for what an environment declares on its
        # verify response.
        agent_name_to_scored = defaultdict(Counter)
        # Rollouts that never reach the main output at all, kept apart from quality.
        agent_name_to_dropped = defaultdict(Counter)
        counts_left = Counter(self._dispatch_name(row) for row in input_rows)
        dispatched_per_agent = Counter(counts_left)
        start_time = time.time()
        environment_routes = Counter(
            row[NG_ENVIRONMENT_SERVER_KEY] for row in input_rows if NG_ENVIRONMENT_SERVER_KEY in row
        )
        if environment_routes:
            print(f"Environment server routes: {dict(environment_routes)}")

        if config.route_failures_to_sidecar:
            print(
                "route_failures_to_sidecar is on: a failed agent /run becomes a sidecar row instead of "
                "ending the run, and its rollout leaves the score.",
                flush=True,
            )

        exporters_enabled = bool(get_exporters())
        upload_spool_fpath = output_fpath.with_suffix(output_fpath.suffix + ".upload.tmp")
        upload_spool = None
        if len(dispatched_per_agent) > 1:
            print(f"Dispatching rollouts round-robin across {len(dispatched_per_agent)} agents")
        completed_count = 0
        sources_dropped = False
        collection_succeeded = False
        latency_tracker = DispatchLatencyTracker()
        run_timing: Optional[RunTimingReport] = None
        if config.dispatch_start_report_within_s is not None:
            run_timing = RunTimingReport(
                total=len(input_rows),
                start_within_s=config.dispatch_start_report_within_s,
                start_min_fraction=config.dispatch_start_report_min_fraction,
                all_started_within_s=config.dispatch_all_started_within_s,
                long_rollout_s=config.long_rollout_s,
                long_rollout_max_fraction=config.long_rollout_max_fraction,
                timed_out_max_fraction=config.timed_out_max_fraction,
                max_waiting_ratio=config.server_queue_max_waiting_ratio if config.server_queue_metrics_urls else None,
            )
        timing_heartbeat: Optional[asyncio.Task] = None
        queue_sampler: Optional[asyncio.Task] = None
        sigterm_previous: Any = _NO_SIGTERM_HOOK
        if config.dispatch_budget_s is not None:
            print(
                f"Dispatch budget: {config.dispatch_budget_s / 60:.0f} min. New rollouts stop being "
                + (
                    f"dispatched with under {config.drain_margin_s / 60:.0f} min left."
                    if config.drain_margin_s is not None
                    else "dispatched once less time remains than this run's observed p75 task duration."
                )
            )

        async with AsyncExitStack() as output_files:
            if (
                token_source is None
                and token_capture_dirs
                and token_capture_config.enabled
                and token_capture_config.token_id_capture.rebuild_response
            ):
                token_source = TokenCaptureStore(token_capture_dirs[0])
                owned_token_source = token_source
            if owned_token_source is not None:
                output_files.push_async_callback(owned_token_source.close)
            output_files.enter_context(store)
            if not config.retain_results_in_memory and config.upload_rollouts and exporters_enabled:
                output_files.callback(
                    lambda: None if collection_succeeded else upload_spool_fpath.unlink(missing_ok=True)
                )
                upload_spool = output_files.enter_context(upload_spool_fpath.open("w+b"))
                for record in store.selected_records("success").values():
                    upload_spool.write(orjson.dumps(_rollout_for_export(record.read())) + b"\n")
            if run_timing is not None:

                async def _timing_heartbeat() -> None:
                    while True:
                        await asyncio.sleep(config.timing_summary_interval_s)
                        # stdout is block-buffered when written to a log file; flush so the table shows up live.
                        print(run_timing.table(), flush=True)

                async def stop_timing() -> None:
                    for task in (timing_heartbeat, queue_sampler):
                        if task is not None:
                            task.cancel()
                    await asyncio.gather(
                        *(task for task in (timing_heartbeat, queue_sampler) if task is not None),
                        return_exceptions=True,
                    )
                    _restore_sigterm(sigterm_previous)
                    print(run_timing.table(), flush=True)

                output_files.push_async_callback(stop_timing)
                if config.timing_summary_interval_s:
                    timing_heartbeat = asyncio.create_task(_timing_heartbeat())
                if config.server_queue_metrics_urls:
                    queue_sampler = asyncio.create_task(
                        _sample_server_queues(
                            run_timing, config.server_queue_metrics_urls, config.server_queue_sample_interval_s
                        )
                    )
                sigterm_previous = _print_on_sigterm(run_timing.table)

            failure_counts: Counter = Counter()
            dispatches = self._run_examples_with_metadata(
                input_rows,
                semaphore=semaphore,
                route_failures_to_sidecar=config.route_failures_to_sidecar,
                record_unrouted_failures=True,
                # The persisted attempt and recovery selector own retries, including uncertain delivery.
                retry_requests=False,
                _max_num_tries=1,
                typed_outcomes=config.route_failures_to_sidecar,
                validate_results=True,
                on_dispatch_batch=store.record_dispatches,
                max_resident_tasks=config.max_resident_rollout_tasks,
                interleave_by_agent=True,
                dispatch_budget_s=config.dispatch_budget_s,
                drain_margin_s=config.drain_margin_s,
                latency_tracker=latency_tracker,
                **({"run_timing": run_timing} if run_timing is not None else {}),
            )
            if hasattr(dispatches, "aclose"):
                output_files.push_async_callback(dispatches.aclose)
            elif hasattr(dispatches, "close"):
                output_files.callback(dispatches.close)
            for future in dispatches:
                completed = await future
                row, result, rollout_latency_ms = completed.row, completed.result, completed.rollout_latency_ms
                if _materialized_taskset(row) is not None and _is_episode_response(result):
                    result = _native_episode_record(row, result)
                if isinstance(result, dict) and result.get(NG_DISPATCH_DRAINED_KEY):
                    # This is a scheduling decision, not a dispatched attempt.
                    completed_count += 1
                    counts_left[self._dispatch_name(row)] -= 1
                    identity = {
                        key: row[key]
                        for key in (
                            TASK_INDEX_KEY_NAME,
                            ROLLOUT_INDEX_KEY_NAME,
                            ROLLOUT_ID_KEY_NAME,
                            ATTEMPT_INDEX_KEY_NAME,
                            AGENT_REF_KEY_NAME,
                        )
                        if key in row
                    }
                    if config.record_omitted_rollouts:
                        store.record_unstarted(omitted_rollout_record(result | identity))
                    if config.retain_results_in_memory:
                        result.update(identity)
                        rows.append(row)
                        results.append(result)
                    continue
                if isinstance(result, dict) and _materialized_taskset(row) is None:
                    _sanitize_legacy_result(result)
                verification_response = completed.verification_response
                diagnostics = completed.diagnostics or _failure_diagnostics(result)
                metric_result = diagnostics
                if isinstance(result, RolloutFailureRecord) and metric_result.get(NG_FAILURE_CLASS_KEY) is None:
                    # Custom typed dispatchers may provide diagnostics without
                    # legacy routing labels; use the typed failure for those.
                    metric_result = {
                        NG_FAILURE_CLASS_KEY: result.sidecar_failure_class,
                        "_ng_failure_kind": result.failure.failure_kind,
                        "_ng_failure_type": result.exception_type,
                    }
                if verification_response is None:
                    verification_response = _judge_failure_response(result)
                # Custom dispatch implementations may return an already-completed
                # future. Associate those outcomes too; normal HTTP dispatch records
                # its event before the request through on_dispatch above.
                store.record_dispatch(row)
                # HTTP results were validated at the request boundary. Custom
                # drivers returning _CompletedRollout own their completed schema.
                if isinstance(result, dict):
                    if "failure" in result and "episode_id" in result:
                        result = _normalize_rollout_outcome(row, result)
                    elif result.get(NG_FAILURE_CLASS_KEY) is not None:
                        result = _failure_outcome(row, result, "agent")
                structured_failure = isinstance(result, RolloutFailureRecord)
                if structured_failure:
                    compatibility = _failure_compatibility_row(result, verification_response)
                    # The canonical alias must agree with the typed record. Keep a distinct
                    # producer class as collector-owned evidence for existing score selectors.
                    legacy_class = diagnostics.get(NG_FAILURE_CLASS_KEY)
                    if isinstance(legacy_class, str) and legacy_class != result.sidecar_failure_class:
                        compatibility[NG_FAILURE_LEGACY_CLASS_KEY] = legacy_class
                    if verification_response is not None and matches_failure_class(
                        compatibility, {"judge_failed", "judge_invalid"}
                    ):
                        compatibility["response"] = verification_response
                    # Keep producer evidence without making it the canonical failure reason.
                    if diagnostics.get("error"):
                        compatibility.pop("error")
                    result = diagnostics | compatibility
                if _materialized_taskset(row) is not None and isinstance(result, dict):
                    # Result or failure, the record names the inputs it ran (task and dataset digests).
                    for key, value in _task_data_provenance(row).items():
                        result.setdefault(key, value)
                if completed.elapsed_s is not None:
                    # Read by the timeouts section of the run health summary. Not `elapsed_seconds`,
                    # which resume reads to order dispatch.
                    result[ELAPSED_S_KEY] = completed.elapsed_s

                result[TASK_INDEX_KEY_NAME] = row[TASK_INDEX_KEY_NAME]
                result[ROLLOUT_INDEX_KEY_NAME] = row[ROLLOUT_INDEX_KEY_NAME]
                if AGENT_REF_KEY_NAME in row:
                    result[AGENT_REF_KEY_NAME] = row[AGENT_REF_KEY_NAME]
                result[RUN_ID_KEY] = store.manifest.run_id
                if TASK_SOURCE_KEY_NAME in row:
                    result[TASK_SOURCE_KEY_NAME] = row[TASK_SOURCE_KEY_NAME]
                if SKILLS_REF_KEY_NAME in row:
                    result[SKILLS_REF_KEY_NAME] = row[SKILLS_REF_KEY_NAME]
                if ATTEMPT_INDEX_KEY_NAME in row:
                    result[ATTEMPT_INDEX_KEY_NAME] = row[ATTEMPT_INDEX_KEY_NAME]
                if ROLLOUT_ID_KEY_NAME in row:
                    # Capture readback recomputes the id from the finished record.
                    # Preserve an explicit id on the result just like the indices.
                    result[ROLLOUT_ID_KEY_NAME] = row[ROLLOUT_ID_KEY_NAME]
                if completed.environment_server is not None:
                    result[NG_ENVIRONMENT_SERVER_KEY] = completed.environment_server
                elif NG_ENVIRONMENT_SERVER_KEY in row:
                    result[NG_ENVIRONMENT_SERVER_KEY] = row[NG_ENVIRONMENT_SERVER_KEY]
                if completed.environment_server_type is not None:
                    result[NG_RESULT_TYPE_KEY] = completed.environment_server_type
                if completed.failure_report is not None:
                    result = FailureReportRow.merged_into(result, completed.failure_report)

                # Set below for a rollout whose trajectory is health-evaluated once the bookkeeping is done.
                health_outcome: Optional[str] = None
                no_persist = bool(result.get(NG_NO_PERSIST_KEY))
                failure_class = result.get(NG_FAILURE_CLASS_KEY)
                # No score exists. Judge failures may still carry real generation evidence.
                no_result = no_persist or structured_failure or failure_class in _NO_RESULT_FAILURE_CLASSES

                _record_policy_model_server(result, global_config, self._agent_name_for_row(row, global_config))

                # Fold this rollout's captured model calls into its record (uniform across agents; no-op
                # when capture is off). Never alters the harness output/reward already in `result`.
                # Preserve failure captures even without observations. Explicit agent or verifier evidence also
                # survives legacy no-persist markers, since reported failures are saved for recovery.
                if capture_dirs and (
                    not no_persist or NG_AGENT_OBSERVATIONS_KEY in result or NG_VERIFIER_INVOCATIONS_KEY in result
                ):
                    # A cancelled call's capture write runs detached, so drain the claims this
                    # rollout still holds before snapshotting; otherwise a late row is lost.
                    rollout_id_for_capture = maybe_rollout_id_from_run_body(result)
                    if rollout_id_for_capture is not None:
                        await await_model_call_capture_quiescence(rollout_id_for_capture, capture_dirs)
                    merge_model_call_capture_into_record(
                        result,
                        capture_dirs,
                        include_payloads=not _has_observation_gap(result, "multimodal_history_redacted"),
                    )

                if (
                    "ng_model_call_capture" in result
                    or NG_AGENT_OBSERVATIONS_KEY in result
                    or NG_TRAJECTORY_KEY in result
                    or NG_VERIFIER_INVOCATIONS_KEY in result
                ):
                    _attach_trajectory_record(row, result)

                # Assembles ng_perf from ng_trajectory when observability is enabled.
                _attach_ng_perf(
                    result, observability_enabled=observability_enabled, rollout_latency_ms=rollout_latency_ms
                )

                # Freeze and rebuild tokens only for participating agents.
                # This step does not retire the frozen snapshot.
                # It leaves harness output and reward unchanged.
                # Direct callers of run_examples finalize each record themselves.
                token_capture_build = None
                recoverable_generation = matches_failure_class(
                    result, {"judge_failed", "judge_invalid"}
                ) and isinstance(result.get("response"), dict)
                if (not no_result or recoverable_generation) and token_capture_enabled_for_row(row):
                    token_capture_build = await finalize_rollout_token_capture(
                        result, token_source, mask_incomplete_when_attributed=mask_incomplete_when_attributed
                    )
                    if token_capture_build is not None and not no_result:
                        finalized_count += 1
                        if token_capture_build.get(MASK_SAMPLE_KEY):
                            masked_count += 1
                            # Aggregate available reasons for the abort message.
                            build_metrics = token_capture_build.get("metrics") or {}
                            if build_metrics.get("capture_incomplete"):
                                mask_reasons["capture_incomplete"] += 1
                            if build_metrics.get("unresolved_parent_calls"):
                                mask_reasons["unresolved_parent_calls"] += 1
                            build_error = token_capture_build.get("error") or build_metrics.get("error")
                            if build_error:
                                mask_reasons[str(build_error)] += 1
                        settings = token_capture_config.token_id_capture
                        if (
                            settings.max_mask_fraction is not None
                            and finalized_count >= settings.mask_fraction_min_samples
                            and masked_count / finalized_count > settings.max_mask_fraction
                        ):
                            raise RuntimeError(
                                f"{masked_count}/{finalized_count} finalized rollouts "
                                f"({masked_count / finalized_count:.1%}) are masked, exceeding "
                                f"token_id_capture.max_mask_fraction={settings.max_mask_fraction}. "
                                f"Mask reasons: {dict(mask_reasons)}. Aborting instead of collecting "
                                "mostly token-less data."
                            )

                if failure_class is not None:
                    result[NG_FAILURE_MAX_ATTEMPTS_KEY] = store.manifest.max_rollout_attempts
                completed_count += 1
                if config.retain_results_in_memory:
                    rows.append(row)
                    results.append(result)
                if upload_spool is not None:
                    upload_spool.write(orjson.dumps(_rollout_for_export(result)) + b"\n")

                dispatch_name = self._dispatch_name(row)
                if completed.run_error is not None:
                    # route_failures_to_sidecar is off, so this failed /run ends the run. Its identity, cause and
                    # report are kept as non-superseding evidence: they do not replace a score or earlier failure.
                    # Admission still stops an unresolved possibly-delivered request from running the agent again.
                    store.record_unstarted(omitted_rollout_record(result) | {NG_UNROUTED_RUN_ERROR_KEY: True})
                    _record_rollout_outcome(result, "omitted", dispatch_name)
                    print(
                        "[rollout_collection] the run stops at a failed /run (route_failures_to_sidecar is off); "
                        f"it is recorded in {failures_path_for(store.output)}. {len(input_rows) - completed_count + 1} "
                        f"of {len(input_rows)} dispatched rollouts have no result; saved evidence governs resume admission.",
                        flush=True,
                    )
                    raise completed.run_error
                if not sources_dropped:
                    # The rows this run adds are not the merged ones an aggregation's shard list names. Every outcome
                    # below writes a row, so the list goes with the first; a run that writes none leaves it as it was.
                    aggregation_sources_path_for(output_fpath).unlink(missing_ok=True)
                    sources_dropped = True
                if no_persist:
                    # A returned suppression without a failure is an intentional
                    # omission. Known kill-shaped failures were converted above.
                    store.record_omission(row, str(result.get("error") or "Producer requested no result persistence"))
                    _record_rollout_outcome(metric_result, "omitted", dispatch_name)
                    if batch_tracker is not None:
                        # This attempt supersedes any failure reported before resume.
                        batch_failure_rows = [
                            failure
                            for failure in batch_failure_rows
                            if (failure[TASK_INDEX_KEY_NAME], failure[ROLLOUT_INDEX_KEY_NAME])
                            != (row[TASK_INDEX_KEY_NAME], row[ROLLOUT_INDEX_KEY_NAME])
                        ]
                elif failure_class is not None:
                    # Failures remain separate from completed generation data.
                    failure_counts[failure_class] += 1
                    # Every dropped rollout says so as it happens, whichever layer classified it.
                    # tqdm.write keeps the line off the progress bar it would otherwise collide with.
                    detail = str(result.get("error") or result.get("_ng_failure_message") or "")[:200]
                    tqdm.write(
                        "🚨 [rollout_collection] rollout dropped from the score: "
                        f"row={json.dumps(_rollout_request_debug_summary(row), sort_keys=True)} "
                        f"class={failure_class} execution {row.get(ATTEMPT_INDEX_KEY_NAME, 0)} error={detail}"
                    )
                    store.record_outcome(result)
                    outcome = "omitted" if store.disposition(row) == "omitted" else "failed"
                    _record_rollout_outcome(metric_result, outcome, dispatch_name)
                    # Health eligibility is separate from scoring: typed failures have no score,
                    # but a reported environment failure may retain an executed trajectory.
                    if outcome == "failed" and (
                        failure_class not in _NO_RESULT_FAILURE_CLASSES
                        or (
                            structured_failure
                            and result["_ng_failure_record"]["source"] == "environment"
                            and (NG_AGENT_OBSERVATIONS_KEY in result or NG_TRAJECTORY_KEY in result)
                        )
                    ):
                        health_outcome = "failed"
                    if batch_tracker is not None:
                        batch_failure_rows.append(_batch_status_row(result, batch_agent_refs))
                else:
                    # Success → main jsonl.
                    store.record_outcome(result, sync=capture_build_can_retire(token_capture_build))
                    persisted_success_keys.add((row[TASK_INDEX_KEY_NAME], row[ROLLOUT_INDEX_KEY_NAME]))
                    _record_rollout_outcome(result, "scored", dispatch_name)
                    health_outcome = "scored"
                    _record_rollout_shape(row, result)
                    if batch_tracker is not None:
                        batch_completed_rows.append(_batch_status_row(result, batch_agent_refs))
                    try:
                        rollout_id = maybe_rollout_id_from_run_body(result)
                    except (TypeError, ValueError) as error:
                        # Preserve capture evidence when the rollout id is invalid.
                        rollout_id = None
                        if not warned_malformed_rollout_id:
                            warned_malformed_rollout_id = True
                            warnings.warn(
                                f"a result carries a malformed rollout id ({error}); "
                                "its token capture will not be retired.",
                                stacklevel=2,
                            )
                    if rollout_id is not None and capture_build_can_retire(token_capture_build):
                        await retire_rollout_token_capture(rollout_id, token_source, token_capture_build)

                counts_left[dispatch_name] -= 1
                if counts_left[dispatch_name] <= 0:
                    counts_left.pop(dispatch_name)

                if batch_tracker is not None:
                    batch_tracker.write_status(batch_completed_rows, batch_failure_rows)

                agent_name = dispatch_name
                if not no_result:
                    # An infrastructure failure is not a score of zero, and not a sample either.
                    metrics = agent_name_to_metrics[agent_name]
                    numeric = {
                        k: v for k, v in result.items() if isinstance(v, (int, float)) and not k.startswith("_")
                    }
                    metrics.update(numeric)
                    agent_name_to_metric_counts[agent_name].update(numeric.keys())
                    agent_name_to_counts[agent_name] += 1

                # Use the persisted classification so terminal skips count as omissions
                # and kill-shaped failures count as failures, just as in final coverage.
                disposition = store.disposition(row)
                if disposition in {"failure", "omitted"}:
                    agent_name_to_dropped[agent_name].update({"failed" if disposition == "failure" else "omitted": 1})
                elif result.get(MASK_SAMPLE_KEY):
                    agent_name_to_scored[agent_name].update({"masked": 1})
                elif "reward" in result:
                    agent_name_to_scored[agent_name].update({"reward": float(result.get("reward") or 0.0), "count": 1})

                current_pct = 100 * completed_count / len(input_rows)
                if pcts_to_print and current_pct >= pcts_to_print[0]:
                    while pcts_to_print and current_pct >= pcts_to_print[0]:
                        pcts_to_print.pop(0)

                    time_taken_s = time.time() - start_time
                    time_taken = timedelta(seconds=int(time_taken_s))
                    rollouts_per_min = completed_count / (time_taken_s / 60)
                    print_str = f"Finished {completed_count} / {len(input_rows)} rollouts ({int(current_pct)}%) in {time_taken} ({rollouts_per_min:.2f} rollouts/min). "

                    top_left = counts_left.most_common()
                    top_left_str = "\n".join(f"{i + 1}. {k}: {v}" for i, (k, v) in enumerate(top_left))
                    print_str += f"""Examples left:
{top_left_str}
"""
                    for agent_name in sorted(agent_name_to_metrics):
                        metrics = agent_name_to_metrics[agent_name]
                        agent_total_samples = dispatched_per_agent[agent_name]
                        agent_sample_pct = 100 * agent_name_to_counts[agent_name] / agent_total_samples
                        metric_counts = agent_name_to_metric_counts[agent_name]
                        avg_metrics = {k: v / metric_counts[k] for k, v in metrics.items()}
                        print_str += f"""Found {agent_name_to_counts[agent_name]} / {agent_total_samples} ({agent_sample_pct:.2f}%) rollouts for `{agent_name}`.
{json.dumps(avg_metrics, indent=4)}
"""
                    # Use tqdm.write here so we can print properly with tqdm being used.
                    tqdm.write(print_str)

                    if get_exporters():
                        step_metrics = {"progress/total/rollouts_per_min": rollouts_per_min}
                        for agent_name, metrics in agent_name_to_metrics.items():
                            scored = agent_name_to_metric_counts[agent_name]["reward"]
                            if not scored:
                                continue
                            step_metrics[f"progress/{agent_name}/reward"] = round(100 * metrics["reward"] / scored, 2)
                            step_metrics[f"progress/{agent_name}/reward_lower_bound"] = round(
                                100 * metrics["reward"] / (counts_left[agent_name] + scored),
                                2,
                            )

                        # Include agents with no successful outcomes so total failures
                        # remain visible even when they have no reward series.
                        for agent_name in sorted(agent_name_to_scored.keys() | agent_name_to_dropped.keys()):
                            step_metrics.update(
                                _masking_step_metrics(
                                    agent_name,
                                    agent_name_to_scored.get(agent_name, Counter()),
                                    agent_name_to_dropped.get(agent_name, Counter()),
                                )
                            )

                        export_metrics(step_metrics, step=int(current_pct))

                # Last, so a cancellation here costs a health sample and never the bookkeeping above.
                health_span_context = completed.dispatch_span_context
                if health_outcome is not None and (health_metrics_enabled or health_span_context is not None):
                    await asyncio.to_thread(
                        _record_rollout_health,
                        result,
                        health_outcome,
                        health_ignored_checks,
                        health_span_context,
                    )

            collection_succeeded = True

        # The collection API already retains returned results. Reuse those
        # objects instead of rereading another copy of every saved trajectory.
        selected_attempts = store.selected_records("success")
        persisted_results = [
            result
            for result in results
            if (logical_rollout_id(result), result.get(ATTEMPT_INDEX_KEY_NAME, 0)) in selected_attempts
        ]
        persisted_rows = store.inputs_for(persisted_results)
        completion = store.coverage()
        print(
            f"Rollout coverage: {completion['successful']}/{completion['expected']} completed, "
            f"of which {completion['measured']} measured, {completion['masked']} masked, "
            f"and {completion['unscored']} unscored; "
            f"{completion['failed']} failed, {completion['intentionally_omitted']} intentionally omitted, "
            f"{completion['not_started']} not started, {completion['missing']} missing outcomes. "
            f"Details: {coverage_path_for(output_fpath)}"
        )
        # Explicitly counted failures can provide a score even when no rollout
        # succeeded. Select attempts before applying that scoring policy.
        counted = await asyncio.to_thread(
            lambda: _counted_failure_rows(store.failures(), config.count_failure_classes_as_zero)
        )
        missing = (
            await asyncio.to_thread(
                _missing_rollout_rows_counted_as_zero, [config.materialized_jsonl_fpath], [], set(), history=store
            )
            if config.count_missing_rollouts_as_zero and not config.disable_aggregation
            else []
        )
        if input_rows and not selected_attempts and not counted and not missing:
            if batch_tracker is not None:
                batch_tracker.write_status(batch_completed_rows, batch_failure_rows, force=True)
            if upload_spool is not None:
                upload_spool_fpath.unlink(missing_ok=True)
            drained_note = (
                f" {latency_tracker.drained} of them were never started because dispatch_budget_s ran out; "
                "resume_from_cache dispatches them in the next run."
                if latency_tracker.drained
                else ""
            )
            raise IncompleteEvaluationError(
                f"None of the {len(input_rows)} dispatched rollouts produced a result "
                f"{dict(failure_counts)}.{drained_note} Inspect {failures_fpath}; the run has no score to report.",
                retryable=store.coverage()["retryable"] > 0,
                made_progress=store.progress_outcomes_recorded > 0,
            )
        print(latency_tracker.summary())

        if config.upload_rollouts and exporters_enabled:  # pragma: no cover
            print("Uploading rollouts. This may take a few minutes if your data is large.")
            if config.retain_results_in_memory:
                upload_results = [
                    _rollout_for_export(result) for result in results if not result.get(NG_DISPATCH_DRAINED_KEY)
                ]
                export_rollouts(upload_results)
            else:
                assert upload_spool is not None
                try:
                    upload_results = await _read_jsonl(upload_spool_fpath)
                    export_rollouts(upload_results)
                    del upload_results
                finally:
                    upload_spool_fpath.unlink(missing_ok=True)

        if not config.retain_results_in_memory and not config.disable_aggregation:
            # Aggregation consumes only successful rows from the main artifact.
            persisted_results = await asyncio.to_thread(store.selected, "success")

        print("Sorting results to ensure consistent ordering")
        rows.sort(key=lambda r: (r[TASK_INDEX_KEY_NAME], r[ROLLOUT_INDEX_KEY_NAME]))
        results.sort(key=lambda r: (r[TASK_INDEX_KEY_NAME], r[ROLLOUT_INDEX_KEY_NAME]))
        persisted_rows.sort(key=lambda r: (r[TASK_INDEX_KEY_NAME], r[ROLLOUT_INDEX_KEY_NAME]))
        persisted_results.sort(key=lambda r: (r[TASK_INDEX_KEY_NAME], r[ROLLOUT_INDEX_KEY_NAME]))

        # Aggregate persisted results plus explicitly counted metrics-only failures and missing
        # rollouts, matching `gym eval aggregate` without changing either rollout artifact.
        imputed: List[Dict[str, Any]] = []
        if config.disable_aggregation:
            print(
                "Skipping aggregate-metrics computation because disable_aggregation=True. "
                "Run `gym eval aggregate` after all shards finish to compute the global metrics."
            )
            aggregate_metrics_fpath = None
        else:
            print("Computing aggregate metrics")
            if config.count_failure_classes_as_zero:
                print(
                    f"Counting {len(counted)} failure row(s) as scored zeros: {config.count_failure_classes_as_zero}"
                )
            if config.count_missing_rollouts_as_zero:
                imputed = missing
                print(f"Counting {len(imputed)} materialized rollout(s) with no row as scored zeros")
                counted.extend(imputed)
            # One run never reuses a (task, rollout), so its rows need no server to tell them apart.
            await asyncio.to_thread(
                _fill_task_fields, counted, persisted_results, [config.materialized_jsonl_fpath], _routing_identity
            )
            # Appending leaves a missing early repeat behind the repeats that did land. The two
            # lists are zipped positionally downstream, so they are reordered together.
            scored_rows = persisted_results + counted
            metadata_rows = persisted_rows + counted if config.retain_results_in_memory else scored_rows
            order = sorted(range(len(scored_rows)), key=lambda i: _rollout_order_key(scored_rows[i]))
            aggregate_metrics_fpath = await self._call_aggregate_metrics(
                [scored_rows[i] for i in order],
                [metadata_rows[i] for i in order],
                output_fpath,
                raise_on_error=False,  # Write batch status before reporting aggregation failures below.
            )

        if batch_tracker is not None:
            batch_tracker.write_status(
                batch_completed_rows,
                batch_failure_rows,
                aggregate_metrics_fpath=aggregate_metrics_fpath,
                aggregation_deferred=config.disable_aggregation,
                force=True,
            )

        _raise_for_aggregation_errors(aggregate_metrics_fpath)

        expected_rollouts = completion["expected"]
        scored_rollouts = completion["successful"] - completion["unscored"] + len(counted)
        store.write_coverage(
            scored=scored_rollouts, failures_counted_as_zero=len(counted) - len(imputed), imputed=len(imputed)
        )
        coverage = _coverage_report(
            expected_rollouts, scored_rollouts, failure_counts, failures_fpath, imputed=len(imputed)
        )
        if get_exporters():
            failure_metrics = await asyncio.to_thread(
                _failure_export_metrics,
                [],
                lambda row: (row.get(TASK_INDEX_KEY_NAME), row.get(ROLLOUT_INDEX_KEY_NAME)),
                persisted_success_keys,
                histories=[store],
            )
            export_metrics(
                {
                    "coverage/expected": expected_rollouts,
                    "coverage/scored": scored_rollouts,
                    "coverage/missing": completion["missing"],
                    "coverage/not_scored": expected_rollouts - scored_rollouts,
                    "coverage/measured": completion["measured"],
                    "coverage/masked": completion["masked"],
                    "coverage/unscored": completion["unscored"],
                    "coverage/failed": completion["failed"],
                    "coverage/omitted": completion["intentionally_omitted"],
                    "coverage/not_started": completion["not_started"],
                    "coverage/missing_outcomes": completion["missing"],
                    "coverage/failure_budget_exhausted": completion["failure_budget_exhausted"],
                    **({"coverage/imputed": len(imputed)} if config.count_missing_rollouts_as_zero else {}),
                    **failure_metrics,
                }
            )

        print(f"""Finished rollout collection! View results at:
Fully materialized inputs: {config.materialized_jsonl_fpath}
Rollouts: {output_fpath}
Aggregate metrics: {aggregate_metrics_fpath}{coverage}""")

        # A producer-marked final zero settles its rollout: resume does not run it again, so it is complete and in
        # the denominator, while health still reports it as a failure and no verifier execution is recorded for it.
        config.check_completion(
            expected=completion["expected"],
            results=persisted_results,
            completed=completion["successful"] + completion["final_zeros"],
            intentionally_omitted=completion["intentionally_omitted"],
            retryable=completion["retryable"] > 0,
            made_progress=store.progress_outcomes_recorded > 0,
        )
        return results

    async def _call_aggregate_metrics(
        self,
        results: List[Dict],
        rows: List[Dict],
        output_fpath: Path,
        *,
        raise_on_error: bool = True,
    ) -> Optional[Path]:
        """Call /aggregate_metrics on the environment server each rollout ran through.

        Rows are grouped by the environment server stamped on them at preprocessing
        (``_ng_environment_server``); a row without the stamp is grouped by the environment server
        that fronts its agent, as before, so the identity decided at dispatch is the one aggregation
        uses, across shards and resumed runs alike. Writes a single _aggregate_metrics.json with one
        entry per environment server (same shape as the old _agent_metrics.json, plus the server
        name). Returns the file path.
        Failed requests retain an entry with empty metric collections and ``aggregation_error``.
        By default, re-raise the first original exception after saving partial metrics, preserving
        its type and traceback for direct callers such as custom evaluation drivers. Collection and
        aggregation entrypoints pass ``raise_on_error=False`` to write batch status first, then call
        ``_raise_for_aggregation_errors`` for a summary error.
        """
        if not results:
            return None

        server_client = self.setup_server_client()
        global_config_dict = server_client.global_config_dict
        available_servers = sorted(
            str(name)
            for name, block in global_config_dict.items()
            if isinstance(block, DictConfig) and ENVIRONMENT_SERVER_TYPE_KEY_NAME in block
        )

        # Group results by the environment server they ran through.
        servers_by_agent = _environment_servers_by_agent(global_config_dict)
        server_results: Dict[str, List[Dict]] = {}
        server_agents: Dict[str, Optional[str]] = {}
        for row, result in zip(rows, results):
            agent_name = (row.get(AGENT_REF_KEY_NAME) or result.get(AGENT_REF_KEY_NAME) or {}).get("name")
            server_name = row.get(NG_ENVIRONMENT_SERVER_KEY) or result.get(NG_ENVIRONMENT_SERVER_KEY)
            if not isinstance(server_name, str):
                if not agent_name:
                    continue
                server_name = _environment_server_for_agent(agent_name, servers_by_agent)
            elif server_name not in available_servers:
                # Shards aggregated under a config that no longer declares the server that produced them.
                raise ValueError(
                    f"Result rows are stamped with environment server {server_name!r}, which is not in the "
                    f"running config (available: {available_servers}); aggregate with the config that produced them"
                )
            server_results.setdefault(server_name, []).append(result)
            if server_name not in server_agents:
                if agent_name is None:
                    # A materialized row names no agent; the environment server's own binding does.
                    agent_name = self._agent_name_for_row({NG_ENVIRONMENT_SERVER_KEY: server_name}, global_config_dict)
                server_agents[server_name] = agent_name

        # One entry per environment server, labelled by the agent it binds so metric names and
        # `agent_ref` keep today's shape. Servers that front the same agent (a session-based server and its
        # legacy_agent twin) are each labelled by their own name, whatever order their rows arrive in.
        labels = label_runs(server_agents)
        first_error: Optional[Exception] = None

        async def _fetch_agent_metrics(server_name: str, agent_name: str, agent_result_list: List[Dict]) -> Dict:
            nonlocal first_error
            # Strip generated content, but keep provider metrics for request-level
            # aggregation (e.g. SPEED-Bench), along with usage and truncation status.
            stripped = []
            for r in agent_result_list:
                entry = {
                    k: v
                    for k, v in r.items()
                    if k
                    not in (
                        "response",
                        "responses_create_params",
                        NG_AGENT_OBSERVATIONS_KEY,
                        "ng_model_call_capture",
                        NG_TRAJECTORY_KEY,
                    )
                }
                response = r.get("response") or {}
                response_metadata = {}
                usage = response.get("usage")
                if usage is not None:
                    response_metadata["usage"] = usage
                incomplete_details = response.get("incomplete_details")
                if incomplete_details is not None:
                    response_metadata["incomplete_details"] = incomplete_details
                metrics = response.get("metrics")
                if metrics is not None:
                    response_metadata["metrics"] = metrics
                if response_metadata:
                    entry["response"] = response_metadata
                stripped.append(entry)

            agg_response = None
            try:
                agg_request = AggregateMetricsRequest(verify_responses=stripped)
                agg_response = await server_client.post(
                    server_name=server_name,
                    url_path="/aggregate_metrics",
                    json=agg_request,
                )
                await raise_for_status(agg_response)
                agg_result = AggregateMetrics.model_validate(await get_response_json(agg_response))
            except Exception as exc:
                if first_error is None:
                    first_error = exc
                logger.exception(
                    "Aggregate-metrics request failed for agent '%s'; writing a repairable error entry.",
                    agent_name,
                )
                return {
                    **_aggregation_error_entry(agent_name, exc, getattr(agg_response, "status", None)),
                    NG_ENVIRONMENT_SERVER_KEY: server_name,
                }

            agent_entry = {
                AGENT_REF_KEY_NAME: {"name": agent_name},
                NG_ENVIRONMENT_SERVER_KEY: server_name,
                "agent_metrics": agg_result.agent_metrics,
                "key_metrics": agg_result.key_metrics,
                "group_level_metrics": agg_result.group_level_metrics,
                "repeat_level_metrics": agg_result.repeat_level_metrics,
            }
            if agg_result.perf_summary is not None:
                agent_entry["perf_summary"] = agg_result.perf_summary
            return agent_entry

        all_agent_metrics: List[Dict] = []
        tasks = [
            _fetch_agent_metrics(server_name, labels[server_name], results_list)
            for server_name, results_list in server_results.items()
        ]
        for coro in asyncio.as_completed(tasks):
            agent_entry = await coro
            all_agent_metrics.append(agent_entry)

            agent_name = agent_entry[AGENT_REF_KEY_NAME]["name"]
            if AGGREGATION_ERROR_KEY in agent_entry:
                print(
                    f"\nAggregate metrics failed for {agent_name}:\n"
                    + json.dumps(agent_entry[AGGREGATION_ERROR_KEY], indent=4)
                )
                continue
            key_metrics = agent_entry.get("key_metrics", {})
            print(f"\nKey metrics for {agent_name}:\n" + json.dumps(key_metrics, indent=4))

        all_agent_metrics.sort(key=lambda entry: entry[AGENT_REF_KEY_NAME]["name"])

        metrics_to_log = flatten_aggregate_metrics(all_agent_metrics)

        export_metrics(metrics_to_log)

        # Write single file with all agents
        metrics_fpath = aggregate_metrics_path_for(output_fpath)
        metrics_fpath.write_bytes(orjson.dumps(all_agent_metrics, option=orjson.OPT_INDENT_2))

        if raise_on_error and first_error is not None:
            raise first_error

        return metrics_fpath

    @staticmethod
    def resolve_task_sources(examples: List[Dict], global_config_dict: DictConfig) -> None:
        """Stamp an agent_ref onto every row that carries only a task_source.

        task_source names the config instance that declared the row's dataset. Resolution is
        :func:`~nemo_gym.config.global_config.resolve_dataset_agent` — the same rules benchmark
        discovery uses, so dispatch can never disagree with the listing. Conflicting `agent:`
        pins across one instance's datasets are a hard error (rows carry only the instance
        name), as are unknown/non-routable instances; +agent_map is the disambiguator.

        Rows that already have an agent_ref are left untouched, so this is a no-op on legacy
        datasets and on already-resolved (materialized) rows. Runs before any dispatch.
        """
        legacy_routed = sum(
            1
            for row in examples
            if (row.get(AGENT_REF_KEY_NAME) or {}).get("name") is not None and row.get(TASK_SOURCE_KEY_NAME) is None
        )
        if legacy_routed:
            warnings.warn(
                f"{legacy_routed} rows routed via their baked-in agent_ref (no task_source). This "
                "legacy path is deprecated: re-collate the dataset with current Gym to produce "
                "task_source-routed rows, or re-route explicitly with +agent_map.",
                DeprecationWarning,
                stacklevel=2,
            )

        unresolved = {
            ts
            for row in examples
            if (row.get(AGENT_REF_KEY_NAME) or {}).get("name") is None
            and (ts := row.get(TASK_SOURCE_KEY_NAME)) is not None
        }
        if not unresolved:
            return

        resolution: Dict[str, str] = {}
        errors: List[str] = []
        for ts in sorted(unresolved):
            block = global_config_dict.get(ts)
            if block is None:
                close = get_close_matches(ts, list(global_config_dict.keys()), n=1)
                errors.append(
                    f"{ts!r}: not in the running config" + (f" (did you mean {close[0]!r}?)" if close else "")
                )
            elif not isinstance(block, DictConfig):
                errors.append(f"{ts!r}: not a server instance")
            elif "responses_api_agents" in block or "resources_servers" in block:
                pins = dataset_agent_pins(global_config_dict, ts)
                if len(pins) > 1:
                    errors.append(
                        f"{ts!r}: its datasets pin conflicting agents ({sorted(pins)}), which task_source "
                        f"alone cannot tell apart; pass +agent_map={{{ts}: <agent>}} to pick one"
                    )
                    continue
                try:
                    resolution[ts] = resolve_dataset_agent(global_config_dict, ts, pin=pins[0] if pins else None)
                except ConfigError as e:
                    errors.append(f"{ts!r}: {e}")
            else:
                errors.append(f"{ts!r}: instance is not an agent or resources server (datasets cannot route here)")

        if errors:
            raise ValueError("Cannot resolve task_source to an agent: " + "; ".join(errors))

        for row in examples:
            if (row.get(AGENT_REF_KEY_NAME) or {}).get("name") is None:
                ts = row.get(TASK_SOURCE_KEY_NAME)
                if ts is not None:
                    row[AGENT_REF_KEY_NAME] = {"name": resolution[ts]}

    @staticmethod
    def _validate_agent_names(examples: List[Dict], global_config_dict: DictConfig) -> None:
        """Fail before any dispatch when a row names an agent absent from the running config.

        Without this, the first bad row dies mid-collection with a raw omegaconf ConfigKeyError
        after valid rows have already been dispatched.
        """
        requested = {name for row in examples if (name := (row.get(AGENT_REF_KEY_NAME) or {}).get("name")) is not None}
        available = {
            str(name)
            for name, block in global_config_dict.items()
            if isinstance(block, DictConfig) and "responses_api_agents" in block
        }
        unknown = sorted(requested - available)
        if not unknown:
            return
        hints = []
        for name in unknown:
            # Naming a non-agent instance (e.g. a resources server via agent_map) is as fatal as a
            # typo: rows route by agent, and only an agent has an environment server in front of it.
            if name in global_config_dict:
                hints.append(f"{name!r} (exists but is not an agent instance)")
                continue
            close = get_close_matches(name, available, n=1)
            hints.append(f"{name!r}" + (f" (did you mean {close[0]!r}?)" if close else ""))
        raise ValueError(
            f"Rows reference agents not present in the running config: {', '.join(hints)}. "
            "Include the agent's config in the run, or re-route with +agent_map/+agent_name."
        )

    @staticmethod
    def _dispatch_name(row: dict[str, Any]) -> str:
        environment_server_name = row.get(NG_ENVIRONMENT_SERVER_KEY)
        if isinstance(environment_server_name, str):
            return environment_server_name
        return row[AGENT_REF_KEY_NAME]["name"]

    @staticmethod
    def _agent_name_for_row(
        row: dict[str, Any],
        global_config_dict: DictConfig,
    ) -> str | None:
        environment_server_name = row.get(NG_ENVIRONMENT_SERVER_KEY)
        if not isinstance(environment_server_name, str):
            return (row.get(AGENT_REF_KEY_NAME) or {}).get("name")
        environment_group = global_config_dict[environment_server_name]["environment_servers"]
        return environment_server_attributed_agent(next(iter(environment_group.values())))

    @classmethod
    def _stamp_environment_server_agent_refs(
        cls,
        examples: list[dict[str, Any]],
        global_config_dict: DictConfig,
    ) -> None:
        """Stamp compatibility-routed rows with the bound agent.

        These rows never reach ``resolve_task_sources``, so without this they carry no
        ``agent_ref`` and results, aggregate metrics and reward profiling lose the agent
        they ran on. A row that already names an agent is left alone; the name is validated
        against the environment server by ``_validate_environment_servers``. Materialized tasks are
        skipped: they carry no agent by design, and their result projection is not the
        legacy shape this key belongs to.
        """
        for row in examples:
            if NG_ENVIRONMENT_SERVER_KEY not in row or _materialized_taskset(row) is not None:
                continue
            if (row.get(AGENT_REF_KEY_NAME) or {}).get("name") is not None:
                continue
            agent_name = cls._agent_name_for_row(row, global_config_dict)
            if agent_name is not None:
                row[AGENT_REF_KEY_NAME] = {"name": agent_name}

    @classmethod
    def _validate_environment_servers(
        cls,
        examples: list[dict],
        global_config_dict: DictConfig,
    ) -> None:
        requested = {
            environment_server
            for row in examples
            if isinstance((environment_server := row.get(NG_ENVIRONMENT_SERVER_KEY)), str)
        }
        if not requested:
            return
        available = {
            str(name)
            for name, block in global_config_dict.items()
            if isinstance(block, DictConfig) and "environment_servers" in block
        }
        unknown = requested - available
        if unknown:
            raise ValueError(f"Environment servers are not present in the running config: {sorted(unknown)}")

        environment_pairings: list[dict[str, Any]] = []
        for row in examples:
            environment_server_name = row.get(NG_ENVIRONMENT_SERVER_KEY)
            if not isinstance(environment_server_name, str):
                continue
            environment_group = global_config_dict[environment_server_name]["environment_servers"]
            environment_config = next(iter(environment_group.values()))
            resources_ref = environment_config.get("resources_server")
            configured_agent = environment_server_attributed_agent(environment_config)
            server_agents = environment_server_agent_names(environment_config)
            configured_resources = resources_ref.get("name") if isinstance(resources_ref, DictConfig) else None

            row_agent = (row.get(AGENT_REF_KEY_NAME) or {}).get("name")
            task_source = row.get(TASK_SOURCE_KEY_NAME)
            if row_agent is not None and server_agents and row_agent not in server_agents:
                raise ValueError(
                    f"Row agent_ref {row_agent!r} is not an agent of environment server "
                    f"{environment_server_name!r}: {', '.join(repr(agent) for agent in server_agents)}"
                )
            if task_source is not None and configured_resources is not None and task_source != configured_resources:
                raise ValueError(
                    f"Row task_source {task_source!r} does not match environment server "
                    f"{environment_server_name!r} resources server {configured_resources!r}"
                )

            if configured_agent is not None:
                environment_pairings.append(
                    {
                        AGENT_REF_KEY_NAME: {"name": configured_agent},
                        TASK_SOURCE_KEY_NAME: configured_resources,
                        # A materialized row's taskset names the dataset entry that declares it, which the
                        # pairing guard checks the agent against (see ``_dataset_admitted_agents``).
                        **{key: row[key] for key in ("task_id", "task_input") if key in row},
                    }
                )

        cls._validate_agent_names(environment_pairings, global_config_dict)
        cls._validate_agent_pairings(environment_pairings, global_config_dict)

    @staticmethod
    def _validate_agent_pairings(examples: List[Dict], global_config_dict: DictConfig) -> None:
        """Fail before dispatch when a row points to agent incompatible with the resources server it runs on.

        The verifier's ``allowed_agents`` is read from the row's ``task_source`` and from the resources server the
        agent binds; the dataset entry that declares the rows may also name the harnesses it admits (a Harbor
        dataset's ``[gym] allowed_agents``, carried into its generated ``datasets:`` entry, whose rows route by
        taskset to an environment server that binds no resources server). That list is checked against the agent's
        own type, so a run config edited to another harness after ``gym eval run`` admitted one is refused here as
        well. A materialized row is matched to its entry by taskset, wherever the entry is declared; a flat row
        meets every entry the agent declares (see ``_dataset_admitted_agents``).
        """
        if pairing_override_enabled(global_config_dict):
            return
        routes = {
            (name, row.get(TASK_SOURCE_KEY_NAME), _materialized_taskset(row))
            for row in examples
            if (name := (row.get(AGENT_REF_KEY_NAME) or {}).get("name")) is not None
        }
        rejected: Dict[Tuple[str, str], str] = {}
        for agent, task_source, taskset in routes:
            # indexing by name is safe because we already validated the agent names
            agents = global_config_dict[agent][AGENT_SERVER_TYPE_KEY_NAME]
            if not agents:
                continue
            # this is only one agent type per agent instance and the check it elsewhere
            agent_type = str(next(iter(agents)))
            reference = OmegaConf.select(agents[agent_type], "resources_server")
            bound = reference.get("name") if isinstance(reference, DictConfig) else None
            for verifier, relation in (
                (task_source, "their task_source"),
                (bound, "the resources server it runs against"),
            ):
                allowed = allowed_agents_for(global_config_dict, verifier)
                if allowed is None or agent_type in allowed:
                    continue
                rejected[(agent, str(verifier))] = (
                    f"  - rows routed to '{agent}' run '{agent_type}', but {relation} "
                    f"'{verifier}' accepts only: {', '.join(allowed)}"
                )
            for dataset, admitted in _dataset_admitted_agents(global_config_dict, agents[agent_type], taskset):
                if agent_type in admitted:
                    continue
                rejected[(agent, f"dataset {dataset}")] = (
                    f"  - rows routed to '{agent}' run '{agent_type}', but its dataset "
                    f"'{dataset}' admits only: {', '.join(admitted)}"
                )
        if not rejected:
            return
        raise ValueError(
            "Rows would be scored by a verifier that does not accept the agent running them:\n"
            + "\n".join(line for _, line in sorted(rejected.items()))
            + "\n\nRoute these rows to an agent running one of the accepted types, or pass "
            f"--allow-unsupported-pairing (or set {ALLOW_UNSUPPORTED_PAIRING_ENV_VAR_NAME}=1) to bypass the check."
        )

    def _run_examples_with_metadata(
        self,
        examples: List[Dict],
        head_server_config: Optional[BaseServerConfig] = None,
        semaphore: Optional[Semaphore] = None,
        route_failures_to_sidecar: bool = False,
        environment_server_name: str | None = None,
        *,
        max_resident_tasks: Optional[int] = None,
        typed_outcomes: bool = False,
        validate_results: bool = False,
        on_dispatch: Optional[Callable[[Dict], None]] = None,
        on_dispatch_batch: Optional[Callable[[List[Dict]], None]] = None,
        interleave_by_agent: bool = False,
        dispatch_budget_s: Optional[float] = None,
        drain_margin_s: Optional[float] = None,
        latency_tracker: Optional["DispatchLatencyTracker"] = None,
        retry_requests: bool = True,
        _max_num_tries: Optional[Literal[1]] = None,
        run_timing: Optional[RunTimingReport] = None,
        record_unrouted_failures: bool = False,
    ) -> Iterator[Future]:  # pragma: no cover
        """
        Internal dispatch shared by ``run_examples`` and Gym's own collection paths.

        ``record_unrouted_failures`` returns a failed `/run` that ``route_failures_to_sidecar`` would route as a
        failure row whose ``run_error`` holds the error, so the collection writer can keep evidence of it before it
        raises the error. Other callers keep receiving the exception.

        When ``max_resident_tasks`` is set, at most that many rollout tasks are admitted
        at once. When unset, all examples are scheduled as before. The collection
        owner closes either iterator on completion, cancellation or error.

        ``dispatch_budget_s`` stops starting rows that many seconds after this call, and
        ``drain_margin_s`` stops sooner for rows that would not have time to finish. A row
        drained this way resolves to a Gym-built ``_dispatch_drained_result``.

        Identical contract to ``run_examples``, but each future resolves to a ``_CompletedRollout``
        that carries ``rollout_latency_ms`` alongside the raw ``/run`` result instead of inside it,
        so internal-only timing never has to be smuggled through (and stripped back out of) a dict
        that a direct caller of ``run_examples`` could also observe.

        ``interleave_by_agent`` is enabled by the full evaluation path. Lower-level callers retain
        the order they supplied.
        """
        server_client = self.setup_server_client(head_server_config)
        if environment_server_name is not None:
            for row in examples:
                row[NG_ENVIRONMENT_SERVER_KEY] = environment_server_name
        self._validate_environment_servers(examples, server_client.global_config_dict)
        self._stamp_environment_server_agent_refs(examples, server_client.global_config_dict)
        direct_agent_examples = [row for row in examples if NG_ENVIRONMENT_SERVER_KEY not in row]
        self.resolve_task_sources(direct_agent_examples, server_client.global_config_dict)
        self._validate_agent_names(direct_agent_examples, server_client.global_config_dict)
        self._validate_agent_pairings(direct_agent_examples, server_client.global_config_dict)
        # Resolve every agent-routed row before dispatch, so an unroutable agent fails the run instead of one future.
        if interleave_by_agent:
            examples = _round_robin_by_agent(examples)
        servers_by_agent = _environment_servers_by_agent(server_client.global_config_dict)
        server_for_agent = {
            agent_name: _environment_server_for_agent(agent_name, servers_by_agent)
            for agent_name in {row[AGENT_REF_KEY_NAME]["name"] for row in direct_agent_examples}
        }
        server_types = {
            str(name): str(next(iter(block[ENVIRONMENT_SERVER_TYPE_KEY_NAME])))
            for name, block in server_client.global_config_dict.items()
            if isinstance(block, DictConfig) and isinstance(block.get(ENVIRONMENT_SERVER_TYPE_KEY_NAME), DictConfig)
        }
        semaphore = semaphore or nullcontext()
        observer = driver_metrics()
        if observer is not None:
            semaphore = MeteredAdmission(semaphore, observer)
            observer.driver.offered += len(examples)
        tracker = latency_tracker if latency_tracker is not None else DispatchLatencyTracker()
        # `is not None`, not truthiness: 0.0 is a real budget that is already
        # spent, and must drain rather than disable the check.
        deadline = (time.monotonic() + dispatch_budget_s) if dispatch_budget_s is not None else None

        # Coalesce requests admitted on the same event-loop turn. Only rows that
        # acquired a slot and passed the drain check enter this batch. Persist the
        # reservations before releasing any request to the server.
        reservation_waiters: list[tuple[Dict, Future]] = []

        def flush_reservations() -> None:
            batch = [(row, waiter) for row, waiter in reservation_waiters if not waiter.cancelled()]
            reservation_waiters.clear()
            if not batch:
                return
            try:
                on_dispatch_batch([row for row, _ in batch])
            except Exception as error:
                for _, waiter in batch:
                    waiter.set_exception(error)
            else:
                for _, waiter in batch:
                    waiter.set_result(None)

        async def reserve_dispatch(row: Dict) -> None:
            if on_dispatch_batch is None:
                if on_dispatch is not None:
                    on_dispatch(row)
                return
            loop = asyncio.get_running_loop()
            waiter = loop.create_future()
            if not reservation_waiters:
                loop.call_soon(flush_reservations)
            reservation_waiters.append((row, waiter))
            await waiter

        async def _post_subroutine(row: Dict) -> _CompletedRollout:
            server_name = (
                self._dispatch_name(row)
                if NG_ENVIRONMENT_SERVER_KEY in row
                else server_for_agent[row[AGENT_REF_KEY_NAME]["name"]]
            )
            server_type = server_types.get(server_name)
            async with semaphore:
                # Drain check happens *after* acquiring a slot, so it sees the
                # real remaining time at the moment this task would start rather
                # than at queueing time.
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    margin = tracker.drain_margin(drain_margin_s)
                    if remaining <= 0 or (margin is not None and remaining < margin):
                        tracker.record_drained()
                        return _CompletedRollout(
                            row=row,
                            result=_dispatch_drained_result(remaining, margin),
                            rollout_latency_ms=None,
                            environment_server=server_name,
                            environment_server_type=server_type,
                        )

                if is_global_aiohttp_client_request_debug_enabled():
                    print(
                        "[rollout_collection] /run dispatch "
                        f"row={json.dumps(_rollout_request_debug_summary(row), sort_keys=True)}",
                        flush=True,
                    )
                started = time.monotonic()
                started_at = time.time()
                if run_timing is not None:
                    run_timing.record_start(started)
                res = None
                stage = "request"
                await reserve_dispatch(row)
                succeeded = False
                dispatch_span_context = None
                try:
                    request_body = _episode_request_body(row) if _materialized_taskset(row) else row
                    dispatch_span = (
                        _dispatch_span(row, self._dispatch_name(row), server_name=server_name)
                        if is_span_group_enabled(GymSpanGroup.ROLLOUT) and is_trace_exporter_active()
                        else nullcontext()
                    )
                    with dispatch_span as dispatch_span_context:
                        res = await server_client.post(
                            server_name=server_name,
                            url_path="/run",
                            json=request_body,
                            **({"_resend_after_disconnect": False} if not retry_requests else {}),
                            **({"_max_num_tries": _max_num_tries} if _max_num_tries is not None else {}),
                        )
                    await raise_for_status(res)
                    stage = "response"
                    result = await get_response_json(res)
                    if (typed_outcomes or validate_results) and _materialized_taskset(row) is not None:
                        stage = "result"
                        result = _native_episode_record(row, result)
                    if run_timing is not None and isinstance(result, Mapping):
                        # Classify the record collection saves, as the run health summary does.
                        episode = _materialized_taskset(row) is not None and _is_episode_response(result)
                        run_timing.record_result(
                            _episode_record(result) if episode else result, time.monotonic() - started
                        )
                    outcome = result
                    if typed_outcomes or validate_results:
                        stage = "result"
                        outcome = _normalize_rollout_outcome(row, result)
                    # Normalization sanitizes legacy fields before diagnostics
                    # can copy them back into the persisted failure envelope.
                    verification_response = _judge_failure_response(result)
                    diagnostics = _failure_diagnostics(result)
                    # Independently-measured task wall-clock (ng_perf.total_latency_ms), not derived
                    # from summed model-call/tool latencies to account for additional overhead.
                    rollout_latency_ms = (time.time() - started_at) * 1000
                    succeeded = True
                    return _CompletedRollout(
                        row=row,
                        result=outcome,
                        rollout_latency_ms=rollout_latency_ms,
                        verification_response=verification_response,
                        diagnostics=diagnostics,
                        environment_server=server_name,
                        environment_server_type=server_type,
                        elapsed_s=time.monotonic() - started,
                        dispatch_span_context=dispatch_span_context,
                        failure_report=FailureHeader.decode(getattr(res, "headers", None)),
                    )
                except Exception as e:
                    print(
                        "[rollout_collection] /run failed "
                        f"status={getattr(res, 'status', None)} "
                        f"row={json.dumps(_rollout_request_debug_summary(row), sort_keys=True)}",
                        flush=True,
                    )
                    unrouted = record_unrouted_failures and isinstance(e, _RUN_FAILURE_ERRORS)
                    if not isinstance(e, (*_RUN_FAILURE_ERRORS, InvalidRolloutResult)) or not (
                        route_failures_to_sidecar or typed_outcomes or unrouted
                    ):
                        if isinstance(e, StaleRolloutResult):
                            e.add_note(
                                "Set +route_failures_to_sidecar=true to save this stale reply as a failure "
                                "and continue independent tasks; resume to retry the unfinished task."
                            )
                        elif isinstance(e, InvalidRolloutResult):
                            e.add_note(
                                "Set +route_failures_to_sidecar=true to save this invalid reply as a failure "
                                "and continue independent tasks; fix the producer before retrying the failed task."
                            )
                        raise
                    if res is not None:
                        res.release()
                    # The status comes from the error when it carries one, and from the response
                    # when the body was the part that failed.
                    status = getattr(e, "status", None) or getattr(res, "status", None)
                    failure = _agent_request_failure_row(e, status)
                    if (
                        (typed_outcomes or validate_results)
                        and isinstance(e, (InvalidRolloutResult, orjson.JSONDecodeError))
                        and not isinstance(e, StaleRolloutResult)
                    ):
                        failure[NG_FAILURE_CLASS_KEY] = ENVIRONMENT_PROTOCOL_VIOLATION
                        failure[NG_TERMINAL_KEY] = True
                    failure_row = failure
                    failed_outcome = _failure_outcome(row, failure, stage) if typed_outcomes or unrouted else failure
                    if (
                        isinstance(failed_outcome, RolloutFailureRecord)
                        and _max_num_tries == 1
                        and isinstance(e, ClientConnectorError)
                    ):
                        # One failed connection attempt cannot have delivered /run. A final error after retries
                        # cannot establish this fact about earlier attempts.
                        failed_outcome = failed_outcome.model_copy(update={"delivery": "not_sent"})
                    return _CompletedRollout(
                        row=row,
                        result=failed_outcome,
                        rollout_latency_ms=None,
                        diagnostics=_failure_diagnostics(failure),
                        environment_server=server_name,
                        environment_server_type=server_type,
                        elapsed_s=time.monotonic() - started,
                        dispatch_span_context=dispatch_span_context,
                        failure_report=_collection_failure_report(
                            e,
                            failure_row[NG_FAILURE_CLASS_KEY],
                            getattr(e, "headers", None) or getattr(res, "headers", None),
                        ),
                        run_error=None if route_failures_to_sidecar else e,
                    )
                finally:
                    # Failed and timed-out tasks inform the adaptive margin too, but only by
                    # raising it: see ``DispatchLatencyTracker.record_failure``.
                    elapsed = time.monotonic() - started
                    if succeeded:
                        tracker.record(elapsed)
                    else:
                        tracker.record_failure(elapsed)

        async def observed_post(row: Dict) -> _CompletedRollout:
            # Create the inner coroutine only after the task starts: cancellation
            # before admission must not leave an unawaited request coroutine.
            return await count_resident(_post_subroutine(row), observer)

        awaitables = map(observed_post if observer is not None else _post_subroutine, examples)
        if max_resident_tasks is not None:
            return _BoundedCompletionIterator(awaitables, max_resident_tasks=max_resident_tasks, total=len(examples))

        return _EagerCompletionIterator(awaitables, total=len(examples))

    def run_outcomes(
        self,
        examples: List[Dict],
        head_server_config: Optional[BaseServerConfig] = None,
        semaphore: Optional[Semaphore] = None,
        *,
        run_id: str | None = None,
    ) -> _RowResultIterator:
        """Yield ``(row, completed_dict | RolloutFailureRecord)`` in completion order.

        Expected HTTP, JSON and result-shape failures become bounded records. Programming
        errors and cancellation still propagate. Call ``preprocess_examples`` first, or
        provide stable explicit rollout ids. This does not persist or retry work; a direct
        caller owns those decisions. Supply ``run_id`` to associate direct calls with
        a run, overriding any row stamp. Without it, existing row stamps are preserved;
        unstamped rows share a new invocation ID. Returned rows are invocation-owned copies;
        input rows are not modified. ``run_examples`` retains its behavior.

        On early stopping, await the iterator's ``aclose()`` to cancel and join unfinished
        client tasks, or ``aclose_and_drain()`` to also retrieve settled, unconsumed outcomes.
        Both have the same server/agent cleanup limitations documented by ``run_examples``.
        """
        owner = run_id or str(uuid4())
        examples = [dict(row, **{RUN_ID_KEY: run_id or row.get(RUN_ID_KEY) or owner}) for row in examples]
        for row in examples:
            if maybe_rollout_id_from_run_body(row) is None:
                raise ValueError("run_outcomes requires rollout ids; call preprocess_examples first.")
            logical_rollout_id(row)  # Enforce the shared EpisodeId rules before any HTTP dispatch.

        return _RowResultIterator(
            self._run_examples_with_metadata(
                examples, head_server_config=head_server_config, semaphore=semaphore, typed_outcomes=True
            ),
        )

    def run_examples(
        self,
        examples: List[Dict],
        head_server_config: Optional[BaseServerConfig] = None,
        semaphore: Optional[Semaphore] = None,
        route_failures_to_sidecar: bool = False,
        environment_server_name: str | None = None,
        *,
        max_resident_tasks: Optional[int] = None,
        dispatch_budget_s: Optional[float] = None,
        drain_margin_s: Optional[float] = None,
        latency_tracker: Optional["DispatchLatencyTracker"] = None,
        retry_requests: bool = True,
    ) -> _RowResultIterator:  # pragma: no cover
        """
        We provide this function as a lower level interface for running rollout collection.

        Rows are dispatched as given: task_sources are resolved and agent names validated here,
        but run-level knobs (``agent_map``, ``fan_out``, ``num_repeats``) are NOT applied — call
        ``preprocess_examples`` first if you need them.

        ``route_failures_to_sidecar`` makes a failed `/run` a failure row instead of an exception
        that ends every rollout still in flight. It defaults off because those rollouts then leave
        the score.

        ``max_resident_tasks`` limits admitted tasks and therefore concurrent requests,
        even when ``semaphore`` allows more. Admission starts when the first returned
        awaitable is awaited. None schedules all examples up front, as with
        ``asyncio.as_completed``.

        To stop early, ``await`` the returned iterator's ``aclose()``: it cancels unfinished
        rollouts and waits for their client tasks in this process to end. Cancelling an in-flight
        `/run` closes its connection; a server with ``ClientDisconnectCancellationMiddleware``
        subsequently cancels its handler. Server/agent cleanup is not awaited or acknowledged,
        depends on the agent, and may continue after return. Sandbox release is not guaranteed.
        Handed-out awaitables may return completed results or raise their exceptions, including
        ``CancelledError``; bounded awaitables first awaited after closure raise ``CancelledError``.
        The error propagates in the awaiting caller's task even without cancelling that task,
        and ``asyncio.gather`` and ``TaskGroup`` treat it as cancellation.
        Without it, abandoned rollouts keep running to completion in the background.

        To preserve completed but unconsumed work when stopping, use
        ``await completions.aclose_and_drain()`` instead. It stops admission before suspending,
        cancels and awaits unfinished client tasks, and returns settled ``(row, result)``
        futures, including failures. Consume those futures and any outcome already claimed
        by an outstanding completion waiter; an unclaimed waiter raises ``CancelledError``.
        Repeated drains return an empty list. This also works with bounded admission without
        starting more rows, and has the same server/agent cleanup limits as ``aclose()``.

        Use ``finally`` to close the iterator if collection stops early::

            completions = helper.run_examples(examples)
            try:
                for future in completions:
                    row, result = await future
                    if should_stop(row, result):
                        break
            finally:
                await completions.aclose()

        ``dispatch_budget_s`` stops starting rows that many seconds after this call, and
        ``drain_margin_s`` stops sooner for rows that would not have time to finish.

        Rows start in the order given, once the returned iterator is first consumed.

        A row that ran resolves to exactly the ``(row, result)`` pair Gym's own `/run` endpoint
        returned — no Gym-private fields are added to ``result``. A row that produced no `/run`
        result resolves to a Gym-built ``result`` carrying Gym-private fields instead: a row drained
        by the dispatch budget gets ``_ng_failure_class="cancelled"`` with the ``_ng_dispatch_drained``
        and ``_ng_no_persist`` markers, and a failed `/run` under ``route_failures_to_sidecar`` gets a
        failure row with ``_ng_failure_*`` fields.

        ``retry_requests=False`` disables automatic HTTP replay of a mutating `/run`.
        Connections that were never established can still be retried safely.
        It does not convert failures to successful result rows.
        """

        return _RowResultIterator(
            self._run_examples_with_metadata(
                examples,
                head_server_config=head_server_config,
                semaphore=semaphore,
                route_failures_to_sidecar=route_failures_to_sidecar,
                environment_server_name=environment_server_name,
                max_resident_tasks=max_resident_tasks,
                dispatch_budget_s=dispatch_budget_s,
                drain_margin_s=drain_margin_s,
                latency_tracker=latency_tracker,
                **({"retry_requests": False} if not retry_requests else {}),
            )
        )

    def setup_server_client(self, head_server_config: Optional[BaseServerConfig] = None) -> ServerClient:
        """Return a client for the head server, reusing the cached one when the helper opted in.

        Without opt-in, every call fetches the configuration from the head server again.
        An opted-in helper keeps one client and fetches again only when the head config differs from the cached one.
        Build a new helper after the configuration behind the same head changes.
        """
        if not self.reuse_server_client:
            return setup_server_client_utils(head_server_config)
        if self._cached_server_client is None or head_server_config != self._cached_head_server_config:
            client = setup_server_client_utils(head_server_config)
            head_snapshot = head_server_config.model_copy(deep=True) if head_server_config is not None else None
            self._cached_server_client = client
            self._cached_head_server_config = head_snapshot
        return self._cached_server_client


class RolloutAggregationConfig(RunHealthConfig, BaseNeMoGymCLIConfig):
    """
    Aggregate metrics across rollout shards produced by `gym eval run --no-serve +disable_aggregation=true`.

    Reads every JSONL file matching `input_glob`, computes aggregate metrics by POSTing to each
    agent server's `/aggregate_metrics` endpoint over the global union of records, and writes a
    single `<output_jsonl_fpath stem>_aggregate_metrics.json` next to the rollouts. By default
    also concatenates all shards into `output_jsonl_fpath`.

    Examples:

    ```bash
    gym eval aggregate \
        "+config_paths=[benchmarks/aime24/config.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]" \
        +input_glob='results/rollouts-rs*-chunk*.jsonl' \
        +output_jsonl_fpath=results/rollouts.jsonl
    ```
    """

    retry_terminal_timeouts: bool = Field(
        default=False,
        description="Accepted for compatibility. Managed runs use their saved policy; legacy coverage is unknown.",
    )
    input_glob: str = Field(
        description=(
            "Glob pattern or comma-separated list of glob patterns matching the rollout shards "
            "to aggregate (e.g. 'results/rollouts-rs*-chunk*.jsonl' or "
            "'results/run1/rollouts.jsonl,results/run2/rollouts.jsonl'). Whitespace around "
            "commas is stripped. Duplicate matches across patterns are deduplicated."
        )
    )
    output_jsonl_fpath: str = Field(
        description=(
            "Path used to derive the aggregate-metrics output location "
            "('<stem>_aggregate_metrics.json' beside the resolved output, preserving unambiguous legacy "
            "alias companions) and, when merge_shards=True, the merged-rollouts file. Metrics-only output "
            "through a file symlink cannot replace an existing target's reports unless it is the only input."
        ),
    )
    batch_manifest_fpath: Optional[str] = Field(
        default=None,
        description=(
            "Optional Eval Factory batch_manifest.json. Gym validates it and refreshes the sibling "
            "batch_status.json after aggregation."
        ),
    )
    materialized_inputs_jsonl_fpath: Optional[str] = Field(
        default=None,
        description=(
            "Full materialized-inputs JSONL used to validate a batch manifest. Defaults to the standard "
            "sibling derived from output_jsonl_fpath."
        ),
    )
    merge_shards: bool = Field(
        default=True,
        description=(
            "Concatenate the matched shard JSONLs into output_jsonl_fpath alongside the metrics file, and list "
            "them in '<stem>_sources.json', which `gym eval health-check` reads for a merged 'rollouts.jsonl'. "
            "output_jsonl_fpath may be a shard only when it is the only one, reached through its own directory (not a "
            "file link from elsewhere), and then nothing is listed."
        ),
    )
    count_failure_classes_as_zero: List[str] = Field(
        default_factory=list,
        description=(
            "Failure classes from the failures sidecar to count in aggregate metrics, e.g. "
            "['agent_run_error'], so a failed rollout lands in the denominator. A row that carries "
            "no reward is scored zero for the metrics only; no artifact is changed. Each shard's "
            "sidecar matches the canonical class or saved original legacy producer class and "
            "is read against that shard's own rows, so runs on different environment "
            "servers that reuse task indices do not hide each other's failures."
        ),
    )
    count_missing_rollouts_as_zero: bool = Field(
        default=False,
        description=(
            "Count a materialized rollout that produced no row at all as a zero, checking each "
            "shard against its own rows, materialized inputs and failures sidecar, so runs on "
            "different environment servers that reuse task indices do not hide each other's lost "
            "rollouts. A rollout another shard landed or recorded for the same environment server "
            "is not counted. Same contract as the "
            "collection-time flag, including that a recorded failure is never counted here and "
            "the same metric-hook limitation."
        ),
    )
    disable_health_check: bool = Field(
        default=False,
        description="Skip post-aggregation rollout quality verification and report writing.",
    )
    health_check_workers: Optional[int] = Field(
        default=None,
        ge=1,
        description="Number of rollout-health worker processes (defaults to min(cpus, 8)).",
    )
    health_check_ignored_checks: List[str] = Field(
        default_factory=list,
        description="Health-check IDs to exclude from execution and verdict derivation.",
    )

    @field_validator("health_check_ignored_checks", mode="before")
    @classmethod
    def _validate_health_check_ignored_checks(cls, value):
        return _normalize_health_check_ignored_checks(value)


def loads_jsonl_line(raw, fpath, line_no: int):
    """Parse one JSONL line, raising a clean `ConfigError` (naming file + line) on malformed JSON."""
    try:
        return orjson.loads(raw)
    except orjson.JSONDecodeError as e:
        raise ConfigError(f"Malformed JSON in '{fpath}' at line {line_no}: {e}") from e


def _expand_input_glob(input_glob: str) -> List[str]:
    """Expand a glob-or-comma-separated-globs string into a sorted, deduplicated list of paths.

    Examples:
      'results/rollouts.jsonl' -> ['results/rollouts.jsonl'] (if it exists)
      'a/*.jsonl, b/*.jsonl'   -> matches of both patterns, deduplicated
    """
    patterns = [p.strip() for p in input_glob.split(",") if p.strip()]
    seen: Dict[Path, str] = {}  # source identity includes its companion recovery files
    for pattern in patterns:
        for path in sorted(glob_module.glob(pattern)):
            resolved = resolve_rollout_path(Path(path), read_only=True)
            if (
                resolved.stem.endswith(("_attempts", "_failures", "_materialized_inputs"))
                and not manifest_path_for(resolved).exists()
            ):
                # Broad shard globs must not score recovery artifacts as results.
                continue
            identity = resolved.resolve()
            if (
                identity in seen
                and Path(seen[identity]) != identity
                and resolved != identity
                and str(resolved) != seen[identity]
            ):
                raise ConfigError(
                    f"Multiple legacy companion layouts refer to the same output: {seen[identity]}, {resolved}"
                )
            if identity not in seen or resolved != identity:
                # A permitted legacy alias owns the only companion files; keep
                # that spelling when the glob also includes its bare target.
                seen[identity] = str(resolved)
    return list(seen.values())


class RolloutAggregationHelper(BaseModel):
    async def run_from_config(self, config: RolloutAggregationConfig) -> Optional[Path]:
        input_paths = _expand_input_glob(config.input_glob)
        if not input_paths:
            raise ConfigPathNotFoundError(f"No shards matched input_glob={config.input_glob!r}")
        # Readers and writers must agree before creating any companions. Keep
        # unambiguous legacy alias layouts, but reject competing companion sets.
        given_fpath = Path(config.output_jsonl_fpath)
        output_fpath = resolve_rollout_path(given_fpath, read_only=True)
        if (
            not config.merge_shards
            and output_fpath != given_fpath.parent.resolve() / given_fpath.name
            and output_fpath.exists()
            and {Path(path).resolve() for path in input_paths} != {output_fpath}
        ):
            raise ConfigError(
                f"Metrics-only aggregation through {given_fpath} would replace metrics of {output_fpath}; "
                "pass that path itself or a new output path. Saved artifacts were not changed."
            )
        # Metrics-only aggregation may target the source basename. Its report
        # must not replace the collector's run-specific completion snapshot.
        report_path = coverage_path_for(
            output_fpath if config.merge_shards else output_fpath.with_stem(output_fpath.stem + "_aggregate")
        )
        if config.merge_shards:
            sources = {
                artifact
                for path in input_paths
                for artifact in (
                    Path(path),
                    manifest_path_for(Path(path)),
                    journal_path_for(Path(path)),
                    materialized_path_for(Path(path)),
                    failures_path_for(Path(path)),
                    coverage_path_for(Path(path)),
                    resubmit_offset_path_for(Path(path)),
                )
            }
            if any(
                output_fpath.resolve() == source.resolve()
                or (output_fpath.exists() and source.exists() and output_fpath.samefile(source))
                for source in sources
            ):
                raise ConfigError(
                    f"Merged output {given_fpath} is one of the shards input_glob matched, or another source "
                    "rollout artifact or its recovery files, and merging would replace it. Merge into another path, "
                    "or set merge_shards=false to recompute metrics in place."
                )
            # A merged projection cannot inherit another run's recovery history.
            # Inspect both names when the destination is a symlink to a saved run.
            for target in {output_fpath, output_fpath.resolve()}:
                for companion in (
                    manifest_path_for(target),
                    journal_path_for(target),
                    materialized_path_for(target),
                    failures_path_for(target),
                ):
                    if companion.exists() or companion.is_symlink():
                        raise ConfigError(
                            f"Merged output has existing recovery artifacts ({companion}); "
                            "choose a new output path to preserve the saved run."
                        )
        for source in input_paths:
            for artifact in (
                Path(source),
                manifest_path_for(Path(source)),
                journal_path_for(Path(source)),
                materialized_path_for(Path(source)),
                failures_path_for(Path(source)),
                coverage_path_for(Path(source)),
                resubmit_offset_path_for(Path(source)),
            ):
                if report_path.resolve() == artifact.resolve() or (
                    report_path.exists() and artifact.exists() and report_path.samefile(artifact)
                ):
                    raise ConfigError(
                        "Aggregate coverage must not overwrite source rollout artifacts; choose a new output path."
                    )
        print(f"Aggregating {len(input_paths)} shard(s):")
        for p in input_paths:
            print(f"  - {p}")

        results: List[Dict] = []
        histories: List[RolloutStore] = []
        legacy_paths: List[Path] = []
        legacy_results: List[Dict] = []
        history_results: Dict[str, List[Dict]] = {}
        shard_keys: Dict[str, set] = {}
        run_ids: set[str] = set()
        # Each shard's size and SHA-256 as read, for the merged file's shard list to name.
        shard_digests: Dict[str, Tuple[int, str]] = {}
        for shard_path in input_paths:
            shard = Path(shard_path)
            history = RolloutStore.read(
                shard, import_legacy=False, retry_terminal_timeouts=config.retry_terminal_timeouts
            )
            if history is not None:
                if history.manifest.run_id in run_ids:
                    raise ConfigError("The same run was supplied through multiple shard paths.")
                run_ids.add(history.manifest.run_id)
                histories.append(history)
                # Imported legacy rows can lack a run tag. The merged projection
                # must still retain the source run's identity for health checks.
                selected = history_results[history.manifest.run_id] = [
                    row | {RUN_ID_KEY: history.manifest.run_id} for row in history.selected("success")
                ]
                results.extend(selected)
                with open(shard_path, "rb") as f:
                    sha256 = hashlib.file_digest(f, "sha256").hexdigest()
                    shard_digests[shard_path] = (f.tell(), sha256)
                continue
            legacy_paths.append(shard)
            keys = shard_keys[str(shard)] = set()
            digest = hashlib.sha256()
            with open(shard_path, "rb") as f:
                for line_no, line in enumerate(f, 1):
                    digest.update(line)
                    line = line.strip()
                    if not line:
                        continue
                    result = loads_jsonl_line(line, shard_path, line_no)
                    keys.add((result.get(TASK_INDEX_KEY_NAME), result.get(ROLLOUT_INDEX_KEY_NAME)))
                    results.append(result)
                    legacy_results.append(result)
                shard_digests[shard_path] = (f.tell(), digest.hexdigest())
        print(f"Loaded {len(results)} rollout record(s) from {len(input_paths)} shard(s)")

        # Sort for deterministic aggregation ordering (matches run_from_config's post-collection sort)
        results.sort(key=lambda r: (r.get(TASK_INDEX_KEY_NAME), r.get(ROLLOUT_INDEX_KEY_NAME)))

        output_fpath.parent.mkdir(parents=True, exist_ok=True)

        failures_fpaths = [failures_path_for(path) for path in legacy_paths]
        latest_failure_rows = _latest_failure_rows(failures_fpaths)
        batch_tracker = None
        if config.batch_manifest_fpath:
            materialized_fpath = (
                Path(config.materialized_inputs_jsonl_fpath)
                if config.materialized_inputs_jsonl_fpath
                else output_fpath.with_stem(output_fpath.stem + "_materialized_inputs").with_suffix(".jsonl")
            )
            if not materialized_fpath.exists():
                raise ConfigPathNotFoundError(
                    f"Materialized inputs not found at '{materialized_fpath}'. Pass "
                    "+materialized_inputs_jsonl_fpath=<path> when aggregating a batch from shards."
                )
            with materialized_fpath.open("rb") as materialized_file:
                materialized_rows = [orjson.loads(line) for line in materialized_file if line.strip()]
            batch_tracker = BatchStatusTracker(Path(config.batch_manifest_fpath), materialized_rows)
            batch_agent_refs = {
                (row[TASK_INDEX_KEY_NAME], row[ROLLOUT_INDEX_KEY_NAME]): row[AGENT_REF_KEY_NAME]
                for row in materialized_rows
            }
            batch_completed_rows = [_batch_status_row(row, batch_agent_refs) for row in results]
            batch_failure_rows = [_batch_status_row(row, batch_agent_refs) for row in latest_failure_rows.values()]
            for history in histories:
                batch_failure_rows.extend(_batch_status_row(row, batch_agent_refs) for row in history.failures())
            batch_tracker.write_status(batch_completed_rows, batch_failure_rows, force=True)

        if config.merge_shards:
            from nemo_gym.rollout_health import write_aggregation_sources

            print(f"Merging shards into {output_fpath}")
            # The merged rows have none of the shards' failures or inventories, so a recheck of this directory reads
            # the shards. Their list names what was read from them and the merged contents, and is written before
            # those take their place. A symlinked output keeps its link, and the merged rows replace its target.
            merged_fpath = output_fpath.resolve()
            # Replacing a merged file keeps its mode, as writing through it did.
            prior_mode = merged_fpath.stat().st_mode & 0o7777 if merged_fpath.exists() else None
            # Created anew under a name no one can guess, and owner-only while it takes an existing output's rows,
            # so it never grants more than that output does. A new output keeps the default creation mode.
            merging_fpath = merged_fpath.with_name(f".{merged_fpath.name}.{secrets.token_hex(8)}.merging")
            fd = os.open(
                merging_fpath, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600 if prior_mode is not None else 0o666
            )
            try:
                with os.fdopen(fd, "wb") as out:
                    for r in results:
                        out.write(orjson.dumps(r) + b"\n")
                    out.flush()
                    os.fsync(out.fileno())
                if prior_mode is not None:
                    os.chmod(merging_fpath, prior_mode)
                # A merge never replaces one of its shards (checked above), so it always lists them.
                write_aggregation_sources(
                    output_fpath, [(Path(path), *shard_digests[path]) for path in input_paths], merging_fpath
                )
                # Readers indexed against the old file must detect replacement rather than read rewritten bytes
                # at stale offsets, so the merged rows replace it instead of rewriting it. If they never land, the
                # list names contents that are not there, and a recheck refuses until the aggregation is rerun.
                merging_fpath.replace(merged_fpath)
            finally:
                merging_fpath.unlink(missing_ok=True)

        # `_call_aggregate_metrics` groups by the `_ng_environment_server` stamp, falling back to
        # AGENT_REF_KEY_NAME; result rows carry both from the run that produced them.
        helper = RolloutCollectionHelper()

        @functools.cache
        def servers_by_agent() -> Mapping[str, list[str]]:
            # With nothing to score and nothing to count, only the coverage report keys rollouts, and an
            # agent's name does for that: the head server is reached for scoring alone.
            if not (results or config.count_failure_classes_as_zero or config.count_missing_rollouts_as_zero):
                return {}
            return _environment_servers_by_agent(helper.setup_server_client().global_config_dict)

        def group(row: Mapping[str, Any]) -> Optional[str]:
            return _metrics_group(row, servers_by_agent)

        # Each shard is checked against its own rows and its own sidecar, and rollouts are keyed by
        # the server that scores them: separate runs number their tasks from 0, so one run's row must
        # not hide another run's failed or lost rollout. A rollout another shard landed for the same
        # server is still not counted, and of the attempts recorded for one rollout the last stands.
        latest_failures: Dict[tuple, Dict[str, Any]] = {}
        for path, failures_fpath in zip(legacy_paths, failures_fpaths):
            for key, row in _latest_failure_rows([failures_fpath]).items():
                if key not in shard_keys[str(path)]:
                    latest_failures[_identity_key(row, group)] = row
        result_ids = (
            {_identity_key(r, group) for r in legacy_results}
            if latest_failures or config.count_missing_rollouts_as_zero
            else set()
        )
        wanted = set(config.count_failure_classes_as_zero)
        # A row that names no agent and no server cannot reach any server's metrics; it stays a dropped one.
        counted = [
            _counted_failure_row(row)
            for identity, row in latest_failures.items()
            if identity not in result_ids and _counts_as_zero(row, wanted) and _routing_identity(row) is not None
        ]
        if config.count_failure_classes_as_zero or counted:
            print(f"Counting {len(counted)} failure row(s) as scored zeros: {config.count_failure_classes_as_zero}")
        materialized_fpaths = [materialized_path_for(path) for path in legacy_paths]
        missing: List[Dict[str, Any]] = []
        if config.count_missing_rollouts_as_zero:
            accounted = result_ids | set(latest_failures)
            for path, materialized_fpath, failures_fpath in zip(legacy_paths, materialized_fpaths, failures_fpaths):
                for zero in _missing_rollout_rows_counted_as_zero(
                    [materialized_fpath], [failures_fpath], set(shard_keys[str(path)])
                ):
                    if _identity_key(zero, group) not in accounted:
                        accounted.add(_identity_key(zero, group))
                        missing.append(zero)
            counted.extend(missing)
        if counted:
            _fill_task_fields(counted, legacy_results, materialized_fpaths, group)

        legacy_counted_ids = {_identity_key(row, group) for row in counted}
        for history in histories:
            added = _counted_failure_rows(history.failures(), config.count_failure_classes_as_zero)
            if config.count_missing_rollouts_as_zero:
                imputed = _missing_rollout_rows_counted_as_zero(
                    [materialized_path_for(history.output)], [], set(), history=history
                )
                missing.extend(imputed)
                added.extend(imputed)
            _fill_task_fields(
                added,
                history_results[history.manifest.run_id],
                [materialized_path_for(history.output)],
                _routing_identity,
            )
            counted.extend(added)

        if config.count_failure_classes_as_zero:
            print(
                f"Counting {len(counted) - len(missing)} failure row(s) as scored zeros: "
                f"{config.count_failure_classes_as_zero}"
            )
        if config.count_missing_rollouts_as_zero:
            print(f"Counting {len(missing)} materialized rollout(s) with no row as scored zeros")
        scored = sorted(results + counted, key=_rollout_order_key)
        aggregate_metrics_fpath = await helper._call_aggregate_metrics(
            scored, scored, output_fpath, raise_on_error=False
        )
        if batch_tracker is not None:
            batch_tracker.write_status(
                batch_completed_rows,
                batch_failure_rows,
                aggregate_metrics_fpath=aggregate_metrics_fpath,
                force=True,
            )

        _raise_for_aggregation_errors(aggregate_metrics_fpath)

        # The shards' own sidecars say which rollouts never made it into the files just scored.
        dropped = Counter(
            row.get(NG_FAILURE_CLASS_KEY) or "unknown"
            for identity, row in latest_failures.items()
            if identity not in result_ids and identity not in legacy_counted_ids
        )
        dropped.update(
            row[NG_FAILURE_CLASS_KEY]
            for history in histories
            for row in history.failures()
            if not _counts_as_zero(row, wanted)
        )
        components = [history.coverage() for history in histories]
        inventory_known = not legacy_paths
        masked_rollouts = sum(
            bool(row.get(MASK_SAMPLE_KEY)) and type(row.get("reward")) in (int, float) for row in results
        )
        unscored_rollouts = sum(type(row.get("reward")) not in (int, float) for row in results)
        scored_rollouts = len(results) - unscored_rollouts + len(counted)
        completion = {
            "schema_version": 1,
            "expected": sum(c["expected"] for c in components) if inventory_known else None,
            "successful": len(results),
            "measured": len(results) - masked_rollouts - unscored_rollouts,
            "masked": masked_rollouts,
            "unscored": unscored_rollouts,
            "failed": sum(c["failed"] for c in components) if inventory_known else None,
            "failed_final": sum(c["failed_final"] for c in components) if inventory_known else None,
            "failed_not_final": sum(c["failed_not_final"] for c in components) if inventory_known else None,
            "failed_by_class": dict(sum((Counter(c["failed_by_class"]) for c in components), Counter()))
            if inventory_known
            else None,
            "intentionally_omitted": sum(c["intentionally_omitted"] for c in components) if inventory_known else None,
            "not_started": sum(c["not_started"] for c in components) if inventory_known else None,
            "missing": sum(c["missing"] for c in components) if inventory_known else None,
            "failure_budget_exhausted": sum(c["failure_budget_exhausted"] for c in components)
            if inventory_known
            else None,
            "max_rollout_attempts": components[0]["max_rollout_attempts"]
            if components and len({c["max_rollout_attempts"] for c in components}) == 1
            else None,
            "retry_terminal_timeouts": components[0]["retry_terminal_timeouts"]
            if inventory_known and components and len({c["retry_terminal_timeouts"] for c in components}) == 1
            else None,
            "complete": inventory_known and all(c["complete"] for c in components),
            "coverage_known": inventory_known,
            "scored": scored_rollouts,
            "failures_counted_as_zero": len(counted) - len(missing),
            "imputed": len(missing),
            "shards": components,
            "shards_without_inventory": [str(path) for path in legacy_paths],
        }
        atomic_write_json(report_path, completion)
        if not inventory_known:
            print(
                "Completion coverage is unknown for legacy shards without validated attempt history; "
                "scores may be partial."
            )
        coverage = _coverage_report(
            completion["expected"] if inventory_known else scored_rollouts + sum(dropped.values()),
            scored_rollouts,
            dropped,
            "the shards' _failures.jsonl sidecars",
            imputed=len(missing),
        )
        if inventory_known:
            coverage += (
                f"\nCoverage: {completion['successful']} completed "
                f"({completion['measured']} measured, {completion['masked']} masked, {completion['unscored']} unscored), "
                f"{completion['failed']} failed, "
                f"{completion['intentionally_omitted']} intentionally omitted, "
                f"{completion['not_started']} not started, {completion['missing']} missing outcomes. "
                f"Details: {report_path}"
            )
        if get_exporters():  # pragma: no cover
            failure_metrics = await asyncio.to_thread(
                _failure_export_metrics,
                failures_fpaths,
                lambda row: _identity_key(row, group),
                {_identity_key(r, group) for r in results} if failures_fpaths else set(),
                histories=histories,
            )
            metrics = {
                **failure_metrics,
                "coverage/scored": scored_rollouts,
                "coverage/known": int(inventory_known),
                **({"coverage/imputed": len(missing)} if config.count_missing_rollouts_as_zero else {}),
                "coverage/measured": completion["measured"],
                "coverage/masked": completion["masked"],
                "coverage/unscored": completion["unscored"],
            }
            if inventory_known:
                metrics["coverage/expected"] = completion["expected"]
                metrics["coverage/missing"] = completion["missing"]
                metrics["coverage/not_scored"] = completion["expected"] - scored_rollouts
                metrics["coverage/failed"] = completion["failed"]
                metrics["coverage/omitted"] = completion["intentionally_omitted"]
                metrics["coverage/not_started"] = completion["not_started"]
                metrics["coverage/missing_outcomes"] = completion["missing"]
                metrics["coverage/failure_budget_exhausted"] = completion["failure_budget_exhausted"]
            export_metrics(metrics)

        print(f"""Finished rollout aggregation! View results at:
Merged rollouts: {output_fpath if config.merge_shards else "<not merged>"}
Aggregate metrics: {aggregate_metrics_fpath}{coverage}""")

        if not config.disable_health_check:
            from nemo_gym.rollout_health import JournalHealthUnavailable, format_health_report, run_health_checks

            try:
                health_result = await asyncio.to_thread(
                    run_health_checks,
                    # Keep each shard's identity, failure history and saved retry policy.
                    [Path(path) for path in input_paths],
                    selected_projection=output_fpath if config.merge_shards and inventory_known else None,
                    run_coverage=components if inventory_known else None,
                    output_dir=output_fpath.parent,
                    workers=config.health_check_workers,
                    ignored_checks=config.health_check_ignored_checks,
                    limits=config.run_health_limits,
                )
            except JournalHealthUnavailable as error:
                print(f"Rollout health checks skipped: {error}")
            except Exception:
                logger.exception(
                    "Rollout health checks failed after aggregation; aggregate artifacts are still available."
                )
            else:
                print(format_health_report(health_result))
                export_run_health_timeouts(health_result)

        return aggregate_metrics_fpath


# Backward-compatibility shims (CLI refactor): these CLI entry points moved to `nemo_gym.cli.eval`.
# Re-exported lazily to avoid a circular import; accessing them emits a DeprecationWarning.
from nemo_gym._moved_paths import moved_names_getattr  # noqa: E402
from nemo_gym.utils.compat import moved_attr_getter  # noqa: E402


_moved_cli_entry_points = moved_attr_getter(
    __name__,
    {
        "collect_rollouts": "nemo_gym.cli.eval",
        "aggregate_rollouts": "nemo_gym.cli.eval",
    },
)
# Names that moved to other modules stay importable from here until their removal release; see nemo_gym/_moved_paths.py.
_moved_names = moved_names_getattr(__name__)


def __getattr__(name: str) -> Any:
    try:
        return _moved_names(name)
    except AttributeError:
        return _moved_cli_entry_points(name)
