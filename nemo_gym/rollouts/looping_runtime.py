# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Passive live looping analysis with bounded admission and owned CPU processes."""

from __future__ import annotations

import copy
import logging
import math
import multiprocessing
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full
from typing import Any

from omegaconf import OmegaConf

from nemo_gym.config.global_config import environment_server_attributed_agent, get_first_server_config_dict
from nemo_gym.config.looping import LoopingConfig
from nemo_gym.metrics.looping import LoopingCounts
from nemo_gym.models.model_call_capture import CaptureStore
from nemo_gym.rollouts.looping_assessment import assess_looping_record
from nemo_gym.rollouts.model_calls import merge_model_call_capture_into_record, model_call_capture_dirs_from_config
from nemo_gym.rollouts.trajectory_builder import _build_trajectory_record, _trajectory_identity
from nemo_gym.trajectory.looping import (
    MAX_OUTPUT_ITEMS,
    MAX_TOOL_CALLS,
    LoopingAssessment,
    LoopingInputLimit,
    check_looping_input_size,
    unknown_looping_assessment,
)


logger = logging.getLogger(__name__)
MAX_CAPTURE_BYTES = 64 * 1024 * 1024


def _file_state(path: Path) -> tuple[int, int, int, int] | None:
    try:
        file_stat = path.stat()
    except FileNotFoundError:
        return None
    return file_stat.st_dev, file_stat.st_ino, file_stat.st_size, file_stat.st_mtime_ns


@dataclass(frozen=True)
class _CaptureInput:
    record: dict[str, Any]
    files: tuple[tuple[Path, tuple[int, int, int, int] | None], ...]


def _capture_input(
    row: Mapping[str, Any], result: Mapping[str, Any], directories: tuple[Path, ...], policy: str | None
) -> _CaptureInput:
    task_id, rollout_id = _trajectory_identity(dict(row))
    record = {
        "trajectory_identity": {"task_id": task_id, "rollout_id": rollout_id},
        "_ng_rollout_id": rollout_id,
        "ng_policy_model_server": policy,
        "response": {"id": (result.get("response") or {}).get("id")},
    }
    observations = result.get("ng_agent_observations")
    if observations is not None:
        records = observations.get("records") or []
        if len(records) > MAX_OUTPUT_ITEMS:
            raise LoopingInputLimit("output_item_limit")
        tool_call_count = sum(observation.get("kind") == "tool_call" for observation in records)
        if tool_call_count > MAX_TOOL_CALLS:
            raise LoopingInputLimit("output_item_limit")
        retained_records = []
        for observation in records:
            if observation.get("kind") not in {"agent_invocation", "tool_call", "context_compaction"}:
                continue
            observation = dict(observation)
            if observation.get("kind") == "agent_invocation":
                conversation = observation.get("conversation") or []
                if len(conversation) > MAX_OUTPUT_ITEMS:
                    raise LoopingInputLimit("output_item_limit")
                observation["conversation"] = [
                    {"type": event["type"], "call_id": event.get("call_id"), "output": event.get("output")}
                    for event in conversation
                    if event.get("type") in {"function_call_output", "custom_tool_call_output"}
                ]
            retained_records.append(observation)
        record["ng_agent_observations"] = {**observations, "records": retained_records}
    check_looping_input_size(record)
    capture_files = []
    for directory in directories:
        capture_path = CaptureStore(directory, create=False).path_for(rollout_id)
        capture_files.append((capture_path, _file_state(capture_path)))
    return _CaptureInput(copy.deepcopy(record), tuple(capture_files))


def _assess_capture(source: _CaptureInput) -> LoopingAssessment:
    try:
        if any(expected_state is None for _, expected_state in source.files):
            return unknown_looping_assessment("capture_missing")
        if any(_file_state(path) != expected_state for path, expected_state in source.files):
            return unknown_looping_assessment("capture_changed")
        for _, expected_state in source.files:
            if expected_state is None:
                continue
            _, _, size_bytes, _ = expected_state
            if size_bytes > MAX_CAPTURE_BYTES:
                return unknown_looping_assessment("capture_byte_limit")
        record = source.record
        merge_model_call_capture_into_record(record, [path.parent for path, _ in source.files], include_payloads=True)
        if any(_file_state(path) != expected_state for path, expected_state in source.files):
            return unknown_looping_assessment("capture_changed")
        trajectory = _build_trajectory_record(record, record, schema_version="2.0", include_payloads=True)
        record["ng_trajectory"] = trajectory.model_dump(mode="json")
        return assess_looping_record(record)
    except (ValueError, TypeError, KeyError, AttributeError, OSError, RecursionError, OverflowError):
        logger.warning("Looping capture analysis failed", exc_info=True)
        return unknown_looping_assessment("analysis_error")


def _worker(inbox: Any, outbox: Any) -> None:
    while True:
        ticket, source = inbox.get()
        assessment = _assess_capture(source)
        outbox.put((ticket, assessment))


class LoopingExecutor:
    """A fixed pool for capture analysis; normal submission and polling never wait for scans."""

    @classmethod
    def from_config(cls, global_config: Mapping[str, Any], *, config: Mapping[str, Any]) -> LoopingExecutor | None:
        """Resolve capture context once; disabled analysis starts no processes."""
        settings = LoopingConfig.model_validate(config)
        if not settings.enabled:
            return None
        global_config = OmegaConf.create(global_config, flags={"allow_objects": True})
        directories = tuple(model_call_capture_dirs_from_config(global_config))
        if not directories:
            raise ValueError("looping analysis requires observability_enabled and model_call_capture_dir")
        policies = {}
        for server_name, server_block in global_config.items():
            if not isinstance(server_block, Mapping) or "responses_api_agents" not in server_block:
                continue
            agent_config = get_first_server_config_dict(global_config, server_name)
            model_reference = agent_config.get("model_server")
            if isinstance(model_reference, Mapping):
                policies[server_name] = model_reference.get("name")
            else:
                policies[server_name] = None
        environment_policies = {}
        for server_name, server_block in global_config.items():
            if not isinstance(server_block, Mapping) or "environment_servers" not in server_block:
                continue
            environment_config = get_first_server_config_dict(global_config, server_name)
            agent_name = environment_server_attributed_agent(environment_config)
            environment_policies[server_name] = policies.get(agent_name)
        policies.update(environment_policies)
        return cls(settings, directories, policies)

    def __init__(self, config: LoopingConfig, directories: tuple[Path, ...], policies: Mapping[str, str | None]):
        context = multiprocessing.get_context("spawn")
        self._inbox = context.Queue(maxsize=config.max_pending)
        self._outbox = context.Queue(maxsize=config.max_pending)
        self._processes = [
            context.Process(target=_worker, args=(self._inbox, self._outbox), daemon=True)
            for _ in range(config.workers)
        ]
        self._directories = directories
        self._policies = policies
        self._limit = config.max_pending
        self._pending: dict[int, bool] = {}
        self._counts = LoopingCounts()
        self._lock = threading.RLock()
        self._epoch = time.time()
        self._ticket = 0
        self._rejected = 0
        self._closed = False
        self._failed = False
        try:
            for process in self._processes:
                process.start()
        except BaseException:
            for process in self._processes:
                if process.pid is not None:
                    process.kill()
                    process.join(timeout=0)
            for channel in (self._inbox, self._outbox):
                channel.close()
            raise

    def _poll(self) -> None:
        while True:
            try:
                ticket, assessment = self._outbox.get_nowait()
            except (Empty, ValueError):
                break
            if ticket in self._pending:
                is_valid = self._pending.pop(ticket)
                self._counts.observe(assessment, valid=is_valid)
        if self._closed or self._failed:
            return
        worker_exited = any(process.exitcode is not None for process in self._processes)
        if not worker_exited:
            return
        self._failed = True
        logger.warning("Looping worker exited; outstanding assessments are unavailable")
        for is_valid in self._pending.values():
            self._counts.observe(unknown_looping_assessment("worker_error"), valid=is_valid)
        self._pending.clear()
        for process in self._processes:
            if process.is_alive():
                process.kill()

    def submit_completed(self, row: Mapping[str, Any], result: Mapping[str, Any]) -> bool:
        """Admit one completed dispatch or record an unavailable assessment."""
        is_valid = not (
            result.get("mask_sample") is True or (result.get("instance_config") or {}).get("mask_sample") is True
        )
        with self._lock:
            self._poll()
            if self._closed:
                reason = "analysis_closed"
            elif self._failed:
                reason = "worker_unavailable"
            elif len(self._pending) >= self._limit:
                reason = "queue_full"
            else:
                reason = None
            if reason is not None:
                self._counts.observe(unknown_looping_assessment(reason), valid=is_valid)
                self._rejected += 1
                return False
            try:
                dispatch_name = row.get("_ng_environment_server")
                if not isinstance(dispatch_name, str):
                    dispatch_name = (row.get("agent_ref") or {}).get("name")
                policy = self._policies.get(dispatch_name)
                source = _capture_input(row, result, self._directories, policy)
                self._inbox.put_nowait((self._ticket, source))
            except (LoopingInputLimit, ValueError, TypeError, KeyError, AttributeError, OSError, Full) as error:
                if isinstance(error, LoopingInputLimit):
                    reason = str(error)
                else:
                    reason = "record_unavailable"
                self._counts.observe(unknown_looping_assessment(reason), valid=is_valid)
                self._rejected += 1
                return False
            self._pending[self._ticket] = is_valid
            self._ticket += 1
            return True

    def metrics_snapshot(self) -> dict[str, float | None]:
        """Read cumulative process-local counts and pending coverage."""
        with self._lock:
            self._poll()
            expected_total = self._counts.counts["selected_total"] + len(self._pending)
            return {
                **self._counts.metrics(expected=expected_total),
                "looping/epoch": self._epoch,
                "looping/pending_total": float(len(self._pending)),
                "looping/submitted_total": float(self._ticket),
                "looping/rejected_total": float(self._rejected),
            }

    def close(self, *, timeout_s: float) -> None:
        """Drain within the caller's budget, then terminate only this executor's processes."""
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("looping close timeout must be positive")
        shutdown_deadline = time.monotonic() + timeout_s
        # Reserve the last fifth of the budget for terminating and joining workers.
        drain_deadline = shutdown_deadline - timeout_s / 5
        with self._lock:
            if self._closed:
                return
            self._closed = True
        while time.monotonic() < drain_deadline:
            with self._lock:
                self._poll()
                if not self._pending:
                    break
            remaining_s = max(0, shutdown_deadline - time.monotonic())
            time.sleep(min(0.01, remaining_s))
        for process in self._processes:
            if process.is_alive():
                # Workers only read evidence; kill also bounds stopped or unresponsive workers.
                process.kill()
        for process in self._processes:
            remaining_s = max(0, shutdown_deadline - time.monotonic())
            process.join(timeout=remaining_s)
        if any(process.is_alive() for process in self._processes):
            logger.warning("Looping workers have not exited after the shutdown deadline")
        with self._lock:
            self._poll()
            for is_valid in self._pending.values():
                self._counts.observe(unknown_looping_assessment("shutdown_timeout"), valid=is_valid)
            self._pending.clear()
        for channel in (self._inbox, self._outbox):
            channel.cancel_join_thread()
            channel.close()
