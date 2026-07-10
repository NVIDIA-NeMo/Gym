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
"""Rollout-keyed capture storage and observability records for model calls."""

from __future__ import annotations

import fcntl
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import orjson
from pydantic import BaseModel, model_validator

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming


class ModelCallCaptureConfig(BaseModel):
    """Run-wide model-call capture settings from Gym's global config."""

    observability_enabled: bool = False
    model_call_capture_dir: Optional[Path] = None

    @model_validator(mode="after")
    def validate_capture_dir(self) -> "ModelCallCaptureConfig":
        if not self.observability_enabled:
            return self
        if self.model_call_capture_dir is None:
            raise ValueError("model_call_capture_dir is required when observability_enabled=true")
        if not self.model_call_capture_dir.is_absolute():
            raise ValueError("model_call_capture_dir must be an absolute path")
        return self


def _validate_rollout_id(rollout_id: str) -> str:
    if not rollout_id or any(not (char.isascii() and (char.isalnum() or char in "._-")) for char in rollout_id):
        raise ValueError(f"Invalid rollout id: {rollout_id!r}")
    return rollout_id


class CaptureStore:
    """Append-only, rollout-keyed JSONL sink for model exchanges."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @property
    def root(self) -> Path:
        return self._root

    def path_for(self, rollout_id: str) -> Path:
        return self._root / f"{_validate_rollout_id(rollout_id)}.capture.jsonl"

    def record(self, rollout_id: str, exchange: dict[str, Any]) -> None:
        """Append one exchange and fsync (durable across a killed box).

        ``flock`` serializes appends across worker processes (a model server may run with
        ``num_workers > 1``, where the in-process lock can't coordinate); the in-process lock
        serializes threads. This does blocking file IO + fsync, so callers run it off the event
        loop (the capture middleware offloads it via ``asyncio.to_thread``).
        """
        line = orjson.dumps(exchange, default=str, option=orjson.OPT_APPEND_NEWLINE)
        path = self.path_for(rollout_id)
        with self._lock:
            with path.open("ab") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    handle.write(line)
                    handle.flush()
                    os.fsync(handle.fileno())
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def read(self, rollout_id: str) -> list[dict[str, Any]]:
        path = self.path_for(rollout_id)
        if not path.exists():
            return []
        exchanges: list[dict[str, Any]] = []
        # Stream line-by-line; a capture can be large (token-ids / logprobs).
        with self._lock:
            with path.open("rb") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
                try:
                    for line in handle:
                        stripped = line.strip()
                        if not stripped:
                            continue
                        exchanges.append(orjson.loads(stripped))
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return exchanges


class ModelCallRecord(BaseModel):
    """Observability record derived from one captured model-server exchange."""

    # HTTP information
    status_code: int
    route: str

    # Gym information
    model_ref: ModelServerRef

    # Model-call record
    request: NeMoGymResponseCreateParamsNonStreaming
    response: Optional[NeMoGymResponse]  # Only present if the call succeeded

    # Used for cases where we never hit a NeMoGymResponsesCreateParams or NeMoGymResponse in a model call e.g. calling an Anthropic model with /v1/messages
    # For those scenarios we always store the raw_request and raw_response and provided a normalized version by converting to Responses
    # For normal Responses routes, this is empty.
    raw_request: Optional[Dict[str, Any]]
    raw_response: Optional[Dict[str, Any]]


def read_model_call_records(store: CaptureStore, rollout_id: str) -> list[ModelCallRecord]:
    """Read captured exchanges in durable append order."""
    return store.read(rollout_id)


def aggregate_model_call_records(calls: list[ModelCallRecord]) -> dict[str, Any]:
    """Aggregate token and latency values from model-call records."""

    def _sum(attr: str) -> Optional[float]:
        values = [getattr(call, attr) for call in calls if getattr(call, attr) is not None]
        return sum(values) if values else None

    return {
        "tokens_in": _sum("tokens_in"),
        "tokens_out": _sum("tokens_out"),
        "tokens_reasoning": _sum("tokens_reasoning"),
        "tokens_total": _sum("tokens_total"),
        "latency_total_ms": _sum("latency_total_ms"),
        "num_calls": len(calls),
    }


def aggregate_model_call_metrics(store: CaptureStore, rollout_id: str) -> dict[str, Any]:
    """Aggregate model-call metrics for one rollout id."""
    return aggregate_model_call_records(read_model_call_records(store, rollout_id))
