# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""The capture record schemas that everything else keys off.

Records are written to the per-rollout capture store wrapped in an envelope:

    {"v": 1, "kind": "model_call" | "tokens" | "tool", "seq": N, "ts": ..., "data": {...}}

`TokenEntry` (kind="tokens") is kept as a separate record from `ModelCallRecord`
(kind="model_call") on purpose: the token ids are large and are only read by
training, so keeping them in their own record lets a reader that only needs
call stats skip them, and lets the token ids later move to a different store
without changing anything that reads ModelCallRecords.
"""

from __future__ import annotations

import time
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# Rollout identity and the correlation carriers now live in nemo_gym.rollout_id.
AUX_TAG_HEADER = "x-nemo-gym-aux"


class ModelCallRecord(BaseModel):
    """One model call's stats and content, used by eval, debugging, and
    reporting. `response` is stored WITHOUT token fields (those live in
    TokenEntry). `seq` is the authoritative per-rollout ordering.

    Named for what it is — one model call — since "step" is overloaded in this
    codebase (env step, GRPO step, turn)."""

    model_config = ConfigDict(extra="allow")

    kind: Literal["model_call"] = "model_call"
    rollout_id: str
    request_id: str
    # Named after the API, not the vendor; an open set keyed to the converter
    # registry: "messages" (/v1/messages) | "chat_completions" | "responses".
    dialect: str = "chat_completions"
    model: str = ""
    status_code: int = 200
    error_category: Optional[str] = None
    latency_ms: float = 0.0
    ttft_ms: Optional[float] = None  # streaming calls
    tokens_in: int = 0
    tokens_out: int = 0
    cache: Optional[dict] = None  # normalized cache stats (cached/read/creation)
    aux_tag: Optional[str] = None  # e.g. "verifier" via x-nemo-gym-aux
    from_untrusted_prefix: bool = False  # arrived via /ng-rollout/<id> (sandbox)
    request: dict = Field(default_factory=dict)
    response: dict = Field(default_factory=dict)
    seq: int = -1  # assigned by the store


TOKEN_FIELDS = ("prompt_token_ids", "generation_token_ids", "generation_log_probs", "routed_experts")


class TokenEntry(BaseModel):
    """One model call's token ids and logprobs, used by training. Today these
    are recorded at the gate from the token fields the model server attaches to
    its response."""

    model_config = ConfigDict(extra="allow")

    kind: Literal["tokens"] = "tokens"
    rollout_id: str
    request_id: str  # joins 1:1 with its ModelCallRecord
    prompt_token_ids: list[int]
    generation_token_ids: list[int]
    generation_log_probs: list[float]
    routed_experts: Optional[Any] = None
    model: str = ""
    weight_version: Optional[int] = None
    seq: int = -1


class ToolObservation(BaseModel):
    """Timing and status for one environment tool call. Available for tools
    the resources server lends; best-effort or absent for tools that run
    entirely inside the harness's own sandbox."""

    model_config = ConfigDict(extra="allow")

    kind: Literal["tool"] = "tool"
    rollout_id: str
    call_id: str
    tool_name: str
    status: str = "ok"
    started_at: float = Field(default_factory=time.time)
    duration_ms: float = 0.0
    sandbox: Optional[dict] = None
    seq: int = -1


KIND_TYPES = {"model_call": ModelCallRecord, "tokens": TokenEntry, "tool": ToolObservation}


def record_from_envelope(env: dict) -> BaseModel:
    cls = KIND_TYPES.get(env.get("kind"))
    if cls is None:
        raise KeyError(f"unknown capture record kind: {env.get('kind')!r}")
    data = dict(env["data"])
    data["seq"] = env["seq"]
    return cls.model_validate(data)
