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

"""Trajectory types, TokenSource implementations, and auxiliary-call
classification.

The builder reads token ids only through the TokenSource interface, so where
the token ids actually come from (the capture store, inline response items, or
some future store) is swappable without changing the builder.
"""

from __future__ import annotations

from typing import Optional, Protocol

from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.observability.capture_reader import CaptureReader
from nemo_gym.observability.records import ModelCallRecord, TokenEntry


class ScopedReward(BaseModel):
    value: float
    scope: str = "rollout"  # rollout | turn | step | span
    source: str = "verify"
    span: Optional[tuple[int, int]] = None


class Trajectory(BaseModel):
    """The trainable record. It has the same shape whether the rollout came
    from an external harness or an in-process Gym agent."""

    model_config = ConfigDict(extra="allow")

    rollout_id: str
    chain_id: str  # "main" | "branch-<n>"
    token_ids: list[int]
    loss_mask: list[int]  # 1 = policy-sampled, else 0
    logprobs: list[Optional[float]]  # aligned; None where mask == 0
    messages: list[dict] = Field(default_factory=list)
    reward: float = 0.0
    rewards: list[ScopedReward] = Field(default_factory=list)
    is_resolved: Optional[bool] = None
    provenance: dict = Field(default_factory=dict)

    def model_post_init(self, __context) -> None:
        n = len(self.token_ids)
        if not (len(self.loss_mask) == len(self.logprobs) == n):
            raise ValueError(f"misaligned trajectory: ids={n} mask={len(self.loss_mask)} lp={len(self.logprobs)}")
        for m, lp in zip(self.loss_mask, self.logprobs):
            if m == 1 and lp is None:
                raise ValueError("mask-1 position missing logprob")


class TokenSource(Protocol):
    def entries(self, rollout_id: str) -> list[TokenEntry]: ...


class CaptureTokenSource:
    """Reads token ids from the capture store (the kind='tokens' records).
    Used for external harnesses, whose calls are captured at the gate."""

    def __init__(self, reader: CaptureReader):
        self.reader = reader

    def entries(self, rollout_id: str) -> list[TokenEntry]:
        return [r for r in self.reader.records(rollout_id, kinds={"tokens"}) if isinstance(r, TokenEntry)]


class InlineItemsTokenSource:
    """Adapts the token-id fields already carried on a returned response's
    output items into TokenEntry records. Used for in-process Gym agents, whose
    model responses carry the token ids inline, so they feed the same builder
    as captured external rollouts."""

    def __init__(self, rollout_id: str, response: dict, model: str = ""):
        self.rollout_id = rollout_id
        self.response = response
        self.model = model

    def entries(self, rollout_id: str) -> list[TokenEntry]:
        out: list[TokenEntry] = []
        seq = 0
        for item in self.response.get("output", []):
            if not isinstance(item, dict):
                continue
            if item.get("generation_token_ids") is None:
                continue
            out.append(
                TokenEntry(
                    rollout_id=self.rollout_id,
                    request_id=f"inline-{seq}",
                    prompt_token_ids=item.get("prompt_token_ids") or [],
                    generation_token_ids=item.get("generation_token_ids") or [],
                    generation_log_probs=item.get("generation_log_probs") or [],
                    routed_experts=item.get("routed_experts"),
                    model=self.model,
                    seq=seq,
                )
            )
            seq += 1
        return out


class DataPlaneTokenSource:
    """Not implemented. Placeholder for reading token ids from a training data
    plane instead of the capture store."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "DataPlaneTokenSource is not implemented yet; use CaptureTokenSource or InlineItemsTokenSource."
        )


DEFAULT_AUX_FINGERPRINTS = ("title generator", "summarize this conversation")


def classify_auxiliary(
    steps: list[ModelCallRecord], policy_model: str = "", fingerprints: tuple[str, ...] = DEFAULT_AUX_FINGERPRINTS
) -> dict[str, str]:
    """Return request_id -> reason for every model call that should be
    excluded from training. A call is auxiliary if it carries an aux tag
    (a verifier or judge call), if it went to a different model than the
    policy, or if its request text matches one of the known non-task
    fingerprints (for example a title generator or conversation summarizer)."""
    aux: dict[str, str] = {}
    for s in steps:
        if s.aux_tag:
            aux[s.request_id] = f"tag:{s.aux_tag}"
            continue
        if policy_model and s.model and s.model != policy_model:
            aux[s.request_id] = f"model:{s.model}"
            continue
        blob = str(s.request).lower()
        for fp in fingerprints:
            if fp in blob:
                aux[s.request_id] = f"fingerprint:{fp}"
                break
    return aux
