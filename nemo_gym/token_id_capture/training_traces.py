# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned, text-only training rows owned by one logical rollout.

Context tokens may appear in several rows. A sampled call's generated span
has exactly one loss-bearing owner, independent of the reconstruction mode.
This module does not assign rewards, advantages, or trainer loss weights.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nemo_gym.token_id_capture.builder import BuildOutput


class SampledSpan(BaseModel):
    """Half-open row positions for one call's owned sampled generation."""

    model_config = ConfigDict(extra="forbid", strict=True)
    model_call_id: str = Field(min_length=1)
    start: int = Field(ge=1)
    end: int = Field(ge=2)


class TrainingTrace(BaseModel):
    """One complete conditioning sequence with aligned behavior evidence."""

    model_config = ConfigDict(extra="forbid", strict=True)
    trace_id: str = Field(min_length=1)
    model_call_ids: list[str]
    token_ids: list[int]
    generation_logprobs: list[float]
    loss_mask: list[int]
    sampled_spans: list[SampledSpan]

    @model_validator(mode="after")
    def _validate_tokens(self) -> "TrainingTrace":
        size = len(self.token_ids)
        if not size or len(self.generation_logprobs) != size or len(self.loss_mask) != size:
            raise ValueError("training trace token, logprob, and mask lengths must match and be nonempty")
        if self.loss_mask[0] != 0 or any(mask not in (0, 1) for mask in self.loss_mask):
            raise ValueError("training trace requires a masked conditioning token and binary loss masks")
        if any(token < 0 for token in self.token_ids) or any(not math.isfinite(lp) for lp in self.generation_logprobs):
            raise ValueError("training trace requires nonnegative tokens and finite logprobs")
        if len(set(self.model_call_ids)) != len(self.model_call_ids):
            raise ValueError("model_call_ids must be unique within a trace")
        owned = [0] * size
        for span in self.sampled_spans:
            if span.model_call_id not in self.model_call_ids or not span.start < span.end <= size:
                raise ValueError("sampled span must reference a captured call and lie within its trace")
            for index in range(span.start, span.end):
                if owned[index]:
                    raise ValueError("sampled spans must not overlap")
                owned[index] = 1
        if owned != self.loss_mask or not any(owned):
            raise ValueError("sampled spans must exactly cover all trainable tokens")
        return self


class TrainingTraceBatch(BaseModel):
    """Wire contract for all retained calls of one logical rollout."""

    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal[1] = 1
    rollout_id: str = Field(min_length=1)
    builder: Literal["prefix_merging", "per_request"]
    traces: list[TrainingTrace] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_ownership(self) -> "TrainingTraceBatch":
        if len({trace.trace_id for trace in self.traces}) != len(self.traces):
            raise ValueError("trace ids must be unique within a rollout")
        owners = [span.model_call_id for trace in self.traces for span in trace.sampled_spans]
        if len(owners) != len(set(owners)):
            raise ValueError("each captured sampled span must have exactly one training owner")
        if set(owners) != {call_id for trace in self.traces for call_id in trace.model_call_ids}:
            raise ValueError("every referenced captured call must have a training owner")
        return self


def project_training_traces(rollout_id: str, out: BuildOutput) -> TrainingTraceBatch:
    """Project safe chains, masking repeated ancestors rather than sampling them twice.

    Chain order depends on call identities, not transport completion order.
    A repeated ancestor remains exact context in another branch, with loss 0.
    Unsupported router-replay evidence is rejected rather than discarded.
    """
    if out.notes.unresolved_retries or out.notes.unresolved_parent_calls:
        raise ValueError("all_traces requires resolved retry selection and parent custody")
    if out.notes.terminal_chain in ("broken", "not_captured"):
        raise ValueError("the declared terminal call is not safely captured")
    owned_calls: set[str] = set()
    traces: list[TrainingTrace] = []
    for chain in sorted(out.chains, key=lambda item: tuple(link.entry.model_call_id for link in item.links)):
        chain.validate()
        token_ids = list(chain.root_prompt)
        logprobs = [0.0] * len(token_ids)
        masks = [0] * len(token_ids)
        spans: list[SampledSpan] = []
        call_ids: list[str] = []
        for link in chain.links:
            entry = link.entry
            if entry.rollout_id != rollout_id:
                raise ValueError("captured call belongs to a different rollout")
            if entry.routed_experts is not None:
                raise ValueError("all_traces does not yet support routed_experts")
            call_ids.append(entry.model_call_id)
            token_ids.extend(link.interstitial)
            logprobs.extend([0.0] * len(link.interstitial))
            masks.extend([0] * len(link.interstitial))
            start = len(token_ids)
            if not start:
                raise ValueError("sampled tokens require a nonempty conditioning prompt")
            token_ids.extend(entry.generation_token_ids)
            owns = entry.model_call_id not in owned_calls
            logprobs.extend(entry.generation_log_probs if owns else [0.0] * len(entry.generation_token_ids))
            masks.extend([int(owns)] * len(entry.generation_token_ids))
            if owns:
                spans.append(SampledSpan(model_call_id=entry.model_call_id, start=start, end=len(token_ids)))
                owned_calls.add(entry.model_call_id)
        if spans:
            traces.append(
                TrainingTrace(
                    trace_id=f"{rollout_id}:{call_ids[-1]}",
                    model_call_ids=call_ids,
                    token_ids=token_ids,
                    generation_logprobs=logprobs,
                    loss_mask=masks,
                    sampled_spans=spans,
                )
            )
    return TrainingTraceBatch(rollout_id=rollout_id, builder=out.notes.builder, traces=traces)
