# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""An episode's generation limit, carried to its agent and model in the correlation URL."""

import base64
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from typing import Iterator

from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.core.correlation import CorrelationContext, encode, parse_path


class EpisodeGenerationBudget(BaseModel):
    """A cap shared by all attempts of one logical rollout, with its explicit stop owner."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    max_calls: int = Field(gt=0)
    rollout_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    agent_server: str = Field(min_length=1)

    def segment(self) -> str:
        return base64.urlsafe_b64encode(self.model_dump_json().encode()).decode().rstrip("=")

    @classmethod
    def from_segment(cls, value: str) -> "EpisodeGenerationBudget":
        if len(value) > 1024:
            raise ValueError("Episode generation budget segment is too long")
        raw = base64.b64decode(value + "=" * (-len(value) % 4), altchars=b"-_", validate=True)
        return cls.model_validate_json(raw)


_CURRENT: ContextVar[EpisodeGenerationBudget | None] = ContextVar("episode_generation_budget", default=None)


def current_generation_budget() -> EpisodeGenerationBudget | None:
    return _CURRENT.get()


@contextmanager
def generation_budget_context(budget: EpisodeGenerationBudget | None) -> Iterator[None]:
    token = _CURRENT.set(budget)
    try:
        yield
    finally:
        _CURRENT.reset(token)


def with_generation_budget(path: str, budget: EpisodeGenerationBudget | None) -> str:
    """Add the explicitly supplied budget, preserving all other correlation segments."""
    if budget is None:
        return path
    path, separator, query = path.partition("?")
    parsed = parse_path(path)
    context, rest = parsed if parsed is not None else (CorrelationContext(), path)
    return encode(replace(context, generation_budget=budget.segment())) + rest + separator + query
