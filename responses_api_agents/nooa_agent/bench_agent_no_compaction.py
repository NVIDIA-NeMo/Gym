# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned BenchAgent configuration for training on complete model histories."""

from typing import Any

from nooa.config.truncation_config import TruncationConfig
from nooa.interactive import SummarizationConfig
from nooa_bench.bench_agent import BenchAgent


class BenchAgentWithoutCompaction(BenchAgent):
    """Disable automatic summaries, history eviction and overflow recovery pruning.

    Delegation constructs ``type(self)`` and shares its LLM, so each child gets
    these same controls. Tool-output rendering limits remain upstream-owned.
    """

    _gym_preserve_training_history = True

    def __init__(self, *args: Any, summarization: SummarizationConfig | None = None, **kwargs: Any) -> None:
        if summarization is not None and summarization.policy != "none":
            raise ValueError("Training BenchAgent requires summarization.policy='none'")
        if "truncation" in kwargs:
            raise ValueError("Training BenchAgent owns history truncation controls")
        super().__init__(
            *args,
            summarization=SummarizationConfig(policy="none"),
            truncation=TruncationConfig(max_context_tokens=None, max_event_tokens=None, response_reserve_tokens=0),
            **kwargs,
        )

        def refuse_context_pruning(ctx_window: int | None, exc: BaseException | None = None) -> None:
            # The pinned runtime otherwise archives events after a provider rejects
            # the context, even with its summarizer and render budgets disabled.
            if exc is not None:
                raise exc
            raise RuntimeError("NOOA context limit reached; automatic history pruning is disabled")

        self.runtime._archive_on_context_error = refuse_context_pruning
