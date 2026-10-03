# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collect provider usage before Hermes can discard a response or compact its history."""

from __future__ import annotations

from functools import wraps
from threading import RLock
from typing import TYPE_CHECKING, TypedDict


if TYPE_CHECKING:
    from openai import OpenAI
    from openai.types.chat import ChatCompletion
    from run_agent import AIAgent


class _InputDetails(TypedDict):
    cached_tokens: int | None


class _OutputDetails(TypedDict):
    reasoning_tokens: int | None


class ResponseUsage(TypedDict):
    """JSON usage shared by the host adapter and the standalone sandbox runner."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_tokens_details: _InputDetails
    output_tokens_details: _OutputDetails


class HermesTokenUsage:
    """Count the root agent's Chat Completions, including retries and final summaries.

    Hermes' session counters omit summaries and some early returns. Instrument its
    existing client and its client factory instead, without modifying global SDK state.
    Auxiliary clients and delegated agents have separate usage and are not included.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._calls = 0
        self._complete = True
        self._input = self._output = 0
        self._cached: int | None = 0
        self._reasoning: int | None = 0

    def instrument(self, agent: AIAgent) -> None:
        """Track each client the pinned Hermes creates for this agent instance."""
        original = getattr(agent, "_create_openai_client", None)
        if not callable(original):
            return
        self._instrument_client(agent.client)

        @wraps(original)
        def create(*args, **kwargs):
            client = original(*args, **kwargs)
            self._instrument_client(client)
            return client

        agent._create_openai_client = create

    def _instrument_client(self, client: OpenAI) -> None:
        original = client.chat.completions.create

        @wraps(original)
        def create(*args, **kwargs):
            response = original(*args, **kwargs)
            self._record(response)
            return response

        client.chat.completions.create = create

    def _record(self, response: ChatCompletion) -> None:
        usage = getattr(response, "usage", None)
        prompt = getattr(usage, "prompt_tokens", None)
        completion = getattr(usage, "completion_tokens", None)
        cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", None)
        reasoning = getattr(getattr(usage, "completion_tokens_details", None), "reasoning_tokens", None)
        with self._lock:
            self._calls += 1
            if type(prompt) is not int or prompt < 0 or type(completion) is not int or completion < 0:
                self._complete = False
                return
            self._input += prompt
            self._output += completion
            self._cached = (
                self._cached + cached if self._cached is not None and type(cached) is int and cached >= 0 else None
            )
            self._reasoning = (
                self._reasoning + reasoning
                if self._reasoning is not None and type(reasoning) is int and reasoning >= 0
                else None
            )

    def snapshot(self) -> ResponseUsage | None:
        """Return reported totals, or unknown if any returned response omitted usage."""
        with self._lock:
            if not self._calls or not self._complete:
                return None
            return {
                "input_tokens": self._input,
                "output_tokens": self._output,
                "total_tokens": self._input + self._output,
                "input_tokens_details": {"cached_tokens": self._cached},
                "output_tokens_details": {"reasoning_tokens": self._reasoning},
            }
