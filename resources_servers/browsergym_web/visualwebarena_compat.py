# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Narrow compatibility hooks for the pinned VisualWebArena evaluator."""

from __future__ import annotations

import importlib
import logging
import time
from collections.abc import Callable
from typing import Any


_ORIGINAL_ATTR = "_nemo_gym_original_chat_completion"
LOG = logging.getLogger("nemo_gym.resources_servers.browsergym_web")


def configure_evaluator_model(model_name: str | None) -> None:
    """Remap VisualWebArena's hard-coded GPT-4 judge to a configured model.

    VisualWebArena 0.0.15 hard-codes ``gpt-4-1106-preview`` inside fuzzy and
    unachievable-answer evaluators. The OpenAI client already honors
    ``OPENAI_BASE_URL`` and ``OPENAI_API_KEY``; this hook changes only the
    model argument and leaves the upstream prompts and score parsing intact.
    """

    if not model_name:
        return

    provider = importlib.import_module("visualwebarena.llms.providers.openai_utils")
    helpers = importlib.import_module("visualwebarena.evaluation_harness.helper_functions")
    current: Callable[..., Any] = provider.generate_from_openai_chat_completion
    original: Callable[..., Any] = getattr(provider, _ORIGINAL_ATTR, current)
    setattr(provider, _ORIGINAL_ATTR, original)

    def generate_with_model_override(*args: Any, **kwargs: Any) -> Any:
        if "model" in kwargs:
            kwargs = dict(kwargs)
            kwargs["model"] = model_name
        elif len(args) >= 2:
            positional = list(args)
            positional[1] = model_name
            args = tuple(positional)
        else:
            raise TypeError("VisualWebArena chat completion call did not provide a model")
        started_at = time.monotonic()
        try:
            return original(*args, **kwargs)
        finally:
            LOG.info(
                "VisualWebArena evaluator request model=%s elapsed_seconds=%.3f",
                model_name,
                time.monotonic() - started_at,
            )

    provider.generate_from_openai_chat_completion = generate_with_model_override
    helpers.generate_from_openai_chat_completion = generate_with_model_override
