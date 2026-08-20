# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Narrow compatibility hooks for the pinned WebArena-family evaluators."""

from __future__ import annotations

import importlib
import logging
import os
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any


_ORIGINAL_ATTR = "_nemo_gym_original_chat_completion"
LOG = logging.getLogger("nemo_gym.resources_servers.browsergym_web")
_RULE_ONLY_EVALUATOR_KEY = "unused-for-rule-only-evaluator"


def configure_evaluator_environment(*, api_key: str, base_url: str | None) -> None:
    """Prepare the legacy OpenAI clients before their modules are imported."""

    os.environ["OPENAI_API_KEY"] = api_key
    if base_url:
        # OPENAI_BASE_URL is used by the pinned OpenAI client. Keep the legacy
        # OPENAI_API_BASE spelling as well because existing evaluator images
        # can contain the older WebArena-era client.
        os.environ["OPENAI_BASE_URL"] = base_url
        os.environ["OPENAI_API_BASE"] = base_url


@contextmanager
def rule_only_evaluator_import_environment(*, base_url: str | None):
    """Temporarily satisfy VisualWebArena's unconditional client import."""

    keys = ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE")
    previous = {key: os.environ.get(key) for key in keys}
    configure_evaluator_environment(
        api_key=_RULE_ONLY_EVALUATOR_KEY,
        base_url=base_url,
    )
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _configure_evaluator_model(*, package: str, benchmark: str, model_name: str | None) -> None:
    """Remap one benchmark's hard-coded GPT-4 judge to a configured model.

    The pinned WebArena and VisualWebArena evaluators hard-code
    ``gpt-4-1106-preview`` inside fuzzy and unachievable-answer evaluators.
    This hook changes only the model argument and leaves the upstream prompts,
    generation options, and score parsing intact.
    """

    if not model_name:
        return

    provider = importlib.import_module(f"{package}.llms.providers.openai_utils")
    helpers = importlib.import_module(f"{package}.evaluation_harness.helper_functions")
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
            raise TypeError(f"{benchmark} chat completion call did not provide a model")
        started_at = time.monotonic()
        try:
            return original(*args, **kwargs)
        finally:
            LOG.info(
                "%s evaluator request model=%s elapsed_seconds=%.3f",
                benchmark,
                model_name,
                time.monotonic() - started_at,
            )

    provider.generate_from_openai_chat_completion = generate_with_model_override
    helpers.generate_from_openai_chat_completion = generate_with_model_override


def configure_webarena_evaluator_model(model_name: str | None) -> None:
    _configure_evaluator_model(package="webarena", benchmark="WebArena", model_name=model_name)


def configure_evaluator_model(model_name: str | None) -> None:
    """Backward-compatible name for the VisualWebArena evaluator hook."""

    _configure_evaluator_model(
        package="visualwebarena",
        benchmark="VisualWebArena",
        model_name=model_name,
    )
