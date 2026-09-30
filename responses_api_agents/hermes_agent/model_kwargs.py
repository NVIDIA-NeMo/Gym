# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared model-request adaptation for host and sandbox Hermes runtimes (stdlib only)."""

import json
import logging
from typing import Any


LOG = logging.getLogger(__name__)


def _model_api_kwargs(
    kwargs: dict[str, Any], *, preserve_reasoning_history: bool, model_enable_thinking: bool | None
) -> dict[str, Any]:
    kwargs = kwargs.copy()
    extra_body = dict(kwargs.get("extra_body") or {})
    metadata = dict(kwargs.get("metadata") or {})
    template_kwargs = json.loads(metadata.pop("chat_template_kwargs", None) or "{}")
    template_kwargs.update(extra_body.pop("chat_template_kwargs", None) or {})
    if "enable_thinking" in template_kwargs:
        received = template_kwargs.pop("enable_thinking")
        if model_enable_thinking is None:
            LOG.warning(
                "Ignoring Hermes enable_thinking=%r; the Model Server has no explicit enable_thinking setting. "
                "Configure thinking on the Model Server, not Hermes.",
                received,
            )
        elif received != model_enable_thinking:
            LOG.warning(
                "Ignoring Hermes enable_thinking=%r: it conflicts with Model Server enable_thinking=%r. "
                "The Model Server setting takes precedence.",
                received,
                model_enable_thinking,
            )
    if preserve_reasoning_history:
        template_kwargs["truncate_history_thinking"] = False
    # Gym rejects chat_template_kwargs at the top level, where the SDK expands extra_body.
    if template_kwargs:
        metadata["chat_template_kwargs"] = json.dumps(template_kwargs)
    if extra_body:
        kwargs["extra_body"] = extra_body
    else:
        kwargs.pop("extra_body", None)
    if metadata:
        kwargs["metadata"] = metadata
    else:
        kwargs.pop("metadata", None)
    return kwargs
