# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""OpenAI API routing layer for api_fetch.

Dispatches incoming REST-style OpenAI/ChatGPT requests to their matching implementation functions.
Modeled after the OpenAI API (https://platform.openai.com/docs/api-reference).
"""

from automationbench.tools.api.impl.chatgpt import (
    openai_chat_completions_create,
    openai_completions_create,
    openai_moderations_create,
    openai_responses_create,
)
from automationbench.utils.routing import make_router

# ---------------------------------------------------------------------------
# Route table and dispatch map
# ---------------------------------------------------------------------------

_ROUTES: list[tuple[str, str, str]] = [
    ("POST", r"openai/v1/chat/completions$", "chat_completions"),
    ("POST", r"openai/v1/completions$", "completions"),
    ("POST", r"openai/v1/responses$", "responses"),
    ("POST", r"openai/v1/moderations$", "moderations"),
]

_HANDLERS = {
    "chat_completions": lambda w, g, p, b: openai_chat_completions_create(w, **b),
    "completions": lambda w, g, p, b: openai_completions_create(w, **b),
    "responses": lambda w, g, p, b: openai_responses_create(w, **b),
    "moderations": lambda w, g, p, b: openai_moderations_create(w, **b),
}


route_chatgpt = make_router(_ROUTES, _HANDLERS)
