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

"""The training-token record and how to pull it off a served response.

A ``TokenEntry`` holds only what a trainer needs from one model call: the exact
prompt token ids the engine ran on, the generated token ids, and one log
probability per generated token. It is deliberately separate from the model-call
capture record used for evaluation (``ModelCallRecord``): the eval record is a
compact request/response summary and never carries token ids, while a
``TokenEntry`` is large and read only when building training data. Keeping them
apart lets eval reads skip the token payloads and lets training token ids move
to a different store later without touching the eval schema.

Both records for the same model call share a ``model_call_id``, so training can
join a ``TokenEntry`` to its ``ModelCallRecord`` when it needs the eval context.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict


# The fields the model server attaches to a served response when token-id return
# is on. ``routed_experts`` is present only for MoE backends that report it.
TOKEN_FIELDS = ("prompt_token_ids", "generation_token_ids", "generation_log_probs", "routed_experts")


class TokenEntry(BaseModel):
    """One model call's captured record: the content-bearing output items (assistant
    text, tool calls) together with the token fields, keyed to its rollout and to the
    ``model_call_id`` the capture middleware minted for the call.

    ``output_items`` holds the served response's output items with their content, so a
    trainer can read the text (e.g. NeMo-RL's invalid-tool-call / malformed-thinking
    penalties) — token ids alone are not sufficient. The top-level token arrays are the
    same fields carried on the generated item, kept here for the builder's chaining.
    """

    model_config = ConfigDict(extra="allow")

    rollout_id: str
    model_call_id: str
    model: str = ""
    prompt_token_ids: list[int]
    generation_token_ids: list[int]
    generation_log_probs: list[float]
    routed_experts: Optional[Any] = None
    # The served response's output items (Responses shape), content preserved.
    output_items: list[dict] = []
    # Non-semantic; a cheap diagnostic for retry/sibling-branch cases.
    created_at: float = 0.0


def response_to_output_items(payload: dict) -> list[dict]:
    """Normalize a served response to a list of content-bearing Responses output items.

    Responses payloads already carry ``output``. Chat payloads carry
    ``choices[*].message``; the assistant message is wrapped as a single Responses
    ``message`` item so the training record is dialect-uniform.
    """
    output = payload.get("output")
    if isinstance(output, list) and output:
        return [item for item in output if isinstance(item, dict)]
    items: list[dict] = []
    for choice in payload.get("choices") or []:
        message = (choice or {}).get("message") or {}
        if not isinstance(message, dict):
            continue
        item = dict(message)
        item.setdefault("type", "message")
        item.setdefault("role", "assistant")
        items.append(item)
    return items


def extract_token_fields(response_json: dict) -> Optional[dict]:
    """Pull the token-id fields off a served response, or ``None`` if absent.

    Handles both shapes a Gym model server can return: a Responses-style
    ``output`` list (the fields ride the last output item that carries them) and
    a chat-completions ``choices[*].message``. Returns ``None`` when no item
    carries token ids (e.g. token-id return is off, or an empty completion).
    """
    candidates: list[dict] = []
    for item in response_json.get("output") or []:
        if isinstance(item, dict) and item.get("generation_token_ids") is not None:
            candidates.append(item)
    for choice in response_json.get("choices") or []:
        message = (choice or {}).get("message") or {}
        if isinstance(message, dict) and message.get("generation_token_ids") is not None:
            candidates.append(message)
    if not candidates:
        return None
    source = candidates[-1]
    return {field: source.get(field) for field in TOKEN_FIELDS}
