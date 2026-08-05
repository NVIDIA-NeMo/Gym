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

from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator


# The fields the model server attaches to a served response when token-id return
# is on. ``routed_experts`` is present only for MoE backends that report it.
TOKEN_FIELDS = ("prompt_token_ids", "generation_token_ids", "generation_log_probs", "routed_experts")

# Bumped whenever a field is added or its meaning changes. Writer and reader are different
# processes and may be different repositories, and records outlive a deploy, so a reader has to
# be able to refuse a record it was not built for. ``extra="allow"`` means an unknown shape
# otherwise decodes cleanly and corrupts training rows in silence. The field is present from the
# first version because a check added later cannot tell an old record from an unversioned one.
#
#   1  rollout and call identity, the token arrays, the output items and their carrier index
TOKEN_ENTRY_RECORD_SCHEMA_VERSION = 1


class TokenEntry(BaseModel):
    """One model call's captured record: the content-bearing output items (assistant
    text, tool calls) together with the token fields, keyed to its rollout and to the
    ``model_call_id`` the capture middleware minted for the call.

    ``output_items`` holds the served response's output items with their content. Token
    ids alone are not enough for a trainer that scores text, such as penalties for an
    invalid tool call or a malformed thinking block.

    The token arrays are stored once, at the top level. The served response carries
    them on the output item that produced the generation, and those per-item copies
    are dropped on write: the builder overwrites them anyway, since an item's prompt
    in a chained trajectory is the running cumulative sequence rather than the prompt
    of the single call. ``token_item_index`` records which item they came off, so the
    builder can put the chain-correct values back on the right one.
    """

    model_config = ConfigDict(extra="allow")

    schema_version: int = TOKEN_ENTRY_RECORD_SCHEMA_VERSION
    rollout_id: str
    model_call_id: str
    model: str = ""
    prompt_token_ids: list[int]
    generation_token_ids: list[int]
    generation_log_probs: list[float]
    routed_experts: Any | None = None
    # The served response's output items (Responses shape), content preserved, token
    # arrays removed.
    output_items: list[dict] = []
    # Index into ``output_items`` of the item the token arrays were taken off, or null
    # when no item carried them. Records written before the arrays were de-duplicated
    # leave this unset and still carry the arrays inline, which the builder handles.
    token_item_index: int | None = None
    # Non-semantic; a cheap diagnostic for retry/sibling-branch cases.
    created_at: float = 0.0

    @model_validator(mode="after")
    def _refuse_a_newer_record(self) -> "TokenEntry":
        """Decode a record older than this reader, refuse one newer.

        Older is safe: a field this reader does not have takes its default and the consumer
        degrades, so a record written before parent links existed simply has none and the builder
        matches token prefixes instead.

        Newer is not, and it is the direction ``extra="allow"`` hides. A field this reader cannot
        see is kept and ignored, so a record whose tokens were written under rules this reader does
        not know decodes clean and trains as though nothing were different. Refusing is loud: the
        read fails, the caller marks that rollout unusable, and the run says which version it saw.
        """
        if self.schema_version > TOKEN_ENTRY_RECORD_SCHEMA_VERSION:
            raise ValueError(
                f"token record is schema_version {self.schema_version}, but this reader understands "
                f"up to {TOKEN_ENTRY_RECORD_SCHEMA_VERSION}. Upgrade the reader, or point it at "
                "records written by a writer it matches."
            )
        return self


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


def strip_token_fields(items: list[dict]) -> tuple[list[dict], int | None]:
    """Drop the token arrays from output items, keeping the content.

    Returns the stripped items and the index of the item the arrays came off, which
    is the last one carrying them, matching what ``extract_token_fields`` reads. The
    arrays are held once on the entry instead: storing them again per item roughly
    doubles a record, and the per-item values are not the ones a trainer sees, since
    the builder replaces an item's prompt with the chain's running sequence.
    """
    index: int | None = None
    stripped: list[dict] = []
    for position, item in enumerate(items):
        if item.get("generation_token_ids") is not None:
            index = position
        stripped.append({key: value for key, value in item.items() if key not in TOKEN_FIELDS})
    return stripped, index


def extract_token_fields(response_json: dict) -> dict | None:
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
