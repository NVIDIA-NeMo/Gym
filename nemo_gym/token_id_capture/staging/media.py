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

"""Compact-space token delta for Megatron VLM calls.

A vision-language engine runs on an *expanded* prompt (one media token per projected
embedding) but tokenizes the chat render in *compact* form (one media token per image
or video). The canonical ``token_ids_delta`` is expanded, which is what a trainer
needs. Splicing a previous turn into the next compact render needs the compact form,
so a call whose engine expanded media tokens also stages ``compact_token_ids_delta``.

It rides ``StagedCallRecord.extras`` and is therefore bound by ``extras_digest``;
text-only calls stage no extras at all. Media tensors themselves reach the sink as
``complete_call_from_response`` attachments, not as extras.
"""

from __future__ import annotations

from collections.abc import Sequence


COMPACT_TOKEN_IDS_DELTA_FIELD = "compact_token_ids_delta"


def build_compact_token_ids_delta(
    compact_prompt_token_ids: Sequence[int],
    generated_token_ids: Sequence[int],
    *,
    compact_prev_len: int,
) -> list[int]:
    """Return the compact-space delta: the new compact prompt suffix plus the generation.

    Mirrors ``build_staging_delta`` in the compact token space. ``compact_prev_len`` is
    the compact length of the parent chain the worker spliced in; the engine prompt must
    extend it.
    """
    if compact_prev_len < 0 or compact_prev_len > len(compact_prompt_token_ids):
        raise ValueError(
            f"compact_prev_len {compact_prev_len} is outside the compact prompt length {len(compact_prompt_token_ids)}"
        )
    if any(type(token_id) is not int for token_id in (*compact_prompt_token_ids, *generated_token_ids)):
        raise ValueError("compact_prompt_token_ids and generated_token_ids must contain only integer token ids")
    return list(compact_prompt_token_ids[compact_prev_len:]) + list(generated_token_ids)


__all__ = ["COMPACT_TOKEN_IDS_DELTA_FIELD", "build_compact_token_ids_delta"]
