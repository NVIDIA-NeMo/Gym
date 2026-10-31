# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Layer-0 readouts of a residual stream, shared by the monitor's frozen reader and the vLLM plugin.

Imports only torch, so the plugin can load it inside a vLLM process that has no Gym installed.
"""

from collections import Counter
from typing import Any, Callable

import torch


# The first positions are attention sinks with atypical activations. The Jacobian lens fit skips them too.
SKIP_FIRST_POSITIONS = 16


def concept_token_ids(tokenizer: Any, concepts: list[str]) -> dict[int, str]:
    """Each concept's single-token forms (casing, leading space) by token id."""
    by_token: dict[int, str] = {}
    for concept in concepts:
        for form in (concept, f" {concept}", concept.capitalize(), f" {concept.capitalize()}"):
            ids = tokenizer.encode(form, add_special_tokens=False)
            if len(ids) == 1 and ids[0] != tokenizer.unk_token_id:
                by_token[ids[0]] = concept
    unwatched = sorted(set(concepts) - set(by_token.values()))
    if len(unwatched) == len(concepts):
        raise ValueError(f"None of the concepts is a single token: {unwatched}")
    if unwatched:
        print(f"[misalignment_monitor] concepts with no single-token form are not watched: {unwatched}", flush=True)
    return by_token


def probe_score(residual: torch.Tensor, probe: dict[str, torch.Tensor], temperature: float) -> float:
    """``sigmoid((w . (h - mu) / sd + b) / T)`` for one position's residual ``h``."""
    features = (residual.float() - probe["mu"]) / probe["sd"]
    logit = (features * probe["weight"].flatten()).sum() + probe["bias"].flatten()[0]
    return float(torch.sigmoid(logit / temperature))


def jspace_hits(
    residual: torch.Tensor,
    lens_logits: Callable[[torch.Tensor], torch.Tensor],
    concept_ids: torch.Tensor,
    *,
    top_k: int,
    first_position: int = 0,
    chunk: int = 256,
) -> list[tuple[int, int, int]]:
    """``(position, token_id, rank)`` for each watched token in a position's lens top-k.

    ``residual`` is ``[positions, d_model]`` and ``lens_logits`` maps a chunk of it to vocabulary logits.
    Positions are numbered from ``first_position``, and those before ``SKIP_FIRST_POSITIONS`` are not read.
    """
    hits = []
    for start in range(max(0, SKIP_FIRST_POSITIONS - first_position), residual.shape[0], chunk):
        top = lens_logits(residual[start : start + chunk]).topk(top_k).indices
        for row, rank in torch.isin(top, concept_ids).nonzero().tolist():
            hits.append((first_position + start + row, int(top[row, rank]), rank))
    return hits


def jspace_reading(
    hits: list[tuple[int, int, int]], concept_by_token: dict[int, str], context: Callable[[int], str]
) -> dict[str, Any]:
    """The layer-0 report for J-space hits. ``context`` gives the text just before a position."""
    examples = [
        {"concept": concept_by_token[token], "rank": rank, "position": position, "context": context(position)}
        for position, token, rank in hits[:10]
    ]
    return {
        "mode": "jspace",
        "concepts": dict(Counter(concept_by_token[token] for _, token, _ in hits)),
        "examples": examples,
        "flagged": bool(hits),
    }
