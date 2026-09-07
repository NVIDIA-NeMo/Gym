# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Arena verdict and bootstrap utilities."""

import re

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit
from tiktoken import Encoding


_VERDICT_LABELS_A_WINS: frozenset[str] = frozenset({"[[A>>B]]", "[[A>B]]"})
_VERDICT_LABELS_TIE: frozenset[str] = frozenset({"[[A=B]]"})
_VERDICT_LABELS_B_WINS: frozenset[str] = frozenset({"[[B>A]]", "[[B>>A]]"})
_VERDICT_LABEL_BOTH_BAD = "[[BB]]"
# lmarena_v2 upweights strong verdicts; lmarena_v3 sets `verdict_weight` to 1.
_VERDICT_LABELS_STRONG: frozenset[str] = frozenset({"[[A>>B]]", "[[B>>A]]"})

_ALL_VERDICT_LABELS = _VERDICT_LABELS_A_WINS | _VERDICT_LABELS_TIE | _VERDICT_LABELS_B_WINS | {_VERDICT_LABEL_BOTH_BAD}

_THINK_PATTERN = re.compile(r"<think>.*?</think>|<thinking>.*?</thinking>", re.DOTALL)
_THINK_CONTENT_PATTERN = re.compile(r"<think>(.*?)</think>|<thinking>(.*?)</thinking>", re.DOTALL)
_THINK_OPEN_PATTERN = re.compile(r"<think>|<thinking>")


def _without_closed_thinking_blocks(text: str) -> tuple[str, re.Match | None]:
    text = _THINK_PATTERN.sub("", text)
    return text, _THINK_OPEN_PATTERN.search(text)


def _strip_thinking_blocks(text: str) -> str:
    text, unclosed = _without_closed_thinking_blocks(text)
    return text[: unclosed.start() if unclosed else None].strip()


def _extract_thinking_content(text: str) -> str:
    """Return text inside all <think>/<thinking> blocks."""
    parts = []
    for m in _THINK_CONTENT_PATTERN.finditer(text):
        content = m.group(1) if m.group(1) is not None else m.group(2)
        if content := content.strip():
            parts.append(content)
    text, unclosed = _without_closed_thinking_blocks(text)
    if unclosed and (content := text[unclosed.end() :].strip()):
        parts.append(content)
    return "\n\n".join(parts)


def _extract_verdict(text: str) -> str | None:
    """Return the rightmost verdict label in *text*, or None if none found.

    The judge states its final decision at the end of its reasoning, so taking
    the rightmost occurrence is more reliable than taking the first.
    """
    last_pos = -1
    last_label: str | None = None
    for label in _ALL_VERDICT_LABELS:
        pos = text.rfind(label)
        if pos >= 0 and pos > last_pos:
            last_pos = pos
            last_label = label
    return last_label


def _score_verdict_as_a(verdict: str | None) -> float:
    """Return [0, 0.5, 1.0] for position-A's perspective (A is the policy model)."""
    if verdict in _VERDICT_LABELS_A_WINS:
        return 1.0
    if verdict in _VERDICT_LABELS_TIE or verdict == _VERDICT_LABEL_BOTH_BAD:
        return 0.5
    return 0.0  # B wins or None


def _score_verdict_as_b(verdict: str | None) -> float:
    """Return [0, 0.5, 1.0] for position-B's perspective (B is the policy model)."""
    if verdict in _VERDICT_LABELS_B_WINS:
        return 1.0
    if verdict in _VERDICT_LABELS_TIE or verdict == _VERDICT_LABEL_BOTH_BAD:
        return 0.5
    return 0.0  # A wins or None


def _weighted_scores_as_a(verdict: str | None, weight: int) -> list[float]:
    """Score position A, repeating strong verdicts `weight` times."""
    score = _score_verdict_as_a(verdict)
    return [score] * (weight if verdict in _VERDICT_LABELS_STRONG else 1)


def _weighted_scores_as_b(verdict: str | None, weight: int) -> list[float]:
    """Return a list of scores for position-B, repeating `weight` times for strong verdicts."""
    score = _score_verdict_as_b(verdict)
    return [score] * (weight if verdict in _VERDICT_LABELS_STRONG else 1)


# v2 uses a Bradley-Terry correction over relative length, header, list, and bold
# features.
# v3 uses reference-length style control instead.

# Regex patterns matching arena-hard-auto/utils/add_markdown_info.py exactly.
_CODE_BLOCK_RE = re.compile(r"```[^`]*```", re.DOTALL)
_HEADER_RES = [re.compile(rf"^#{{{n}}}\s", re.MULTILINE) for n in range(1, 7)]
_ORDERED_LIST_RE = re.compile(r"^\s*\d+\.\s", re.MULTILINE)
_UNORDERED_LIST_RE = re.compile(r"^\s*[-*+]\s", re.MULTILINE)
_BOLD_STAR_RE = re.compile(r"\*\*[^*\n]+\*\*")
_BOLD_UNDER_RE = re.compile(r"__[^_\n]+__")


def _extract_style_counts(text: str, encoding: Encoding) -> np.ndarray:
    """Count tokens, headers, lists, and bold spans for v2 style control."""
    token_len = len(encoding.encode(text, disallowed_special=()))
    stripped = _CODE_BLOCK_RE.sub("", text)
    header_count = sum(len(r.findall(stripped)) for r in _HEADER_RES)
    list_count = len(_ORDERED_LIST_RE.findall(stripped)) + len(_UNORDERED_LIST_RE.findall(stripped))
    bold_count = len(_BOLD_STAR_RE.findall(stripped)) + len(_BOLD_UNDER_RE.findall(stripped))
    return np.array((token_len, header_count, list_count, bold_count), dtype=float)


def _compute_raw_style_feature(policy_text: str, baseline_text: str, encoding: Encoding) -> np.ndarray:
    """Compute the 4-element raw (un-normalised) style feature for one judgment."""
    model = _extract_style_counts(policy_text, encoding)
    baseline = _extract_style_counts(baseline_text, encoding)
    feature = np.zeros(4)
    total_tokens = model[0] + baseline[0]
    feature[0] = (model[0] - baseline[0]) / total_tokens if total_tokens else 0.0
    model_density = model[1:] / (model[0] + 1.0)
    baseline_density = baseline[1:] / (baseline[0] + 1.0)
    feature[1:] = (model_density - baseline_density) / (model_density + baseline_density + 1.0)
    return feature


def _bt_neg_ll_from_logits(logits: np.ndarray, y: np.ndarray) -> float:
    """Return Bradley-Terry binary cross-entropy."""
    p = np.clip(expit(logits), 1e-12, 1.0 - 1e-12)
    return float(-np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _fit_bt_with_offset(offsets: np.ndarray, scores: np.ndarray) -> float:
    """Fit one Bradley-Terry quality parameter with fixed style offsets."""
    return float(
        minimize_scalar(
            lambda theta: _bt_neg_ll_from_logits(theta + offsets, scores),
            bounds=(-15.0, 15.0),
            method="bounded",
        ).x
    )


def _bootstrap(
    score_groups: list[np.ndarray],
    offset_groups: list[np.ndarray] | None = None,
    n_rounds: int = 100,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap individual judge games."""
    scores = {"all": score_groups}
    offsets = None if offset_groups is None else {"all": offset_groups}
    return _bootstrap_per_category(scores, offsets, n_rounds, seed)


def _bootstrap_per_category(
    cat_scores: dict[str, list[np.ndarray]],
    cat_offsets: dict[str, list[np.ndarray]] | None = None,
    n_rounds: int = 100,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap individual judge games within each category."""
    categories = sorted(cat_scores.keys())
    flat_scores = {category: np.concatenate(cat_scores[category]) for category in categories}
    flat_offsets = (
        None if cat_offsets is None else {category: np.concatenate(cat_offsets[category]) for category in categories}
    )
    rng = np.random.default_rng(seed)
    results = np.zeros(n_rounds)
    for i in range(n_rounds):
        estimates = []
        for category in categories:
            scores = flat_scores[category]
            indices = rng.integers(0, len(scores), size=len(scores))
            if flat_offsets is None:
                estimates.append(float(scores[indices].mean()))
            else:
                theta = _fit_bt_with_offset(flat_offsets[category][indices], scores[indices])
                estimates.append(float(expit(theta)))
        results[i] = float(np.mean(estimates))
    pt_est = float(results.mean())
    ci_lower = min(float(np.percentile(results, 2.5)), pt_est)
    ci_upper = max(float(np.percentile(results, 97.5)), pt_est)
    return pt_est, ci_lower, ci_upper
