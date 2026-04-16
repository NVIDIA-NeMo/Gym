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
"""Arena evaluation methodology: verdict scoring, style feature extraction,
Bradley-Terry model fitting, and bootstrap confidence intervals.

No nemo_gym imports — this module can be imported standalone (e.g. by precompute scripts).
"""

import re
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit


try:
    import tiktoken as _tiktoken

    _TIKTOKEN_ENC = _tiktoken.encoding_for_model("gpt-4o")
except ImportError:
    raise ImportError("tiktoken is required to compute style features")


# ── Verdict labels ────────────────────────────────────────────────────────────

# All possible verdict labels the judge can output.
_VERDICT_LABELS_A_WINS: frozenset[str] = frozenset({"[[A>>B]]", "[[A>B]]"})
_VERDICT_LABELS_TIE: frozenset[str] = frozenset({"[[A=B]]"})
_VERDICT_LABELS_B_WINS: frozenset[str] = frozenset({"[[B>A]]", "[[B>>A]]"})
_VERDICT_LABEL_BOTH_BAD = "[[BB]]"
# Strong verdicts — counted `verdict_weight` times in the reward average.
_VERDICT_LABELS_STRONG: frozenset[str] = frozenset({"[[A>>B]]", "[[B>>A]]"})

_ALL_VERDICT_LABELS = _VERDICT_LABELS_A_WINS | _VERDICT_LABELS_TIE | _VERDICT_LABELS_B_WINS | {_VERDICT_LABEL_BOTH_BAD}

# Default weight for strong verdicts (>>) in the reward average.
_DEFAULT_VERDICT_WEIGHT: int = 3

# Mapping from arena-hard-auto judgment file labels (no brackets) to the bracketed verdict
# labels used by the judge prompt.  Used by precompute_style_constants.py to convert
# pre-computed reference judgments into the same scoring path as live evaluation.
#
# In arena-hard-auto games[1] the evaluated model plays as A; in games[0] it plays as B.
# Both games use this same mapping — pass the result to _weighted_scores_as_a (games[1])
# or _weighted_scores_as_b (games[0]) respectively.
_AH_AUTO_LABEL_TO_VERDICT: dict[str, str] = {
    "A>>B": "[[A>>B]]",
    "A>B": "[[A>B]]",
    "A=B": "[[A=B]]",
    "B=A": "[[A=B]]",
    "A<B": "[[B>A]]",
    "B>A": "[[B>A]]",
    "A<<B": "[[B>>A]]",
    "B>>A": "[[B>>A]]",
    "B<A": "[[A>B]]",
    "B<<A": "[[A>>B]]",
    "BB": "[[BB]]",
}

# Strip / extract <think>/<thinking> blocks emitted by reasoning models.
_THINK_PATTERN = re.compile(r"<think>.*?</think>|<thinking>.*?</thinking>", re.DOTALL)
_THINK_CONTENT_PATTERN = re.compile(r"<think>(.*?)</think>|<thinking>(.*?)</thinking>", re.DOTALL)


def _strip_thinking_blocks(text: str) -> str:
    return _THINK_PATTERN.sub("", text).strip()


def _extract_thinking_content(text: str) -> str:
    """Return the concatenated content inside all <think>/<thinking> blocks (tags stripped).

    Used to preserve reasoning text in rollouts for debugging.  The returned string is
    never passed to the judge — only the stripped answer is.
    """
    parts = []
    for m in _THINK_CONTENT_PATTERN.finditer(text):
        content = m.group(1) if m.group(1) is not None else m.group(2)
        if content := content.strip():
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
    if verdict in _VERDICT_LABELS_TIE:
        return 0.5
    return 0.0  # B wins, both bad, or None


def _score_verdict_as_b(verdict: str | None) -> float:
    """Return [0, 0.5, 1.0] for position-B's perspective (B is the policy model)."""
    if verdict in _VERDICT_LABELS_B_WINS:
        return 1.0
    if verdict in _VERDICT_LABELS_TIE:
        return 0.5
    return 0.0  # A wins, both bad, or None


def _weighted_scores_as_a(verdict: str | None, weight: int) -> list[float]:
    """Return a list of scores for position-A, repeating `weight` times for strong verdicts.

    Strong wins/losses (>>) contribute `weight` items to the flattened average,
    giving them more influence on the final score.
    A None verdict (parse failure) is treated as a weak loss — it contributes a single 0.0,
    not repeated `weight` times.
    """
    score = _score_verdict_as_a(verdict)
    return [score] * (weight if verdict in _VERDICT_LABELS_STRONG else 1)


def _weighted_scores_as_b(verdict: str | None, weight: int) -> list[float]:
    """Return a list of scores for position-B, repeating `weight` times for strong verdicts."""
    score = _score_verdict_as_b(verdict)
    return [score] * (weight if verdict in _VERDICT_LABELS_STRONG else 1)


# ── Style control ─────────────────────────────────────────────────────────────
#
# Style constants (norm_mean, norm_std, coefs) are pre-computed from a multi-model
# dataset using scripts/precompute_style_constants.py.
#
# Feature dimensions (4):
#   0 – token length differential : (model_len - base_len) / (model_len + base_len)
#   1 – header density differential
#   2 – list density differential
#   3 – bold density differential
#
# Density differential for dimension d:
#   model_density = count_d / (token_len + 1)
#   diff          = (model_density - base_density) / (model_density + base_density + 1)
#
# Regression: Bradley-Terry model fit on (model-one-hot | z-scored style features).
# Style coefficients capture how much each style axis shifts the win probability.

# Regex patterns matching arena-hard-auto/utils/add_markdown_info.py exactly.
_CODE_BLOCK_RE = re.compile(r"```[^`]*```", re.DOTALL)
_HEADER_RES = [re.compile(rf"^#{{{n}}}\s", re.MULTILINE) for n in range(1, 7)]
_ORDERED_LIST_RE = re.compile(r"^\s*\d+\.\s", re.MULTILINE)
_UNORDERED_LIST_RE = re.compile(r"^\s*[-*+]\s", re.MULTILINE)
_BOLD_STAR_RE = re.compile(r"\*\*[^*\n]+\*\*")
_BOLD_UNDER_RE = re.compile(r"__[^_\n]+__")


def _extract_style_metadata(text: str) -> dict[str, Any]:
    """Extract 4-element style metadata.

    Token count uses tiktoken. Code blocks are stripped before counting markdown elements
    so that code inside fences is not counted.
    """
    token_len = len(_TIKTOKEN_ENC.encode(text, disallowed_special=()))

    stripped = _CODE_BLOCK_RE.sub("", text)
    header_count = sum(len(r.findall(stripped)) for r in _HEADER_RES)
    list_count = len(_ORDERED_LIST_RE.findall(stripped)) + len(_UNORDERED_LIST_RE.findall(stripped))
    bold_count = len(_BOLD_STAR_RE.findall(stripped)) + len(_BOLD_UNDER_RE.findall(stripped))

    return {"token_len": token_len, "header_count": header_count, "list_count": list_count, "bold_count": bold_count}


def _raw_style_feature_from_counts(
    token_len_m: int,
    header_m: int,
    list_m: int,
    bold_m: int,
    token_len_b: int,
    header_b: int,
    list_b: int,
    bold_b: int,
) -> np.ndarray:
    """Compute the 4-element raw (un-normalised) style feature from pre-computed counts."""
    feat = np.zeros(4)
    total = token_len_m + token_len_b
    feat[0] = (token_len_m - token_len_b) / total if total > 0 else 0.0
    for dim, (mc, bc) in enumerate(((header_m, header_b), (list_m, list_b), (bold_m, bold_b)), start=1):
        m_dens = mc / (token_len_m + 1.0)
        b_dens = bc / (token_len_b + 1.0)
        feat[dim] = (m_dens - b_dens) / (m_dens + b_dens + 1.0)
    return feat


def _compute_raw_style_feature(policy_text: str, baseline_text: str) -> np.ndarray:
    """Compute the 4-element raw (un-normalised) style feature for one judgment."""
    pm = _extract_style_metadata(policy_text)
    bm = _extract_style_metadata(baseline_text)
    return _raw_style_feature_from_counts(
        pm["token_len"],
        pm["header_count"],
        pm["list_count"],
        pm["bold_count"],
        bm["token_len"],
        bm["header_count"],
        bm["list_count"],
        bm["bold_count"],
    )


def _bt_neg_ll_from_logits(logits: np.ndarray, y: np.ndarray) -> float:
    """Binary cross-entropy neg log-likelihood for a Bradley-Terry model.

    Args:
        logits: Pre-computed log-odds (theta + style_offset, or X @ beta).
        y:      Binary outcome labels in [0, 1].
    """
    p = np.clip(expit(logits), 1e-12, 1.0 - 1e-12)
    return float(-np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _bt_neg_ll_grad(X: np.ndarray, beta: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Gradient of _bt_neg_ll_from_logits w.r.t. beta when logits = X @ beta.

    Used by precompute_style_constants.py for BFGS joint fitting over the full
    (n_models + 4) parameter vector.  Not needed by the server's 1-D scalar fit.
    """
    p = expit(X @ beta)
    return X.T @ (p - y)


def _fit_bt_with_offset(offsets: np.ndarray, scores: np.ndarray) -> float:
    """Fit a single BT quality parameter θ with pre-computed style offsets fixed.

    Minimises −Σ [y log σ(θ + o) + (1−y) log(1 − σ(θ + o))].
    Returns θ (convert to win rate via expit(θ)).
    """
    return float(
        minimize_scalar(
            lambda theta: _bt_neg_ll_from_logits(theta + offsets, scores),
            bounds=(-15.0, 15.0),
            method="bounded",
        ).x
    )


def _bootstrap(
    scores: np.ndarray,
    offsets: np.ndarray | None = None,
    n_rounds: int = 100,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Run bootstrap resampling; return (mean, p2.5, p97.5).

    If *offsets* is provided, fits a style-controlled BT model per sample and
    converts the quality coefficient to a win-rate probability.
    Otherwise computes the boostrap mean of scores.
    """
    rng = np.random.default_rng(seed)
    N = len(scores)
    results = np.zeros(n_rounds)
    for i in range(n_rounds):
        idx = rng.integers(0, N, size=N)
        if offsets is not None:
            theta = _fit_bt_with_offset(offsets[idx], scores[idx])
            results[i] = float(expit(theta))
        else:
            results[i] = float(scores[idx].mean())
    pt_est = float(np.mean(results))
    # Clamp so that ci_lower <= pt_est <= ci_upper always holds despite floating-point rounding.
    ci_lower = min(float(np.percentile(results, 2.5)), pt_est)
    ci_upper = max(float(np.percentile(results, 97.5)), pt_est)
    return pt_est, ci_lower, ci_upper


def _bootstrap_per_category(
    cat_scores: dict[str, np.ndarray],
    cat_offsets: dict[str, np.ndarray] | None = None,
    n_rounds: int = 100,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap per-category win rates; return (mean, p2.5, p97.5) of their unweighted average.

    For a single category this is equivalent to ``_bootstrap()``.
    For multi-category datasets, this gives equal weight to each category
    regardless of question count.
    """
    rng = np.random.default_rng(seed)
    categories = sorted(cat_scores.keys())
    results = np.zeros(n_rounds)
    for i in range(n_rounds):
        cat_wrs: list[float] = []
        for cat in categories:
            s = cat_scores[cat]
            N = len(s)
            idx = rng.integers(0, N, size=N)
            if cat_offsets is not None:
                theta = _fit_bt_with_offset(cat_offsets[cat][idx], s[idx])
                cat_wrs.append(float(expit(theta)))
            else:
                cat_wrs.append(float(s[idx].mean()))
        results[i] = float(np.mean(cat_wrs))
    pt_est = float(np.mean(results))
    ci_lower = min(float(np.percentile(results, 2.5)), pt_est)
    ci_upper = max(float(np.percentile(results, 97.5)), pt_est)
    return pt_est, ci_lower, ci_upper
