# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Primary metrics for ChemReason-Bench, ported literally from upstream.

Derived from ``eval/eval.py`` in https://github.com/Khadaz/ChemReason-Bench at
commit ``c0b9ac2933708fcca47b1795492952cbf280e194`` (2026-05-08), which the
project licenses under Apache-2.0. Constants (``EN_STOP``, ``DEFAULT_UCUM``,
punctuation regexes, tolerances) and the metric bodies reproduce that file so
that scores are comparable to the published table by construction rather than
by coincidence.

Scope: only the six PRIMARY metrics named in ``eval/eval.py`` ``primary_map``
(line 813) are implemented here, because those are the ones the published
Primary-Overall macro-average is built from. Upstream's secondary metrics
(Kendall tau, BLEU, ROUGE-L, BERTScore, AUROC/AUPRC/ECE/Brier, log-loss, MRR)
are deliberately omitted: they do not enter the headline, and BERTScore in
particular would pull a neural model into an otherwise dependency-free scorer.

    task type             primary metric
    ------------------    ----------------------
    ordering              pairwise_accuracy
    contrastive_choice    top1_accuracy
    step_validation       f1_positive
    condition_validation  f1_positive
    step_completion       step_completion_score
    rationalization       coverage_f1

Two upstream inconsistencies were resolved deliberately:

1. ``eval/eval_config.yaml`` states the step-completion formula as
   ``0.5 * action_em + 0.5 * slot_f1``. Both ``eval/eval.py`` (line 631) and the
   paper (appendix, "Raw = 0.8*ActionEM + 0.2*SlotF1, SCS = Raw*(1-FER)") use
   0.8/0.2 with a format-error penalty. The config string is stale; the code and
   paper agree and are what produced the published numbers, so 0.8/0.2 is used.
2. ``f1_positive`` is a per-corpus quantity, not a per-row one -- it needs the
   full confusion matrix. Same for ``step_completion_score``, whose format-error
   penalty is a corpus-level rate. Rows are therefore scored into contributions
   here and reduced across the corpus by the caller.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# --------------------------------------------------------------------------
# Upstream constants, reproduced verbatim (eval/eval.py lines 237-243, 332-356)
# --------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[\.,;:!\?\(\)\[\]\{\}\-\/_`'\"\^\|]")
_WS_RE = re.compile(r"\s+")

EN_STOP = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "if",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "am",
    "to",
    "of",
    "in",
    "on",
    "for",
    "by",
    "with",
    "as",
    "at",
    "from",
    "that",
    "this",
    "these",
    "those",
    "it",
    "its",
    "their",
    "which",
    "who",
    "whom",
    "than",
    "then",
    "so",
    "such",
    "not",
    "no",
    "yes",
    "can",
    "could",
    "should",
    "would",
    "may",
    "might",
    "will",
    "shall",
    "do",
    "does",
    "did",
    "have",
    "has",
    "had",
    "we",
    "you",
    "they",
    "he",
    "she",
    "i",
    "me",
    "my",
    "your",
    "our",
    "us",
}

DEFAULT_UCUM = {
    # volume
    "μl": "uL",
    "µl": "uL",
    "ul": "uL",
    "μL": "uL",
    "µL": "uL",
    "uL": "uL",
    "ml": "mL",
    "mL": "mL",
    "l": "L",
    "L": "L",
    # mass
    "mg": "mg",
    "g": "g",
    "kg": "kg",
    "µg": "ug",
    "μg": "ug",
    "ug": "ug",
    "ng": "ng",
    # substance amount
    "mol": "mol",
    "mmol": "mmol",
    "µmol": "umol",
    "μmol": "umol",
    "umol": "umol",
    # time
    "s": "s",
    "sec": "s",
    "secs": "s",
    "second": "s",
    "seconds": "s",
    "min": "min",
    "mins": "min",
    "minute": "min",
    "minutes": "min",
    "h": "h",
    "hr": "h",
    "hour": "h",
    "hours": "h",
    # temperature
    "°c": "C",
    "c": "C",
    "°f": "F",
    "f": "F",
    "k": "K",
    "C": "C",
    "F": "F",
    "K": "K",
}

TIME_UNITS = {"s", "min", "h"}
TEMP_UNITS = {"C", "F", "K"}
_OTHER_LEGAL_UNITS = {"mg", "g", "kg", "ug", "ng", "uL", "mL", "L", "mol", "mmol", "umol"}

TASK_TYPES = (
    "ordering",
    "contrastive_choice",
    "step_validation",
    "condition_validation",
    "step_completion",
    "rationalization",
)

PRIMARY_METRIC_BY_TASK = {
    "ordering": "pairwise_accuracy",
    "contrastive_choice": "top1_accuracy",
    "step_validation": "f1_positive",
    "condition_validation": "f1_positive",
    "step_completion": "step_completion_score",
    "rationalization": "coverage_f1",
}

# Tasks whose published primary metric is the mean of the `gen` and `lm`
# protocols (paper appendix F.3.4: m_t = 0.5 * (m_gen + m_lm)). Scoring only one
# protocol and reporting it as Primary-Overall is NOT comparable to the paper.
DUAL_PROTOCOL_TASKS = frozenset({"step_validation", "condition_validation", "contrastive_choice"})


@dataclass(frozen=True)
class UnitPolicy:
    abs_tol_temp: float = 1.0
    rel_tol_amount: float = 0.05
    rel_tol_time: float = 0.10


DEFAULT_UNIT_POLICY = UnitPolicy()


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


# --------------------------------------------------------------------------
# Text normalization (eval/eval.py lines 245-278)
# --------------------------------------------------------------------------


def normalize_text(s: Optional[str], lower: bool = True, strip_punct: bool = True) -> str:
    if s is None:
        return ""
    if lower:
        s = s.lower()
    if strip_punct:
        s = _PUNCT_RE.sub(" ", s)
    return _WS_RE.sub(" ", s).strip()


def tokenize_en(s: Optional[str], stopwords: Optional[set] = None) -> List[str]:
    toks = [t for t in normalize_text(s).split() if t]
    if stopwords:
        toks = [t for t in toks if t not in stopwords]
    return toks


def token_f1(pred: Optional[str], truth: Optional[str], stopwords: set) -> float:
    """Bag-of-tokens F1 after normalization and stopword removal.

    NOTE for interpretation: this metric pays for lexical overlap alone. A
    response that echoes the prompt's chemistry vocabulary scores above zero
    without explaining anything, so `coverage_f1` must never be read as a
    correctness rate. Upstream nonetheless makes it the primary metric for
    RATIONALIZATION, so it is reproduced as-is.
    """
    pt = tokenize_en(pred, stopwords)
    gt = tokenize_en(truth, stopwords)
    if not pt and not gt:
        return 1.0
    if not pt or not gt:
        return 0.0
    pc, gc = Counter(pt), Counter(gt)
    overlap = sum((pc & gc).values())
    prec = safe_div(overlap, sum(pc.values()))
    rec = safe_div(overlap, sum(gc.values()))
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


# --------------------------------------------------------------------------
# ORDERING -- pairwise_accuracy (eval/eval.py lines 86-120)
# --------------------------------------------------------------------------


def pairwise_accuracy(pred: List[Any], truth: List[Any]) -> float:
    """Fraction of correctly ordered pairs among the model's *legal* step ids.

    Upstream is deliberately conservative: ids absent from the gold order are
    dropped rather than filled in, and fewer than two surviving ids scores 0.
    """
    truth_pos = {str(k): i for i, k in enumerate(truth)}
    pred_seq = [str(p) for p in pred if str(p) in truth_pos]
    if len(pred_seq) < 2:
        return 0.0
    n = len(pred_seq)
    total_pairs = n * (n - 1) // 2
    if total_pairs == 0:
        return 0.0
    correct = 0
    for i in range(n):
        for j in range(i + 1, n):
            if truth_pos[pred_seq[i]] < truth_pos[pred_seq[j]]:
                correct += 1
    return safe_div(correct, total_pairs)


# --------------------------------------------------------------------------
# STEP_COMPLETION -- slot F1 and the composite score (eval/eval.py 388-520, 617-640)
# --------------------------------------------------------------------------


def norm_unit(u: Any, ucum: Optional[Dict[str, str]] = None) -> Any:
    if not isinstance(u, str):
        return u
    ucum = DEFAULT_UCUM if ucum is None else ucum
    key = u.strip()
    return ucum.get(key.lower(), key)


def units_legal(unit: Optional[str]) -> bool:
    if unit is None:
        return False
    u = str(unit)
    return u in TIME_UNITS or u in TEMP_UNITS or u in _OTHER_LEGAL_UNITS


def compare_numeric_slot(
    kind: str,
    pred_val: Any,
    pred_unit: Any,
    gt_val: Any,
    gt_unit: Any,
    policy: UnitPolicy = DEFAULT_UNIT_POLICY,
) -> bool:
    if pred_unit is None or gt_unit is None or str(pred_unit) != str(gt_unit):
        return False
    try:
        pv = float(pred_val)
        gv = float(gt_val)
    except (TypeError, ValueError):
        return False
    if kind == "temperature":
        return abs(pv - gv) <= policy.abs_tol_temp
    if kind == "time":
        denom = abs(gv) if gv != 0 else 1.0
        return abs(pv - gv) <= policy.rel_tol_time * denom
    if kind == "amount":
        denom = abs(gv) if gv != 0 else 1.0
        return abs(pv - gv) <= policy.rel_tol_amount * denom
    return False


_NUMERIC_SLOT_GROUPS = (
    ("amount", "amount_value", "amount_unit"),
    ("time", "duration_value", "duration_unit"),
    ("temperature", "temperature_value", "temperature_unit"),
)
_TOKEN_SLOT_FIELDS = ("duration_token", "temperature_token")


def slot_f1(
    pred_slots: Optional[Dict[str, Any]],
    gt_slots: Optional[Dict[str, Any]],
    ucum: Optional[Dict[str, str]] = None,
    policy: UnitPolicy = DEFAULT_UNIT_POLICY,
) -> Tuple[float, bool]:
    """Returns ``(slot_f1, fatal_error_flag)``.

    Reagents match as an unordered multiset; amount/time/temperature match as
    grouped value+unit pairs under tolerance; ``*_token`` fields match exactly.
    ``fatal`` fires when the prediction carries a unit outside the legal set,
    and feeds the corpus-level format-error rate.
    """
    ucum = DEFAULT_UCUM if ucum is None else ucum
    pred = dict(pred_slots or {})
    gt = dict(gt_slots or {})

    for k in list(pred):
        if k.endswith("_unit"):
            pred[k] = norm_unit(pred[k], ucum)
    for k in list(gt):
        if k.endswith("_unit"):
            gt[k] = norm_unit(gt[k], ucum)

    fatal = any(k.endswith("_unit") and v is not None and not units_legal(v) for k, v in pred.items())

    def reagent_values(d: Dict[str, Any]) -> List[str]:
        return [v.strip() for k, v in d.items() if (k == "reagent" or k.startswith("reagent_")) and isinstance(v, str)]

    pred_reags = reagent_values(pred)
    gt_reags = reagent_values(gt)

    tp = fp = fn = 0

    if gt_reags or pred_reags:
        pc, gc = Counter(pred_reags), Counter(gt_reags)
        match = sum((pc & gc).values())
        tp += match
        fp += max(0, sum(pc.values()) - match)
        fn += max(0, sum(gc.values()) - match)

    for kind, v_key, u_key in _NUMERIC_SLOT_GROUPS:
        gt_has = v_key in gt or u_key in gt
        pr_has = v_key in pred or u_key in pred
        if not gt_has and not pr_has:
            continue
        if (
            gt_has
            and pr_has
            and compare_numeric_slot(kind, pred.get(v_key), pred.get(u_key), gt.get(v_key), gt.get(u_key), policy)
        ):
            tp += 1
        else:
            fp += int(pr_has)
            fn += int(gt_has)

    for tkey in _TOKEN_SLOT_FIELDS:
        gt_has = tkey in gt
        pr_has = tkey in pred
        if not gt_has and not pr_has:
            continue
        if gt_has and pr_has and str(pred[tkey]) == str(gt[tkey]):
            tp += 1
        else:
            fp += int(pr_has)
            fn += int(gt_has)

    prec = safe_div(tp, tp + fp)
    rec = safe_div(tp, tp + fn)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return f1, fatal


def step_completion_score(action_em: float, slot_f1_mean: float, format_error_rate: float) -> float:
    """``SCS = (0.8*ActionEM + 0.2*SlotF1) * (1 - FER)``, per paper and eval.py.

    All three inputs are corpus means; see the module docstring on why the
    config file's 0.5/0.5 string is not used.
    """
    raw = 0.8 * action_em + 0.2 * slot_f1_mean
    penalty = max(0.0, min(1.0, 1.0 - format_error_rate))
    return raw * penalty


# --------------------------------------------------------------------------
# Per-row scoring
# --------------------------------------------------------------------------


def score_row(task_type: str, prediction: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, Any]:
    """Score one prediction against one gold record.

    Returns a dict carrying both a per-row ``reward`` in [0, 1] and whatever
    corpus-level contributions that task needs (confusion-matrix cells for the
    binary tasks, action/slot/fatal terms for step completion). The per-row
    reward is a usable signal on its own, but the *published* metric for the
    binary and step-completion tasks is only defined over a corpus, so the
    caller must reduce the contributions rather than average the rewards.
    """
    if task_type == "ordering":
        pred_order = prediction.get("predicted_order") or prediction.get("order") or []
        gold_order = ground_truth.get("correct_order") or []
        acc = pairwise_accuracy(pred_order, gold_order)
        return {"reward": acc, "pairwise_accuracy": acc}

    if task_type == "contrastive_choice":
        gold_idx = ground_truth.get("correct_option_idx")
        try:
            hit = float(int(prediction.get("predicted_option_idx", -1)) == int(gold_idx))
        except (TypeError, ValueError):
            hit = 0.0
        return {"reward": hit, "top1_hit": hit}

    if task_type in ("step_validation", "condition_validation"):
        gold_label = 1 if bool(ground_truth.get("label")) else 0
        pred_label = 1 if bool(prediction.get("label")) else 0
        return {
            "reward": float(pred_label == gold_label),
            "gold_label": gold_label,
            "pred_label": pred_label,
        }

    if task_type == "step_completion":
        act_pred = str(prediction.get("action", "")).strip().upper()
        act_true = str(ground_truth.get("action", "")).strip().upper()
        action_em = float(act_pred == act_true)
        f1, fatal = slot_f1(prediction.get("slots"), ground_truth.get("slots"))
        # Per-row reward mirrors the corpus formula with this row's own fatal flag.
        return {
            "reward": step_completion_score(action_em, f1, float(fatal)),
            "action_em": action_em,
            "slot_f1": f1,
            "format_error": float(fatal),
        }

    if task_type == "rationalization":
        cov = token_f1(
            prediction.get("gold_rationale", ""),
            ground_truth.get("gold_rationale", ""),
            EN_STOP,
        )
        return {"reward": cov, "coverage_f1": cov}

    raise ValueError(f"unknown task_type: {task_type!r}")


# --------------------------------------------------------------------------
# Corpus reduction
# --------------------------------------------------------------------------


def f1_positive_from_labels(gold: List[int], pred: List[int]) -> float:
    tp = sum(1 for g, p in zip(gold, pred) if g == 1 and p == 1)
    fp = sum(1 for g, p in zip(gold, pred) if g == 0 and p == 1)
    fn = sum(1 for g, p in zip(gold, pred) if g == 1 and p == 0)
    prec = safe_div(tp, tp + fp)
    rec = safe_div(tp, tp + fn)
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def reduce_task(task_type: str, rows: List[Dict[str, Any]]) -> float:
    """Reduce per-row contributions to that task's published primary metric."""
    if not rows:
        return 0.0
    if task_type == "ordering":
        return sum(r["pairwise_accuracy"] for r in rows) / len(rows)
    if task_type == "contrastive_choice":
        return sum(r["top1_hit"] for r in rows) / len(rows)
    if task_type in ("step_validation", "condition_validation"):
        return f1_positive_from_labels([r["gold_label"] for r in rows], [r["pred_label"] for r in rows])
    if task_type == "step_completion":
        n = len(rows)
        return step_completion_score(
            sum(r["action_em"] for r in rows) / n,
            sum(r["slot_f1"] for r in rows) / n,
            sum(r["format_error"] for r in rows) / n,
        )
    if task_type == "rationalization":
        return sum(r["coverage_f1"] for r in rows) / len(rows)
    raise ValueError(f"unknown task_type: {task_type!r}")


def primary_overall(per_task: Dict[str, float]) -> float:
    """Macro-average over the six task families (eval/eval.py line 838).

    Upstream records 0.0 for a task that is present in the gold but produced no
    usable metric, so a task missing from ``per_task`` is scored 0 rather than
    silently dropped from the denominator.
    """
    values = [float(per_task.get(t, 0.0)) for t in TASK_TYPES]
    return sum(values) / len(values) if values else 0.0
