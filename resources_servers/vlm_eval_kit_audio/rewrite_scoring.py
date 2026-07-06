"""Scoped native-Gym reimplementation of OCRBench_v2 scoring.

Self-contained port (stdlib only) of the exact code paths in mcore's
``vlmeval/dataset/utils/ocrbrnch_v2_eval.py`` + ``Ocrbench_v2/vqa_metric.py``
that our 15-row subsample exercises.

Coverage
--------
Categories (``verifier_extra["category"]``):
    - "APP agent en"  (the ONLY category present in
      resources_servers/vlm_eval_kit_audio/data/ocrbenchv2_validation.jsonl)

Eval modes for that category:
    - "without eval" / absent  -> ``vqa_evaluation`` (substring match for
      answers under 5 words, else ANLS with 0.5 threshold)
    - "multiple choice"        -> alpha-only exact match (trivial, ported
      because it shares the same engine branch)

Anything else (any other category, or eval == "case sensitive") raises
``NotImplementedError``.
"""

import ast

# Categories that share the "APP agent en" branch in mcore's
# process_predictions (line 50). We only claim support for the ones in our
# subsample; the rest of that branch would need vqa_evaluation_case_sensitive.
_SUPPORTED_CATEGORIES = {"APP agent en"}


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Verbatim port of Ocrbench_v2/vqa_metric.py::levenshtein_distance."""
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def _vqa_evaluation(predict, answers):
    """Verbatim port of Ocrbench_v2/vqa_metric.py::vqa_evaluation.

    Note the asymmetry faithfully preserved from upstream: in the list
    branch, short answers (<5 words) get substring-only matching (no ANLS
    fallback), while the non-list branch falls back to ANLS. Our caller
    always wraps answers into a list (as mcore's OCRBench_v2.evaluate does),
    so only the list branch is live.
    """
    score = 0
    if isinstance(answers, list):
        for j in range(len(answers)):
            if isinstance(answers[j], (int, float)):
                answers[j] = str(answers[j])
            answer = answers[j].lower().strip().replace("\n", " ")
            if isinstance(predict, (int, float)):
                predict = str(predict)
            predict = predict.lower().strip().replace("\n", " ")
            if len(answer.split()) < 5:
                if answer in predict:
                    score = 1
            else:
                dist = _levenshtein_distance(predict, answer)
                length = max(len(predict), len(answer))
                anls = 0.0 if length == 0 else float(dist) / float(length)
                anls = 1 - anls
                if anls >= 0.5 and anls > score:
                    score = anls
    else:
        answers = answers.lower().strip().replace("\n", " ")
        predict = predict.lower().strip().replace("\n", " ")
        if len(answers.split()) < 5:
            if answers in predict:
                score = 1
            else:
                dist = _levenshtein_distance(predict, answers)
                length = max(len(predict), len(answers))
                anls = 0.0 if length == 0 else float(dist) / float(length)
                anls = 1 - anls
                if anls >= 0.5 and anls > score:
                    score = anls
    return score


def _maybe_literal(value, sentinel):
    """Mirror of the field decoding in mcore's OCRBench_v2.evaluate."""
    if isinstance(value, str) and value != sentinel:
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    return value


def score_ocrbench_v2_rewrite(verifier_extra: dict, prediction: str) -> float:
    """Score one OCRBench_v2 sample. Scoped rewrite of the mcore engine.

    Covers ONLY category "APP agent en" with eval "without eval"/absent
    (vqa_evaluation path) or "multiple choice". All other categories and
    eval modes raise NotImplementedError.
    """
    category = verifier_extra.get("category")
    if category not in _SUPPORTED_CATEGORIES:
        raise NotImplementedError(f"category {category!r} not covered by this scoped rewrite")

    answers = _maybe_literal(verifier_extra.get("answer"), None)
    if not isinstance(answers, list):
        answers = [answers]
    predict = prediction if isinstance(prediction, str) else str(prediction)

    eval_field = verifier_extra.get("eval", "without eval")
    if not eval_field or eval_field == "without eval":
        return float(_vqa_evaluation(predict, answers))
    if eval_field == "multiple choice":
        # Port of ocrbrnch_v2_eval.py lines 55-68.
        assert len(answers) == 1
        if not isinstance(prediction, str):
            return 0.0
        pred_alpha = "".join(c for c in prediction if c.isalpha())
        return 1.0 if pred_alpha == answers[0] else 0.0
    raise NotImplementedError(f"eval mode {eval_field!r} not covered by this scoped rewrite")
