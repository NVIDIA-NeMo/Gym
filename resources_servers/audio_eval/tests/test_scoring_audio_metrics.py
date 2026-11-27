# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Aggregation behavior of AudioMetrics and the BaseMetrics pass@k / majority@k machinery."""

import math

import pytest

from resources_servers.audio_eval._scoring.evaluation.metrics.audio_metrics import AudioMetrics, compute_corpus_bleu
from resources_servers.audio_eval._scoring.evaluation.metrics.base import BaseMetrics


def wer_row(errors, ref_words, subs=0, ins=0, dels=0, **extra):
    return {
        "wer": errors / ref_words,
        "wer_errors": errors,
        "wer_ref_words": ref_words,
        "wer_substitutions": subs,
        "wer_insertions": ins,
        "wer_deletions": dels,
        "is_correct": errors / ref_words < 0.5,
        **extra,
    }


def metrics_of(rows_per_example, **kwargs):
    metrics = AudioMetrics(**kwargs)
    for rows in rows_per_example:
        metrics.update(rows)
    return metrics.get_metrics()


# ----------------------------------------------------------------------------- BLEU


def test_compute_corpus_bleu_single_and_mixed_tokenizers():
    pytest.importorskip("sacrebleu")
    from sacrebleu import corpus_bleu

    hyps = ["the quick brown fox jumps", "hello there my friend"]
    refs = ["the quick brown fox jumps", "hello there my friend"]
    assert compute_corpus_bleu(hyps, refs, ["13a", "13a"]) == pytest.approx(100.0)

    mixed = compute_corpus_bleu(
        ["the quick brown fox jumps", "你好世界"],
        ["the quick brown fox jumps", "你好世界"],
        ["13a", "zh"],
    )
    assert mixed == pytest.approx(100.0)
    imperfect_h = ["the quick brown fox jumps", "你好世界朋友"]
    imperfect_r = ["the quick brown fox jumps", "你好世界"]
    expected = (
        corpus_bleu(imperfect_h[:1], [imperfect_r[:1]], tokenize="13a").score
        + corpus_bleu(imperfect_h[1:], [imperfect_r[1:]], tokenize="zh").score
    ) / 2
    assert compute_corpus_bleu(imperfect_h, imperfect_r, ["13a", "zh"]) == pytest.approx(expected)


# ----------------------------------------------------------------------------- judge parsing


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Explanation... Rating: 4", (True, 4.0)),
        ("rating: 2.5", (False, 2.5)),
        ("Rating: 9", (True, 5.0)),  # clamped to 5
        ("Judgement: Yes", (True, 5.0)),
        ("judgement: no", (False, 0.0)),
        ("I think yes, it matches", (True, 5.0)),
        ("clearly no match", (False, 0.0)),
        ("unclear", (False, 0.0)),
    ],
)
def test_extract_judge_result(text, expected):
    assert AudioMetrics()._extract_judge_result(text) == expected


def test_get_score_dict_selects_judge_or_automatic_correctness():
    m = AudioMetrics()
    assert m._get_score_dict({"category": "open", "judgement": "Rating: 4", "is_correct": False}) == {
        "judge_correct": True,
        "judge_rating": 4.0,
        "correct": True,
    }
    # Closed-form rows ignore a judgement and use the evaluator's is_correct.
    assert m._get_score_dict({"judgement": "Rating: 5", "is_correct": False}) == {"correct": False}
    assert m._get_score_dict({"is_correct": True}) == {"correct": True}
    assert m._get_score_dict({"is_correct": None}) == {"correct": False}
    assert m._get_score_dict({}) == {"correct": False}
    # Open rows without a judgement fall back to is_correct.
    assert m._get_score_dict({"category": "open", "is_correct": True}) == {"correct": True}


# ----------------------------------------------------------------------------- WER aggregation


def test_corpus_wer_is_micro_and_macro_is_mean_of_rows():
    out = metrics_of(
        [
            [wer_row(1, 4, subs=1, subset_for_metrics="a")],
            [wer_row(3, 6, subs=1, ins=1, dels=1, subset_for_metrics="b")],
            [wer_row(0, 10, subset_for_metrics="a")],
        ]
    )["pass@1"]
    assert out["num_entries"] == 3
    assert out["wer"] == 20.0  # 4 errors / 20 reference words
    assert out["ref_words"] == 20
    assert (out["substitutions"], out["insertions"], out["deletions"]) == (2, 1, 1)
    assert out["wer_macro"] == round(100 * (0.25 + 0.5 + 0.0) / 3, 2)
    assert out["wer::a"] == round(100 * 1 / 14, 2)
    assert out["wer::a_errors"] == 1 and out["wer::a_ref_words"] == 14 and out["wer::a_num_entries"] == 2
    assert out["wer::a_substitutions"] == 1
    assert out["wer::b"] == 50.0
    assert (out["wer::b_substitutions"], out["wer::b_insertions"], out["wer::b_deletions"]) == (1, 1, 1)
    # success_rate/correct come from is_correct (WER < 0.5 -> all three rows correct)
    assert out["correct"] == pytest.approx(100 * 2 / 3)  # the 3-of-6 row is not below the 0.5 threshold
    assert out["success_rate"] == out["correct"]


def test_wer_none_rows_are_not_in_macro_and_zero_ref_words_are_skipped():
    out = metrics_of(
        [
            [
                {
                    "wer": None,
                    "is_correct": None,
                    "wer_errors": 0,
                    "wer_ref_words": 0,
                    "wer_substitutions": 0,
                    "wer_insertions": 0,
                    "wer_deletions": 0,
                    "subset_for_metrics": "z",
                }
            ]
        ]
    )["pass@1"]
    assert "wer" not in out and "wer_macro" not in out and "wer::z" not in out
    assert out["correct"] == 0.0


def test_cpwer_aggregation_with_subsets():
    row = {
        "cpwer": 0.1,
        "cpwer_errors": 1,
        "cpwer_substitutions": 1,
        "cpwer_insertions": 0,
        "cpwer_deletions": 0,
        "cpwer_ref_words": 10,
        "is_correct": False,
    }
    out = metrics_of(
        [
            [{**row, "subset_for_metrics": "s1"}],
            [
                {
                    **row,
                    "cpwer": 0.3,
                    "cpwer_errors": 3,
                    "cpwer_ref_words": 10,
                    "cpwer_deletions": 2,
                    "cpwer_substitutions": 1,
                    "subset_for_metrics": "s2",
                }
            ],
            [{**row, "cpwer": None}],
        ]
    )["pass@1"]
    assert out["cpwer"] == round(100 * 5 / 30, 2)
    assert out["cpwer_ref_words"] == 30 and out["cpwer_errors"] == 5
    assert out["cpwer_deletions"] == 2 and out["cpwer_substitutions"] == 3
    assert out["cpwer_macro"] == 20.0
    assert out["cpwer::s1"] == 10.0 and out["cpwer::s2"] == 30.0
    assert out["cpwer::s2_deletions"] == 2 and out["cpwer::s2_num_entries"] == 1
    assert out["cpwer::s1_ref_words"] == 10 and out["cpwer::s1_errors"] == 1


def test_structured_wer_streams_and_subsets():
    def row(name, errors, ref_words, subset=None, **extra):
        d = {
            name: errors / ref_words,
            f"{name}_errors": errors,
            f"{name}_ref_words": ref_words,
            f"{name}_substitutions": errors,
            "is_correct": True,
        }
        if subset:
            d["subset_for_metrics"] = subset
        d.update(extra)
        return d

    out = metrics_of(
        [
            [row("strict_wer", 1, 10, "x")],
            [row("strict_wer", 3, 10, "x")],
            [row("canonical_wer", 2, 8)],
            [{"is_correct": True, "canonical_wer": None, "content_wer": 0.25}],
        ]
    )["pass@1"]
    assert out["strict_wer"] == 20.0 and out["strict_wer_errors"] == 4 and out["strict_wer_ref_words"] == 20
    assert out["strict_wer_substitutions"] == 4 and out["strict_wer_insertions"] == 0
    assert out["strict_wer::x"] == 20.0 and out["strict_wer::x_num_entries"] == 2
    assert out["strict_wer::x_deletions"] == 0
    assert out["strict_wer_macro"] == 20.0
    assert out["canonical_wer"] == 25.0 and out["canonical_wer_macro"] == 25.0
    assert "canonical_wer::x" not in out
    assert out["content_wer_macro"] == 25.0
    assert "content_wer" not in out


def test_asr_pc_totals_per_and_macro_fallbacks():
    full = {
        "wer_c": 0.5,
        "wer_c_errors": 2,
        "wer_c_ref_words": 4,
        "wer_c_substitutions": 2,
        "wer_c_insertions": 0,
        "wer_c_deletions": 0,
        "wer_pc": 0.75,
        "wer_pc_errors": 3,
        "wer_pc_ref_words": 4,
        "wer_pc_substitutions": 1,
        "wer_pc_insertions": 1,
        "wer_pc_deletions": 1,
        "per": 0.5,
        "per_correct": 1,
        "per_substitutions": 1,
        "per_insertions": 0,
        "per_deletions": 0,
        "per_errors": 1,
        "per_operations": 2,
        "is_correct": False,
    }
    out = metrics_of(
        [
            [full],
            [
                {
                    **full,
                    "wer_c": 0.0,
                    "wer_c_errors": 0,
                    "wer_pc": 0.25,
                    "wer_pc_errors": 1,
                    "per": 0.0,
                    "per_correct": 2,
                    "per_substitutions": 0,
                    "per_errors": 0,
                }
            ],
        ]
    )["pass@1"]
    assert out["wer_c"] == 25.0 and out["wer_c_errors"] == 2 and out["wer_c_ref_words"] == 8
    assert out["wer_c_macro"] == 25.0
    assert out["wer_pc"] == 50.0 and out["wer_pc_macro"] == 50.0
    assert out["wer_pc_deletions"] == 2 and out["wer_pc_insertions"] == 2
    assert out["per"] == 25.0 and out["per_correct"] == 3 and out["per_operations"] == 4
    assert out["per_errors"] == 1 and out["per_macro"] == 25.0
    # Without count fields the macro mean of per-row scores is used for the headline value.
    fb = metrics_of([[{"wer_c": 0.2, "wer_pc": 0.4, "per": 0.1, "is_correct": True}]])["pass@1"]
    assert fb["wer_c"] == 20.0 and fb["wer_pc"] == 40.0 and fb["per"] == 10.0
    assert fb["wer_c_macro"] == 20.0 and fb["per_macro"] == 10.0


def test_bleu_comet_and_bleu_subsets():
    pytest.importorskip("sacrebleu")
    rows = [
        {
            "bleu": 1.0,
            "pred_text": "the quick brown fox jumps",
            "text": "the quick brown fox jumps",
            "bleu_tokenize": "13a",
            "subset_for_metrics": "de",
            "comet": 0.8,
            "is_correct": True,
        },
        {
            "bleu": 1.0,
            "pred_text": "hello there my friend",
            "text": "hello there my friend",
            "bleu_tokenize": "13a",
            "subset_for_metrics": "de",
            "comet": 0.6,
            "is_correct": True,
        },
        {
            "bleu": 0.0,
            "pred_text": "你好世界",
            "text": "你好世界",
            "bleu_tokenize": "zh",
            "subset_for_metrics": "zh",
            "is_correct": True,
        },
    ]
    out = metrics_of([[r] for r in rows])["pass@1"]
    assert out["bleu"] == pytest.approx(100.0, abs=0.01)
    assert out["bleu::de"] == pytest.approx(100.0, abs=0.01) and out["bleu::de_num_entries"] == 2
    assert out["bleu::zh_num_entries"] == 1
    assert out["comet"] == 70.0


def test_cer_totals_subsets_and_macro():
    def cer_row(errors, chars, subset=None, cer=None):
        d = {
            "cer": errors / chars if cer is None else cer,
            "cer_errors": errors,
            "cer_ref_words": chars,
            "cer_substitutions": errors,
            "cer_insertions": 0,
            "cer_deletions": 0,
            "is_correct": True,
        }
        if subset:
            d["subset_for_metrics"] = subset
        return d

    out = metrics_of([[cer_row(1, 10, "ja")], [cer_row(3, 10, "ja")], [cer_row(0, 20)]])["pass@1"]
    assert out["cer"] == 10.0 and out["cer_ref_chars"] == 40
    assert out["cer_substitutions"] == 4 and out["cer_insertions"] == 0 and out["cer_deletions"] == 0
    assert out["cer_macro"] == round(100 * (0.1 + 0.3 + 0.0) / 3, 2)
    assert out["cer::ja"] == 20.0 and out["cer::ja_errors"] == 4 and out["cer::ja_ref_chars"] == 20
    assert out["cer::ja_substitutions"] == 4 and out["cer::ja_num_entries"] == 2
    assert out["cer::ja_insertions"] == 0 and out["cer::ja_deletions"] == 0


def test_hallucination_pc_rate_and_char_rate():
    rows = [
        {
            "task_type": "Hallucination",
            "hallucination_rate": 1.0,
            "is_correct": False,
            "generation": " " + "a" * 100 + " ",
            "audio_duration": 30.0,
        },
        {
            "task_type": "Hallucination",
            "hallucination_rate": 0.0,
            "is_correct": True,
            "predicted_answer": "b" * 50,
            "generation": "ignored",
            "audio_duration": 30.0,
        },
        {
            "task_type": "Hallucination",
            "hallucination_rate": 0.0,
            "is_correct": True,
            "generation": "zzz",
            "audio_duration": 0.0,
        },
        {"pc_rate": 0.5, "punct_f1": 0.6, "cap_accuracy": 0.4, "is_correct": False, "generation": "x"},
        {"pc_rate": 1.0, "punct_f1": 1.0, "cap_accuracy": 1.0, "is_correct": True, "generation": "y"},
    ]
    out = metrics_of([[r] for r in rows])["pass@1"]
    assert out["hallucination_rate"] == round(100 * 1 / 3, 2)
    assert out["char_rate"] == 150.0  # (100 + 50) chars over one minute of audio
    assert out["pc_rate"] == 75.0 and out["punct_f1"] == 80.0 and out["cap_accuracy"] == 70.0


def test_judge_scores_and_success_rate_from_judgement():
    rows = [
        {"category": "open", "judgement": "Rating: 5", "generation": "a"},
        {"category": "open", "judgement": "Rating: 1", "generation": "b"},
        {"category": "open", "judgement": "Judgement: Yes", "generation": "c"},
    ]
    out = metrics_of([[r] for r in rows])["pass@1"]
    assert out["judge_score"] == pytest.approx((5 + 1 + 5) / 3 * 20)
    assert out["judge_correct"] == pytest.approx(100 * 2 / 3)
    assert out["success_rate"] == pytest.approx(100 * 2 / 3)
    assert "accuracy" not in out


def test_success_rate_falls_back_to_judge_correct_when_no_correct_key(monkeypatch):
    class JudgeOnly(AudioMetrics):
        def _get_score_dict(self, prediction):
            return {"judge_correct": bool(prediction["jc"])}

    m = JudgeOnly()
    m.update([{"jc": True, "generation": "x"}])
    m.update([{"jc": False, "generation": "y"}])
    assert m.get_metrics()["pass@1"]["success_rate"] == 50.0


def test_reference_wer_variants_exclude_count_fields():
    rows = [
        {
            "wer": 0.5,
            "wer_tn": 0.5,
            "wer_itn": 0.0,
            "wer_errors": 1,
            "wer_ref_words": 2,
            "wer_substitutions": 1,
            "wer_insertions": 0,
            "wer_deletions": 0,
            "is_correct": False,
        },
        {
            "wer": 0.0,
            "wer_tn": 0.25,
            "wer_itn": None,
            "wer_errors": 0,
            "wer_ref_words": 2,
            "wer_substitutions": 0,
            "wer_insertions": 0,
            "wer_deletions": 0,
            "is_correct": True,
        },
    ]
    out = metrics_of([[r] for r in rows])["pass@1"]
    assert out["wer_tn"] == 37.5
    assert out["wer_itn"] == 0.0
    assert out["wer"] == 25.0 and out["wer_macro"] == 25.0


# ----------------------------------------------------------------------------- common metrics / pass@k


def test_update_common_metrics_tokens_and_timing():
    m = AudioMetrics()
    m.update(
        [
            {
                "is_correct": True,
                "generation": "a",
                "num_generated_tokens": 10,
                "generation_start_time": 5.0,
                "generation_end_time": 7.0,
            }
        ]
    )
    m.update(
        [
            {
                "is_correct": False,
                "generation": "b",
                "num_generated_tokens": 5,
                "generation_start_time": 6.0,
                "generation_end_time": 12.9,
            }
        ]
    )
    out = m.get_metrics()["pass@1"]
    assert out["num_entries"] == 2 and out["avg_tokens"] == 7  # int(15 / 2)
    assert out["gen_seconds"] == 7  # int(12.9 - 5.0)
    empty = AudioMetrics()
    agg = {}
    empty.update_common_metrics(agg)
    assert agg == {"num_entries": 0, "avg_tokens": 0}


def test_no_answer_is_halved_and_counts_empty_generations():
    out = metrics_of([[{"is_correct": True, "generation": "x"}], [{"is_correct": False, "generation": "  "}]])
    # BaseMetrics counts pass@1 no_answer once per entry; audio metrics halve it (documented double-counting fix).
    assert out["pass@1"]["no_answer"] == pytest.approx(25.0)


def test_pass_at_k_majority_and_statistics_with_multiple_attempts():
    examples = [
        [{"is_correct": True, "generation": "a"}, {"is_correct": False, "generation": "b"}],
        [{"is_correct": False, "generation": "c"}, {"is_correct": False, "generation": "d"}],
        [{"is_correct": True, "generation": "e"}, {"is_correct": True, "generation": "e"}],
    ]
    out = metrics_of(examples, max_k=2)
    assert out["pass@2"]["correct"] == pytest.approx(100 * 2 / 3)
    assert out["pass@1"]["correct"] == pytest.approx(100 * (0.5 + 0 + 1) / 3)
    assert out["pass@1[avg-of-2]"]["correct"] == pytest.approx(100 * (0.5 + 0 + 1) / 3)
    # Majority picks the most common answer; ties average the tied answers' correctness.
    assert out["majority@2"]["correct"] == pytest.approx(100 * (0.5 + 0 + 1) / 3)
    stats = out["pass@1[avg-of-2]"]["correct_statistics"]
    # Run 1 scores [1,0,1] -> 2/3, run 2 [0,0,1] -> 1/3.
    assert stats["avg"] == pytest.approx((0.5 + 0 + 1) / 3)
    assert stats["std_dev_across_runs"] == pytest.approx(np_std([2 / 3, 1 / 3]))
    assert stats["std_err_across_runs"] == pytest.approx(np_std([2 / 3, 1 / 3]) / math.sqrt(2))
    assert stats["avg_sample_std_dev"] == pytest.approx((math.sqrt(0.5) + 0.0 + 0.0) / 3)


def np_std(values):
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def test_update_rejects_wrong_attempt_count_and_infers_k():
    m = AudioMetrics(max_k=2)
    with pytest.raises(ValueError, match="Expected 2 predictions, but got 1"):
        m.update([{"is_correct": True, "generation": "a"}])
    inferred = AudioMetrics(max_k=0)
    inferred.update([{"is_correct": True, "generation": "a"}, {"is_correct": True, "generation": "a"}])
    assert inferred.max_k == 2


# ----------------------------------------------------------------------------- BaseMetrics directly


class _Dummy(BaseMetrics):
    def _get_score_dict(self, prediction):
        return {"correct": prediction["score"]}


def test_base_metrics_abstract_score_dict_and_reset():
    class Bare(BaseMetrics):
        pass

    bare = Bare()
    with pytest.raises(NotImplementedError, match="Needs be implemented"):
        bare._get_score_dict({})
    d = _Dummy()
    d.update([{"score": True}])
    d.reset()
    assert d.total == 0 and d.max_k == 0 and not d.eval_dict and not d.all_scores


def _dummy(max_k=0, **kwargs):
    d = _Dummy(**kwargs)
    d.max_k = max_k
    return d


def _feed(d, predictions, predicted_answers=None):
    d.update(predictions)
    d._compute_pass_at_k(predictions, predicted_answers)


def test_base_metrics_common_metrics_non_float_values_and_tokens():
    d = _dummy(compute_no_answer=False)
    _feed(
        d,
        [
            {
                "score": 1,
                "num_generated_tokens": 9,
                "num_reasoning_tokens": 4,
                "num_answer_tokens": 5,
                "generation_start_time": 1.0,
                "generation_end_time": 3.5,
            }
        ],
    )
    _feed(d, [{"score": 0, "num_generated_tokens": 3}])
    out = d.get_metrics()["pass@1"]
    assert out["num_entries"] == 2
    assert out["avg_tokens"] == 6  # int(12 / 2)
    assert out["gen_seconds"] == 2
    assert out["correct"] == 50.0
    assert "no_answer" not in out
    assert d.all_scores["reasoning_tokens"] == [[4], [0]]
    # Fallback: with no breakdown, all generated tokens count as answer tokens.
    assert d.all_scores["answer_tokens"] == [[5], [3]]
    no_tokens = _dummy()
    _feed(no_tokens, [{"score": 1}])
    assert "avg_tokens" not in no_tokens.get_metrics()["pass@1"]


def test_base_metrics_non_binary_scores_use_max_for_pass_at_k():
    d = _dummy()
    _feed(d, [{"score": 0.2}, {"score": 0.9}, {"score": 0.5}])
    out = d.get_metrics()
    assert out["pass@1"]["correct"] == pytest.approx(20.0)
    assert out["pass@2"]["correct"] == pytest.approx(90.0)
    assert out["pass@3"]["correct"] == pytest.approx(90.0)
    assert out["pass@1[avg-of-3]"]["correct"] == pytest.approx(100 * (0.2 + 0.9 + 0.5) / 3)
    stats = out["pass@1[avg-of-3]"]["correct_statistics"]
    assert stats["avg"] == pytest.approx((0.2 + 0.9 + 0.5) / 3)


def test_base_metrics_majority_tie_break_and_no_valid_answers():
    d = _dummy(max_k=2)
    d.total = 1
    d._compute_majority_at_k(predictions=[{"score": 1}, {"score": 0}], predicted_answers=["a", "b"])
    # Two answers tie with count 1, so the majority score is their mean.
    assert d.eval_dict["majority@2"]["correct"] == 0.5
    assert d.eval_dict["majority@2"]["no_answer"] == 0
    d.eval_dict.clear()
    d._compute_majority_at_k(predictions=[{"score": 1}, {"score": 1}], predicted_answers=[None, None])
    assert d.eval_dict["majority@2"]["correct"] == 0
    assert d.eval_dict["majority@2"]["no_answer"] == 1
    d.eval_dict.clear()
    d._compute_majority_at_k(predictions=[{"score": 1}, {"score": 0}], predicted_answers=["x", "x"])
    assert d.eval_dict["majority@2"]["correct"] == 0.5  # same answer scored differently: mean of its scores


def test_base_metrics_pass_at_k_binary_probability():
    d = _dummy(max_k=3)
    d.total = 1
    d._compute_pass_at_k(predictions=[{"score": 1}, {"score": 0}, {"score": 0}], predicted_answers=["a", None, None])
    # pass@k = 1 - C(2,k)/C(3,k): k=1 -> 1/3, k=2 -> 2/3, k=3 -> 1 (fewer incorrect than k).
    assert d.eval_dict["pass@1"]["correct"] == pytest.approx(1 / 3)
    assert d.eval_dict["pass@2"]["correct"] == pytest.approx(2 / 3)
    assert d.eval_dict["pass@3"]["correct"] == 1
    assert d.eval_dict["pass@1"]["no_answer"] == 0 and d.eval_dict["pass@3"]["no_answer"] == 0
    assert d.eval_dict["pass@1[avg-of-2]"]["no_answer"] == 0.5


def test_base_metrics_std_requires_consistent_attempt_counts():
    d = _dummy(max_k=2)
    d.all_scores["correct"].append([1.0])
    with pytest.raises(AssertionError, match="expected 2"):
        d._add_std_metrics({})


def test_base_metrics_passes_non_float_values_through_unscaled():
    d = _dummy()
    d.total = 4
    d.eval_dict["pass@1"]["count"] = 3  # an int is reported as-is, a float as a percentage of total
    d.eval_dict["pass@1"]["frac"] = 1.0
    out = d.get_metrics()["pass@1"]
    assert out["count"] == 3
    assert out["frac"] == 25.0
