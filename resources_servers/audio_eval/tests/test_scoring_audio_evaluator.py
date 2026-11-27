# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Row-level scoring behavior of the audio evaluator (ASR, ASR-PC, CER, MER, BLEU, cpWER, ...)."""

import asyncio

import pytest


pytest.importorskip("jiwer")
pytest.importorskip("whisper_normalizer")
pytest.importorskip("kaldialign")

from resources_servers.audio_eval._scoring.evaluation.evaluator import audio as A
from resources_servers.audio_eval._scoring.evaluation.evaluator.audio import AudioEvaluator, AudioEvaluatorConfig


def run_row(row, **settings):
    return asyncio.run(AudioEvaluator(settings).eval_single(row))


# ----------------------------------------------------------------------------- text helpers


def test_remove_symbols_and_diacritics_maps_special_letters_and_strips_marks():
    assert A.remove_symbols_and_diacritics("Café œ, ß!") == "Cafe oe  ss "
    assert A.remove_symbols_and_diacritics("a-b", keep="-") == "a-b"
    assert A.remove_symbols_and_diacritics("Łódź") == "Lodz"


def test_remove_symbols_keeps_diacritics():
    assert A.remove_symbols("café, ok!") == "café  ok "


def test_remove_symbols_keep_marks_preserves_thai_vowels_and_drops_punct():
    thai = "สวัสดี"
    assert A.remove_symbols_keep_marks(thai + ", ok.") == thai + "  ok "


def test_multilingual_normalizer_variants():
    norm = A.MultilingualTextNormalizer()
    assert norm("Hello, [noise] World (aside)!") == "hello world"
    assert norm("Café") == "cafe"
    assert A.MultilingualTextNormalizer(remove_diacritics=False)("Café, ok") == "café ok"
    assert A.MultilingualTextNormalizer(preserve_marks=True)("สวัสดี, ok.") == "สวัสดี ok"
    pytest.importorskip("num2words")
    assert norm("I have 10 000 cats and 3 dogs", lang="en") == "i have ten thousand cats and three dogs"
    # Numbers that num2words cannot render in a language are left as digits.
    assert norm("room 5", lang="xx-not-a-language") == "room 5"


def test_extract_asr_text_and_prefix_stripping():
    assert A.extract_asr_text("  language English<asr_text> hello there ") == "hello there"
    assert A.extract_asr_text("plain") == "plain"
    assert A.strip_helpful_prefixes("I do not have access to audio files") == ""
    assert A.strip_helpful_prefixes('The audio says "hello world".') == "hello world"
    assert A.strip_helpful_prefixes("The audio says: 'hello there.'") == "hello there"
    assert A.strip_helpful_prefixes("The audio says:'hello there'") == "hello there"
    assert A.strip_helpful_prefixes("it is 'hello world'.") == "hello world"
    assert A.strip_helpful_prefixes("it is five o'clock") == "it is five o'clock"
    srt = "1 00:00:01,000 --> 00:00:02,500 hello\n2 00:00:03,000 --> 00:00:04,000 world"
    assert A.strip_helpful_prefixes(srt).split() == ["hello", "world"]
    assert A.strip_helpful_prefixes("00:00:01,000 --> 00:00:02,000 hi") == "hi"
    assert A.strip_helpful_prefixes("a\\n1 00:00:01 b") == "a 00:00:01 b"


def test_mixed_segment_counts_continua_per_unit_and_latin_per_word():
    assert A.mixed_segment("我 love 北京 city") == ["我", "love", "北", "京", "city"]
    assert A.mixed_segment("ab cd") == ["ab", "cd"]
    # Thai tone/vowel marks stay attached to their base only in grapheme mode.
    thai = "ที่"
    assert len(A.mixed_segment(thai)) == len(thai)
    assert A.mixed_segment(thai, grapheme=True) == [thai]


# ----------------------------------------------------------------------------- ASR-PC


def test_punctuation_error_counts_perfect_and_mismatched():
    perfect = A.punctuation_error_counts("Hello, world.", "Hello, world.")
    assert perfect["per"] == 0.0
    assert perfect["per_correct"] == 2
    assert perfect["per_operations"] == 2
    # Reference has "," and "."; hypothesis swaps "," for "?" and drops ".".
    res = A.punctuation_error_counts("Hello, world.", "Hello? world")
    assert res["per_substitutions"] == 1
    assert res["per_deletions"] == 1
    assert res["per_insertions"] == 0
    assert res["per_errors"] == 2
    assert res["per_operations"] == 2
    assert res["per"] == 1.0
    # Words that differ do not count as punctuation errors; extra punctuation is an insertion.
    res = A.punctuation_error_counts("hello world", "hello, world")
    assert res["per_insertions"] == 1
    assert res["per_operations"] == 1
    # Empty on both sides has no operations.
    assert A.punctuation_error_counts("", "")["per"] == 0.0
    # Unmatched words on both sides exercise substitution / insertion / deletion backtraces.
    res = A.punctuation_error_counts("a b c.", "x")
    assert res["per_deletions"] == 1


def test_evaluate_asr_pc_exact_and_masked():
    exact = A.evaluate_asr_pc("Hello, World.", "Hello, World.")
    assert exact["wer"] == exact["wer_c"] == exact["wer_pc"] == 0.0
    assert exact["is_correct"] is True
    assert exact["text"] == "hello world"
    miss = A.evaluate_asr_pc("Hello, World.", "Hello World")
    assert miss["wer"] == 0.0
    assert miss["wer_c"] == 0.0
    assert miss["wer_pc_errors"] == 2
    assert miss["wer_pc_ref_words"] == 4
    assert miss["wer_pc"] == 0.5
    assert miss["is_correct"] is False
    assert miss["per_deletions"] == 2


# ----------------------------------------------------------------------------- CER whitespace


def test_resolve_cer_whitespace_mode():
    assert A.resolve_cer_whitespace_mode(None) == "keep"
    assert A.resolve_cer_whitespace_mode(True) == "all"
    assert A.resolve_cer_whitespace_mode(False) == "keep"
    assert A.resolve_cer_whitespace_mode("han_adjacent") == "han_adjacent"
    for bad in ("false", 3):
        with pytest.raises(ValueError, match="Invalid CER whitespace override"):
            A.resolve_cer_whitespace_mode(bad)


def test_resolve_sample_cer_whitespace_mode():
    assert A.resolve_sample_cer_whitespace_mode({}) == "keep"
    assert A.resolve_sample_cer_whitespace_mode({"use_mer": True}) == "all"
    assert A.resolve_sample_cer_whitespace_mode({"cer_whitespace_mode": "han_adjacent"}) == "han_adjacent"
    assert A.resolve_sample_cer_whitespace_mode({"cer_strip_whitespace": True}) == "all"
    assert A.resolve_sample_cer_whitespace_mode({"cer_strip_whitespace": False, "use_mer": True}) == "keep"
    with pytest.raises(ValueError, match="Set only one"):
        A.resolve_sample_cer_whitespace_mode({"cer_whitespace_mode": "keep", "cer_strip_whitespace": True})
    with pytest.raises(ValueError, match="must be one of"):
        A.resolve_sample_cer_whitespace_mode({"cer_whitespace_mode": 1})
    with pytest.raises(ValueError, match="must be a boolean"):
        A.resolve_sample_cer_whitespace_mode({"cer_strip_whitespace": "yes"})


# ----------------------------------------------------------------------------- SOT / cpWER


def test_parse_sot_speaker_streams_modes():
    text = "<spk:0> hello there <spk:1> hi <spk:0> again"
    streams = A.parse_sot_speaker_streams(text, mode="strict")
    assert dict(streams) == {"s0": "hello there again", "s1": "hi"}
    # Untagged text before the first tag is dropped in strict mode, kept as default speaker in canonical.
    assert dict(A.parse_sot_speaker_streams("lead <spk:1> x", mode="strict")) == {"s1": "x"}
    assert dict(A.parse_sot_speaker_streams("lead <spk:1> x")) == {"s0": "lead", "s1": "x"}
    assert dict(A.parse_sot_speaker_streams("no tags", mode="canonical", default_speaker="sx")) == {"sx": "no tags"}
    assert dict(A.parse_sot_speaker_streams("no tags", mode="strict")) == {}
    assert dict(A.parse_sot_speaker_streams("", mode="canonical")) == {}
    assert dict(A.parse_sot_speaker_streams("[S2] a [s3] b", strict_tag_style="s_bracket", mode="strict")) == {
        "s2": "a",
        "s3": "b",
    }
    assert dict(A.parse_sot_speaker_streams("Speaker 4: hey")) == {"s4": "hey"}
    assert dict(A.parse_sot_speaker_streams("[s1] a <spk:2> b", mode="strict", strict_tag_style="both")) == {
        "s1": "a",
        "s2": "b",
    }
    with pytest.raises(ValueError, match="Unsupported SOT parse mode"):
        A.parse_sot_speaker_streams("x", mode="bogus")
    with pytest.raises(ValueError, match="Unsupported SOT strict tag style"):
        A.parse_sot_speaker_streams("x", mode="strict", strict_tag_style="bogus")


def test_resolve_speaker_tag_style():
    assert A.resolve_speaker_tag_style("<spk:*>") == "spk"
    assert A.resolve_speaker_tag_style("") == "spk"
    assert A.resolve_speaker_tag_style(" [S*] ") == "s_bracket"
    assert A.resolve_speaker_tag_style("both") == "both"
    with pytest.raises(ValueError, match="Unsupported speaker_token"):
        A.resolve_speaker_tag_style("???")


def test_normalize_cpwer_stream_modes():
    assert A.normalize_cpwer_stream("  A   b ", mode="none") == "A b"
    assert A.normalize_cpwer_stream("Hello, World", mode="chime8") == A.normalize_cpwer_stream("hello world")
    with pytest.raises(ValueError, match="Invalid cpwer_normalization"):
        A.normalize_cpwer_stream("x", mode="bogus")


def test_evaluate_msasr_counts_and_empty_reference():
    ref = "<spk:0> hello there <spk:1> good day"
    hyp = "<spk:0> hello there <spk:1> good morning"
    res = A.evaluate_msasr(ref, hyp, cpwer_normalization="none")
    assert res["cpwer_errors"] == 1
    assert res["cpwer_substitutions"] == 1
    assert res["cpwer_ref_words"] == 4
    assert res["cpwer"] == 0.25
    assert res["is_correct"] is False
    assert res["reference_streams"] == {"s0": "hello there", "s1": "good day"}
    assert res["hypothesis_streams"] == {"s0": "hello there", "s1": "good morning"}
    exact = A.evaluate_msasr(ref, ref, cpwer_normalization="none")
    assert exact["cpwer"] == 0.0 and exact["is_correct"] is True
    assert exact["text"] == "hello there good day"
    # A reference with no speaker tags cannot be scored.
    empty = A.evaluate_msasr("untagged", "<spk:0> x", cpwer_normalization="none")
    assert empty == {"cpwer": None, "is_correct": None, "text": "", "pred_text": "x"}


def test_evaluate_msasr_pairs_reference_and_hypothesis_speakers_by_the_best_permutation():
    """cpWER is invariant to the speaker labels: the hypothesis labels need not match the reference labels."""
    ref = "<spk:0> hello there <spk:1> good day to you"
    # The labels are swapped: matched by label every word is wrong, matched by the best permutation none is.
    swapped = A.evaluate_msasr(ref, "<spk:0> good day to you <spk:1> hello there", cpwer_normalization="none")
    assert swapped["cpwer_errors"] == 0 and swapped["cpwer"] == 0.0 and swapped["is_correct"] is True
    aligned_wrong = A.evaluate_msasr(ref, "<spk:0> good day to you <spk:1> hello", cpwer_normalization="none")
    assert aligned_wrong["cpwer_errors"] == 1 and aligned_wrong["cpwer_deletions"] == 1
    # One more hypothesis speaker than reference speakers: the surplus stream is paid for as insertions.
    split = A.evaluate_msasr(ref, "<spk:0> good day to you <spk:1> hello <spk:2> there", cpwer_normalization="none")
    assert split["cpwer_ref_words"] == 6
    assert split["cpwer_errors"] == 2 and split["cpwer_insertions"] == 1 and split["cpwer_deletions"] == 1
    assert split["cpwer"] == pytest.approx(2 / 6)


# ----------------------------------------------------------------------------- normalization / WER


def test_resolve_asr_normalization_mode():
    cfg = AudioEvaluatorConfig(normalization_mode="multilingual")
    assert A.resolve_asr_normalization_mode(cfg) == "multilingual"
    cfg = AudioEvaluatorConfig(normalization_mode="multilingual", apply_normalization=False)
    assert A.resolve_asr_normalization_mode(cfg) == "none"
    cfg = AudioEvaluatorConfig(normalization_mode="no_tn_itn", apply_normalization=False)
    assert A.resolve_asr_normalization_mode(cfg) == "no_tn_itn"


def test_preprocess_asr_text_modes():
    assert A.preprocess_asr_text("  Hello,   World ", mode="none") == "Hello, World"
    assert A.preprocess_asr_text("Hello, World! It's", mode="no_tn_itn") == "hello world its"
    assert A.preprocess_asr_text("Thé, Mañana. ¿Ok?", mode="lower_nopunct") == "thé mañana ok"
    assert A.preprocess_asr_text("Twenty five dollars", mode="standard") == "$25"
    assert A.preprocess_asr_text("Hello, World", mode="multilingual", lang="en") == "hello world"
    assert A.preprocess_asr_text("Hello, World", mode="multilingual", lang=None) == "hello world"
    with pytest.raises(ValueError, match="Invalid normalization_mode"):
        A.preprocess_asr_text("x", mode="bogus")


def test_preprocess_asr_text_multilingual_non_english():
    pytest.importorskip("num2words")
    out = A.preprocess_asr_text("Café 3", mode="multilingual", lang="fr")
    assert out == "café trois"
    out = A.preprocess_asr_text("Café", mode="multilingual", lang="fr", remove_diacritics=True)
    assert out == "cafe"


def test_wer_with_counts_plain_and_merge_compounds():
    res = A._wer_with_counts("a b c d", "a x c")
    assert res["wer_errors"] == 2
    assert res["wer_substitutions"] == 1
    assert res["wer_deletions"] == 1
    assert res["wer_insertions"] == 0
    assert res["wer_ref_words"] == 4
    assert res["wer"] == 0.5
    merged = A._wer_with_counts("note book", "notebook", merge_compounds=True)
    assert merged["wer_errors"] == 0
    assert merged["wer_ref_words"] == 2


def test_wer_with_counts_merge_compounds_falls_back_without_batch_error_rate(monkeypatch):
    import kaldialign

    monkeypatch.delattr(kaldialign, "batch_error_rate")
    res = A._wer_with_counts("the note book", "the notebook", merge_compounds=True)
    assert res["wer_errors"] == 0
    assert res["wer_ref_words"] == 3
    assert res["wer"] == 0.0
    res = A._wer_with_counts("", "x", merge_compounds=True)
    assert res["wer"] == 0.0 or res["wer_ref_words"] == 0


def test_cer_with_counts_and_key_prefix_validation():
    res = A._cer_with_counts("abcd", "abxd")
    assert res["cer_errors"] == 1
    assert res["cer_substitutions"] == 1
    assert res["cer_ref_words"] == 4
    assert res["cer"] == 0.25
    assert A._cer_with_counts("ab", "ab", key_prefix="wer")["wer"] == 0.0
    with pytest.raises(AssertionError):
        A._cer_with_counts("ab", "ab", key_prefix="bogus")


def test_evaluate_asr_scores_and_drops_empty_reference():
    res = A.evaluate_asr("The cat sat.", "the cat sat down", normalization_mode="none")
    assert res["wer_errors"] == 3  # "The"/"the", "sat."/"sat" and the extra "down"
    res = A.evaluate_asr("The cat sat.", "the cat sat", normalization_mode="standard")
    assert res["wer"] == 0.0
    assert res["is_correct"] is True
    assert res["text"] == "the cat sat" and res["pred_text"] == "the cat sat"
    # Empty hypothesis is replaced by a placeholder so every reference word is an error.
    res = A.evaluate_asr("one two", "", normalization_mode="none")
    assert res["pred_text"] == "empty"
    assert res["wer"] == 1.0
    assert res["is_correct"] is False
    # Empty reference is dropped (None), not scored.
    assert A.evaluate_asr("", "anything", normalization_mode="none") == {
        "wer": None,
        "is_correct": None,
        "text": "",
        "pred_text": "anything",
    }
    assert A.evaluate_asr("  ", "", normalization_mode="none")["pred_text"] == ""


def test_evaluate_asr_compound_paths():
    res = A.evaluate_asr("note book", "notebook", normalization_mode="none", normalize_compound=True)
    assert res["wer"] == 0.0
    res = A.evaluate_asr("note book", "", normalization_mode="none", merge_compounds=True)
    assert res["pred_text"] == ""
    assert res["wer"] == 1.0


# ----------------------------------------------------------------------------- BLEU


def test_resolve_bleu_tokenize():
    assert A.resolve_bleu_tokenize(None) == "13a"
    assert A.resolve_bleu_tokenize("de") == "13a"
    assert A.resolve_bleu_tokenize("ja") == "ja-mecab"
    assert A.resolve_bleu_tokenize("zh-CN") == "zh"
    assert A.resolve_bleu_tokenize("cmn_hans_cn") == "zh"
    assert A.resolve_bleu_tokenize("ko_kr") == "ko-mecab"


def test_evaluate_translation_scores_and_tokenizer():
    pytest.importorskip("sacrebleu")
    same = A.evaluate_translation(" the quick brown fox jumps ", "the quick brown fox jumps", "de")
    assert same["bleu"] == pytest.approx(1.0)
    assert same["is_correct"] is True
    assert same["bleu_tokenize"] == "13a"
    assert same["text"] == "the quick brown fox jumps"
    diff = A.evaluate_translation("the quick brown fox", "completely different words", None)
    assert diff["bleu"] < 0.3 and diff["is_correct"] is False
    zh = A.evaluate_translation("你好世界", "你好世界", "zh")
    assert zh["bleu_tokenize"] == "zh"
    assert zh["bleu"] == pytest.approx(1.0)


def test_evaluate_translation_reports_scoring_errors(monkeypatch):
    import sacrebleu

    def boom(*args, **kwargs):
        raise RuntimeError("tokenizer exploded")

    monkeypatch.setattr(sacrebleu, "sentence_bleu", boom)
    res = A.evaluate_translation(" ref ", " hyp ", "de")
    assert res == {
        "bleu": 0.0,
        "is_correct": False,
        "error": "tokenizer exploded",
        "text": "ref",
        "pred_text": "hyp",
        "bleu_tokenize": "13a",
    }


# ----------------------------------------------------------------------------- CER / MER


def test_evaluate_cer_whitespace_modes_and_validation():
    keep = A.evaluate_cer("ab cd", "abcd", whitespace_mode="keep")
    assert keep["cer_errors"] == 1 and keep["cer_ref_words"] == 5 and keep["cer"] == 0.2
    allw = A.evaluate_cer("ab cd", "abcd", whitespace_mode="all")
    assert allw["cer"] == 0.0 and allw["text"] == "abcd"
    boolean = A.evaluate_cer("ab cd", "abcd", strip_whitespace=True)
    assert boolean["cer"] == 0.0
    assert A.evaluate_cer("ab cd", "abcd", strip_whitespace=False)["cer"] == 0.2
    han = A.evaluate_cer("你 好 ok", "你好 ok", whitespace_mode="han_adjacent")
    assert han["cer"] == 0.0
    with pytest.raises(ValueError, match="strip_whitespace must be a boolean"):
        A.evaluate_cer("a", "a", strip_whitespace="yes")
    with pytest.raises(ValueError, match="Set only one"):
        A.evaluate_cer("a", "a", strip_whitespace=True, whitespace_mode="all")
    with pytest.raises(ValueError, match="cannot be combined"):
        A.evaluate_cer("a", "a", normalize_compound=True, whitespace_mode="han_adjacent")


def test_evaluate_cer_empty_reference_prefix_and_compounds():
    assert A.evaluate_cer("", "abc", key_prefix="wer") == {
        "wer": None,
        "is_correct": None,
        "text": "",
        "pred_text": "abc",
    }
    res = A.evaluate_cer("abc", "abd", key_prefix="wer")
    assert res["wer_errors"] == 1 and res["is_correct"] is True
    res = A.evaluate_cer("note book", "notebook", normalize_compound=True)
    assert res["cer"] == 0.0


def test_evaluate_mer_mixed_script_counting():
    # Mandarin counts per character, English per word: 4 reference tokens, 1 wrong.
    res = A.evaluate_mer("我爱 python code", "我恨 python code")
    assert res["wer_ref_words"] == 4
    assert res["wer_errors"] == 1
    assert res["wer"] == 0.25
    assert res["is_correct"] is True
    assert res["text"] == "我爱 python code"
    empty_hyp = A.evaluate_mer("我爱你", "")
    assert empty_hyp["pred_text"] == "empty"
    assert empty_hyp["wer_errors"] == 3
    assert A.evaluate_mer("", "x") == {"wer": None, "is_correct": None, "text": "", "pred_text": "x"}
    comp = A.evaluate_mer("note book", "notebook", normalize_compound=True)
    assert comp["wer"] == 0.0
    graph = A.evaluate_mer("ที่ ok", "ที่ ok", grapheme=True)
    assert graph["wer"] == 0.0 and graph["wer_ref_words"] == 2


# ----------------------------------------------------------------------------- hallucination / PC rate


def test_evaluate_hallucination_rate_thresholds():
    missing = A.evaluate_hallucination("ref", "hyp", None)
    assert missing["error"] == "missing_audio_duration" and missing["is_correct"] is True
    assert A.evaluate_hallucination("ref", "hyp", {"audio_duration": 0})["error"] == "missing_audio_duration"
    normal = A.evaluate_hallucination("ref", "a" * 100, {"audio_duration": 10.0})
    assert normal["char_rate"] == 600.0
    assert normal["hallucination_rate"] == 0.0 and normal["is_correct"] is True
    bad = A.evaluate_hallucination("ref", "a" * 300, {"audio_duration": 10.0})
    assert bad["char_rate"] == 1800.0
    assert bad["hallucination_rate"] == 1.0 and bad["is_correct"] is False


def test_evaluate_pc_rate_punctuation_and_capitalization():
    perfect = A.evaluate_pc_rate("Hello, World.", "Hello, World.")
    assert perfect["punct_f1"] == 1.0 and perfect["cap_accuracy"] == 1.0
    assert perfect["pc_rate"] == 1.0 and perfect["is_correct"] is True
    # Same punctuation count and positions, but lowercase: F1 = 1, caps = 0.
    lower = A.evaluate_pc_rate("Hello, World.", "hello, world.")
    assert lower["punct_f1"] == 1.0 and lower["cap_accuracy"] == 0.0
    assert lower["pc_rate"] == 0.5 and lower["is_correct"] is False
    # Different word counts zero the capitalization score; one of two reference marks is matched.
    skew = A.evaluate_pc_rate("Hi, there.", "Hi, there extra")
    assert skew["cap_accuracy"] == 0.0
    assert skew["punct_precision"] == 1.0 and skew["punct_recall"] == 0.5
    assert skew["punct_f1"] == pytest.approx(0.667, abs=1e-3)
    none = A.evaluate_pc_rate("", "")
    assert none["pc_rate"] == 0.0 and none["punct_f1"] == 0.0 and none["cap_accuracy"] == 0.0


# ----------------------------------------------------------------------------- text coercion


def test_coerce_text_field_variants():
    assert A.coerce_text_field(None) == ""
    assert A.coerce_text_field("  x ") == "x"
    assert A.coerce_text_field({"text": " t "}) == "t"
    assert A.coerce_text_field({"answer": 5}) == "5"
    assert A.coerce_text_field({"text": None, "reference": "r"}) == "r"
    assert A.coerce_text_field({"a": "x", "b": None, "c": ["y", "z"]}) == "x y z"
    assert A.coerce_text_field(["a", None, ("b", "c")]) == "a b c"
    assert A.coerce_text_field(7) == "7"


# ----------------------------------------------------------------------------- AudioEvaluator.eval_single


def test_eval_single_asr_row():
    res = run_row(
        {"task_type": "ASR", "generation": "<asr_text>the cat sat down", "expected_answer": "The cat sat."},
        normalization_mode="standard",
    )
    assert res["predicted_answer"] == "the cat sat down"
    assert res["wer_errors"] == 1 and res["wer_ref_words"] == 3
    assert res["wer"] == pytest.approx(1 / 3)
    assert res["is_correct"] is True


def test_eval_single_asr_normalization_disabled_and_prefix_stripping():
    row = {"task_type": "ASR", "generation": 'The audio says "Hello"', "expected_answer": "Hello"}
    with_prefix = run_row(row, strip_helpful_prefixes=True, normalization_mode="none")
    assert with_prefix["wer"] == 0.0
    without = run_row(row, normalization_mode="none")
    assert without["wer"] == 4.0  # one substitution plus three insertions over one reference word
    off = run_row(
        {"task_type": "ASR", "generation": "hello", "expected_answer": "Hello"},
        apply_normalization=False,
    )
    assert off["wer"] == 1.0  # normalization off keeps case


def test_eval_single_asr_pc_row():
    res = run_row({"task_type": "ASR-PC", "generation": "Hello World", "expected_answer": "Hello, World."})
    assert res["wer"] == 0.0
    assert res["wer_pc_errors"] == 2 and res["wer_pc"] == 0.5
    assert res["is_correct"] is False


def test_eval_single_ms_asr_row():
    res = run_row(
        {
            "task_type": "MS-ASR",
            "generation": "<spk:0> hello there <spk:1> good morning",
            "expected_answer": "<spk:0> hello there <spk:1> good day",
        },
        cpwer_normalization="none",
    )
    assert res["cpwer"] == 0.25 and res["cpwer_errors"] == 1
    assert res["predicted_answer"] == "<spk:0> hello there <spk:1> good morning"
    default_norm = run_row(
        {"task_type": "MS-ASR", "generation": "[s0] Hello.", "expected_answer": "[s0] hello"},
        speaker_token="[s*]",
    )
    assert default_norm["cpwer"] == 0.0


def test_eval_single_asr_leaderboard_with_reference_fields():
    row = {
        "task_type": "ASR_LEADERBOARD",
        "generation": "twenty five dollars",
        "expected_answer": "25 dollars",
        "text_tn": "25 dollars",
        "text_itn": "twenty five dollars",
        "plain": "twenty five dollars",
    }
    res = run_row(row, reference_fields=["text_tn", "text_itn", "plain"], normalization_mode="none")
    assert res["wer"] == pytest.approx(2 / 2)
    assert res["wer_tn"] == 1.0 and res["is_correct_tn"] is False
    assert res["wer_itn"] == 0.0 and res["is_correct_itn"] is True
    assert res["wer_plain"] == 0.0
    plain = run_row(row, normalization_mode="none")
    assert "wer_tn" not in plain
    with pytest.raises(KeyError):
        run_row({"task_type": "ASR_LEADERBOARD", "generation": "a", "expected_answer": "a"}, reference_fields=["nope"])


def test_eval_single_translation_rows():
    pytest.importorskip("sacrebleu")
    for task in ("AST", "Translation", "Multilingual-AST"):
        res = run_row(
            {
                "task_type": task,
                "generation": "the quick brown fox jumps",
                "expected_answer": "the quick brown fox jumps",
                "extra_fields": {"tgt_lang": "de"},
            }
        )
        assert res["bleu"] == pytest.approx(1.0) and res["bleu_tokenize"] == "13a"
    zh = run_row(
        {"task_type": "AST", "generation": "你好", "expected_answer": "你好", "extra_fields": {"tgt_lang": "zh"}}
    )
    assert zh["bleu_tokenize"] == "zh"
    nolang = run_row({"task_type": "AST", "generation": "a b c d", "expected_answer": "a b c d"})
    assert nolang["bleu_tokenize"] == "13a"


def test_eval_single_multilingual_asr_wer_path():
    res = run_row(
        {
            "task_type": "Multilingual-ASR",
            "generation": "the cat sat",
            "expected_answer": "The cat sat.",
            "extra_fields": {"src_lang": "en_us"},
        },
        normalization_mode="multilingual",
    )
    assert res["wer"] == 0.0 and res["is_correct"] is True
    assert res["cer"] == 0.0 and res["cer_ref_words"] > 0
    # Non-English default turns on compound normalization; merge_compounds is opt-in.
    pytest.importorskip("num2words")
    res = run_row(
        {
            "task_type": "Multilingual-ASR",
            "generation": "Un café",
            "expected_answer": "un cafe",
            "extra_fields": {"src_lang": "fr", "remove_diacritics": True, "merge_compounds": True},
        },
        normalization_mode="multilingual",
    )
    assert res["wer"] == 0.0


def test_eval_single_multilingual_asr_cer_path():
    res = run_row(
        {
            "task_type": "Multilingual-ASR",
            "generation": "你好吗",
            "expected_answer": "你好嘛",
            "extra_fields": {"src_lang": "cmn_hans_cn", "use_cer": True, "cer_whitespace_mode": "han_adjacent"},
        },
        normalization_mode="multilingual",
    )
    # CER is folded into the headline wer* keys and also reported under cer*.
    assert res["wer_errors"] == 1 and res["wer_ref_words"] == 3
    assert res["wer"] == pytest.approx(1 / 3)
    assert res["cer"] == pytest.approx(1 / 3)
    assert set(res) >= {"cer_errors", "cer_ref_words", "cer_substitutions", "cer_insertions", "cer_deletions"}


def test_eval_single_multilingual_asr_mixed_error_rate_path():
    res = run_row(
        {
            "task_type": "Multilingual-ASR",
            "generation": "我恨 python code",
            "expected_answer": "我爱 Python Code",
            "extra_fields": {"src_lang": "cmn", "use_mer": True, "mer_grapheme": True},
        },
        normalization_mode="multilingual",
    )
    assert res["wer_errors"] == 1 and res["wer_ref_words"] == 4
    assert res["wer"] == 0.25
    # The parallel CER uses all-whitespace lower_nopunct normalization for code-switched rows.
    assert res["cer_errors"] == 1
    assert res["cer"] == pytest.approx(1 / res["cer_ref_words"])
    # preserve_marks via explicit flag on a monolingual row
    flagged = run_row(
        {
            "task_type": "Multilingual-ASR",
            "generation": "สวัสดี",
            "expected_answer": "สวัสดี",
            "extra_fields": {"src_lang": "th", "preserve_marks": True},
        },
        normalization_mode="multilingual",
    )
    assert flagged["wer"] == 0.0 and flagged["cer"] == 0.0


def test_eval_single_cer_row():
    res = run_row(
        {
            "task_type": "CER",
            "generation": "abxd",
            "expected_answer": "abcd",
            "extra_fields": {"cer_strip_whitespace": True},
        }
    )
    assert res["cer"] == 0.25 and res["cer_errors"] == 1 and res["is_correct"] is True
    assert run_row({"task_type": "CER", "generation": "a b", "expected_answer": "ab"})["cer"] == 0.5


def test_eval_single_hallucination_and_pc_rate_rows():
    res = run_row(
        {"task_type": "Hallucination", "generation": "a" * 300, "expected_answer": "ref", "audio_duration": 10.0}
    )
    assert res["hallucination_rate"] == 1.0 and res["char_rate"] == 1800.0
    assert res["ref_char_rate"] == pytest.approx(18.0)
    assert res["hyp_char_rate"] == pytest.approx(1800.0)
    assert res["char_rate_diff"] == pytest.approx(1782.0)
    nodur = run_row({"task_type": "Hallucination", "generation": "abc", "expected_answer": "ref"})
    assert nodur["error"] == "missing_audio_duration" and "ref_char_rate" not in nodur
    pc = run_row({"task_type": "PC-Rate", "generation": "Hello, World.", "expected_answer": "Hello, World."})
    assert pc["pc_rate"] == 1.0


def test_eval_single_unknown_task_defers_to_judge():
    assert run_row({"task_type": "QA", "generation": "x", "expected_answer": "y"}) == {
        "requires_judge": True,
        "is_correct": False,
    }
    assert run_row({"generation": "x"}) == {"requires_judge": True, "is_correct": False}
    kept = run_row({"task_type": "QA", "generation": "x", "requires_judge": False, "is_correct": True})
    assert kept == {}


def test_evaluator_config_defaults_and_unknown_keys():
    ev = AudioEvaluator({"normalization_mode": "none", "speaker_token": "[s*]"})
    assert ev.eval_config.normalization_mode == "none"
    assert ev.eval_config.cpwer_normalization == "chime8"
    assert ev.eval_config.apply_normalization is True
    with pytest.raises(TypeError):
        AudioEvaluator({"not_a_setting": 1})


# ----------------------------------------------------------------------------- registered modes and task types


def test_registered_normalization_modes_and_task_types_extend_the_evaluator(monkeypatch):
    monkeypatch.setattr(A, "_EXTRA_NORMALIZATION_MODES", {})
    monkeypatch.setattr(A, "_EXTRA_TASK_TYPES", {})
    with pytest.raises(ValueError, match="Invalid normalization_mode"):
        A.preprocess_asr_text("x", mode="shout")

    A.register_normalization_mode("shout", lambda text, **kwargs: text.upper() + kwargs.get("suffix", ""))
    assert A.preprocess_asr_text("  a   b ", mode="shout", suffix="!") == "A B !"
    assert "shout" in A.valid_normalization_modes()
    assert A.evaluate_asr("A b", "a B", normalization_mode="shout")["wer"] == 0.0
    with pytest.raises(ValueError, match="already defined"):
        A.register_normalization_mode("shout", str.upper)
    with pytest.raises(ValueError, match="already defined"):
        A.register_normalization_mode("standard", str.upper)

    seen = {}

    def handler(sample, generation, config):
        seen.update(sample=sample, generation=generation, config=config)
        return {"is_correct": generation == "7", "custom": True}

    A.register_task_type("Custom", handler)
    row = {"task_type": "Custom", "generation": "7", "expected_answer": "7", "audio_duration": 2.0}
    res = run_row(row)
    assert res["is_correct"] is True and res["custom"] is True
    # The common tail still applies to a registered task type.
    assert res["ref_char_rate"] == pytest.approx(30.0)
    assert (
        seen["sample"]["task_type"] == "Custom"
        and seen["generation"] == "7"
        and isinstance(seen["config"], AudioEvaluatorConfig)
    )
    with pytest.raises(ValueError, match="already defined"):
        A.register_task_type("Custom", handler)
