# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from NVIDIA-NeMo/Skills (Apache-2.0), nemo_skills/evaluation/evaluator/contextasr.py
# at commit da85a881d972e6fec847b90cf553a0bf9bf10638: https://github.com/NVIDIA-NeMo/Skills
# The evaluation logic is ported from ContextASR-Bench (MIT): https://github.com/MrSupW/ContextASR-Bench
# at commit 897de87bd4eb430de28dca807fc725958c7ebc85. Upstream notice:
#
#   MIT License
#
#   Copyright (c) 2025 He Wang
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.
#
# Modifications (NVIDIA-authored): the Skills `BaseEvaluator` and `BaseEvaluatorConfig` imports are replaced by this
# package's `BaseEvaluator`, and the unused `ContextASREvaluatorConfig` (which only subclassed the Skills config
# through `nested_dataclass`) is removed. `evaluate_contextasr_sample` reads the generation with `.get`, so a row
# without one scores as an empty hypothesis. Normalization, entity matching and the WER, NE-WER and NE-FNR counts are
# unchanged.

"""ContextASR-Bench evaluator: WER, NE-WER, and NE-FNR metrics.

Ports the official evaluation logic from the ContextASR-Bench paper:
- Text normalization: contraction expansion, punctuation removal, single-letter merging
- Entity extraction: exact match and fuzzy match (edit-distance based)
- WER: word-level edit distance (I+D+S) / T
- NE-WER: WER on fuzzy-matched entity token sequences
- NE-FNR: 1 - (exact entity matches / total entities)
"""

import math
from collections import defaultdict

import editdistance
import regex as re

from resources_servers.audio_eval._scoring.evaluation.evaluator.base import BaseEvaluator


EN_PUNCS_MID_STOP = re.escape(',;(){}[]"|:')
EN_PUNCS_END_STOP = re.escape("!?.")
EN_PUNCS_NON_STOP = re.escape('#$%&*+/<=>@\\^_`~"')
CN_PUNCS_MID_STOP = re.escape("，；､、丶｟｠《》（）｢｣［］｛｝「｣『』【】〔〕〖〗〘〙〚〛〈〉｜：：")
CN_PUNCS_END_STOP = re.escape("！？｡。")
CN_PUNCS_NON_STOP = re.escape('＂＃＄％＆＇＊＋－／＜＝＞＠＼＾＿｀～〃〜〝〞〟〰〾〿‛""„‟…‧﹏·•・′″–—―')
ALL_PUNCTUATIONS = (
    EN_PUNCS_MID_STOP
    + EN_PUNCS_END_STOP
    + EN_PUNCS_NON_STOP
    + CN_PUNCS_MID_STOP
    + CN_PUNCS_END_STOP
    + CN_PUNCS_NON_STOP
)


def _merge_single_letters(text):
    """Combine adjacent single letters separated by spaces."""
    words = text.split()
    current = []
    result = []
    for word in words:
        first_char = word[0]
        remaining = word[1:] if len(word) > 1 else ""
        if first_char.islower() or first_char.isupper():
            if remaining == "" or remaining == "s" or remaining == "'s":
                current.append(first_char)
                if remaining:
                    current.append(remaining)
            else:
                if current:
                    result.append("".join(current))
                    current = []
                result.append(word)
        else:
            if current:
                result.append("".join(current))
                current = []
            result.append(word)
    if current:
        result.append("".join(current))
    return " ".join(result)


def simple_tokenize(text):
    """Normalize English text: expand contractions, remove punctuation, merge single letters, lowercase.

    This replicates the ContextASR-Bench paper's normalization exactly.
    """
    import contractions

    if text.isupper():
        text = text.lower()
    text = re.sub(r"^(O')\s|\s(O')$|\s(O')\s", " O ", text)
    text = re.sub(r"^(o')\s|\s(o')$|\s(o')\s", " o ", text)
    text = contractions.fix(text, leftovers=False, slang=False)
    text = re.sub(rf"[{ALL_PUNCTUATIONS}]", " ", text)
    text = text.replace("-", " ")
    text = text.replace("'", " ")
    ckj = r"\p{Han}\p{Hangul}\p{Hiragana}\p{Katakana}"
    latin = r"\p{IsLatin}"
    text = re.sub(rf"(?<=[{ckj}])(?=[{ckj}])", " ", text)
    text = re.sub(rf"(?<=[{ckj}])(?={latin})", " ", text)
    text = re.sub(rf"(?<={latin})(?=[{ckj}])", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = _merge_single_letters(text)
    return text.lower()


def extract_entities(text, entities_list, entity2count=None):
    """Extract entities that appear exactly in the text, preserving order."""
    text_tokens = text.split()
    entities_with_tokens = [(entity.split(), entity) for entity in entities_list]

    match_entity2count = defaultdict(int)
    matches = []
    n = len(text_tokens)

    for i in range(n):
        for e_tokens, e_str in entities_with_tokens:
            length = len(e_tokens)
            if i + length > n:
                continue
            if text_tokens[i : i + length] == e_tokens:
                if entity2count and match_entity2count[e_str] >= entity2count[e_str]:
                    continue
                match_entity2count[e_str] += 1
                matches.append((i, length, e_str))

    matches.sort(key=lambda x: (x[0], x[1]))
    return [entity for (_, _, entity) in matches]


def extract_entities_fuzzy(text, entities_list):
    """Fuzzy-match entities in text using edit distance, preserving order."""
    text_tokens = text.split()
    match_positions = []

    for entity in entities_list:
        entity_tokens = entity.split()
        n = len(entity_tokens)
        if n == 0:
            continue
        max_dist = math.ceil(n / 2) - 1
        min_len = max(1, n - max_dist)
        max_len = n + max_dist
        lengths_to_search = [n] + list(range(n - 1, min_len - 1, -1)) + list(range(n + 1, max_len + 1))

        next_start = 0
        for start in range(len(text_tokens)):
            if start < next_start:
                continue
            for length in lengths_to_search:
                end = start + length
                if end > len(text_tokens):
                    break
                window = text_tokens[start:end]
                distance = editdistance.eval(window, entity_tokens)
                if distance <= max_dist:
                    next_start = end
                    window_text = " ".join(window)
                    search = re.search(re.escape(entity), window_text)
                    if search:
                        matched_entity = entity
                        next_start -= len(window_text[search.end() :].strip().split())
                    else:
                        matched_entity = window_text
                    match_positions.append((start, matched_entity))
                    break

    match_positions.sort(key=lambda x: (x[0], len(x[1].split())))
    seen = set()
    ordered = []
    for pos, entity in match_positions:
        if (pos, entity) not in seen:
            seen.add((pos, entity))
            ordered.append(entity)
    return ordered


def calculate_wer(hyp_tokens, ref_tokens):
    """Word-level edit distance. Returns (wer, insertions, deletions, substitutions)."""
    n_hyp = len(hyp_tokens)
    n_ref = len(ref_tokens)

    if n_ref == 0:
        return (0.0, 0, 0, 0) if n_hyp == 0 else (float(n_hyp), n_hyp, 0, 0)

    dp = [[0] * (n_ref + 1) for _ in range(n_hyp + 1)]
    ins = [[0] * (n_ref + 1) for _ in range(n_hyp + 1)]
    dels = [[0] * (n_ref + 1) for _ in range(n_hyp + 1)]
    subs = [[0] * (n_ref + 1) for _ in range(n_hyp + 1)]

    for i in range(1, n_hyp + 1):
        dp[i][0] = i
        ins[i][0] = i
    for j in range(1, n_ref + 1):
        dp[0][j] = j
        dels[0][j] = j

    for i in range(1, n_hyp + 1):
        for j in range(1, n_ref + 1):
            if hyp_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                ins[i][j] = ins[i - 1][j - 1]
                dels[i][j] = dels[i - 1][j - 1]
                subs[i][j] = subs[i - 1][j - 1]
            else:
                insertion = dp[i - 1][j] + 1
                deletion = dp[i][j - 1] + 1
                substitution = dp[i - 1][j - 1] + 1
                dp[i][j] = min(insertion, deletion, substitution)
                if dp[i][j] == substitution:
                    ins[i][j] = ins[i - 1][j - 1]
                    dels[i][j] = dels[i - 1][j - 1]
                    subs[i][j] = subs[i - 1][j - 1] + 1
                elif dp[i][j] == deletion:
                    ins[i][j] = ins[i][j - 1]
                    dels[i][j] = dels[i][j - 1] + 1
                    subs[i][j] = subs[i][j - 1]
                else:
                    ins[i][j] = ins[i - 1][j] + 1
                    dels[i][j] = dels[i - 1][j]
                    subs[i][j] = subs[i - 1][j]

    wer = dp[n_hyp][n_ref] / n_ref
    return (wer, ins[n_hyp][n_ref], dels[n_hyp][n_ref], subs[n_hyp][n_ref])


def evaluate_contextasr_sample(data_point):
    """Evaluate a single ContextASR-Bench sample.

    Returns per-sample metrics dict with raw counts for corpus-level aggregation.
    """
    reference = data_point["expected_answer"]
    generation = (data_point.get("generation") or "").strip()
    entity_list = data_point["entity_list"]

    norm_ref = simple_tokenize(reference)
    norm_hyp = simple_tokenize(generation)

    ref_tokens = norm_ref.split()
    hyp_tokens = norm_hyp.split()

    # WER
    wer, wer_i, wer_d, wer_s = calculate_wer(hyp_tokens, ref_tokens)
    wer_errors = wer_i + wer_d + wer_s
    wer_ref_words = len(ref_tokens)

    # Entity normalization
    norm_entities = []
    for entity in entity_list:
        norm_entity = simple_tokenize(entity)
        if norm_entity and norm_entity in norm_ref:
            norm_entities.append(norm_entity)

    updates = {
        "wer": wer,
        "wer_errors": wer_errors,
        "wer_ref_words": wer_ref_words,
        "is_correct": wer < 0.5,
        "text": norm_ref,
        "pred_text": norm_hyp,
    }

    if norm_entities:
        ref_entities = extract_entities(norm_ref, norm_entities)
        entity2count = defaultdict(int)
        for e in ref_entities:
            entity2count[e] += 1

        hyp_exact_entities = extract_entities(norm_hyp, norm_entities, entity2count)
        hyp_fuzzy_entities = extract_entities_fuzzy(norm_hyp, norm_entities)

        # NE-WER: WER on fuzzy entity token sequences
        ref_entity_text = " ".join(ref_entities)
        hyp_fuzzy_text = " ".join(hyp_fuzzy_entities)
        ne_ref_tokens = ref_entity_text.split()
        ne_hyp_tokens = hyp_fuzzy_text.split()

        if ne_ref_tokens:
            ne_wer, ne_i, ne_d, ne_s = calculate_wer(ne_hyp_tokens, ne_ref_tokens)
            ne_wer_errors = ne_i + ne_d + ne_s
            ne_wer_ref_words = len(ne_ref_tokens)
        else:
            ne_wer = 0.0
            ne_wer_errors = 0
            ne_wer_ref_words = 0

        # NE-FNR: 1 - exact_hits / total_entities
        ne_total = len(ref_entities)
        ne_hits = len(hyp_exact_entities)
        ne_fnr = 1.0 - (ne_hits / ne_total) if ne_total > 0 else 0.0

        updates.update(
            {
                "ne_wer": ne_wer,
                "ne_wer_errors": ne_wer_errors,
                "ne_wer_ref_words": ne_wer_ref_words,
                "ne_fnr": ne_fnr,
                "ne_fnr_hits": ne_hits,
                "ne_fnr_total": ne_total,
            }
        )
    else:
        updates.update(
            {
                "ne_wer": 0.0,
                "ne_wer_errors": 0,
                "ne_wer_ref_words": 0,
                "ne_fnr": 0.0,
                "ne_fnr_hits": 0,
                "ne_fnr_total": 0,
            }
        )

    return updates


class ContextASREvaluator(BaseEvaluator):
    """Evaluator for ContextASR-Bench: WER, NE-WER, NE-FNR."""

    async def eval_single(self, data_point: dict) -> dict:
        """Evaluate a single sample, returning WER/NE-WER/NE-FNR metrics."""
        return evaluate_contextasr_sample(data_point)
