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
"""Translation verifier scored by an LLM judge.

``verify()`` sends the source segment, the reference translation, and the
policy model's candidate translation to a judge model (any OpenAI-Responses
-compatible endpoint, e.g. a self-hosted vLLM server) and asks it to rate
adequacy + fluency on a 0-100 scale. That score, normalized to [0, 1], is the
RL reward. Corpus-level sentence-BLEU and chrF (via sacrebleu) are also
computed per row as cheap, judge-independent diagnostics -- they do NOT
affect the reward.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from fastapi import FastAPI
from pydantic import BaseModel
from sacrebleu import sentence_bleu, sentence_chrf

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.failure_kinds import JUDGE_UNPARSEABLE
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import get_response_json


LOG = logging.getLogger(__name__)


# --- Tokenizer selection (sentence-BLEU diagnostic only) -----------------------
# ``13a`` is sacrebleu's default, whitespace/punctuation-based tokenizer -- it
# doesn't segment Brahmic scripts (Devanagari, Bengali, Tamil, etc.) into
# meaningful sub-word units, so BLEU on Indic targets would otherwise be
# scored at (roughly) the word level with no normalization. ``flores200``
# is a multilingual SentencePiece model covering 200 languages, including
# all Indic languages, and is the standard tokenizer for Indic-language BLEU
# in MT literature (e.g. FLORES-200, NLLB). It needs the ``sentencepiece``
# package (see requirements.txt) and downloads its model file to
# ``~/.sacrebleu/models`` on first use.
# ``ja-mecab`` / ``ko-mecab`` need sacrebleu's [ja]/[ko] extras; ``zh`` is built in.
#
# Languages are identified by FLORES-200 code (e.g. ``kan_Knda``, ``eng_Latn``)
# -- the 3-letter subtag before the underscore is the lookup key, so the
# script suffix (``_Deva``, ``_Arab``, ...) doesn't need to match exactly.
_TOKENIZER_BY_FLORES_SUBTAG = {
    "jpn": "ja-mecab",
    "kor": "ko-mecab",
    "zho": "zh",
    # Indic languages (Indo-Aryan + Dravidian + others in Brahmic/Perso-Arabic
    # scripts), by FLORES-200 subtag.
    "hin": "flores200",  # Hindi
    "ben": "flores200",  # Bengali
    "tam": "flores200",  # Tamil
    "tel": "flores200",  # Telugu
    "mar": "flores200",  # Marathi
    "guj": "flores200",  # Gujarati
    "kan": "flores200",  # Kannada
    "mal": "flores200",  # Malayalam
    "pan": "flores200",  # Punjabi
    "ory": "flores200",  # Odia
    "asm": "flores200",  # Assamese
    "urd": "flores200",  # Urdu
    "npi": "flores200",  # Nepali
    "san": "flores200",  # Sanskrit
    "doi": "flores200",  # Dogri
    "gom": "flores200",  # Konkani (Goan)
    "snd": "flores200",  # Sindhi
    "brx": "flores200",  # Bodo
    "kas": "flores200",  # Kashmiri
    "mai": "flores200",  # Maithili
    "mni": "flores200",  # Manipuri (Meitei)
}

# Display names for the judge prompt only -- not stored per-row.
_FLORES_LANG_NAMES = {
    "eng": "English",
    "hin": "Hindi",
    "ben": "Bengali",
    "tam": "Tamil",
    "tel": "Telugu",
    "mar": "Marathi",
    "guj": "Gujarati",
    "kan": "Kannada",
    "mal": "Malayalam",
    "pan": "Punjabi",
    "ory": "Odia",
    "asm": "Assamese",
    "urd": "Urdu",
    "npi": "Nepali",
    "san": "Sanskrit",
    "doi": "Dogri",
    "gom": "Konkani",
    "snd": "Sindhi",
    "brx": "Bodo",
    "kas": "Kashmiri",
    "mai": "Maithili",
    "mni": "Manipuri",
    "jpn": "Japanese",
    "kor": "Korean",
    "zho": "Chinese",
}


def _flores_subtag(lang_code: str) -> str:
    return lang_code.split("_", 1)[0]


def _tokenizer_for(tgt_lang: str) -> str:
    return _TOKENIZER_BY_FLORES_SUBTAG.get(_flores_subtag(tgt_lang), "13a")


def _lang_name(lang_code: str) -> str:
    return _FLORES_LANG_NAMES.get(_flores_subtag(lang_code), lang_code)


# --- Thinking-preamble handling ---------------------------------------------
# Reasoning models emit a pre-answer reasoning preamble wrapped in
# <think>...</think>. vLLM's reasoning parser strips the opening <think>
# tag but keeps the closing </think>, so the raw response looks like
#   "We need to translate ... </think>\nProlog"
# Drop the preamble before scoring, or both BLEU and the judge score are
# computed against the reasoning text instead of the translation.


def _strip_reasoning_preamble(text: str) -> str:
    """Remove a pre-answer reasoning preamble.

    Three cases:
      1. ``</think>`` present: return everything after the *last* occurrence
         (the actual answer, with the preamble dropped).
      2. ``<think>`` present but no ``</think>``: reasoning started but didn't
         close -- the model truncated mid-reasoning. Return empty string so
         the rollout counts as no-answer.
      3. Neither tag present: no inline reasoning preamble (e.g. the endpoint
         returned reasoning as a structured ``output[i].type="reasoning"``
         block and ``output_text`` already contains only the answer). Return
         the text unchanged.
    """
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1].lstrip("\n")
    if "<think>" in text:
        return ""
    return text


# --- Request / response shapes ------------------------------------------------


class TranslationWithJudgeResourcesServerConfig(BaseResourcesServerConfig):
    judge_model_server: ModelServerRef
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    strip_reasoning: bool = True


class TranslationWithJudgeRunRequest(BaseRunRequest):
    # Mirrors the curriculum dataset's own row schema (see
    # prepare_translation_curriculum.py) rather than inventing separate field
    # names -- ``prompt`` is the same messages list wrapped into
    # ``responses_create_params.input``, kept here too so verify() can pull
    # the raw source segment back out for the judge prompt.
    prompt: List[Dict[str, Any]]
    solution: str
    src_lang: str  # FLORES-200 code, e.g. "eng_Latn" -- language of `prompt`
    tgt_lang: str  # FLORES-200 code, e.g. "kan_Knda" -- language of `solution`
    direction: Optional[str] = None  # "src2tgt" | "tgt2src" -- provenance only, not used
    prompt_style: Optional[str] = None
    dataset_type: Optional[str] = None
    pass_rate: Optional[float] = None
    gen_failed: Optional[bool] = None
    # `__subset__` / `__split__` also ride along in the on-disk JSONL for
    # provenance, but pydantic reserves leading-and-trailing-underscore names
    # for private attributes, so they aren't declared as fields here.


class TranslationWithJudgeVerifyRequest(TranslationWithJudgeRunRequest, BaseVerifyRequest):
    pass


class JudgeEvaluation(BaseModel):
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    response: NeMoGymResponse


class TranslationWithJudgeVerifyResponse(TranslationWithJudgeVerifyRequest, BaseVerifyResponse):
    # Model's translation, post-strip-reasoning if enabled.
    generation: str
    # Judge's raw 0-100 quality score. None if the generation was empty or
    # the judge's response didn't contain a parseable score.
    judge_score: Optional[float] = None
    # Diagnostic-only metrics (0-100), not used for reward.
    sentence_bleu: float
    sentence_chrf: float
    judge_evaluation: Optional[JudgeEvaluation] = None


# --- Server -------------------------------------------------------------------


class TranslationWithJudgeResourcesServer(SimpleResourcesServer):
    # Adapted from Arena Hard / math_with_judge's equivalence-judge prompts,
    # rewritten as a direct 0-100 quality rating for translation adequacy +
    # fluency instead of a binary equivalence verdict.
    JUDGE_SYSTEM_MESSAGE: ClassVar[
        str
    ] = """You are an expert translation quality evaluator. You will be shown a source segment, the source and target languages, a reference (human) translation, and a candidate translation produced by an AI system. Rate the candidate translation's quality on a scale from 0 to 100:

- 100: perfect -- fully accurate and fluent, reads naturally in the target language
- 70-99: minor issues -- meaning is fully preserved, only small awkwardness or stylistic imperfections
- 40-69: moderate issues -- partial meaning loss, unnatural phrasing, or noticeable errors
- 1-39: major issues -- meaning is substantially distorted or the text is largely unreadable in the target language
- 0: no usable translation -- empty, in the wrong language, or unrelated to the source

Judge primarily on adequacy (is the source meaning preserved?) and fluency (does it read naturally in the target language?). The reference translation is one valid rendering, not the only correct one -- do not penalize a candidate solely for differing in wording from the reference, as long as the meaning and quality are preserved.

After your reasoning, end your response with a single line in exactly this format:
Score: <integer from 0 to 100>"""

    JUDGE_PROMPT_TEMPLATE: ClassVar[str] = """Source language: {source_lang_name}
Target language: {target_lang_name}

Source segment:
{source_text}

Reference translation:
{reference_translation}

Candidate translation:
{candidate_translation}"""

    _SCORE_RE: ClassVar[re.Pattern] = re.compile(r"score\s*[:=]\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))", re.IGNORECASE)

    config: TranslationWithJudgeResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        return super().setup_webserver()

    async def verify(self, body: TranslationWithJudgeVerifyRequest) -> TranslationWithJudgeVerifyResponse:
        raw = body.response.output_text or ""
        if self.config.strip_reasoning:
            raw = _strip_reasoning_preamble(raw)
        generation = raw.strip()

        if not generation:
            # A genuine policy failure (the model produced no translation), not an
            # infrastructure one -- stays an unmasked, real zero.
            return TranslationWithJudgeVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                generation="",
                judge_score=None,
                sentence_bleu=0.0,
                sentence_chrf=0.0,
                judge_evaluation=None,
            )

        tokenize = _tokenizer_for(body.tgt_lang)
        bleu = sentence_bleu(generation, [body.solution], tokenize=tokenize).score
        chrf = sentence_chrf(generation, [body.solution]).score

        source_text = body.prompt[-1]["content"] if body.prompt else ""
        judge_score, judge_evaluation = await self._judge_translation(
            source_text=source_text,
            reference_translation=body.solution,
            candidate_translation=generation,
            source_lang_name=_lang_name(body.src_lang),
            target_lang_name=_lang_name(body.tgt_lang),
        )
        reward = (judge_score / 100.0) if judge_score is not None else 0.0
        # An unparseable judge response is an infrastructure/judge failure, not a policy
        # one -- mask it so it is reported as lost coverage instead of a real zero score.
        judge_unparseable = judge_score is None

        return TranslationWithJudgeVerifyResponse(
            **body.model_dump(),
            reward=reward,
            generation=generation,
            judge_score=judge_score,
            sentence_bleu=bleu,
            sentence_chrf=chrf,
            judge_evaluation=judge_evaluation,
            mask_sample=judge_unparseable,
            failure_kind=JUDGE_UNPARSEABLE if judge_unparseable else None,
            failure_reason="Judge response contained no parseable `Score: <0-100>` line."
            if judge_unparseable
            else None,
        )

    async def _judge_translation(
        self,
        source_text: str,
        reference_translation: str,
        candidate_translation: str,
        source_lang_name: str,
        target_lang_name: str,
    ) -> Tuple[Optional[float], JudgeEvaluation]:
        config = self.config
        responses_create_params = config.judge_responses_create_params.model_copy(deep=True)

        judge_prompt = self.JUDGE_PROMPT_TEMPLATE.format(
            source_lang_name=source_lang_name,
            target_lang_name=target_lang_name,
            source_text=source_text,
            reference_translation=reference_translation,
            candidate_translation=candidate_translation,
        )
        responses_create_params.input = [
            NeMoGymEasyInputMessage(role="system", content=self.JUDGE_SYSTEM_MESSAGE),
            NeMoGymEasyInputMessage(role="user", content=judge_prompt),
        ]

        response = await self.server_client.post(
            server_name=config.judge_model_server.name,
            url_path="/v1/responses",
            json=responses_create_params,
        )
        judge_response = NeMoGymResponse.model_validate(await get_response_json(response))
        judge_evaluation = JudgeEvaluation(responses_create_params=responses_create_params, response=judge_response)

        return self._parse_judge_score(judge_response.output_text or ""), judge_evaluation

    @classmethod
    def _parse_judge_score(cls, judge_text: str) -> Optional[float]:
        """Parse the last ``Score: X`` occurrence and clip to [0, 100].

        Returns None if the judge's response contains no parseable score --
        verify() treats that as reward=0.0 rather than guessing.
        """
        matches = list(cls._SCORE_RE.finditer(judge_text))
        if not matches:
            return None
        score = float(matches[-1].group(1))
        return max(0.0, min(100.0, score))

    # --- Aggregate metrics ---------------------------------------------------

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Aggregate per-row judge_score / sentence_bleu / sentence_chrf by language pair.

        Output keys, mirroring wmt_translation's per-pair + cross-pair shape:

          <src>-><tgt>/judge_score  <src>-><tgt>/judge_score_std_dev_across_runs
          <src>-><tgt>/bleu  <src>-><tgt>/chrf
          xx->xx/judge_score  <src>->xx/judge_score  xx-><tgt>/judge_score
          ... same pattern for /bleu and /chrf
        """
        if not tasks:
            return {}

        rollout_counts = [len(r) for r in tasks]
        max_k = min(rollout_counts) if rollout_counts else 0

        per_pair: Dict[Tuple[str, str], List[List[Dict[str, float]]]] = defaultdict(
            lambda: [list() for _ in range(max_k)]
        )
        for task_rollouts in tasks:
            for k, rollout in enumerate(task_rollouts):
                if k >= max_k:
                    break
                src = rollout.get("src_lang")
                tgt = rollout.get("tgt_lang")
                if not src or not tgt:
                    continue
                per_pair[(src, tgt)][k].append(
                    {
                        "judge_score": rollout.get("judge_score"),
                        "bleu": rollout.get("sentence_bleu"),
                        "chrf": rollout.get("sentence_chrf"),
                    }
                )

        def _mean_std(values: List[float]) -> Tuple[float, float]:
            if not values:
                return (0.0, 0.0)
            n = len(values)
            mean = sum(values) / n
            if n < 2:
                return (mean, 0.0)
            var = sum((v - mean) ** 2 for v in values) / n
            return (mean, var**0.5)

        # Per-run mean per (pair, k, metric) -- the corpus-level average of that
        # metric across all rows sharing a rollout index, for a given pair.
        per_run_mean: Dict[Tuple[str, str], Dict[str, List[float]]] = {}
        for pair, runs in per_pair.items():
            per_metric: Dict[str, List[float]] = {"judge_score": [], "bleu": [], "chrf": []}
            for run_rows in runs:
                for metric in per_metric:
                    # judge_score is None here only for the empty-generation case (a real
                    # policy failure, unmasked) -- unparseable-judge rows are mask_sample=True
                    # and never reach this method (the framework filters them out upstream).
                    # Counts as 0 rather than being dropped, matching verify()'s own
                    # reward=0.0 treatment for that case.
                    values = [row[metric] if row[metric] is not None else 0.0 for row in run_rows]
                    if values:
                        per_metric[metric].append(sum(values) / len(values))
            per_run_mean[pair] = per_metric

        metrics: Dict[str, Any] = {}
        all_pairs = sorted(per_pair.keys())

        for src, tgt in all_pairs:
            pair_label = f"{src}->{tgt}"
            for metric, key in (("judge_score", "judge_score"), ("bleu", "bleu"), ("chrf", "chrf")):
                m, s = _mean_std(per_run_mean[(src, tgt)][metric])
                metrics[f"{pair_label}/{key}"] = m
                metrics[f"{pair_label}/{key}_std_dev_across_runs"] = s

        def _aggregate(pair_filter, metric: str) -> List[float]:
            filtered_pairs = [p for p in all_pairs if pair_filter(p)]
            if not filtered_pairs:
                return []
            min_runs = min(len(per_run_mean[p][metric]) for p in filtered_pairs)
            runs: List[float] = []
            for k in range(min_runs):
                per_pair_k = [per_run_mean[p][metric][k] for p in filtered_pairs if k < len(per_run_mean[p][metric])]
                if per_pair_k:
                    runs.append(sum(per_pair_k) / len(per_pair_k))
            return runs

        src_langs = sorted({p[0] for p in all_pairs})
        tgt_langs = sorted({p[1] for p in all_pairs})

        for metric, key in (("judge_score", "judge_score"), ("bleu", "bleu"), ("chrf", "chrf")):
            m, s = _mean_std(_aggregate(lambda p: True, metric))
            metrics[f"xx->xx/{key}"] = m
            metrics[f"xx->xx/{key}_std_dev_across_runs"] = s

            for src in src_langs:
                m, s = _mean_std(_aggregate(lambda p, _s=src: p[0] == _s, metric))
                metrics[f"{src}->xx/{key}"] = m
                metrics[f"{src}->xx/{key}_std_dev_across_runs"] = s
            for tgt in tgt_langs:
                m, s = _mean_std(_aggregate(lambda p, _t=tgt: p[1] == _t, metric))
                metrics[f"xx->{tgt}/{key}"] = m
                metrics[f"xx->{tgt}/{key}_std_dev_across_runs"] = s

        return metrics

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Headline metrics: global + per-source judge score, plus BLEU/chrF for context."""
        keys_of_interest = (
            "xx->xx/judge_score",
            "eng_Latn->xx/judge_score",
            "xx->xx/bleu",
            "eng_Latn->xx/bleu",
            "xx->xx/chrf",
            "eng_Latn->xx/chrf",
        )
        return {k: agent_metrics[k] for k in keys_of_interest if k in agent_metrics}


if __name__ == "__main__":
    TranslationWithJudgeResourcesServer.run_webserver()
