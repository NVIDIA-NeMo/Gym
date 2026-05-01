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
"""AudioBench-judge resources server: LLM-as-a-judge for open-ended audio tasks.

Used by the AudioBench "judge" half (instruction-following on audio, audio QA,
emotion/gender/accent recognition phrased as open-ended QA, etc.). Pairs an
audio benchmark row's ``expected_answer`` and ``question`` with the model's
generated answer and asks an LLM judge to rate the model's answer 0–5 on
alignment with the reference.

The rating prompt is byte-for-byte the AudioBench upstream prompt, also
mirrored verbatim in NeMo Skills' ``judge/audiobench.yaml``. Score 0–5 is
mapped to ``judge_score = avg(rating) * 20`` (0–100) for headline reporting,
and ``is_correct = rating >= 3`` for binary pass@k accounting — same shape as
NeMo Skills' ``audio_metrics.AudioMetrics._extract_judge_result``.
"""

import logging
import re
from typing import Any, ClassVar, Dict, List, Optional

from pydantic import BaseModel

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.reward_profile import compute_pass_majority_metrics, highest_k_metrics
from nemo_gym.server_utils import get_response_json


LOG = logging.getLogger(__name__)


# Rating output (primary AudioBench format) and the legacy binary
# Judgement: Yes/No fallback that ``audiobench_binary.yaml`` produces.
RATING_PATTERN: re.Pattern[str] = re.compile(
    r"Rating:\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE
)
JUDGEMENT_PATTERN: re.Pattern[str] = re.compile(
    r"Judgement:\s*(Yes|No)", re.IGNORECASE
)


def _extract_assistant_text(response: NeMoGymResponse) -> str:
    """Concatenate every ``output_text`` part of a Responses-API response.

    Same shape as ``asr_with_pc.app._extract_assistant_text``; we duplicate
    the helper rather than import to keep server isolation (each server has
    its own venv).
    """
    parts: List[str] = []
    for output_item in response.output:
        if output_item.type != "message":
            continue
        for content_item in output_item.content:
            if content_item.type != "output_text":
                continue
            parts.append(content_item.text)
    return "".join(parts)


def extract_judge_result(judgement_text: str) -> tuple[bool, float]:
    """Extract ``(is_correct, rating)`` from judge output text.

    Mirrors NeMo Skills' ``AudioMetrics._extract_judge_result`` byte-for-byte:

      * ``Rating: X`` (X in 0..5) — primary AudioBench format
      * ``Judgement: Yes/No``    — legacy binary format mapped to 5.0/0.0
      * fallback plain ``yes``/``no`` anywhere in the text
      * otherwise: ``(False, 0.0)``
    """
    rating_match = RATING_PATTERN.search(judgement_text)
    if rating_match:
        rating = float(rating_match.group(1))
        rating = max(0.0, min(5.0, rating))
        return rating >= 3.0, rating

    judgement_match = JUDGEMENT_PATTERN.search(judgement_text)
    if judgement_match:
        is_yes = judgement_match.group(1).lower() == "yes"
        return is_yes, 5.0 if is_yes else 0.0

    if re.search(r"\byes\b", judgement_text, re.IGNORECASE):
        return True, 5.0
    if re.search(r"\bno\b", judgement_text, re.IGNORECASE):
        return False, 0.0

    return False, 0.0


class AudioBenchJudgeConfig(BaseResourcesServerConfig):
    judge_model_server: ModelServerRef
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming


class AudioBenchJudgeRunRequest(BaseRunRequest):
    question: str
    expected_answer: str


class AudioBenchJudgeVerifyRequest(AudioBenchJudgeRunRequest, BaseVerifyRequest):
    pass


class JudgeEvaluation(BaseModel):
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    response: NeMoGymResponse


class AudioBenchJudgeVerifyResponse(BaseVerifyResponse):
    expected_answer: str
    question: str
    generation: str
    judge_rating: float
    judge_correct: bool
    judge_evaluation: Optional[JudgeEvaluation]


class AudioBenchJudgeResourcesServer(SimpleResourcesServer):
    """Open-ended audio QA scoring via an LLM judge with 0–5 rating.

    Verbatim copy of AudioBench's ``llama3_70b_as_judge_binary``-derived
    rating prompt — byte-equivalent with NeMo Skills' ``judge/audiobench.yaml``.
    """

    # Byte-for-byte the same template as NeMo Skills'
    # ``nemo_skills/prompt/config/judge/audiobench.yaml``.
    JUDGE_PROMPT_TEMPLATE: ClassVar[
        str
    ] = """[Reference Answer]
{expected_answer}

[Model Answer]
{generation}

[Question]
{question}

[Task]
Rate the model's answer based on its alignment with the reference answer, focusing on accuracy and relevance to the reference provided. Please be critical on the details. If the model response is something like 'cannot decide', please rate as 0.
Criteria: Assess if the model's response mirrors the reference in terms of content, accuracy, and relevance.
Score0: The answer is refusing to give concrete results, providing something like 'cannot decide'.
Score0: The answer is completely misaligned, providing incorrect or irrelevant information compared to the reference.
Score1: The answer shows minimal alignment, often misunderstanding or providing irrelevant details unrelated to the reference.
Score2: The answer recognizes the topic but diverges significantly from the reference in accuracy or relevance.
Score3: The answer aligns with the reference generally but lacks detail or precise accuracy in some aspects.
Score4: The answer is mostly accurate and relevant, closely following the reference but could be clearer or more detailed.
Score5: The answer is highly accurate, detailed, and matches the reference answer perfectly, capturing its essence and detail.

Your response should be formatted as follows:
Explanation: (Provide a concise explanation of your rating, comparing the reference answer with the model's response. "The reference answer is [XXX], while the model's answer is [YYY]. I think ...")
Rating: (int)"""

    config: AudioBenchJudgeConfig

    async def verify(self, body: AudioBenchJudgeVerifyRequest) -> AudioBenchJudgeVerifyResponse:
        generation = _extract_assistant_text(body.response).strip()

        judge_rating, judge_correct, judge_evaluation = await self._run_judge(
            question=body.question,
            expected_answer=body.expected_answer,
            generation=generation,
        )

        return AudioBenchJudgeVerifyResponse(
            **body.model_dump(),
            generation=generation,
            judge_rating=judge_rating,
            judge_correct=judge_correct,
            judge_evaluation=judge_evaluation,
            reward=1.0 if judge_correct else 0.0,
        )

    async def _run_judge(
        self,
        question: str,
        expected_answer: str,
        generation: str,
    ) -> tuple[float, bool, Optional[JudgeEvaluation]]:
        """Call the judge model once with the AudioBench rating prompt.

        AudioBench's reference judge does a single forward pass (no two-sided
        order swap like Arena Hard). We mirror that — extra calls per row are
        non-trivial cost on large eval sets like 34-dataset audiobench.judge.
        """
        if not generation:
            # Empty generation → rating 0; skip the LLM call to save tokens.
            return 0.0, False, None

        config = self.config
        responses_create_params = config.judge_responses_create_params.model_copy(deep=True)

        judge_prompt = self.JUDGE_PROMPT_TEMPLATE.format(
            question=question,
            expected_answer=expected_answer,
            generation=generation,
        )
        responses_create_params.input = [
            NeMoGymEasyInputMessage(
                role="user",
                content=judge_prompt,
            ),
        ]

        try:
            response = await self.server_client.post(
                server_name=config.judge_model_server.name,
                url_path="/v1/responses",
                json=responses_create_params,
            )
            judge_response = NeMoGymResponse.model_validate(await get_response_json(response))
        except Exception as e:  # noqa: BLE001
            # If the judge call itself fails (auth, 429, etc.), treat as
            # incorrect with rating 0 rather than poisoning the whole eval.
            LOG.warning("audiobench_judge: judge call failed (%s); scoring rating=0", e)
            return 0.0, False, None

        judge_evaluation = JudgeEvaluation(
            responses_create_params=responses_create_params, response=judge_response
        )

        judge_text = _extract_assistant_text(judge_response)
        is_correct, rating = extract_judge_result(judge_text)
        return rating, is_correct, judge_evaluation

    # ──────────────────────────────────────────────────────────────────────
    # Aggregate metrics
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _score_fn(r: dict) -> Dict[str, Any]:
        """Per-rollout scores: judge accuracy and 0–5 rating.

        ``compute_pass_majority_metrics`` multiplies score values by 100, so
        we return ``rating / 5`` for the AudioBench-style ``judge_score``
        — yields 0–100 in the aggregated metrics, matching AudioBench's
        ``avg_rating * 20`` headline formula.
        """
        scores: Dict[str, Any] = {
            "accuracy": float(r.get("judge_correct", False)),
        }
        rating = r.get("judge_rating")
        if rating is not None:
            # 0..5 → 0..1 here; framework scales to 0..100 in the aggregate.
            scores["judge_score"] = float(rating) / 5.0
        gen = (r.get("generation") or "").strip()
        scores["no_answer"] = 0.0 if gen else 1.0
        return scores

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        return compute_pass_majority_metrics(
            tasks,
            score_fn=self._score_fn,
            answer_key=None,
        )[0]

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Headline metrics: pass@k accuracy + AudioBench-style judge_score."""
        key: Dict[str, Any] = {}
        for name in ("mean/input_tokens", "mean/output_tokens"):
            if name in agent_metrics:
                key[name] = agent_metrics[name]

        key.update(highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]"))
        key.update(highest_k_metrics(agent_metrics, "pass@{k}", exclude_names=["no_answer"]))
        return key


if __name__ == "__main__":
    AudioBenchJudgeResourcesServer.run_webserver()
