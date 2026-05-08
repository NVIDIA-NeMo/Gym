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

"""
SimpleQA Resources Server.

Evaluates short-form factual QA using an LLM judge that grades answers on a
three-tier scale: CORRECT (A), INCORRECT (B), NOT_ATTEMPTED (C). Mirrors the
SimpleQA-Verified judge protocol from
https://www.kaggle.com/code/nanliao7/simpleqa-verified-benchmark-starter-code

Computes:
- reward: 1.0 for CORRECT, 0.0 otherwise
- per-rollout {is_correct, is_incorrect, is_not_attempted} indicators
- pass@k metrics for each verdict tier
- F1 = 2*P*R / (P + R) where P = correct/(correct+incorrect),
                         R = correct/total
- accuracy_given_attempted = correct/(correct+incorrect)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import yaml
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.reward_profile import compute_pass_majority_metrics, highest_k_metrics


_DEFAULT_JUDGE_PROMPT_PATH = str(Path(__file__).parent / "prompts" / "judge.yaml")


def extract_text_from_response(response: NeMoGymResponse) -> str:
    """Return the last assistant message text.

    Reasoning-trace handling is the model server's responsibility — start
    vLLM with --reasoning-parser <name> so <think>/<thinking> blocks are
    split into a separate output item before the response leaves the
    server. This function returns the message content verbatim.
    """
    for output in reversed(response.output):
        if getattr(output, "type", None) == "message" and getattr(output, "role", None) == "assistant":
            content = getattr(output, "content", None)
            texts: list[str] = []
            if isinstance(content, list):
                for c in content:
                    text = getattr(c, "text", None)
                    if isinstance(text, str):
                        texts.append(text)
            elif isinstance(content, str):
                texts = [content]
            if texts:
                return "\n".join(texts).strip()
    return ""


_VALID_GRADES = ("A", "B", "C")


def parse_judge_grade(judge_text: str) -> str:
    """Parse the single-letter grade (A/B/C) from the judge's response.

    Falls back to "C" (NOT_ATTEMPTED) when the output cannot be reliably
    parsed — matches Skills' SimpleQAMetrics behavior of treating
    unparseable judgements as NOT_ATTEMPTED (DEFAULT_GRADE_IF_UNPARSEABLE = "C").
    """
    # Skills' is_correct_judgement_label_matching matches either the whole
    # cleaned string or judgement[0] — covered by the cleaned[:1] check on
    # the stripped string and on the last line.
    cleaned = judge_text.strip()
    if cleaned[:1] in _VALID_GRADES:
        return cleaned[:1]

    last_line = cleaned.rsplit("\n", 1)[-1].strip()
    if last_line[:1] in _VALID_GRADES:
        return last_line[:1]

    for letter in _VALID_GRADES:
        if letter in cleaned:
            return letter

    return "C"


# ---------------------------------------------------------------------------
# Config, request / response models
# ---------------------------------------------------------------------------


class SimpleQAConfig(BaseResourcesServerConfig):
    judge_model_server: ModelServerRef
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming

    judge_prompt_path: str = Field(
        default=_DEFAULT_JUDGE_PROMPT_PATH,
        description="Path to a YAML file containing the judge prompt under a 'user' key. "
        "Placeholders: {question}, {expected_answer}, {generation}.",
    )
    use_chat_completions_for_judge: bool = Field(
        default=False,
        description="Use /v1/chat/completions instead of /v1/responses for the judge model. "
        "Required for endpoints that don't support the OpenAI Responses API (e.g., NVIDIA API).",
    )


class SimpleQARunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    id: Optional[Union[int, str]] = None
    question: Optional[str] = None
    expected_answer: Optional[str] = None


class SimpleQAVerifyRequest(SimpleQARunRequest, BaseVerifyRequest):
    pass


class SimpleQAVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    extracted_answer: Optional[str] = None
    expected_answer: Optional[str] = None
    verdict: Optional[str] = None
    judge_output: Optional[str] = None
    is_correct: float = 0.0
    is_incorrect: float = 0.0
    is_not_attempted: float = 0.0


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class SimpleQAServer(SimpleResourcesServer):
    config: SimpleQAConfig

    def model_post_init(self, context):
        prompt_data = yaml.safe_load(Path(self.config.judge_prompt_path).read_text())
        self._judge_prompt_template = prompt_data["user"]
        return super().model_post_init(context)

    @staticmethod
    def _score_fn(result: dict) -> dict:
        """Score function for compute_pass_majority_metrics.

        Maps SimpleQA judge verdicts to named scores. ``correct`` is the
        primary pass@k signal (matches Skills' SimpleQAMetrics). The other
        two channels expose pass@k for INCORRECT and NOT_ATTEMPTED so the
        full per-tier breakdown is available.
        """
        verdict = result.get("verdict", "not_attempted")
        return {
            "correct": 1.0 if verdict == "correct" else 0.0,
            "incorrect": 1.0 if verdict == "incorrect" else 0.0,
            "not_attempted": 1.0 if verdict == "not_attempted" else 0.0,
        }

    def compute_metrics(self, tasks: List[List[dict]]) -> dict:
        """Compute SimpleQA metrics: pass@k per tier, F1, accuracy_given_attempted."""
        metrics, _, _, _ = compute_pass_majority_metrics(
            tasks,
            score_fn=self._score_fn,
            answer_key="extracted_answer",
        )

        # Derive F1 + accuracy_given_attempted at every aggregation level
        # (matches Skills' SimpleQAMetrics.get_metrics). At each prefix, P is
        # the per-attempted-question accuracy (= correct/(correct+incorrect))
        # and R is the per-question accuracy (= correct); F1 = 2PR / (P+R).
        # compute_pass_majority_metrics returns values in 0-100; rescale to
        # 0-1 for the F1 calculation, then re-scale the result back.
        for key in list(metrics.keys()):
            if "/correct" not in key:
                continue
            agg = key.rsplit("/correct", 1)[0]
            correct_key = f"{agg}/correct"
            incorrect_key = f"{agg}/incorrect"
            if incorrect_key not in metrics:
                continue
            correct = metrics[correct_key] / 100.0
            incorrect = metrics[incorrect_key] / 100.0
            attempted = correct + incorrect
            accuracy_given_attempted = correct / attempted if attempted > 0 else 0.0
            denom = accuracy_given_attempted + correct
            f1 = (2 * accuracy_given_attempted * correct / denom) if denom > 0 else 0.0
            metrics[f"{agg}/f1"] = 100.0 * f1
            metrics[f"{agg}/accuracy_given_attempted"] = 100.0 * accuracy_given_attempted

        return metrics

    def get_key_metrics(self, agent_metrics: dict) -> dict:
        """Select headline metrics for SimpleQA."""
        key: dict = {}
        for name in ("mean/input_tokens", "mean/output_tokens"):
            if name in agent_metrics:
                key[name] = agent_metrics[name]
        key.update(highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]"))
        key.update(highest_k_metrics(agent_metrics, "pass@{k}", exclude_names=["no_answer"]))
        return key

    async def verify(self, body: SimpleQAVerifyRequest) -> SimpleQAVerifyResponse:
        # Reasoning-trace handling is the model server's responsibility:
        # start vLLM with --reasoning-parser <name> so <think>...</think>
        # blocks are split off before the response reaches us.
        generation = extract_text_from_response(body.response)

        question = body.question or ""
        expected_answer = body.expected_answer or ""

        judge_prompt = self._judge_prompt_template.format(
            question=question,
            expected_answer=expected_answer,
            generation=generation,
        )

        if self.config.use_chat_completions_for_judge:
            chat_params = NeMoGymChatCompletionCreateParamsNonStreaming(
                messages=[{"role": "user", "content": judge_prompt}],
                max_tokens=self.config.judge_responses_create_params.max_output_tokens or 64,
                temperature=self.config.judge_responses_create_params.temperature or 0.0,
                top_p=self.config.judge_responses_create_params.top_p or 1.0,
            )
            response_obj = await self.server_client.post(
                server_name=self.config.judge_model_server.name,
                url_path="/v1/chat/completions",
                json=chat_params,
            )
            chat_response = NeMoGymChatCompletion.model_validate(await response_obj.json())
            content = chat_response.choices[0].message.content if chat_response.choices else None
            judge_text = content.strip() if content else ""
        else:
            msgs: List[NeMoGymEasyInputMessage] = [
                NeMoGymEasyInputMessage(role="user", content=judge_prompt),
            ]
            request_params = self.config.judge_responses_create_params.model_copy(deep=True)
            request_params.input = msgs

            response_obj = await self.server_client.post(
                server_name=self.config.judge_model_server.name,
                url_path="/v1/responses",
                json=request_params,
            )
            judge_response = NeMoGymResponse.model_validate(await response_obj.json())
            judge_text = extract_text_from_response(judge_response)

        grade = parse_judge_grade(judge_text)

        if grade == "A":
            verdict = "correct"
            reward = 1.0
        elif grade == "B":
            verdict = "incorrect"
            reward = 0.0
        else:
            verdict = "not_attempted"
            reward = 0.0

        is_correct = 1.0 if verdict == "correct" else 0.0
        is_incorrect = 1.0 if verdict == "incorrect" else 0.0
        is_not_attempted = 1.0 if verdict == "not_attempted" else 0.0

        return SimpleQAVerifyResponse(
            **body.model_dump(exclude={"expected_answer", "extracted_answer"}),
            reward=reward,
            extracted_answer=generation,
            expected_answer=expected_answer,
            verdict=verdict,
            judge_output=judge_text,
            is_correct=is_correct,
            is_incorrect=is_incorrect,
            is_not_attempted=is_not_attempted,
        )


if __name__ == "__main__":
    SimpleQAServer.run_webserver()
