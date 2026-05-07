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
FrontierScience Judge Resources Server.

Single-pass LLM-judge verifier for free-form science olympiad answers.
Mirrors NeMo Skills' `frontierscience-olympiad` benchmark verification:
the judge sees the problem, reference answer, and attempted answer, then
emits ``Judgement: YES`` or ``Judgement: NO`` on its final line. The
verdict is parsed by anchoring on the last "Judgement:" occurrence.

The judge prompt is loaded from a YAML file at startup and is configurable
via the ``judge_prompt_path`` config field — defaults to the verbatim
Skills prompt under ``prompts/judge.yaml``.

Source: https://cdn.openai.com/pdf/2fcd284c-b468-4c21-8ee0-7a783933efcc/frontierscience-paper.pdf
"""

from __future__ import annotations

import re
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
from nemo_gym.reward_profile import (
    compute_pass_majority_metrics,
    compute_subset_metrics,
    highest_k_metrics,
)


_DEFAULT_JUDGE_PROMPT_PATH = str(Path(__file__).parent / "prompts" / "judge.yaml")

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINKING_TAG_RE = re.compile(r"<thinking>.*?</thinking>", re.DOTALL)
_JUDGEMENT_RE = re.compile(r"Judgement:\s*(YES|NO)", re.IGNORECASE)


def _strip_thinking_traces(text: str) -> str:
    """Remove <think>...</think> and <thinking>...</thinking> blocks."""
    text = _THINK_TAG_RE.sub("", text)
    text = _THINKING_TAG_RE.sub("", text)
    text = re.sub(r"^.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"^.*?</thinking>", "", text, flags=re.DOTALL)
    return text.strip()


def extract_text_from_response(response: NeMoGymResponse, strip_thinking: bool = True) -> str:
    """Return the last assistant message text, optionally stripping thinking traces."""
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
                full_text = "\n".join(texts).strip()
                return _strip_thinking_traces(full_text) if strip_thinking else full_text
    return ""


def parse_judgement(judge_text: str) -> Optional[str]:
    """Parse ``Judgement: YES`` / ``Judgement: NO`` from the judge response.

    Returns ``"YES"``, ``"NO"``, or ``None`` if neither marker is present.
    The last occurrence wins, mirroring Skills' is_correct_judgement which
    looks at the final line.
    """
    if not judge_text:
        return None
    matches = list(_JUDGEMENT_RE.finditer(judge_text))
    if not matches:
        return None
    return matches[-1].group(1).upper()


class FrontierScienceJudgeConfig(BaseResourcesServerConfig):
    judge_model_server: ModelServerRef
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming

    judge_prompt_path: str = Field(
        default=_DEFAULT_JUDGE_PROMPT_PATH,
        description=(
            "Path to a YAML file containing the judge prompt under a 'user' key. "
            "Placeholders: {question}, {expected_answer}, {generation}."
        ),
    )
    use_chat_completions_for_judge: bool = Field(
        default=False,
        description=(
            "Use /v1/chat/completions instead of /v1/responses for the judge model. "
            "Required for endpoints that don't support the OpenAI Responses API."
        ),
    )


class FrontierScienceJudgeRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    id: Optional[Union[int, str]] = None
    subject: Optional[str] = None
    question: Optional[str] = None
    expected_answer: Optional[str] = None


class FrontierScienceJudgeVerifyRequest(FrontierScienceJudgeRunRequest, BaseVerifyRequest):
    pass


class FrontierScienceJudgeVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    extracted_answer: Optional[str] = None
    expected_answer: Optional[str] = None
    verdict: Optional[str] = None
    judge_output: Optional[str] = None


class FrontierScienceJudgeServer(SimpleResourcesServer):
    config: FrontierScienceJudgeConfig

    def model_post_init(self, context):
        prompt_data = yaml.safe_load(Path(self.config.judge_prompt_path).read_text())
        self._judge_prompt_template = prompt_data["user"]
        return super().model_post_init(context)

    @staticmethod
    def _score_fn(result: dict) -> dict:
        """Map verify response to a single named score for pass@k metrics."""
        return {"accuracy": float(result.get("reward", 0.0))}

    def compute_metrics(self, tasks: List[List[dict]]) -> dict:
        """Compute pass@k / majority@k plus per-subject stratification.

        The Skills `MathMetrics` evaluator emits per-subject pass@k
        breakdowns via ``subset_for_metrics`` (which is the dataset's
        ``subject`` field). We mirror that with ``compute_subset_metrics``
        keyed on the same field — the per-row ``subject`` value is
        propagated through the rollout dict by the agent.
        """
        metrics, _, _, _ = compute_pass_majority_metrics(
            tasks,
            score_fn=self._score_fn,
            answer_key="extracted_answer",
        )
        subset_metrics = compute_subset_metrics(
            tasks,
            subset_key="subject",
            score_fn=self._score_fn,
            answer_key="extracted_answer",
        )
        metrics.update(subset_metrics)
        return metrics

    def get_key_metrics(self, agent_metrics: dict) -> dict:
        """Select headline metrics for frontierscience-olympiad."""
        key: dict = {}
        for name in ("mean/input_tokens", "mean/output_tokens"):
            if name in agent_metrics:
                key[name] = agent_metrics[name]
        key.update(highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]"))
        key.update(highest_k_metrics(agent_metrics, "pass@{k}", exclude_names=["no_answer"]))
        key.update(highest_k_metrics(agent_metrics, "majority@{k}", exclude_names=["no_answer"]))
        return key

    async def verify(self, body: FrontierScienceJudgeVerifyRequest) -> FrontierScienceJudgeVerifyResponse:
        # Skills' parse_reasoning=True: when </think> is missing but the
        # model started reasoning (<think> present), treat as no answer
        # (truncated mid-CoT). With --reasoning-parser deepseek_r1 vLLM
        # already strips this; the post-process keeps the server correct
        # against unparsed endpoints.
        raw_text = extract_text_from_response(body.response, strip_thinking=False)
        generation = extract_text_from_response(body.response)
        has_open = "<think>" in raw_text or "<thinking>" in raw_text
        has_close = "</think>" in raw_text or "</thinking>" in raw_text
        if has_open and not has_close:
            generation = ""

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
                max_tokens=self.config.judge_responses_create_params.max_output_tokens or 2048,
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

        verdict = parse_judgement(judge_text)
        reward = 1.0 if verdict == "YES" else 0.0

        return FrontierScienceJudgeVerifyResponse(
            **body.model_dump(exclude={"expected_answer", "extracted_answer"}),
            reward=reward,
            extracted_answer=generation if generation else None,
            expected_answer=expected_answer,
            verdict=verdict,
            judge_output=judge_text,
        )


if __name__ == "__main__":
    FrontierScienceJudgeServer.run_webserver()
