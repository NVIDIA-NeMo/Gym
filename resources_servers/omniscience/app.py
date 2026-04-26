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

"""
Omniscience Resources Server.

Evaluates factual knowledge using an LLM judge that grades answers on a
four-tier scale: CORRECT (A), INCORRECT (B), PARTIAL_ANSWER (C),
NOT_ATTEMPTED (D).

Computes:
- reward: 1.0 for CORRECT, 0.0 otherwise
- omniscience_index: (correct - incorrect) / total  (per-sample: +1, 0, or -1)
- is_hallucination: 1.0 when INCORRECT (confident wrong answer)

Based on the AA-Omniscience benchmark:
https://huggingface.co/datasets/ArtificialAnalysis/AA-Omniscience-Public
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Union

import yaml
from fastapi import FastAPI
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


# ---------------------------------------------------------------------------
# Default judge prompt path (relative to this file)
# ---------------------------------------------------------------------------

_DEFAULT_JUDGE_PROMPT_PATH = str(Path(__file__).parent / "prompts" / "judge.yaml")


# ---------------------------------------------------------------------------
# Thinking-trace stripping
# ---------------------------------------------------------------------------

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINKING_TAG_RE = re.compile(r"<thinking>.*?</thinking>", re.DOTALL)


def _strip_thinking_traces(text: str) -> str:
    """Remove <think>...</think> and <thinking>...</thinking> blocks."""
    text = _THINK_TAG_RE.sub("", text)
    text = _THINKING_TAG_RE.sub("", text)
    text = re.sub(r"^.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"^.*?</thinking>", "", text, flags=re.DOTALL)
    return text.strip()


def extract_text_from_response(response: NeMoGymResponse, strip_thinking: bool = True) -> str:
    """Return the last assistant message text.

    Args:
        response: The model response.
        strip_thinking: If True, remove <think>/<thinking> blocks (default).
            Set False to preserve reasoning traces — needed when the judge
            should see the same raw text that Skills passes with
            parse_reasoning=False.
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
                full_text = "\n".join(texts).strip()
                return _strip_thinking_traces(full_text) if strip_thinking else full_text
    return ""


def parse_judge_grade(judge_text: str) -> str:
    """Parse the single-letter grade (A/B/C/D) from the judge's response.

    Returns "A", "B", "C", or "D". Falls back to "B" (INCORRECT) when the
    output cannot be reliably parsed.
    """
    cleaned = judge_text.strip()
    if cleaned in ("A", "B", "C", "D"):
        return cleaned

    last_line = cleaned.rsplit("\n", 1)[-1].strip()
    for letter in ("A", "B", "C", "D"):
        if last_line == letter:
            return letter

    for letter in ("A", "B", "C", "D"):
        if letter in cleaned:
            return letter

    return "B"


# ---------------------------------------------------------------------------
# Config, request / response models
# ---------------------------------------------------------------------------


class OmniscienceConfig(BaseResourcesServerConfig):
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


class OmniscienceRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    id: Optional[Union[int, str]] = None
    domain: Optional[str] = None
    topic: Optional[str] = None
    question: Optional[str] = None
    expected_answer: Optional[str] = None


class OmniscienceVerifyRequest(OmniscienceRunRequest, BaseVerifyRequest):
    pass


class OmniscienceVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    extracted_answer: Optional[str] = None
    expected_answer: Optional[str] = None
    verdict: Optional[str] = None
    judge_output: Optional[str] = None
    is_correct: float = 0.0
    is_incorrect: float = 0.0
    is_partial: float = 0.0
    is_not_attempted: float = 0.0
    omniscience_index: float = 0.0
    is_hallucination: float = 0.0


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class OmniscienceServer(SimpleResourcesServer):
    config: OmniscienceConfig

    def model_post_init(self, context):
        prompt_data = yaml.safe_load(Path(self.config.judge_prompt_path).read_text())
        self._judge_prompt_template = prompt_data["user"]
        return super().model_post_init(context)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        return app

    @staticmethod
    def _omni_score_fn(result: dict) -> dict:
        """Score function for compute_pass_majority_metrics.

        Maps judge verdicts to named scores matching Skills' OmniMetrics.
        """
        verdict = result.get("verdict", "incorrect")
        return {
            "judge_correct": 1.0 if verdict == "correct" else 0.0,
            "judge_incorrect": -1.0 if verdict == "incorrect" else 0.0,
            "judge_partially_correct": 1.0 if verdict == "partial" else 0.0,
            "judge_abstained": 1.0 if verdict == "not_attempted" else 0.0,
        }

    def compute_metrics(self, tasks: List[List[dict]]) -> dict:
        """Compute omniscience metrics: pass@k, omni_index, hallucination rate."""
        metrics, _, _, _ = compute_pass_majority_metrics(
            tasks,
            score_fn=self._omni_score_fn,
            answer_key="extracted_answer",
        )

        # Compute derived omniscience metrics for each aggregation level
        for prefix in list(metrics.keys()):
            if "/judge_correct" in prefix:
                agg = prefix.rsplit("/judge_correct", 1)[0]
                correct = metrics.get(f"{agg}/judge_correct", 0)
                incorrect = -metrics.get(f"{agg}/judge_incorrect", 0)
                non_correct = (
                    incorrect
                    + metrics.get(f"{agg}/judge_partially_correct", 0)
                    + metrics.get(f"{agg}/judge_abstained", 0)
                )
                metrics[f"{agg}/judge_omni_index"] = correct - incorrect
                metrics[f"{agg}/judge_omni_hallucination"] = 100 * incorrect / non_correct if non_correct > 0 else 0

        return metrics

    def get_key_metrics(self, agent_metrics: dict) -> dict:
        """Select headline metrics for omniscience benchmark."""
        key: dict = {}
        for name in ("mean/input_tokens", "mean/output_tokens"):
            if name in agent_metrics:
                key[name] = agent_metrics[name]
        key.update(highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]"))
        key.update(highest_k_metrics(agent_metrics, "pass@{k}", exclude_names=["no_answer"]))
        return key

    async def verify(self, body: OmniscienceVerifyRequest) -> OmniscienceVerifyResponse:
        # Match Skills' parse_reasoning=True behavior:
        # 1. If </think> present: strip reasoning, keep answer after </think> (done by _strip_thinking_traces)
        # 2. If </think> absent: empty string (model never finished reasoning)
        raw_text = extract_text_from_response(body.response, strip_thinking=False)
        generation = extract_text_from_response(body.response)  # strips thinking traces
        if "</think>" not in raw_text and "</thinking>" not in raw_text:
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
        elif grade == "C":
            verdict = "partial"
            reward = 0.0
        elif grade == "D":
            verdict = "not_attempted"
            reward = 0.0
        else:
            verdict = "incorrect"
            reward = 0.0

        is_correct = 1.0 if verdict == "correct" else 0.0
        is_incorrect = 1.0 if verdict == "incorrect" else 0.0
        is_partial = 1.0 if verdict == "partial" else 0.0
        is_not_attempted = 1.0 if verdict == "not_attempted" else 0.0
        omniscience_index = is_correct - is_incorrect
        is_hallucination = is_incorrect

        return OmniscienceVerifyResponse(
            **body.model_dump(exclude={"expected_answer", "extracted_answer"}),
            reward=reward,
            extracted_answer=generation,
            expected_answer=expected_answer,
            verdict=verdict,
            judge_output=judge_text,
            is_correct=is_correct,
            is_incorrect=is_incorrect,
            is_partial=is_partial,
            is_not_attempted=is_not_attempted,
            omniscience_index=omniscience_index,
            is_hallucination=is_hallucination,
        )


if __name__ == "__main__":
    OmniscienceServer.run_webserver()
