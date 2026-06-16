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

Single-pass LLM-judge verifier for FrontierScience. In ``olympiad`` mode
it mirrors NeMo Skills' `frontierscience-olympiad` benchmark verification:
the judge sees the problem, reference answer, and attempted answer, then
emits ``Judgement: YES`` or ``Judgement: NO`` on its final line. In
``research`` mode it parses a 10-point research rubric score and maps scores
at or above the configured threshold to ``reward=1.0``.

The judge prompt is loaded from a YAML file at startup and is configurable
via the ``judge_prompt_path`` config field. The default is the verbatim
Skills short-answer prompt under ``prompts/judge.yaml``.

Source: https://cdn.openai.com/pdf/2fcd284c-b468-4c21-8ee0-7a783933efcc/frontierscience-paper.pdf
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Literal, Optional, Union

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
_BRACKETED_SCORE_RE = re.compile(
    r"(?:final[_\s-]*)?score\s*\[\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*\]\s*/\s*"
    r"(?:max[_\s-]*possible[_\s-]*)?score\s*\[\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*\]",
    re.IGNORECASE,
)
_DENOMINATED_SCORE_RE = re.compile(
    r"([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*(?:/|out\s+of)\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))",
    re.IGNORECASE,
)
_LABELED_SCORE_RE = re.compile(
    r"(?:final\s+)?score\s*[:=]\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))",
    re.IGNORECASE,
)


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


def parse_rubric_score(judge_text: str, max_score: float = 10.0) -> Optional[float]:
    """Parse a rubric score and normalize it onto ``max_score`` points.

    Accepts the formats the research prompt asks for (``Score: 7/10``) plus
    a few common judge variants such as ``Score: 7 out of 10`` and
    ``FINAL_SCORE[7] / MAX_POSSIBLE_SCORE[10]``. The last score-like line wins.
    """
    if not judge_text:
        return None

    for line in reversed(judge_text.splitlines()):
        if "score" not in line.lower():
            continue

        for pattern in (_BRACKETED_SCORE_RE, _DENOMINATED_SCORE_RE):
            match = pattern.search(line)
            if not match:
                continue
            numerator = float(match.group(1))
            denominator = float(match.group(2))
            if denominator <= 0:
                return None
            return numerator / denominator * max_score

        match = _LABELED_SCORE_RE.search(line)
        if match:
            return float(match.group(1))

    return None


def _resolve_prompt_path(path: str) -> Path:
    prompt_path = Path(path)
    if prompt_path.is_absolute():
        return prompt_path

    repo_root = Path(__file__).resolve().parents[2]
    candidates = (
        Path.cwd() / prompt_path,
        Path(__file__).parent / prompt_path,
        repo_root / prompt_path,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return prompt_path


class FrontierScienceJudgeConfig(BaseResourcesServerConfig):
    judge_model_server: ModelServerRef
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    judge_mode: Literal["olympiad", "research"] = Field(
        default="olympiad",
        description=(
            "olympiad parses Judgement: YES/NO directly; research parses a 0-10 score and "
            "marks the attempt correct at rubric_pass_score_threshold or above."
        ),
    )
    rubric_pass_score_threshold: float = Field(
        default=7.0,
        description="Minimum rubric score, on the rubric_max_score scale, that counts as correct.",
    )
    rubric_max_score: float = Field(
        default=10.0,
        description="Maximum rubric score used to normalize parsed judge scores.",
    )

    judge_prompt_path: str = Field(
        default=_DEFAULT_JUDGE_PROMPT_PATH,
        description=(
            "Path to a YAML file containing the judge prompt under a 'user' key. "
            "Placeholders: {question}, {expected_answer}, {rubric}, {generation}."
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
    rubric: Optional[str] = None


class FrontierScienceJudgeVerifyRequest(FrontierScienceJudgeRunRequest, BaseVerifyRequest):
    pass


class FrontierScienceJudgeVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    extracted_answer: Optional[str] = None
    expected_answer: Optional[str] = None
    verdict: Optional[str] = None
    judge_output: Optional[str] = None
    rubric_score: Optional[float] = None
    rubric_score_normalized: Optional[float] = None
    invalid_judge_response: bool = False


class FrontierScienceJudgeServer(SimpleResourcesServer):
    config: FrontierScienceJudgeConfig

    def model_post_init(self, context):
        prompt_data = yaml.safe_load(_resolve_prompt_path(self.config.judge_prompt_path).read_text())
        self._judge_prompt_template = prompt_data["user"]
        return super().model_post_init(context)

    @staticmethod
    def _score_fn(result: dict) -> dict:
        """Map verify response to a single named score for pass@k metrics."""
        scores = {"accuracy": float(result.get("reward", 0.0))}
        if result.get("rubric_score_normalized") is not None:
            scores["rubric_score"] = float(result["rubric_score_normalized"])
        return scores

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

        rubric = body.rubric or expected_answer

        judge_prompt = self._judge_prompt_template.format(
            question=question,
            expected_answer=expected_answer,
            rubric=rubric,
            generation=generation,
        )

        if self.config.use_chat_completions_for_judge:
            chat_params_kwargs = {
                "messages": [{"role": "user", "content": judge_prompt}],
                "max_tokens": self.config.judge_responses_create_params.max_output_tokens or 2048,
            }
            if self.config.judge_responses_create_params.temperature is not None:
                chat_params_kwargs["temperature"] = self.config.judge_responses_create_params.temperature
            elif self.config.judge_mode == "olympiad":
                chat_params_kwargs["temperature"] = 0.0
            if self.config.judge_responses_create_params.top_p is not None:
                chat_params_kwargs["top_p"] = self.config.judge_responses_create_params.top_p
            elif self.config.judge_mode == "olympiad":
                chat_params_kwargs["top_p"] = 1.0
            chat_params = NeMoGymChatCompletionCreateParamsNonStreaming(**chat_params_kwargs)
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
        rubric_score = None
        rubric_score_normalized = None
        invalid_judge_response = False

        if self.config.judge_mode == "research":
            rubric_score = parse_rubric_score(judge_text, max_score=self.config.rubric_max_score)
            if rubric_score is None:
                invalid_judge_response = True
                reward = 1.0 if verdict == "YES" else 0.0
            else:
                rubric_score_normalized = max(0.0, min(rubric_score, self.config.rubric_max_score))
                rubric_score_normalized /= self.config.rubric_max_score
                reward = 1.0 if rubric_score >= self.config.rubric_pass_score_threshold else 0.0
                if verdict is None:
                    verdict = "YES" if reward else "NO"
        else:
            reward = 1.0 if verdict == "YES" else 0.0

        return FrontierScienceJudgeVerifyResponse(
            **body.model_dump(exclude={"expected_answer", "extracted_answer"}),
            reward=reward,
            extracted_answer=generation if generation else None,
            expected_answer=expected_answer,
            verdict=verdict,
            judge_output=judge_text,
            rubric_score=rubric_score,
            rubric_score_normalized=rubric_score_normalized,
            invalid_judge_response=invalid_judge_response,
        )


if __name__ == "__main__":
    FrontierScienceJudgeServer.run_webserver()
