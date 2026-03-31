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
import asyncio
import os
import re
from contextlib import nullcontext
from typing import Optional

from openai import AsyncOpenAI

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
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
from nemo_gym.server_utils import get_response_json


VALID_TAGS = ["[review_needed_senior_swe]", "[review_needed_junior_swe]", "[skip_review]"]
TAG_PATTERN = re.compile(r"\[(review_needed_senior_swe|review_needed_junior_swe|skip_review)\]")
SUMMARY_PATTERN = re.compile(
    r"##\s*AI-generated summary[^\n]*\n+(.*?)(?=\n##|\[(?:review_needed_senior_swe|review_needed_junior_swe|skip_review)\])",
    re.DOTALL | re.IGNORECASE,
)
THINKING_PATTERN = re.compile(r"<think(?:ing)?>(.*?)</think(?:ing)?>", re.DOTALL)
SCORE_PATTERN = re.compile(r"Score:\s*(\d+)", re.IGNORECASE)

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator assessing the quality of PR (Pull Request) summaries.

Your task is to compare a candidate summary against a reference summary and score the candidate on a scale of 1-5.

Scoring criteria:
- 5 (Excellent): Candidate captures all key information from the reference, is well-written, and may even add helpful context
- 4 (Good): Candidate captures most key information, minor omissions or differences that don't affect understanding
- 3 (Adequate): Candidate captures the main point but misses some important details or has minor inaccuracies
- 2 (Poor): Candidate misses significant information or contains noticeable inaccuracies
- 1 (Very Poor): Candidate is largely incorrect, misses the main point, or is incomprehensible

Focus on:
1. Factual accuracy - Does the candidate correctly describe what changed?
2. Completeness - Are all important changes mentioned?
3. Clarity - Is the summary easy to understand?

Do NOT penalize for:
- Different wording that conveys the same meaning
- Minor formatting differences
- Additional helpful context in the candidate"""


def strip_thinking(text: str) -> str:
    """Remove <think>/<thinking> blocks from model output."""
    return THINKING_PATTERN.sub("", text).strip()


def extract_tag(response: str) -> Optional[str]:
    """Extract classification tag from response."""
    if not response:
        return None
    match = TAG_PATTERN.search(response)
    if match:
        return f"[{match.group(1)}]"
    return None


def extract_summary(response: str) -> Optional[str]:
    """Extract summary text from response.

    Looks for content between "## AI-generated summary" header and either
    the next "##" header or the classification tag.
    """
    if not response:
        return None

    match = SUMMARY_PATTERN.search(response)
    if match:
        summary = match.group(1).strip()
        if summary:
            return summary

    # Fallback: content before the tag
    tag_match = TAG_PATTERN.search(response)
    if tag_match:
        content_before_tag = response[: tag_match.start()].strip()
        content_before_tag = re.sub(
            r"^##\s*AI-generated summary[^\n]*\n*",
            "",
            content_before_tag,
            flags=re.IGNORECASE,
        ).strip()
        content_before_tag = re.split(r"\n##", content_before_tag)[0].strip()
        if content_before_tag:
            return content_before_tag

    return None


class CodeRabbitPRConfig(BaseResourcesServerConfig):
    judge_score_threshold: float = 4.0
    judge_model_server: ModelServerRef
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    judge_prompt_template_fpath: str = "prompt_templates/pr_summary_judge.txt"
    judge_system_message: Optional[str] = None
    judge_endpoint_max_concurrency: Optional[int] = 64
    judge_external_base_url: Optional[str] = None
    judge_external_model: Optional[str] = None
    judge_external_api_key_env: str = "JUDGE_API_KEY"


class CodeRabbitPRVerifyRequest(BaseVerifyRequest):
    verifier_metadata: Optional[dict] = None


class CodeRabbitPRVerifyResponse(BaseVerifyResponse):
    predicted_tag: Optional[str] = None
    predicted_summary: Optional[str] = None
    ground_truth_tag: Optional[str] = None
    ground_truth_summary: Optional[str] = None
    tag_correct: bool = False
    judge_score: Optional[float] = None


class CodeRabbitPRServer(SimpleResourcesServer):
    config: CodeRabbitPRConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.config.judge_endpoint_max_concurrency is not None:
            self._judge_semaphore = asyncio.Semaphore(value=self.config.judge_endpoint_max_concurrency)
        else:
            self._judge_semaphore = nullcontext()

        with open(self.config.judge_prompt_template_fpath, "r") as f:
            self._judge_prompt_template = f.read().strip()

        self._judge_system_message = self.config.judge_system_message or JUDGE_SYSTEM_PROMPT

        if self.config.judge_external_base_url:
            api_key = os.environ.get(self.config.judge_external_api_key_env, "")
            self._external_judge_client = AsyncOpenAI(
                base_url=self.config.judge_external_base_url,
                api_key=api_key,
            )
        else:
            self._external_judge_client = None

    async def _generate_judge_score(self, reference: str, candidate: str) -> Optional[float]:
        """Call the judge model to score a candidate summary against a reference.

        Returns a score from 1-5, or None if parsing fails.
        """
        user_prompt = self._judge_prompt_template.format(reference=reference, candidate=candidate)

        if self._external_judge_client:
            return await self._generate_judge_score_external(user_prompt)

        responses_create_params = self.config.judge_responses_create_params.model_copy(deep=True)
        msgs: list[NeMoGymEasyInputMessage] = []
        if self._judge_system_message:
            msgs.append(NeMoGymEasyInputMessage(role="system", content=self._judge_system_message))
        msgs.append(NeMoGymEasyInputMessage(role="user", content=user_prompt))
        responses_create_params.input = msgs

        async with self._judge_semaphore:
            try:
                response = await self.server_client.post(
                    server_name=self.config.judge_model_server.name,
                    url_path="/v1/responses",
                    json=responses_create_params,
                )
                judge_response = NeMoGymResponse.model_validate(await get_response_json(response))
            except Exception as e:
                print(
                    f"DEBUG: CodeRabbitPRServer: judge model server HTTP POST error: {type(e).__name__} {e}",
                    flush=True,
                )
                return None

        # Extract text from judge response
        try:
            last_output = judge_response.output[-1]
            if getattr(last_output, "type", None) != "message":
                return None
            last_content = last_output.content[-1]
            text = getattr(last_content, "text", "")
        except Exception:
            return None

        return self._parse_score(text)

    async def _generate_judge_score_external(self, user_prompt: str) -> Optional[float]:
        """Call the external judge API (OpenAI-compatible) to score a summary."""
        messages = []
        if self._judge_system_message:
            messages.append({"role": "system", "content": self._judge_system_message})
        messages.append({"role": "user", "content": user_prompt})

        async with self._judge_semaphore:
            try:
                completion = await self._external_judge_client.chat.completions.create(
                    model=self.config.judge_external_model,
                    messages=messages,
                    max_tokens=256,
                    temperature=0.0,
                )
                text = completion.choices[0].message.content or ""
            except Exception as e:
                print(
                    f"DEBUG: CodeRabbitPRServer: external judge API error: {type(e).__name__} {e}",
                    flush=True,
                )
                return None

        return self._parse_score(text)

    @staticmethod
    def _parse_score(text: str) -> Optional[float]:
        """Extract numeric score from judge response text."""
        if not text:
            return None
        match = SCORE_PATTERN.search(text)
        if match:
            score = int(match.group(1))
            if 1 <= score <= 5:
                return float(score)
        return None

    async def verify(self, body: CodeRabbitPRVerifyRequest) -> CodeRabbitPRVerifyResponse:
        # Extract model output text
        model_output = ""
        for output_item in body.response.output:
            if output_item.type != "message":
                continue
            for content_item in output_item.content:
                if content_item.type == "output_text":
                    model_output += content_item.text

        metadata = body.verifier_metadata or {}
        gt_tag = metadata.get("ground_truth_tag", "")
        gt_summary = metadata.get("ground_truth_summary", "")

        if not model_output or not model_output.strip():
            return CodeRabbitPRVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                predicted_tag=None,
                predicted_summary=None,
                ground_truth_tag=gt_tag,
                ground_truth_summary=gt_summary,
                tag_correct=False,
                judge_score=None,
            )

        cleaned = strip_thinking(model_output)
        predicted_tag = extract_tag(cleaned)
        predicted_summary = extract_summary(cleaned)

        tag_correct = predicted_tag is not None and predicted_tag == gt_tag

        judge_score = None
        if tag_correct and predicted_summary and gt_summary:
            judge_score = await self._generate_judge_score(reference=gt_summary, candidate=predicted_summary)

        reward = (
            1.0
            if (tag_correct and judge_score is not None and judge_score >= self.config.judge_score_threshold)
            else 0.0
        )

        return CodeRabbitPRVerifyResponse(
            **body.model_dump(),
            reward=reward,
            predicted_tag=predicted_tag,
            predicted_summary=predicted_summary,
            ground_truth_tag=gt_tag,
            ground_truth_summary=gt_summary,
            tag_correct=tag_correct,
            judge_score=judge_score,
        )


if __name__ == "__main__":
    CodeRabbitPRServer.run_webserver()
