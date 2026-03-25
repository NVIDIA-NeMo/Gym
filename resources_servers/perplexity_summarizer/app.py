# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
from __future__ import annotations

import asyncio
import json
import logging
import re
from contextlib import nullcontext
from typing import Literal, Optional

from aiolimiter import AsyncLimiter
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

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
from nemo_gym.server_utils import get_response_json
from resources_servers.perplexity_summarizer.prompts import format_judge_prompt, get_judge_type


class PerplexitySearchConfig(BaseResourcesServerConfig):
    perplexity_api_key: str
    search_max_concurrency: int = 64
    search_rate_limit_qps: Optional[int] = None
    max_search_results: int = 10
    max_tokens_per_page: int = 500

    judge_type: Literal["llm", "reward_model"] = "llm"
    reward_model_server: Optional[ModelServerRef] = None

    judge_model_server: ModelServerRef
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    judge_endpoint_max_concurrency: Optional[int] = 64
    judge_temperature: Optional[float] = None
    judge_top_p: Optional[float] = None
    judge_max_output_tokens: Optional[int] = None
    judge_reasoning_effort: Optional[str] = "medium"


class PerplexitySearchRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    dataset_name: str
    example_id: Optional[str] = None
    query: str
    instruction: Optional[str] = None
    reference_answer: Optional[str] = None
    ground_truth: Optional[str] = None


class PerplexitySearchVerifyRequest(PerplexitySearchRequest, BaseVerifyRequest):
    pass


class JudgeResult(BaseModel):
    correct: int
    reasoning: str
    failure_mode: Optional[str] = None


class PerplexitySearchVerifyResponse(BaseVerifyResponse):
    judge_result: Optional[JudgeResult] = None
    judge_raw_output: Optional[str] = None


class SearchWebRequest(BaseModel):
    queries: list[str]


class SearchWebResponse(BaseModel):
    search_results: str


def _parse_judge_output(raw_output: str, dataset_name: str) -> Optional[JudgeResult]:
    """Parse judge output — supports free-text and JSON fallback.

    IF datasets (user_if, abstention): look for "followed: yes/no".
    Correctness datasets (frames, facts, etc.): look for "correct: yes/no".
    Falls back to JSON parsing if free-text patterns aren't found.
    """
    text = raw_output.strip()

    # Try free-text patterns first
    if get_judge_type(dataset_name) == "if":
        match = re.search(r"followed:\s*(yes|no)", text, re.IGNORECASE)
        if match:
            followed = match.group(1).lower() == "yes"
            return JudgeResult(correct=1 if followed else 0, reasoning=text, failure_mode=None)

    match = re.search(r"correct:\s*(yes|no)", text, re.IGNORECASE)
    if match:
        correct = match.group(1).lower() == "yes"
        return JudgeResult(correct=1 if correct else 0, reasoning=text, failure_mode=None)

    # JSON fallback — handle ```json``` code blocks and raw JSON
    json_text = text
    if "```json" in json_text:
        start = json_text.find("```json") + 7
        end = json_text.find("```", start)
        if end > start:
            json_text = json_text[start:end].strip()
    elif "```" in json_text:
        start = json_text.find("```") + 3
        end = json_text.find("```", start)
        if end > start:
            json_text = json_text[start:end].strip()
    try:
        parsed = json.loads(json_text)
        return JudgeResult(
            correct=int(parsed.get("correct", 0)),
            reasoning=str(parsed.get("reasoning", text)),
            failure_mode=parsed.get("failure_mode"),
        )
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    return JudgeResult(correct=0, reasoning=text, failure_mode="could not parse judge output")


class PerplexitySearchResourcesServer(SimpleResourcesServer):
    config: PerplexitySearchConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._search_semaphore = asyncio.Semaphore(self.config.search_max_concurrency)
        self._search_rate_limiter = (
            AsyncLimiter(self.config.search_rate_limit_qps, time_period=1)
            if self.config.search_rate_limit_qps is not None
            else None
        )
        self._judge_semaphore = (
            asyncio.Semaphore(self.config.judge_endpoint_max_concurrency)
            if self.config.judge_endpoint_max_concurrency is not None
            else nullcontext()
        )
        self._perplexity_client = None

    def _get_perplexity_client(self):
        if self._perplexity_client is None:
            from perplexity import AsyncPerplexity

            self._perplexity_client = AsyncPerplexity(api_key=self.config.perplexity_api_key)
        return self._perplexity_client

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/search_web")(self.search_web)
        return app

    async def search_web(self, body: SearchWebRequest) -> SearchWebResponse:
        if not body.queries:
            return SearchWebResponse(search_results="[]")
        client = self._get_perplexity_client()
        try:
            async with self._search_semaphore:
                if self._search_rate_limiter is not None:
                    await self._search_rate_limiter.acquire()
                response = await client.search.create(
                    query=body.queries,
                    max_results=self.config.max_search_results,
                    max_tokens_per_page=self.config.max_tokens_per_page,
                )
            results = response.results if isinstance(response.results, list) else [response.results]
            parsed = [
                {
                    "id": f"web:{i}",
                    "url": getattr(r, "url", ""),
                    "title": getattr(r, "title", ""),
                    "content": getattr(r, "snippet", ""),
                }
                for i, r in enumerate(results)
            ]
            return SearchWebResponse(search_results=json.dumps(parsed))
        except Exception as e:
            logging.warning("search error: %s %s", type(e).__name__, e)
            return SearchWebResponse(search_results=f"Error: {type(e).__name__} - {e}")

    @staticmethod
    def _extract_response_text(response) -> str:
        """Extract assistant text from response output items.

        Tries output_text first, then falls back to manually extracting text
        from message output items (needed when output_text is not populated).
        """
        text = response.output_text
        if text and text.strip():
            return text.strip()
        # Fallback: extract from output message items
        for item in response.output:
            if hasattr(item, "type") and item.type == "message" and hasattr(item, "content"):
                for part in item.content:
                    if hasattr(part, "text") and part.text and part.text.strip():
                        return part.text.strip()
            elif isinstance(item, dict) and item.get("type") == "message":
                for part in item.get("content", []):
                    t = part.get("text", "")
                    if t and t.strip():
                        return t.strip()
        return ""

    async def verify(self, body: PerplexitySearchVerifyRequest) -> PerplexitySearchVerifyResponse:
        response_text = self._extract_response_text(body.response)
        if not response_text:
            return PerplexitySearchVerifyResponse(**body.model_dump(), reward=0.0)

        if self.config.judge_type == "reward_model":
            return await self._verify_with_reward_model(body, response_text)
        return await self._verify_with_llm_judge(body, response_text)

    async def _verify_with_reward_model(
        self, body: PerplexitySearchVerifyRequest, response_text: str
    ) -> PerplexitySearchVerifyResponse:
        raise NotImplementedError("Reward model judge not yet implemented.")

    async def _verify_with_llm_judge(
        self, body: PerplexitySearchVerifyRequest, response_text: str
    ) -> PerplexitySearchVerifyResponse:
        judge_prompt = format_judge_prompt(
            dataset_name=body.dataset_name,
            query=body.query,
            response=response_text,
            ground_truth=body.ground_truth,
            instruction=body.instruction,
            reference_answer=body.reference_answer,
        )

        judge_raw_output = await self._call_judge(judge_prompt)
        judge_result = _parse_judge_output(judge_raw_output, body.dataset_name) if judge_raw_output else None
        reward = 1.0 if judge_result is not None and judge_result.correct == 1 else 0.0

        return PerplexitySearchVerifyResponse(
            **body.model_dump(),
            reward=reward,
            judge_result=judge_result,
            judge_raw_output=judge_raw_output,
        )

    async def _call_judge(self, prompt: str) -> Optional[str]:
        cfg = self.config
        params = cfg.judge_responses_create_params.model_copy(deep=True)
        if cfg.judge_temperature is not None:
            params.temperature = cfg.judge_temperature
        if cfg.judge_top_p is not None:
            params.top_p = cfg.judge_top_p
        if cfg.judge_max_output_tokens is not None:
            params.max_output_tokens = cfg.judge_max_output_tokens
        if cfg.judge_reasoning_effort is not None:
            params.reasoning = {"effort": cfg.judge_reasoning_effort}
        params.input = [
            NeMoGymEasyInputMessage(role="user", content=prompt),
        ]
        async with self._judge_semaphore:
            try:
                response = await self.server_client.post(
                    server_name=cfg.judge_model_server.name,
                    url_path="/v1/responses",
                    json=params,
                )
                judge_response = NeMoGymResponse.model_validate(await get_response_json(response))
            except Exception as e:
                logging.warning("judge error: %s %s", type(e).__name__, e)
                return None
        return judge_response.output_text or None


if __name__ == "__main__":
    PerplexitySearchResourcesServer.run_webserver()
