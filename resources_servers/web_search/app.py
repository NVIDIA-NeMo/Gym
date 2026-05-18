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
"""Web Search resources server.

Port of NeMo-RLVR's `nemo_rl/environments/web_search_environment.py`.

Multi-step environment where the model can call a `search` tool (backed by the
Serper API) multiple times before providing a final answer.

Verification supports two ground-truth styles:
- `mcq`: Extract option letter from \\boxed{} or "Answer: X" pattern.
- `natural_text`: LLM judge (configurable model server) — mirrors RLVR's
  ``run_verifier_batch`` flow with a strict JSON ``{"score": 0|1}`` parser.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import httpx
from fastapi import FastAPI
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
from nemo_gym.server_utils import get_response_json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def strip_thinking(text: str) -> str:
    """Strip <think>...</think> content from model output."""
    if "<think>" in text and "</think>" not in text:
        return ""
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    return text


def extract_boxed_answer(text: str) -> Optional[str]:
    r"""Extract the answer from \boxed{...} format."""
    pattern = r"\\boxed\{([^}]+)\}"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    return None


def verify_mcq(model_output: str, ground_truth: str) -> float:
    """Verify MCQ-style answer: extract option letter and compare."""
    clean = strip_thinking(model_output)
    if not clean:
        return 0.0

    # Try \boxed{} first
    predicted = extract_boxed_answer(clean)
    if not predicted:
        # Fallback: "Answer is/: (X)" pattern
        matches = re.findall(r"[Aa]nswer\s*(?:is|:)?\s*\(?([A-Za-z])\)?", clean)
        predicted = matches[-1] if matches else ""

    if not predicted:
        return 0.0

    return 1.0 if predicted.strip().upper() == ground_truth.strip().upper() else 0.0


def parse_judge_score(text: str) -> Optional[int]:
    """Parse ``{"score": 0|1}`` from a judge response.

    Mirrors RLVR's parser: strip <think>, regex for ``"score": <0|1>``, fall back
    to literal substring match. Returns ``None`` if no score is recoverable —
    callers should treat that as a verifier failure.
    """
    clean = strip_thinking(text)
    match = re.search(r'"score"\s*:\s*([01])', clean)
    if match:
        return int(match.group(1))
    if '"score": 1' in clean or '"score":1' in clean:
        return 1
    if '"score": 0' in clean or '"score":0' in clean:
        return 0
    return None


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


# Site filters — appended to the user's ``query`` before sending to Serper.
# Verbatim from sovereign-tool-calling's ``serper_search.py``: India-tuned
# (``espn.in``, ``screener.in``, ``moneycontrol.com``).
_WEATHER_SITE = "site:accuweather.com"
_SPORTS_SITE = "site:espn.in OR site:sofascore.com OR site:cricbuzz.com"
_STOCK_SITE = "site:screener.in OR site:moneycontrol.com"


# Per-search-type configuration. Mirrors ``SEARCH_CONFIG`` in
# nemo-rlvr/sovereign_tool_calling/tools/search/engines/serper_search.py.
#
# - ``endpoint``: the Serper API path. ``general``/``weather``/``sports``/``stock``
#   share ``/search``; ``scholar``/``news`` use dedicated endpoints.
# - ``result_key``: which top-level key on the JSON response holds the result
#   list. ``news`` puts results under ``"news"``, everything else under
#   ``"organic"``.
# - ``site_filter``: appended to the query string when set, narrowing Google's
#   index to a curated set of domains.
_SEARCH_TYPE_CONFIG: dict[str, dict[str, Optional[str]]] = {
    "general": {"endpoint": "/search", "result_key": "organic", "site_filter": None},
    "weather": {"endpoint": "/search", "result_key": "organic", "site_filter": _WEATHER_SITE},
    "sports":  {"endpoint": "/search", "result_key": "organic", "site_filter": _SPORTS_SITE},
    "stock":   {"endpoint": "/search", "result_key": "organic", "site_filter": _STOCK_SITE},
    "scholar": {"endpoint": "/scholar", "result_key": "organic", "site_filter": None},
    "news":    {"endpoint": "/news", "result_key": "news", "site_filter": None},
}

# Backwards-compatible alias for tests that introspected the old map.
_SEARCH_TYPE_ENDPOINTS: dict[str, str] = {
    k: v["endpoint"] for k, v in _SEARCH_TYPE_CONFIG.items()
}

# Serper caps ``num`` at 10 per request.
_MAX_NUM_RESULTS = 10


class SearchRequest(BaseModel):
    query: str
    num_results: int = 5
    # One of: ``general`` (default), ``news``, ``scholar``, ``weather``,
    # ``sports``, ``stock``. Mirrors RLVR's ``WebSearchEnvironment`` semantics.
    # Unknown values silently fall back to ``general``.
    search_type: str = "general"


class SearchResponse(BaseModel):
    results: str  # JSON string of search results


class WebSearchRunRequest(BaseRunRequest):
    ground_truth: Union[str, dict]
    question: str = ""


class WebSearchVerifyRequest(WebSearchRunRequest, BaseVerifyRequest):
    pass


class WebSearchJudgeEvaluation(BaseModel):
    """Captured judge call for natural-text rows. ``response`` is None when the HTTP call failed."""

    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    response: Optional[NeMoGymResponse] = None


class WebSearchVerifyResponse(WebSearchVerifyRequest, BaseVerifyResponse):
    """Inherits ``ground_truth`` / ``question`` from the request (so they
    propagate through to the rollouts JSONL for downstream inspection) and
    ``reward`` / ``responses_create_params`` / ``response`` from the base
    verify response. Same pattern as ``TavilySearchVerifyResponse``.
    """

    ground_truth_style: str
    predicted_answer: Optional[str]
    verification_failed: bool
    judge_evaluation: Optional[WebSearchJudgeEvaluation] = None


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class WebSearchResourcesServerConfig(BaseResourcesServerConfig):
    """Configuration for the web-search resources server.

    The judge fields (``judge_model_server``, ``judge_responses_create_params``,
    ``judge_prompt_template_fpath``, ``judge_endpoint_max_concurrency``) are
    optional. When ``judge_model_server`` is not configured, natural-text
    verification falls back to a case-insensitive substring match — mirrors
    RLVR's "no verifier_url" behavior, useful for smoke tests without an LLM.
    """

    serper_api_key: str = ""

    # LLM judge for natural_text rows (matches stem_qa_local_verifier's pattern).
    judge_model_server: Optional[ModelServerRef] = None
    judge_responses_create_params: Optional[NeMoGymResponseCreateParamsNonStreaming] = None
    judge_prompt_template_fpath: str = "prompt_templates/web_search_judge.txt"
    judge_endpoint_max_concurrency: Optional[int] = 64


class WebSearchResourcesServer(SimpleResourcesServer):
    config: WebSearchResourcesServerConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.config.judge_endpoint_max_concurrency is not None:
            self._judge_semaphore = asyncio.Semaphore(
                value=self.config.judge_endpoint_max_concurrency
            )
        else:
            self._judge_semaphore = nullcontext()

        # Resolve template path relative to this file when not absolute.
        template_path = Path(self.config.judge_prompt_template_fpath)
        if not template_path.is_absolute():
            template_path = Path(__file__).resolve().parent / template_path
        # The template file is optional — only required when an LLM judge is wired up.
        if template_path.exists():
            with open(template_path, "r") as f:
                self._judge_prompt_template = f.read()
        else:
            self._judge_prompt_template = None

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/search")(self.search)
        return app

    # -----------------------------------------------------------------------
    # /search route
    # -----------------------------------------------------------------------

    async def search(self, body: SearchRequest) -> SearchResponse:
        """Execute a web search via Serper API.

        ``body.search_type`` selects endpoint + site filter (1:1 with
        ``sovereign_tool_calling/tools/search/engines/serper_search.py``):

        - ``general`` → ``/search``, no site filter.
        - ``weather`` → ``/search`` + ``site:accuweather.com``.
        - ``sports``  → ``/search`` + ``site:espn.in OR site:sofascore.com OR site:cricbuzz.com``.
        - ``stock``   → ``/search`` + ``site:screener.in OR site:moneycontrol.com``.
        - ``scholar`` → ``/scholar``, no site filter.
        - ``news``    → ``/news``, results under ``data["news"]``.

        The payload also includes ``gl=in`` / ``hl=en`` (matches RLVR's
        India-tuned defaults) and ``num`` is capped at 10 (Serper limit).
        """
        api_key = self.config.serper_api_key or os.environ.get("SERPER_API_KEY", "")
        if not api_key:
            return SearchResponse(
                results=json.dumps({"error": "SERPER_API_KEY not configured"})
            )

        cfg = _SEARCH_TYPE_CONFIG.get(body.search_type, _SEARCH_TYPE_CONFIG["general"])
        url = f"https://google.serper.dev{cfg['endpoint']}"

        # Build the actual query string Serper sees.
        search_query = body.query
        if cfg["site_filter"]:
            search_query = f"{body.query} {cfg['site_filter']}"

        headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "q": search_query,
            "num": min(body.num_results, _MAX_NUM_RESULTS),
            "gl": "in",
            "hl": "en",
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()

            formatted = self._format_search_results(
                data, body.query, body.search_type, body.num_results
            )
            return SearchResponse(results=formatted)
        except httpx.HTTPStatusError as e:
            logger.warning("Serper API HTTP error: %s", e)
            return SearchResponse(
                results=json.dumps({"error": f"Search API error: {e.response.status_code}"})
            )
        except Exception as e:
            logger.warning("Search failed: %s", e)
            return SearchResponse(
                results=json.dumps({"error": f"Search failed: {str(e)}"})
            )

    def _format_search_results(
        self,
        data: dict,
        query: str,
        search_type: str,
        num_results: int,
    ) -> str:
        """Format a raw Serper API response, matching RLVR's ``_parse_results``.

        Returns an indented JSON string. Top-level shape:

        .. code-block:: json

            {
              "query": "...",
              "search_type": "...",
              "total_results": <int>,
              "timestamp": "<ISO-8601>",
              "results": [
                {"position": 1, "title": "...", "url": "...", "snippet": "..."},
                ...
              ]
            }

        Per-type extras: ``scholar`` rows include ``citedBy``; ``news`` rows
        include ``date`` and ``source``. Mirrors
        ``sovereign_tool_calling/tools/search/engines/serper_search.py::_parse_results``
        verbatim — same field names, same shape, so the model sees what it saw
        in RLVR rollouts.
        """
        cfg = _SEARCH_TYPE_CONFIG.get(search_type, _SEARCH_TYPE_CONFIG["general"])
        result_key = cfg["result_key"]

        if result_key not in data:
            return json.dumps(
                {
                    "results": [],
                    "query": query,
                    "search_type": search_type,
                    "message": f"No {result_key} search results found",
                },
                indent=2,
                ensure_ascii=False,
            )

        items: list[dict] = []
        for idx, item in enumerate(data[result_key]):
            entry = {
                "position": idx + 1,
                "title": item.get("title", "No title"),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
            }
            if search_type == "scholar":
                entry["citedBy"] = item.get("citedBy", 0)
            elif search_type == "news":
                entry["date"] = item.get("date", "")
                entry["source"] = item.get("source", "")
            items.append(entry)
            if num_results and len(items) >= num_results:
                break

        return json.dumps(
            {
                "query": query,
                "search_type": search_type,
                "total_results": len(items),
                "timestamp": datetime.utcnow().isoformat(),
                "results": items,
            },
            indent=2,
            ensure_ascii=False,
        )

    # -----------------------------------------------------------------------
    # /verify route
    # -----------------------------------------------------------------------

    async def verify(self, body: WebSearchVerifyRequest) -> WebSearchVerifyResponse:
        """Verify the model's final answer against ground truth.

        - ``mcq`` rows: extract a letter and compare locally.
        - ``natural_text`` rows: call the configured ``judge_model_server`` and
          parse its ``{"score": 0|1}`` JSON. Falls back to substring match when
          no judge is configured (smoke-test mode).
        """
        final_response_text = self._extract_final_assistant_text(body)

        # Parse ground truth (accepts JSON string, dict, or bare string).
        ground_truth = body.ground_truth
        if isinstance(ground_truth, str):
            try:
                ground_truth = json.loads(ground_truth)
            except (json.JSONDecodeError, TypeError):
                ground_truth = {"style": "mcq", "value": ground_truth}

        if isinstance(ground_truth, dict):
            gt_style = str(ground_truth.get("style", "mcq"))
            gt_value = str(ground_truth.get("value", ""))
        else:
            gt_style = "mcq"
            gt_value = str(ground_truth)

        if gt_style == "mcq":
            return self._verify_mcq(body, final_response_text, gt_value)
        if gt_style == "natural_text":
            return await self._verify_natural_text(body, final_response_text, gt_value)

        logger.warning("Unknown ground truth style: %s", gt_style)
        return self._make_response(
            body,
            reward=0.0,
            ground_truth_style=gt_style,
            predicted_answer=None,
            verification_failed=True,
            judge_evaluation=None,
        )

    # -----------------------------------------------------------------------
    # mcq path
    # -----------------------------------------------------------------------

    def _verify_mcq(
        self, body: WebSearchVerifyRequest, text: str, gt_value: str
    ) -> WebSearchVerifyResponse:
        reward = 0.0
        predicted: Optional[str] = None
        verification_failed = False
        try:
            reward = verify_mcq(text, gt_value)
            clean = strip_thinking(text)
            predicted = extract_boxed_answer(clean)
            if not predicted:
                matches = re.findall(
                    r"[Aa]nswer\s*(?:is|:)?\s*\(?([A-Za-z])\)?", clean
                )
                predicted = matches[-1] if matches else None
        except Exception as e:
            logger.warning("MCQ verification failed: %s", e)
            verification_failed = True
            reward = 0.0
        return self._make_response(
            body,
            reward=reward,
            ground_truth_style="mcq",
            predicted_answer=predicted,
            verification_failed=verification_failed,
            judge_evaluation=None,
        )

    # -----------------------------------------------------------------------
    # natural_text path
    # -----------------------------------------------------------------------

    async def _verify_natural_text(
        self, body: WebSearchVerifyRequest, text: str, gt_value: str
    ) -> WebSearchVerifyResponse:
        clean = strip_thinking(text)
        predicted = clean if clean else None

        # Empty response (or unclosed <think>) -> reward 0, not a verifier failure.
        if not clean:
            return self._make_response(
                body,
                reward=0.0,
                ground_truth_style="natural_text",
                predicted_answer=predicted,
                verification_failed=False,
                judge_evaluation=None,
            )

        # Smoke-test fallback: substring match when no judge is configured.
        if (
            self.config.judge_model_server is None
            or self.config.judge_responses_create_params is None
            or self._judge_prompt_template is None
        ):
            reward = 1.0 if gt_value.strip().lower() in clean.strip().lower() else 0.0
            return self._make_response(
                body,
                reward=reward,
                ground_truth_style="natural_text",
                predicted_answer=predicted,
                verification_failed=False,
                judge_evaluation=None,
            )

        # LLM judge.
        question = body.question or self._extract_user_question(body) or ""
        prompt = self._judge_prompt_template.format(
            question=question,
            predicted_answer=clean,
            ground_truth=gt_value,
        )
        responses_create_params = self.config.judge_responses_create_params.model_copy(deep=True)
        responses_create_params.input = [
            NeMoGymEasyInputMessage(role="user", content=prompt)
        ]

        async with self._judge_semaphore:
            try:
                http_response = await self.server_client.post(
                    server_name=self.config.judge_model_server.name,
                    url_path="/v1/responses",
                    json=responses_create_params,
                )
                judge_response = NeMoGymResponse.model_validate(
                    await get_response_json(http_response)
                )
            except Exception as e:
                logger.warning("Web-search judge HTTP call failed: %s", e)
                return self._make_response(
                    body,
                    reward=0.0,
                    ground_truth_style="natural_text",
                    predicted_answer=predicted,
                    verification_failed=True,
                    judge_evaluation=WebSearchJudgeEvaluation(
                        responses_create_params=responses_create_params,
                        response=None,
                    ),
                )

        evaluation = WebSearchJudgeEvaluation(
            responses_create_params=responses_create_params,
            response=judge_response,
        )

        try:
            grading_text = judge_response.output[-1].content[-1].text
            score = parse_judge_score(grading_text)
            if score is None:
                logger.warning(
                    "Web-search judge returned unparseable response: %s",
                    grading_text[:200],
                )
                return self._make_response(
                    body,
                    reward=0.0,
                    ground_truth_style="natural_text",
                    predicted_answer=predicted,
                    verification_failed=True,
                    judge_evaluation=evaluation,
                )
            return self._make_response(
                body,
                reward=float(score),
                ground_truth_style="natural_text",
                predicted_answer=predicted,
                verification_failed=False,
                judge_evaluation=evaluation,
            )
        except Exception as e:
            logger.warning("Web-search judge response parse failed: %s", e)
            return self._make_response(
                body,
                reward=0.0,
                ground_truth_style="natural_text",
                predicted_answer=predicted,
                verification_failed=True,
                judge_evaluation=evaluation,
            )

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    @staticmethod
    def _extract_final_assistant_text(body: WebSearchVerifyRequest) -> str:
        """Walk the response output backwards for the LAST assistant message text.

        Skips ``function_call`` / ``reasoning`` outputs so we never confuse a
        tool-call argument blob for the final answer.
        """
        if not body.response.output:
            return ""
        for output in reversed(body.response.output):
            if getattr(output, "type", None) != "message":
                continue
            if getattr(output, "role", None) != "assistant":
                continue
            content = getattr(output, "content", None)
            if isinstance(content, list) and content:
                # Multiple text blocks in one message — join them.
                texts = [
                    getattr(c, "text", None)
                    for c in content
                    if getattr(c, "text", None) is not None
                ]
                if texts:
                    return "\n".join(texts).strip()
            elif isinstance(content, str):
                return content.strip()
        return ""

    @staticmethod
    def _extract_user_question(body: WebSearchVerifyRequest) -> Optional[str]:
        """Find the first user message in ``responses_create_params.input``."""
        for m in body.responses_create_params.input or []:
            role = getattr(m, "role", None)
            if role is None and isinstance(m, dict):
                role = m.get("role")
            if role == "user":
                content = getattr(m, "content", None)
                if content is None and isinstance(m, dict):
                    content = m.get("content")
                if isinstance(content, str):
                    return content
        return None

    def _make_response(
        self,
        body: WebSearchVerifyRequest,
        *,
        reward: float,
        ground_truth_style: str,
        predicted_answer: Optional[str],
        verification_failed: bool,
        judge_evaluation: Optional[WebSearchJudgeEvaluation],
    ) -> WebSearchVerifyResponse:
        return WebSearchVerifyResponse(
            **body.model_dump(),
            reward=reward,
            ground_truth_style=ground_truth_style,
            predicted_answer=predicted_answer,
            verification_failed=verification_failed,
            judge_evaluation=judge_evaluation,
        )


if __name__ == "__main__":
    WebSearchResourcesServer.run_webserver()
