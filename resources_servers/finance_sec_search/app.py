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
Finance SEC Search Resource Server.

Provides tools for searching SEC filings by ticker symbol or company name.
Caches ticker mappings and filing metadata locally to minimize SEC.gov calls.
"""

import asyncio
import atexit
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter, deque
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Literal, NamedTuple, Optional, Sequence, TypeVar, Union

import aiohttp
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from starlette.requests import Request

from nemo_gym.client import SESSION_ID_KEY, get_response_json
from nemo_gym.config import ModelServerRef
from nemo_gym.core import (
    JUDGE_FAILED,
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.server_runtime.base_server import refuse_unsupported_num_workers
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from nemo_gym.verification import JUDGE_FAILURES, call_judge, is_judge_failure, reraise_judge_errors
from resources_servers.sec_local_index.cache import atomic_write, make_shared_dirs
from resources_servers.sec_local_index.edgar_search_service import EdgarSearchService
from resources_servers.sec_local_index.html_text import html_to_text
from resources_servers.sec_local_index.live_edgar_search import LiveEdgarSearch
from resources_servers.sec_local_index.local_edgar_search import (
    LocalEdgarSearch,
    canonical_url_key,
)
from resources_servers.sec_local_index.sec_urls import parse_sec_archives_url


logger = logging.getLogger(__name__)

FILING_READ_SOURCES = ("cache", "sec-corpus", "live")
FILING_READ_LOG_INTERVAL_SEC = 1800.0

# The Vals v1 benchmark's evaluation cutoff. Dates beyond it are clamped so a
# rollout cannot see filings the benchmark's answers do not account for.
DEFAULT_MAX_END_DATE = "2025-04-07"

# The judge explains first and ends with its verdict as "[[N]]" (#2852).
_JUDGE_RATING_RE = re.compile(r"\[\[(\d+)\]\]")


def _extract_judge_rating(judge_text: str) -> Optional[int]:
    """Return the judge's verdict: the LAST ``[[N]]`` in its output, or None when there is none.

    The judge may quote the candidate or restate the rubric on the way to its verdict, so the first
    ``[[N]]`` is not reliable; the verdict is the one it ends with.
    """
    matches = _JUDGE_RATING_RE.findall(judge_text or "")
    return int(matches[-1]) if matches else None


# reward_mode 'rubric' grades a row's rubric as finance_agent_v2's /verify does. Keep RubricJudgement,
# aggregate_rubric_scores and the rubric judge methods in step with finance_agent_v2's, so a rubric
# scores the same on either server.

# Ceiling on the escalated judge output budget (see _judge_attempt).
_JUDGE_MAX_OUTPUT_TOKENS_CAP = 32768
# Failed-reply text kept per criterion for debugging; bounded because these ride
# along in every rollout row.
_MAX_FAILED_REPLY_SAMPLES = 3
_FAILED_REPLY_SAMPLE_CHARS = 600


class RubricJudgement(BaseModel):
    """Per-criterion verdict plus everything needed to audit it.

    ``score`` is None when the judge never produced enough parsable verdicts. That is
    a judge failure, not a "not met": reward drops to 0.0 but ``judge_error`` is set
    so those rows stay filterable.
    """

    criteria: str
    operator: Optional[str] = None
    # Upstream weights, recorded per criterion so Partial Credit can be recomputed
    # from the rollout alone. Never shown to the judge; applied after grading.
    severity: float = 1.0
    must_pass: bool = False
    score: Optional[int] = None
    # Successful verdicts in call order.
    votes: List[int] = Field(default_factory=list)
    votes_for: int = 0
    votes_against: int = 0
    # False when the judge contradicted itself; aggregated into a disagreement rate.
    unanimous: Optional[bool] = None
    attempts_used: int = 0
    parse_failures: int = 0
    api_failures: int = 0
    # One entry per successful vote, in the same order as ``votes``.
    evidence: List[str] = Field(default_factory=list)
    reasons: List[str] = Field(default_factory=list)
    # First few failed replies, clipped: a parse failure is only diagnosable from
    # the raw text.
    failed_reply_samples: List[str] = Field(default_factory=list)
    error: Optional[str] = None


def aggregate_rubric_scores(judgements: Sequence[RubricJudgement]) -> Dict[str, Any]:
    """Aggregate per-criterion verdicts into Vals's Partial Credit and friends.

    Partial Credit is the severity-weighted pass fraction, forced to 0.0 when any
    ``must_pass`` criterion failed. Only an affirmative pass counts, so an unresolved
    criterion weighs as a miss and gates like one; verify() flags those rows with
    ``judge_error`` rather than dropping them, which would inflate broken runs.

    Kept module-level and free of server state so stored verdicts can be rescored
    offline through this exact code.
    """
    total = len(judgements)
    passed = sum(1 for j in judgements if j.score == 1)
    unresolved = sum(1 for j in judgements if j.score is None)

    weight_total = sum(j.severity for j in judgements)
    weight_passed = sum(j.severity for j in judgements if j.score == 1)
    weighted_fraction = (weight_passed / weight_total) if weight_total else None

    dealbreakers_total = sum(1 for j in judgements if j.must_pass)
    dealbreakers_failed = sum(1 for j in judgements if j.must_pass and j.score != 1)

    return {
        "rubric_total": total,
        "rubric_passed": passed,
        "rubric_unresolved": unresolved,
        "rubric_fraction": (passed / total) if total else None,
        "rubric_all_pass": unresolved == 0 and passed == total,
        "rubric_partial_credit": 0.0 if dealbreakers_failed else (weighted_fraction or 0.0),
        "rubric_weighted_fraction": weighted_fraction,
        "rubric_weight_total": weight_total,
        "rubric_weight_passed": weight_passed,
        "rubric_dealbreakers_total": dealbreakers_total,
        "rubric_dealbreakers_failed": dealbreakers_failed,
    }


class _JudgeAttempt(NamedTuple):
    """One judge call's outcome.

    ``score`` None means retry; ``starved`` marks the failures a larger output budget
    can fix (hit max_output_tokens or emitted nothing), which gates escalation.
    """

    score: Optional[int]
    error: Optional[str]
    reply: str
    starved: bool


class FinanceAgentResourcesServerConfig(BaseResourcesServerConfig):
    """Configuration for Finance SEC Search resource server."""

    cache_dir: Optional[str] = Field(
        default=None,
        description="Path for caching ticker mappings and filing metadata. Defaults to ~/.cache/nemo_gym/finance_sec_search/ if not set. Relative paths are resolved from cwd.",
    )
    use_cache: bool = Field(default=False, description="Keep False to always fetch fresh filings (used for eval).")
    user_agent: str = Field(
        default="Gym-SEC-Search/1.0 (research@nvidia.com)", description="User-Agent header for SEC.gov requests"
    )
    requests_per_second: int = Field(default=10, description="Rate limit for SEC.gov requests")
    tavily_api_key: Optional[str] = Field(default=None, description="Tavily API key for web search")
    retrieval_model_server: Optional[ModelServerRef] = Field(
        default=None, description="Model server for retrieve_information LLM calls"
    )
    judge_model_server: Optional[ModelServerRef] = Field(default=None, description="Reference to judge model server")
    judge_responses_create_params: Optional[NeMoGymResponseCreateParamsNonStreaming] = Field(
        default=None, description="Parameters for judge model requests"
    )
    judge_prompt_template: Optional[str] = Field(
        default=None,
        description="Inline judge prompt template. Takes priority over judge_prompt_template_fpath. "
        "Supports {question}, {expected_answer}, {generated_answer} placeholders.",
    )
    judge_prompt_template_fpath: str = Field(
        default="prompt_templates/finance_sec_search_judge.yaml",
        description="Fallback file path for judge prompt template (used when judge_prompt_template is not set)",
    )
    retrieval_responses_create_params: Optional[NeMoGymResponseCreateParamsNonStreaming] = Field(
        default=None, description="Parameters for retrieval model requests (temperature, top_p, etc.)"
    )
    retrieval_system_prompt: Optional[str] = Field(
        default=None,
        description="Inline retrieval system prompt. Takes priority over retrieval_system_prompt_fpath.",
    )
    retrieval_system_prompt_fpath: str = Field(
        default="prompt_templates/finance_sec_search_retrieval.yaml",
        description="Fallback file path for retrieval system prompt (used when retrieval_system_prompt is not set)",
    )
    large_doc_threshold_chars: int = Field(
        default=100000,
        description="If the document is larger than this threshold characters, give a warning to the model to use char ranges.",
    )
    reward_mode: Literal["binary", "scaled", "rubric"] = Field(
        default="binary",
        description="How judge ratings map to rewards. "
        "'binary': only [[2]] → 1.0, else 0.0. "
        "'scaled': [[0]] → 0.0, [[1]] → 0.5, [[2]] → 1.0. "
        "'rubric': each criterion of the row's rubric is judged on its own, as finance_agent_v2 does, and the "
        "reward is its Partial Credit: the severity-weighted share of criteria met, 0.0 if a must_pass one failed.",
    )
    rubric_judge_prompt_template: Optional[str] = Field(
        default=None,
        description="reward_mode 'rubric': inline rubric judge prompt template. Takes priority over "
        "rubric_judge_prompt_template_fpath. Supports {question}, {generated_answer}, "
        "{criterion} placeholders and grades one criterion per call.",
    )
    rubric_judge_prompt_template_fpath: str = Field(
        default="prompt_templates/finance_sec_search_rubric_judge.yaml",
        description="reward_mode 'rubric': fallback file path for the rubric judge prompt template.",
    )
    judge_required_successes: int = Field(
        default=3,
        description="reward_mode 'rubric': successful (parsed) judge verdicts needed per criterion before "
        "voting. Keep odd so the majority cannot tie.",
    )
    judge_max_attempts: int = Field(
        default=10,
        description="reward_mode 'rubric': hard cap on judge calls per criterion. Whichever comes first, "
        "judge_required_successes or this, ends the criterion. Falling short leaves the criterion "
        "unresolved (score null).",
    )
    judge_max_concurrency: int = Field(
        default=4,
        description="reward_mode 'rubric': max concurrent judge calls per rollout, across criteria. "
        "1 makes judging fully sequential.",
    )
    retrieval_max_output_tokens: Optional[int] = Field(
        default=None,
        description="Max output tokens for retrieve_information LLM calls. Increase for thinking models. "
        "Set to null/None to leave it unset so the retrieval call inherits the full generation budget "
        "(used for eval); set an integer to cap it (used for training).",
    )
    retrieval_model_context_length: int = Field(
        default=131072,
        description="Context window (in tokens) of the retrieval model. Used to compute prompt size limits.",
    )
    max_filing_results: int = Field(
        default=200,
        description="Maximum number of filing metadata entries returned by sec_filing_search.",
    )
    request_timeout: int = Field(default=30, description="Per-request timeout in seconds for SEC.gov calls")
    max_connections_per_host: int = Field(default=10, description="Max concurrent connections to SEC.gov")
    max_retries: int = Field(default=3, description="Max retries for transient SEC.gov errors (403, 429, 503)")
    sec_dump_path: Optional[str] = Field(
        default=None,
        description="Path to pre-fetched SEC dump directory (read-only). Used as fallback for filing content cache misses.",
    )
    max_rollout_time_seconds: Optional[float] = Field(
        default=None,
        description="Per-rollout wall-clock time budget in seconds. When exceeded, tool calls return an error "
        "asking the model to submit immediately. Set to None to disable.",
    )
    edgar_search_mode: Optional[Literal["live", "local"]] = Field(
        default=None,
        description="Where edgar_search reads filings from. 'live' queries sec-api.io and needs sec_api_key. "
        "'local' reads local_edgar_index_path and makes no network call. Left unset, edgar_search is "
        "unavailable. Other tools are unaffected.",
    )
    sec_api_key: Optional[str] = Field(
        default=None,
        description="sec-api.io key for edgar_search in live mode.",
    )
    local_edgar_index_path: Optional[str] = Field(
        default=None,
        description="Read-only SQLite FTS5 index used by edgar_search in local mode.",
    )
    local_edgar_metrics_dir: Optional[str] = Field(
        default=None,
        description="Optional directory for per-search latency records.",
    )
    local_edgar_metadata_path: Optional[str] = Field(
        default=None,
        description="Metadata sidecar for the local EDGAR index, built by "
        "resources_servers/sec_local_index/scripts/build_local_edgar_metadata.py. Defaults to the index path plus "
        "'.metadata' when that file exists. Searches are far slower without it.",
    )
    max_end_date: Optional[str] = Field(
        default=None,
        description="Maximum allowed end_date for all date-filtered tools (web_search, edgar_search, etc.). "
        "When set, dates beyond this are clamped and omitted end_dates default to this value. "
        "Set to null (default) to disable clamping.",
    )
    cap_tool_dates_at_as_of_date: bool = Field(
        default=False,
        description="Cap the start_date and end_date of every sec_filing_search, edgar_search and web_search "
        "call at the row's as_of_date (YYYY-MM-DD), as finance_agent_v2 does; max_end_date still applies. "
        "A session keeps the as_of_date of its first seed. Off (default): a row's as_of_date is ignored.",
    )
    supplementary_tickers_fpath: Optional[str] = Field(
        default=None,
        description="Optional JSON overlay of extra ticker mappings in SEC company_tickers.json schema. "
        "Merged onto the live or cached SEC registry at startup.",
    )
    judge_call_timeout: Optional[float] = Field(
        default=60.0,
        description="Per-call timeout in seconds for judge LLM requests. "
        "Prevents stale TCP connections from blocking the rollout indefinitely. "
        "Set to None to disable.",
    )


def _coerce_stringified_collection(v: Any) -> Any:
    """Deserialize a stringified list/dict into its native Python type.

    Tool-call parsers may serialize nested arguments as strings rather than
    native types.  This handles two common formats:
      1. JSON strings:  '["a", "b"]'  or  '[{"key": "v"}]'
      2. Python repr:   "['a', 'b']"  or  "[{'key': 'v'}]"

    Returns the parsed object when successful, or the original value
    unchanged (letting Pydantic's normal validation handle it).
    """
    if not isinstance(v, str):
        return v
    import ast

    try:
        parsed = json.loads(v)
        if isinstance(parsed, (list, dict)):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        parsed = ast.literal_eval(v)
        if isinstance(parsed, (list, dict)):
            return parsed
    except (ValueError, SyntaxError):
        pass
    return v


class FinanceAgentSearchRequest(BaseModel):
    """Request model for SEC filing search."""

    ticker: str = Field(description="Stock ticker symbol (e.g., 'AAPL', 'MSFT', 'NVDA')")
    form_types: Optional[List[str]] = Field(
        default=None,
        description="(optional) Limits search to specific EDGAR form types (e.g., ['10-K', '10-Q', '8-K']). Default: all form types.",
    )
    start_date: Optional[str] = Field(
        default=None,
        description="(optional) Filter filings on or after this date (YYYY-MM-DD)",
    )
    end_date: Optional[str] = Field(
        default=None,
        description="(optional) Filter filings on or before this date (YYYY-MM-DD)",
    )

    @field_validator("form_types", mode="before")
    @classmethod
    def _coerce_form_types(cls, v: Any) -> Any:
        return _coerce_stringified_collection(v)


class FinanceAgentSearchResponse(BaseModel):
    """Response model for SEC filing search."""

    results: str = Field(description="JSON string of filing results")


class EdgarSearchRequest(BaseModel):
    """Request model for local full-text EDGAR search."""

    search_query: str = Field(description="Case-insensitive search term or phrase")
    form_types: Optional[List[str]] = Field(default=None, description="Optional EDGAR form types")
    ciks: Optional[List[str]] = Field(default=None, description="Optional CIK filters")
    start_date: Optional[str] = Field(default="1900-01-01", description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(default=None, description="End date (YYYY-MM-DD)")
    page: int = Field(default=1, ge=1, description="Result page number")
    top_n_results: int = Field(default=100, ge=1, le=100, description="Maximum results to return")

    @field_validator("form_types", "ciks", mode="before")
    @classmethod
    def _coerce_filters(cls, v: Any) -> Any:
        return _coerce_stringified_collection(v)


class EdgarSearchResponse(BaseModel):
    """Response model for local full-text EDGAR search."""

    results: str = Field(description="JSON string of sec-api-compatible search results")


class RetrieveInformationRequest(BaseModel):
    """Request model for retrieve_information tool."""

    prompt: str = Field(
        description=(
            "An LLM prompt applied to your saved documents. You MUST include at least "
            "one data-storage key using the exact double-brace format {{key_name}} -- "
            "for example: 'Summarize this 10-K filing: {{company_10k}}'. The full text "
            "stored under each key replaces its {{key_name}} placeholder before the "
            "prompt is sent. If you do not use this exact {{key_name}} format, the tool "
            "will fail."
        )
    )
    input_character_ranges: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Optional list of character ranges: [{'key': 'doc', 'start': 0, 'end': 100000}]"
    )

    @field_validator("input_character_ranges", mode="before")
    @classmethod
    def _coerce_input_character_ranges(cls, v: Any) -> Any:
        return _coerce_stringified_collection(v)


class RetrieveInformationResponse(BaseModel):
    """Response model for retrieve_information tool."""

    results: str = Field(description="LLM response text from querying stored documents")


class SubmitFinalResultRequest(BaseModel):
    """Request model for submit_final_result tool."""

    final_result: str = Field(description="The final result to submit")


class SubmitFinalResultResponse(BaseModel):
    """Response model for submit_final_result tool."""

    results: str = Field(description="Confirmation of submission")


class WebSearchRequest(BaseModel):
    """Request model for web_search tool."""

    search_query: str = Field(description="The query to search for")
    start_date: Optional[str] = Field(default=None, description="Start date for search range (YYYY-MM-DD)")
    end_date: Optional[str] = Field(default=None, description="End date for search range (YYYY-MM-DD)")
    number_of_results: Optional[int] = Field(default=10, ge=1, le=20, description="Number of search results to return")


class WebSearchResponse(BaseModel):
    """Response model for web_search tool."""

    results: str = Field(description="JSON string with search results")


_DatedToolRequest = TypeVar("_DatedToolRequest", FinanceAgentSearchRequest, EdgarSearchRequest, WebSearchRequest)


class ParseHtmlPageRequest(BaseModel):
    """Request model for parse_html_page tool."""

    url: str = Field(description="The URL of the HTML page to parse")
    key: str = Field(description="The key to use when saving the result in the conversation's data storage.")


class ParseHtmlPageResponse(BaseModel):
    """Response model for parse_html_page tool."""

    results: str = Field(description="Status message about data storage operation")


class FinanceAgentSeedSessionRequest(BaseSeedSessionRequest):
    """Seed request: the row, of which only ``as_of_date`` is read."""

    # Any type, so a row's as_of_date of any shape never fails a seed while cap_tool_dates_at_as_of_date is off.
    as_of_date: Optional[Any] = Field(
        default=None,
        description="cap_tool_dates_at_as_of_date only: YYYY-MM-DD date the question is answered as of. "
        "Null leaves tool dates uncapped.",
    )


def cap_tool_dates(args: Dict[str, Any], as_of_date: str) -> Dict[str, Any]:
    """Return ``args`` with ``end_date`` and ``start_date`` no later than ``as_of_date`` (YYYY-MM-DD)."""
    capped = dict(args)
    end_date = capped.get("end_date")
    if not isinstance(end_date, str) or not end_date or end_date > as_of_date:
        capped["end_date"] = as_of_date
    start_date = capped.get("start_date")
    if isinstance(start_date, str) and start_date > as_of_date:
        capped["start_date"] = as_of_date
    return capped


class FinanceAgentRunRequest(BaseRunRequest):
    """Run request with question and expected answer."""

    question: str
    expected_answer: str


class FinanceAgentVerifyRequest(FinanceAgentRunRequest, BaseVerifyRequest):
    """Verify request for SEC search tasks."""

    # Any type, so a row's rubric of any shape never fails a request in the other reward modes, and
    # excluded from model_dump(), so no response the other modes write carries it.
    rubric: Optional[Any] = Field(
        default=None,
        exclude=True,
        description="reward_mode 'rubric' only: JSON string of the row's rubric criteria, as finance_agent_v2 "
        "reads it. Each entry has 'criteria' (the claim to grade), 'operator', and optionally 'modifiers' with "
        "'severity' (weight) and 'category' ('must_pass' marks a dealbreaker).",
    )


class FinanceAgentVerifyResponse(BaseVerifyResponse):
    """Verify response for SEC search tasks."""

    expected_answer: str
    judge_rating: Optional[int] = None
    judge_text: Optional[str] = None


class FinanceAgentRubricVerifyResponse(FinanceAgentVerifyResponse):
    """Verify response for reward_mode 'rubric', with finance_agent_v2's rubric fields.

    ``reward`` is Vals's Partial Credit: the severity-weighted pass fraction, forced
    to 0.0 if any ``must_pass`` criterion failed.
    """

    # Full per-criterion record — the debugging trail and the disagreement signal.
    rubric_judgements: Optional[List[RubricJudgement]] = None
    rubric_total: Optional[int] = None
    rubric_passed: Optional[int] = None
    rubric_unresolved: Optional[int] = None
    # Unweighted pass fraction.
    rubric_fraction: Optional[float] = None
    rubric_all_pass: Optional[bool] = None
    # The reward, restated under its Vals name so the leaderboard metric is greppable.
    rubric_partial_credit: Optional[float] = None
    # Partial Credit before gating; the gap between the two is what the dealbreakers
    # cost.
    rubric_weighted_fraction: Optional[float] = None
    rubric_weight_total: Optional[float] = None
    rubric_weight_passed: Optional[float] = None
    rubric_dealbreakers_total: Optional[int] = None
    rubric_dealbreakers_failed: Optional[int] = None
    # Set when scoring could not complete (no rubric, or an unresolved criterion), so
    # a judge failure is filterable rather than read as a genuine miss.
    judge_error: Optional[str] = None


# ============================================================================
# Rate Limiter
# ============================================================================


class RateLimiter:
    """Sliding window rate limiter for SEC.gov compliance."""

    def __init__(self, max_requests: int = 10, window_seconds: float = 1.0):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: deque = deque()
        self.lock = asyncio.Lock()

    async def acquire(self):
        """Wait until a request slot is available."""
        while True:
            async with self.lock:
                now = time.monotonic()
                while self.requests and (now - self.requests[0]) >= self.window_seconds:
                    self.requests.popleft()
                if len(self.requests) < self.max_requests:
                    self.requests.append(now)
                    return
                sleep_time = self.window_seconds - (now - self.requests[0])
            await asyncio.sleep(max(sleep_time, 0.01))


def per_worker_rate_limiter(requests_per_second: int, num_workers: int) -> RateLimiter:
    """The rate limiter for one worker, so that all workers together stay within `requests_per_second`.

    SEC.gov's limit applies to the whole server, but each uvicorn worker keeps its own limiter.
    Each worker therefore gets an equal share of the limit, rounded down,
    so in any one-second window the workers together admit at most `requests_per_second` requests.
    The server refuses more workers than `requests_per_second`, where a share would be below one request.
    """
    if num_workers > requests_per_second:
        raise ValueError(f"{num_workers} workers cannot share a limit of {requests_per_second} requests per second")
    return RateLimiter(max_requests=requests_per_second // num_workers, window_seconds=1.0)


# ============================================================================
# Finance SEC Search Resource Server
# ============================================================================


class FinanceAgentResourcesServer(SimpleResourcesServer):
    """
    SEC EDGAR Filing Search Resource Server.
    - /sec_filing_search: Search for SEC filings by ticker or company name
    - /edgar_search: Search the local EDGAR full-text index
    - /parse_html_page: Fetch, parse, and store any HTML page (SEC URLs use XBRL-aware parsing + disk cache)
    - /retrieve_information: Query stored documents via LLM prompt with {{key}} syntax
    - /web_search: Tavily web search
    - /submit_final_result: Submit the final answer
    """

    config: FinanceAgentResourcesServerConfig

    def multiple_workers_unsupported_reason(self) -> Optional[str]:
        num_workers = self.config.num_workers or 1
        if num_workers <= self.config.requests_per_second:
            return None
        # Several workers are allowed up to requests_per_second, so the reason names that limit:
        # the refusal's "needs num_workers=1" prefix alone would understate it.
        return (
            f"it runs with at most requests_per_second ({self.config.requests_per_second}) workers, not {num_workers}, "
            "because with more, each worker's own rate limiter would admit a request at once, "
            "so the workers together could exceed SEC.gov's limit"
        )

    def model_post_init(self, context):
        """Initialize after Pydantic model creation."""
        # Refuse an unsupported worker count before building the per-worker rate limiter.
        refuse_unsupported_num_workers(self)
        if not self.config.cache_dir:
            default = Path.home() / ".cache" / "nemo_gym" / "finance_sec_search"
            logger.warning(
                "cache_dir not set; defaulting to %s. "
                "This path is ephemeral in containers and not shared across Slurm jobs. "
                "Set cache_dir to a shared absolute path for production/multi-seed use.",
                default,
            )
            self._cache_dir = default
        else:
            self._cache_dir = Path(self.config.cache_dir)
            if not self._cache_dir.is_absolute():
                self._cache_dir = Path.cwd() / self._cache_dir
                logger.info("Resolved relative cache_dir to %s", self._cache_dir)
        self._filings_metadata_dir = self._cache_dir / "filings_metadata"
        self._filings_dir = self._cache_dir / "filings"
        self._tickers_file = self._cache_dir / "tickers.json"
        # Only materialize the on-disk cache when caching is enabled. With
        # use_cache=False every request fetches live and no dirs are created.
        if self.config.use_cache:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            make_shared_dirs(self._filings_metadata_dir)
            make_shared_dirs(self._filings_dir)

        self._rate_limiter = per_worker_rate_limiter(self.config.requests_per_second, self.config.num_workers or 1)

        self._tickers: Dict[str, Dict[str, str]] = {}  # ticker -> {"cik": ..., "name": ...}
        self._filings_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}  # cik -> {acc_nodash -> filing_meta}
        # Filled in by edgar_search, read by filing fetches. Separate from
        # _filings_cache, which is per-accession and mirrored to disk, because
        # this is per-document (exhibits included) and process-local.
        self._dump_paths: Dict[str, str] = {}  # canonical document key -> path below sec_dump_path
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()
        self._filings_locks: Dict[str, asyncio.Lock] = {}
        self._initialized = False

        # session_id -> {key -> parsed text content}; scoped by HTTP session cookie
        self._data_storage: Dict[str, Dict[str, str]] = {}
        self._session_start_times: Dict[str, float] = {}
        # session_id -> the as_of_date its tool dates are capped at (cap_tool_dates_at_as_of_date).
        self._as_of_dates: Dict[str, str] = {}

        # Inline template takes priority over file
        if self.config.judge_prompt_template:
            self._judge_prompt_template = self.config.judge_prompt_template.strip()
        else:
            with open(self.config.judge_prompt_template_fpath, "r") as f:
                data = yaml.safe_load(f)
            self._judge_prompt_template = data["judge_prompt_template"].strip()

        if self.config.reward_mode == "rubric":
            if self.config.rubric_judge_prompt_template:
                self._rubric_judge_prompt_template = self.config.rubric_judge_prompt_template.strip()
            else:
                with open(self.config.rubric_judge_prompt_template_fpath, "r") as f:
                    self._rubric_judge_prompt_template = yaml.safe_load(f)["rubric_judge_prompt_template"].strip()

        if self.config.retrieval_system_prompt:
            self._retrieval_system_prompt = self.config.retrieval_system_prompt.strip()
        else:
            with open(self.config.retrieval_system_prompt_fpath, "r") as f:
                data = yaml.safe_load(f)
            self._retrieval_system_prompt = data["retrieval_system_prompt"].strip()

        self._tavily = None
        if self.config.tavily_api_key:
            try:
                from tavily import TavilyClient

                self._tavily = TavilyClient(api_key=self.config.tavily_api_key)
                logger.info("Tavily web search initialized successfully")
            except ImportError:
                logger.warning(
                    "tavily_api_key is configured but the 'tavily' package is not installed. "
                    "web_search will be unavailable. Install with: pip install tavily"
                )
        else:
            logger.info("No tavily_api_key configured — web_search will be unavailable")

        self._filing_read_sources: Counter[str] = Counter()
        self._filing_read_logged_at: Optional[float] = None

        self._local_edgar_search: Optional[LocalEdgarSearch] = None
        self._edgar_search_service: Optional[EdgarSearchService] = None
        cutoff = self.config.max_end_date or DEFAULT_MAX_END_DATE
        mode = self.config.edgar_search_mode
        if mode == "local":
            if not self.config.local_edgar_index_path:
                raise ValueError(
                    "edgar_search_mode is 'local' but local_edgar_index_path is not set. Local mode serves "
                    "edgar_search entirely from that index; without it every search would fail mid-rollout."
                )
            self._local_edgar_search = LocalEdgarSearch(
                self.config.local_edgar_index_path,
                max_end_date=cutoff,
                metrics_dir=self.config.local_edgar_metrics_dir,
                metadata_path=self.config.local_edgar_metadata_path,
            )
            self._edgar_search_service = EdgarSearchService(
                self._local_edgar_search,
                max_end_date=cutoff,
                on_results=self._record_dump_paths,
            )
            logger.info(
                "edgar_search: local mode, index %s (metadata sidecar: %s, coverage %s)",
                self.config.local_edgar_index_path,
                self._local_edgar_search.metadata_path or "none",
                self._local_edgar_search.coverage,
            )
        elif mode == "live":
            if not self.config.sec_api_key:
                raise ValueError(
                    "edgar_search_mode is 'live' but sec_api_key is not set. Set the key, or leave "
                    "edgar_search_mode unset to run without edgar_search."
                )
            self._edgar_search_service = EdgarSearchService(
                LiveEdgarSearch(
                    self.config.sec_api_key,
                    session_provider=self._get_session,
                    max_retries=self.config.max_retries,
                    request_timeout=self.config.request_timeout,
                ),
                max_end_date=cutoff,
            )
            logger.info("edgar_search: live mode against sec-api.io")
        else:
            logger.info("edgar_search: unavailable — edgar_search_mode is not set")

    def _get_session_storage(self, session_id: str) -> Dict[str, str]:
        """Get or create the data storage dict for a session."""
        if session_id not in self._data_storage:
            self._data_storage[session_id] = {}
        return self._data_storage[session_id]

    def _check_time_budget(self, session_id: str) -> Optional[str]:
        """Return an error message if the rollout has exceeded its time budget, else None."""
        if not self.config.max_rollout_time_seconds:
            return None
        start = self._session_start_times.get(session_id)
        if start is None:
            return None
        elapsed = time.monotonic() - start
        if elapsed > self.config.max_rollout_time_seconds:
            logger.warning(
                "Session %s exceeded time budget (%.0fs > %.0fs)",
                session_id,
                elapsed,
                self.config.max_rollout_time_seconds,
            )
            return json.dumps(
                {
                    "error": f"Time budget exhausted ({elapsed:.0f}s / {self.config.max_rollout_time_seconds:.0f}s). "
                    "No further tool calls will be executed. Call submit_final_result immediately with your best answer."
                }
            )
        return None

    def _capped_at_as_of_date(self, request: Request, body: _DatedToolRequest) -> _DatedToolRequest:
        """``body`` with its dates capped at the session's as_of_date, when it has one."""
        as_of_date = self._as_of_dates.get(request.session.get(SESSION_ID_KEY, ""))
        if as_of_date is None:
            return body
        return body.model_copy(
            update=cap_tool_dates({"start_date": body.start_date, "end_date": body.end_date}, as_of_date)
        )

    def _checked_as_of_date(self, value: Any) -> str:
        """``value`` if it is a YYYY-MM-DD calendar date, which finance_agent_v2's seed also requires."""
        if isinstance(value, str) and self._DATE_RE.match(value):
            try:
                date.fromisoformat(value)
                return value
            except ValueError:
                pass
        raise HTTPException(status_code=422, detail=f"as_of_date must be a YYYY-MM-DD date, got {value!r}")

    async def seed_session(self, request: Request, body: FinanceAgentSeedSessionRequest) -> BaseSeedSessionResponse:
        """Reset per-question data storage for this session.

        With cap_tool_dates_at_as_of_date, the session's first seed also sets the date its tool calls are
        capped at, and a later seed never moves it: the agent forwards any tool name the model emits, so the
        model can reach this endpoint mid-rollout.
        """
        session_id = request.session[SESSION_ID_KEY]
        if (
            self.config.cap_tool_dates_at_as_of_date
            and body.as_of_date is not None
            and session_id not in self._as_of_dates
        ):
            self._as_of_dates[session_id] = self._checked_as_of_date(body.as_of_date)
        self._data_storage[session_id] = {}
        self._session_start_times[session_id] = time.monotonic()
        logger.debug("seed_session: reset data storage for session %s", session_id)
        if len(self._data_storage) > 128:
            logger.warning(
                "data_storage has %d active sessions — possible leak (verify cleanup failing?)",
                len(self._data_storage),
            )
        return await super().seed_session(body)

    def setup_webserver(self) -> FastAPI:
        """Register API routes."""
        app = super().setup_webserver()

        self._load_tickers_or_fail()

        app.post("/sec_filing_search")(self.sec_filing_search)
        app.post("/edgar_search")(self.edgar_search)
        app.post("/parse_html_page")(self.parse_html_page)
        app.post("/retrieve_information")(self.retrieve_information)
        app.post("/submit_final_result")(self.submit_final_result)
        app.post("/web_search")(self.web_search)

        @app.post("/{tool_name}")
        async def handle_unknown_tool(tool_name: str):
            return {
                "results": json.dumps(
                    {
                        "error": f"Tool '{tool_name}' does not exist. Available tools: "
                        "sec_filing_search, edgar_search, parse_html_page, "
                        "retrieve_information, submit_final_result, web_search"
                    }
                )
            }

        # Process-level hook: not every FastAPI version exposes a shutdown
        # handler API.
        atexit.register(self._log_filing_read_sources)

        return app

    def _log_filing_read_sources(self) -> None:
        logger.warning(
            "SEC filing reads by source: %s",
            " ".join(f"{source}={self._filing_read_sources[source]}" for source in FILING_READ_SOURCES),
        )

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create the shared HTTP session."""
        async with self._session_lock:
            if self._session is None or self._session.closed:
                connector = aiohttp.TCPConnector(
                    limit=50,
                    limit_per_host=self.config.max_connections_per_host,
                )
                timeout = aiohttp.ClientTimeout(total=self.config.request_timeout * 2)
                self._session = aiohttp.ClientSession(
                    headers={"User-Agent": self.config.user_agent},
                    connector=connector,
                    timeout=timeout,
                )
            return self._session

    async def _fetch_with_retry(self, url: str) -> Optional[str]:
        """Fetch URL with rate limiting, retries, and per-request timeout."""
        session = await self._get_session()
        req_timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)

        for attempt in range(self.config.max_retries):
            await self._rate_limiter.acquire()
            try:
                async with session.get(url, timeout=req_timeout) as response:
                    if response.status == 200:
                        raw = await response.read()
                        encoding = response.charset or "utf-8"
                        try:
                            return raw.decode(encoding)
                        except (UnicodeDecodeError, LookupError):
                            return raw.decode("latin-1")
                    if response.status in (403, 429, 503):
                        logger.warning(
                            "SEC.gov %d on attempt %d/%d for %s",
                            response.status,
                            attempt + 1,
                            self.config.max_retries,
                            url,
                        )
                        await asyncio.sleep(2**attempt)
                        continue
                    logger.warning("SEC.gov %d (non-retryable) for %s", response.status, url)
                    return None
            except (aiohttp.ClientError, asyncio.TimeoutError):
                logger.warning(
                    "Fetch error on attempt %d/%d for %s",
                    attempt + 1,
                    self.config.max_retries,
                    url,
                    exc_info=True,
                )
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(2**attempt)
        return None

    def _load_tickers_or_fail(self):
        """Load ticker mappings at startup. Raises RuntimeError on failure.

        Tries the on-disk cache first, then fetches from SEC with 5 retries
        and exponential backoff.  Called from setup_webserver so the server
        never starts without valid ticker data.
        """
        SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
        MAX_RETRIES = 5

        raw = None

        if self.config.use_cache and self._tickers_file.exists():
            try:
                with open(self._tickers_file, "r") as f:
                    raw = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Cached tickers.json is corrupt (%s), re-downloading", e)
                raw = None

        if raw is None:
            for attempt in range(MAX_RETRIES):
                try:
                    req = urllib.request.Request(SEC_TICKERS_URL, headers={"User-Agent": self.config.user_agent})
                    with urllib.request.urlopen(req, timeout=30) as resp:
                        data = resp.read().decode("utf-8")
                    raw = json.loads(data)
                    if self.config.use_cache:
                        self._atomic_write(self._tickers_file, json.dumps(raw))
                    break
                except (urllib.error.URLError, json.JSONDecodeError, OSError) as e:
                    wait = 2**attempt
                    logger.warning(
                        "Ticker download attempt %d/%d failed: %s (retrying in %ds)",
                        attempt + 1,
                        MAX_RETRIES,
                        e,
                        wait,
                    )
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(wait)

        if not raw:
            raise RuntimeError(
                "Failed to load SEC ticker data after retries. Server cannot start without company_tickers.json."
            )

        raw = self._overlay_supplementary_tickers(raw)

        for item in raw.values():
            self._tickers[item["ticker"].upper()] = {"cik": str(item["cik_str"]).zfill(10), "name": item["title"]}
        self._initialized = True
        logger.info("Loaded %d ticker mappings", len(self._tickers))

    def _overlay_supplementary_tickers(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Merge the overlay onto an SEC registry. Overlay wins; one row per ticker.

        Registry keys are positional and carry no meaning, so overlay keys are
        namespaced to keep them from colliding with SEC's own numbering.
        """
        fpath = self.config.supplementary_tickers_fpath
        if not fpath:
            return raw
        path = Path(fpath)
        if not path.is_file():
            raise RuntimeError(f"supplementary_tickers_fpath not found: {path}")
        with open(path, "r") as f:
            extra = json.load(f)
        if not isinstance(extra, dict):
            raise RuntimeError(f"supplementary_tickers_fpath must be a JSON object: {path}")
        overridden = {item["ticker"].upper() for item in extra.values()}
        merged = {key: item for key, item in raw.items() if item["ticker"].upper() not in overridden}
        for key, item in extra.items():
            merged[f"supplementary-{key}"] = item
        logger.info("Overlaid %d supplementary ticker mappings from %s", len(extra), path)
        return merged

    async def _resolve_ticker(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Look up a ticker symbol. Returns company info dict or None."""
        query = ticker.strip().upper()
        info = self._tickers.get(query)
        if info is None:
            return None
        return {"cik": info["cik"], "ticker": query, "name": info["name"]}

    # ========================================================================
    # Filing Metadata
    # ========================================================================

    def _get_company_cache_path(self, cik: str) -> Path:
        """Cache file path for a company's filing metadata (CIK zero-padded to 10 digits)."""
        return self._filings_metadata_dir / f"{str(cik).zfill(10)}.json"

    @staticmethod
    def _atomic_write(path: Path, content: str, encoding: str = "utf-8") -> None:
        atomic_write(path, content, encoding)

    @staticmethod
    def _read_cache(path: Path) -> Optional[str]:
        """Read a cache entry; an unreadable entry is a miss, not a tool failure."""
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError as e:
            logger.warning("Ignoring unreadable cache entry %s: %s", path, e)
            return None

    def _write_cache(self, path: Path, content: str) -> None:
        """Write a cache entry; a failed write leaves the result uncached rather than failing the tool."""
        try:
            self._atomic_write(path, content)
        except OSError as e:
            logger.warning("Could not cache %s: %s", path, e)

    @staticmethod
    def _parse_filings_columns(columns: Dict[str, Any], cik: str, ticker: str) -> Dict[str, Dict[str, Any]]:
        """Parse SEC columnar filing data into a dict keyed by accession number (no dashes)."""
        acc_numbers = columns.get("accessionNumber", [])
        forms = columns.get("form", [])
        dates = columns.get("filingDate", [])
        report_dates = columns.get("reportDate", [])
        primary_docs = columns.get("primaryDocument", [])

        filings: Dict[str, Dict[str, Any]] = {}
        for acc, form, fdate, rdate, pdoc in zip(acc_numbers, forms, dates, report_dates, primary_docs):
            acc_nodash = acc.replace("-", "")
            filings[acc_nodash] = {
                "ticker": ticker,
                "cik": cik,
                "form": form,
                "filing_date": fdate,
                "report_date": rdate,
                "accession_number": acc,
                "primary_document": pdoc,
                "filing_url": f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{acc_nodash}/{pdoc}",
            }
        return filings

    async def _get_company_filings(self, cik: str, ticker: str) -> Dict[str, Dict[str, Any]]:
        """Get filings for a company. Memory cache → disk cache → SEC.gov.

        Uses per-CIK locking so concurrent requests for the same company
        coalesce into a single fetch instead of stampeding SEC.gov.
        """
        cik_padded = str(cik).zfill(10)

        if cik_padded in self._filings_cache:
            return self._filings_cache[cik_padded]

        lock = self._filings_locks.setdefault(cik_padded, asyncio.Lock())
        async with lock:
            if cik_padded in self._filings_cache:
                return self._filings_cache[cik_padded]

            cache_path = self._get_company_cache_path(cik)
            cached = self._read_cache(cache_path) if self.config.use_cache else None
            if cached is not None:
                filings = json.loads(cached)
                self._filings_cache[cik_padded] = filings
                return filings

            data = await self._fetch_with_retry(f"https://data.sec.gov/submissions/CIK{cik_padded}.json")
            if not data:
                logger.warning("SEC submissions API unavailable for CIK %s (%s)", cik, ticker)
                return {}

            try:
                filings_data = json.loads(data).get("filings", {})
                recent = filings_data.get("recent", {})

                filings = self._parse_filings_columns(recent, cik, ticker)

                for file_ref in filings_data.get("files", []):
                    filename = file_ref.get("name", "")
                    if not filename:
                        continue
                    extra_data = await self._fetch_with_retry(f"https://data.sec.gov/submissions/{filename}")
                    if extra_data:
                        try:
                            extra = json.loads(extra_data)
                            filings.update(self._parse_filings_columns(extra, cik, ticker))
                        except json.JSONDecodeError:
                            logger.warning("Failed to parse supplementary file %s for CIK %s", filename, cik)

                if filings and self.config.use_cache:
                    self._write_cache(cache_path, json.dumps(filings))
                self._filings_cache[cik_padded] = filings
                return filings
            except (json.JSONDecodeError, KeyError):
                logger.warning("Failed to parse SEC submissions for CIK %s (%s)", cik, ticker, exc_info=True)
                return {}

    # ========================================================================
    # Dump Fallback
    # ========================================================================

    async def _read_dump_file(self, dump_path: Path) -> Optional[str]:
        """Read and parse one file from the read-only dump, or None."""
        if not dump_path.is_file():
            return None

        def _read_and_parse() -> str:
            return self._parse_html_to_text(dump_path.read_text(encoding="utf-8"))

        try:
            return await asyncio.get_running_loop().run_in_executor(None, _read_and_parse)
        except OSError:
            logger.warning("Failed to read dump file %s", dump_path)
            return None

    async def _record_dump_paths(self, results: List[Dict[str, Any]]) -> None:
        """Remember where edgar_search's hits live in the dump, for later reads.

        Failures are non-fatal: the search result stands, reads just fall back.
        """
        if not self.config.sec_dump_path or self._local_edgar_search is None:
            return
        urls = [
            url
            for url in (str(result.get("filingUrl") or "") for result in results)
            if url and canonical_url_key(url) not in self._dump_paths
        ]
        if not urls:
            return
        try:
            self._dump_paths.update(await self._local_edgar_search.dump_paths_for_urls_async(urls))
        except Exception:
            logger.warning("Failed to resolve dump paths for %d search results", len(urls), exc_info=True)

    async def _lookup_dump(self, url: str) -> Optional[str]:
        """Try to read a filing from the pre-fetched SEC dump (read-only).

        Returns parsed plain text or None.
        """
        if not self.config.sec_dump_path:
            return None
        root = Path(self.config.sec_dump_path)

        # A path edgar_search recorded. The index holds a row per document, so
        # this is the only route that can name an exhibit rather than a filing's
        # primary document.
        key = canonical_url_key(url)
        relative_path = self._dump_paths.get(key) if key else None
        if relative_path:
            candidate = Path(relative_path)
            if candidate.is_absolute() or ".." in candidate.parts:
                logger.warning("Rejected unsafe dump path %s", relative_path)
            else:
                text_content = await self._read_dump_file(root / candidate)
                if text_content is not None:
                    return text_content
                logger.warning("Index recorded %s but the corpus has no readable file there", relative_path)
            # The index named this URL's own document, so the rebuild below could
            # only answer with a different one.
            return None

        # Otherwise rebuild the path from filing metadata that sec_filing_search
        # left in memory. Uses report_date for the year and form.replace("/", "_")
        # for the form folder, matching download_filings.py. That metadata is keyed
        # per filing, so the filename can only ever be the primary document.
        parts = self._parse_sec_url(url)
        if not parts:
            return None

        metadata = self._filings_cache.get(parts["cik"])
        if not metadata:
            return None

        filing_meta = metadata.get(parts["accession_number"].replace("-", ""))
        if not filing_meta:
            return None

        # Reading primary-document.html for an exhibit URL returns the wrong text
        # and reports success, so decline whenever the URL names another document.
        document = parts.get("document", "").strip()
        primary_document = str(filing_meta.get("primary_document", "")).strip()
        if document and primary_document and document.lower() != primary_document.lower():
            return None

        ticker = filing_meta.get("ticker", "")
        form = filing_meta.get("form", "").replace("/", "_")
        report_date = filing_meta.get("report_date", "")
        year = report_date[:4] if len(report_date) >= 4 else ""
        accession = filing_meta.get("accession_number", "")

        if not all([ticker, form, year, accession]):
            return None

        return await self._read_dump_file(root / ticker / form / year / accession / "primary-document.html")

    # ========================================================================
    # URL Parsing
    # ========================================================================

    def _parse_sec_url(self, url: str) -> Optional[Dict[str, str]]:
        """Parse SEC URL to extract CIK, accession number, and document filename."""
        parsed = parse_sec_archives_url(url)
        if parsed is None:
            return None
        return {
            "cik": parsed.padded_cik,
            "accession_number": parsed.dashed_accession,
            "document": parsed.document_path,
        }

    def _url_to_filing_path(self, url: str) -> Optional[Path]:
        """Convert a SEC EDGAR URL to its local cache file path.

        Keyed by (CIK, accession, document): distinct documents under one accession
        (e.g. edgar_search sub-documents/exhibits) map to distinct cache files.
        Returns None if the URL doesn't match the expected SEC format.
        """
        parsed = parse_sec_archives_url(url)
        if parsed is None:
            return None
        # cache_filename never contains a bare "%", so this cannot clash with a document.
        filename = parsed.cache_filename or "%index"
        return self._filings_dir / parsed.padded_cik / parsed.accession / f"{filename}.txt"

    # ========================================================================
    # sec_filing_search Endpoint
    # ========================================================================

    async def sec_filing_search(self, request: Request, body: FinanceAgentSearchRequest) -> FinanceAgentSearchResponse:
        """Search for SEC filings by ticker symbol.

        Returns filing metadata entries (sorted by date, newest first),
        capped at max_filing_results. Supports optional form_types,
        start_date, and end_date filters.
        """
        if timeout_msg := self._check_time_budget(request.session.get(SESSION_ID_KEY, "")):
            return FinanceAgentSearchResponse(results=timeout_msg)
        body = self._capped_at_as_of_date(request, body)

        company = await self._resolve_ticker(body.ticker)

        if not company:
            return FinanceAgentSearchResponse(
                results=json.dumps(
                    {
                        "error": f"No company found for ticker '{body.ticker}'",
                        "suggestion": "Use the exact stock ticker symbol (e.g., 'AAPL' for Apple, 'MSFT' for Microsoft). "
                        "Note: only companies listed at https://www.sec.gov/files/company_tickers.json are supported.",
                    }
                )
            )

        filings = await self._get_company_filings(company["cik"], company["ticker"])
        form_types = body.form_types

        all_results = []
        for filing in filings.values():
            if form_types and filing["form"] not in form_types:
                continue

            all_results.append(
                {
                    "ticker": company["ticker"],
                    "company_name": company["name"],
                    "form": filing["form"],
                    "filing_date": filing.get("filing_date", ""),
                    "report_date": filing.get("report_date", ""),
                    "accession_number": filing.get("accession_number", ""),
                    "filing_url": filing.get("filing_url", ""),
                }
            )

        all_results.sort(key=lambda x: x["filing_date"], reverse=True)

        if body.start_date:
            all_results = [r for r in all_results if r["filing_date"] >= body.start_date]
        if body.end_date:
            all_results = [r for r in all_results if r["filing_date"] <= body.end_date]

        all_results = all_results[: self.config.max_filing_results]

        if not all_results:
            filters = []
            if form_types:
                filters.append(f"form types {form_types}")
            if body.start_date:
                filters.append(f"start_date={body.start_date}")
            if body.end_date:
                filters.append(f"end_date={body.end_date}")
            filter_msg = f" with {', '.join(filters)}" if filters else ""
            return FinanceAgentSearchResponse(
                results=json.dumps(
                    {
                        "error": f"No filings found for '{body.ticker}'{filter_msg}",
                        "suggestion": "Try broadening your search: remove form_types filter, widen the date range, or check the ticker symbol.",
                    }
                )
            )

        return FinanceAgentSearchResponse(results=json.dumps(all_results, indent=2))

    # ========================================================================
    # edgar_search Endpoint
    # ========================================================================

    async def edgar_search(self, request: Request, body: EdgarSearchRequest) -> EdgarSearchResponse:
        """Full-text search EDGAR, from the local index or sec-api.io."""
        if timeout_msg := self._check_time_budget(request.session.get(SESSION_ID_KEY, "")):
            return EdgarSearchResponse(results=timeout_msg)

        if self._edgar_search_service is None:
            return EdgarSearchResponse(
                results=json.dumps({"error": "edgar_search is not available. edgar_search_mode is not set."})
            )

        body = self._capped_at_as_of_date(request, body)
        return EdgarSearchResponse(results=await self._edgar_search_service.run(body.model_dump()))

    # ========================================================================
    # parse_html_page Endpoint
    # ========================================================================

    @staticmethod
    def _parse_html_to_text(html_content: str) -> str:
        """Extract plain text from HTML."""
        return html_to_text(html_content)

    async def _parse_html_page(self, url: str) -> str:
        """Fetch a URL and extract plain text, reusing the shared session."""
        html_content = await self._fetch_with_retry(url)
        if not html_content:
            raise RuntimeError(f"Failed to fetch {url}. The server may be temporarily unavailable.")
        return self._parse_html_to_text(html_content)

    async def _fetch_sec_filing_text(self, url: str) -> str:
        """Fetch a SEC filing URL using the disk-cache / dump / live pipeline.

        Uses the same generic HTML parser as non-SEC URLs
        but caches the parsed text to disk for subsequent calls.

        Raises on failure (caller handles the exception).
        """
        file_path = self._url_to_filing_path(url)
        if file_path is None:
            raise ValueError(f"Invalid SEC URL format: {url}")

        text_content = self._read_cache(file_path) if self.config.use_cache else None
        source = "cache" if text_content is not None else None

        if text_content is None:
            text_content = await self._lookup_dump(url)
            if text_content is not None:
                source = "sec-corpus"
                if self.config.use_cache:
                    self._write_cache(file_path, text_content)

        if text_content is None:
            html_content = await self._fetch_with_retry(url)
            if not html_content:
                raise RuntimeError(
                    f"Failed to download filing from {url}. The SEC server may be temporarily unavailable."
                )

            text_content = await asyncio.get_running_loop().run_in_executor(
                None, self._parse_html_to_text, html_content
            )
            source = "live"
            if self.config.use_cache:
                self._write_cache(file_path, text_content)

        if not text_content:
            raise ValueError("Filing content was empty after parsing.")

        self._filing_read_sources[source] += 1
        # Logs the first read, then at most one line per interval, so a count
        # survives a server that is killed without a clean shutdown.
        now = time.monotonic()
        if self._filing_read_logged_at is None or now - self._filing_read_logged_at >= FILING_READ_LOG_INTERVAL_SEC:
            self._filing_read_logged_at = now
            self._log_filing_read_sources()
        return text_content

    async def _save_tool_output(self, output: str, key: str, state: dict[str, Any]) -> str:
        if not output:
            raise ValueError("HTML output was empty")

        tool_result = ""
        if key in state:
            tool_result = (
                "WARNING: The key already exists in the data storage. The new result overwrites the old one.\n"
            )
        tool_result += f"SUCCESS: The result has been saved to the data storage under the key: {key}.\n"

        state[key] = output

        keys_list = "\n".join(state.keys())
        tool_result += f"The data_storage currently contains the following keys:\n{keys_list}\n"

        return tool_result

    async def parse_html_page(self, request: Request, body: ParseHtmlPageRequest) -> ParseHtmlPageResponse:
        """Parse an HTML page from any URL and store in session data storage.

        SEC URLs are detected automatically and routed through the
        disk-cache / dump / live-download pipeline (with caching).
        All URLs use the same generic BeautifulSoup parser.
        """
        if timeout_msg := self._check_time_budget(request.session.get(SESSION_ID_KEY, "")):
            return ParseHtmlPageResponse(results=timeout_msg)

        url, key = body.url, body.key
        if not url:
            return ParseHtmlPageResponse(results="ERROR: url is required.")
        if not key:
            return ParseHtmlPageResponse(results="ERROR: key is required.")

        storage = self._get_session_storage(request.session[SESSION_ID_KEY])

        try:
            if self._parse_sec_url(url):
                text_output = await self._fetch_sec_filing_text(url)
            else:
                text_output = await self._parse_html_page(url)
            result_msg = await self._save_tool_output(text_output, key, storage)
        except Exception as e:
            error_msg = str(e)
            logger.warning("parse_html_page failed for %s: %s", url, error_msg)
            return ParseHtmlPageResponse(results=error_msg)

        return ParseHtmlPageResponse(results=result_msg)

    # ========================================================================
    # retrieve_information Endpoint (LLM-based document querying)
    # ========================================================================

    async def retrieve_information(
        self, request: Request, body: RetrieveInformationRequest
    ) -> RetrieveInformationResponse:
        """Query stored documents using LLM-based prompting."""
        if timeout_msg := self._check_time_budget(request.session.get(SESSION_ID_KEY, "")):
            return RetrieveInformationResponse(results=timeout_msg)

        if not self.config.retrieval_model_server:
            return RetrieveInformationResponse(
                results="ERROR: Retrieval model not configured. Set retrieval_model_server in config."
            )

        storage = self._get_session_storage(request.session[SESSION_ID_KEY])
        prompt = body.prompt

        # Extract {{key}} placeholders from prompt
        keys_in_prompt = re.findall(r"\{\{([^{}]+)\}\}", prompt)
        if not keys_in_prompt:
            return RetrieveInformationResponse(
                results="ERROR: Your prompt must include at least one key from data storage "
                "in the format {{key_name}}. Please try again with the correct format. "
                "You can add documents to the data storage with parse_html_page."
            )

        # Validate all keys exist in data storage
        for key in keys_in_prompt:
            if key not in storage:
                available = ", ".join(storage.keys()) if storage else ""
                return RetrieveInformationResponse(
                    results=f"ERROR: The key '{key}' was not found in the data storage. "
                    f"Available keys are: {available}. "
                    "Use the parse_html_page tool to add keys to the data storage."
                )

        ranges_dict: Dict[str, tuple] = {}
        for r in body.input_character_ranges or []:
            if isinstance(r, dict) and all(k in r for k in ("key", "start", "end")):
                ranges_dict[r["key"]] = (r["start"], r["end"])

        final_prompt = prompt
        for key in keys_in_prompt:
            content = storage[key]
            if key in ranges_dict:
                start, end = ranges_dict[key]
                content = content[start:end]
            final_prompt = final_prompt.replace("{{" + key + "}}", content)

        try:
            retrieval_params = (
                self.config.retrieval_responses_create_params or NeMoGymResponseCreateParamsNonStreaming(input=[])
            ).model_copy(deep=True)
            retrieval_params.input = [
                NeMoGymEasyInputMessage(role="system", content=self._retrieval_system_prompt),
                NeMoGymEasyInputMessage(role="user", content=final_prompt),
            ]
            if retrieval_params.max_output_tokens is None:
                retrieval_params.max_output_tokens = self.config.retrieval_max_output_tokens

            llm_response = await self.server_client.post(
                server_name=self.config.retrieval_model_server.name,
                url_path="/v1/responses",
                json=retrieval_params,
            )

            # Surface HTTP-level failures (e.g. 4xx context-overflow from vLLM)
            # explicitly rather than letting them fall through to the vague
            # "no output" branch.  Body is capped to avoid polluting agent
            # context with multi-KB error bodies (e.g. vLLM HTML pages).
            if not llm_response.ok:
                body_text = (await llm_response.text())[:500]
                return RetrieveInformationResponse(
                    results=f"ERROR: Retrieval LLM HTTP {llm_response.status}: {body_text}"
                )

            llm_response_json = await get_response_json(llm_response)
            llm_response_obj = NeMoGymResponse.model_validate(llm_response_json)

            result_text = ""
            for output_item in llm_response_obj.output:
                if getattr(output_item, "type", None) == "message":
                    for content_item in getattr(output_item, "content", []):
                        if getattr(content_item, "type", None) == "output_text":
                            result_text += getattr(content_item, "text", "")

            if not result_text:
                # Include any diagnostic the server returned so the agent can
                # see why output was empty (e.g. incomplete_details.reason ==
                # "max_output_tokens" or content_filter).  Bare "no output"
                # masks these.
                diagnostic_parts: List[str] = []
                incomplete_details = getattr(llm_response_obj, "incomplete_details", None)
                if incomplete_details is not None:
                    reason = getattr(incomplete_details, "reason", None)
                    if reason:
                        diagnostic_parts.append(f"incomplete_details.reason={reason}")
                status = getattr(llm_response_obj, "status", None)
                if status:
                    diagnostic_parts.append(f"status={status}")
                error_field = getattr(llm_response_obj, "error", None)
                if error_field is not None:
                    diagnostic_parts.append(f"error={error_field}")
                diagnostic = (" (" + ", ".join(diagnostic_parts) + ")") if diagnostic_parts else ""
                return RetrieveInformationResponse(results=f"ERROR: Retrieval LLM returned no output.{diagnostic}")

            return RetrieveInformationResponse(results=result_text)

        except Exception as e:
            return RetrieveInformationResponse(results=f"ERROR: Retrieval LLM call failed: {str(e)}")

    # ========================================================================
    # Date Validation Helper
    # ========================================================================

    _DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

    def _validate_and_clamp_dates(
        self,
        start_date: Optional[str],
        end_date: Optional[str],
        default_start: str = "1900-01-01",
    ) -> tuple:
        max_end_date = getattr(self.config, "max_end_date", None)

        if end_date:
            if not self._DATE_RE.match(end_date):
                raise ValueError(f"Invalid end_date format: '{end_date}'. Expected YYYY-MM-DD.")
            if max_end_date and end_date > max_end_date:
                end_date = max_end_date
        elif max_end_date:
            end_date = max_end_date

        if start_date:
            if not self._DATE_RE.match(start_date):
                raise ValueError(f"Invalid start_date format: '{start_date}'. Expected YYYY-MM-DD.")
            if max_end_date and start_date > max_end_date:
                start_date = max_end_date
        else:
            start_date = default_start

        if start_date and end_date and start_date > end_date:
            raise ValueError(f"start_date '{start_date}' is later than end_date '{end_date}'.")

        return start_date, end_date

    async def submit_final_result(self, body: SubmitFinalResultRequest) -> SubmitFinalResultResponse:
        """Accept the agent's final answer submission."""
        final_result = body.final_result
        if not final_result:
            return SubmitFinalResultResponse(results="ERROR: final_result is required. Please provide your answer.")
        return SubmitFinalResultResponse(results=json.dumps({"success": True, "result": final_result}))

    async def web_search(self, request: Request, body: WebSearchRequest) -> WebSearchResponse:
        """Search the web using Tavily."""
        if timeout_msg := self._check_time_budget(request.session.get(SESSION_ID_KEY, "")):
            return WebSearchResponse(results=timeout_msg)

        if self._tavily is None:
            return WebSearchResponse(
                results=json.dumps(
                    {
                        "error": "web_search is not available. Use sec_filing_search, parse_html_page, and retrieve_information instead.",
                    }
                )
            )

        search_query = body.search_query
        if not search_query or not search_query.strip():
            return WebSearchResponse(results=json.dumps({"error": "search_query is required and cannot be empty."}))

        body = self._capped_at_as_of_date(request, body)
        try:
            _, end_date = self._validate_and_clamp_dates(None, body.end_date)
        except ValueError as e:
            return WebSearchResponse(results=json.dumps({"error": str(e)}))

        kwargs: Dict[str, Any] = {}
        if body.start_date:
            try:
                start_date, end_date = self._validate_and_clamp_dates(body.start_date, end_date)
            except ValueError as e:
                return WebSearchResponse(results=json.dumps({"error": str(e)}))
            kwargs["start_date"] = start_date

        num_results = min(body.number_of_results or 10, 20)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                raw = await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: self._tavily.search(
                        query=search_query,
                        search_depth="fast",
                        end_date=end_date,
                        max_results=num_results,
                        chunks_per_source=1,
                        **kwargs,
                    ),
                )
                results = raw.get("results", [])
                return WebSearchResponse(results=json.dumps(results, default=str))
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"web_search attempt {attempt + 1} failed: {e}. Retrying in {2**attempt}s...")
                    await asyncio.sleep(2**attempt)
                else:
                    logger.error(f"web_search failed after {max_retries} attempts: {e}")
                    return WebSearchResponse(results=json.dumps({"error": str(e)}))

    # ========================================================================
    # Verify Endpoint
    # ========================================================================

    async def _call_judge(self, judge_params: NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymResponse:
        # reraise_judge_errors also covers the wait_for timeout, which sits outside call_judge.
        return await reraise_judge_errors(
            asyncio.wait_for(
                call_judge(
                    self.server_client,
                    server_name=self.config.judge_model_server.name,
                    url_path="/v1/responses",
                    json=judge_params,
                    response_model=NeMoGymResponse,
                ),
                timeout=self.config.judge_call_timeout,
            )
        )

    # reward_mode 'rubric'
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_modifiers(entry: Dict[str, Any]) -> tuple[float, bool]:
        """Read ``(severity, must_pass)`` off one rubric entry's ``modifiers``.

        Defaults to ``(1.0, False)``, which degenerates to the unweighted mean for a
        rubric without modifiers. A bad severity is coerced rather than raised — one malformed
        weight should not discard a completed trajectory — and zero is excluded because
        a zero-weight criterion still gates when it is a dealbreaker.
        """
        modifiers = entry.get("modifiers")
        if not isinstance(modifiers, dict):
            return 1.0, False
        raw = modifiers.get("severity")
        try:
            severity = float(raw)
        except (TypeError, ValueError):
            severity = 1.0
        if not severity > 0 or severity != severity or severity == float("inf"):
            logger.warning("Rubric criterion has unusable severity %r — weighting it 1.0", raw)
            severity = 1.0
        return severity, modifiers.get("category") == "must_pass"

    @classmethod
    def _parse_rubric(cls, rubric: Optional[str]) -> List[Dict[str, Any]]:
        """Parse the dataset's rubric JSON into criteria dicts with their weights.

        Returns [] for anything unusable, which verify() treats as an unscorable dry
        run rather than a zero.
        """
        if not rubric:
            return []
        try:
            parsed = json.loads(rubric)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Rubric is not valid JSON — cannot score")
            return []
        if not isinstance(parsed, list):
            logger.warning("Rubric JSON is %s, expected a list — cannot score", type(parsed).__name__)
            return []

        criteria: List[Dict[str, Any]] = []
        for entry in parsed:
            if not isinstance(entry, dict):
                continue
            text = entry.get("criteria")
            if isinstance(text, str) and text.strip():
                severity, must_pass = cls._parse_modifiers(entry)
                criteria.append(
                    {
                        "criteria": text.strip(),
                        "operator": entry.get("operator"),
                        "severity": severity,
                        "must_pass": must_pass,
                    }
                )
        return criteria

    @staticmethod
    def _json_objects(text: str) -> List[str]:
        """Return the top-level ``{...}`` spans of ``text``, in order.

        Brace counting with string tracking, not a regex: the judge quotes the answer
        verbatim, and finance answers carry LaTeX subscripts like ``EBITDAR_{WMT}``.
        A ``{.*?}`` match ends at that inner brace and, since findall does not
        backtrack, misaligns every later candidate.
        """
        spans: List[str] = []
        depth = 0
        start = -1
        in_string = False
        escaped = False
        for i, ch in enumerate(text):
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}" and depth:
                depth -= 1
                if depth == 0:
                    spans.append(text[start : i + 1])
        return spans

    @classmethod
    def _extract_score(cls, judge_text: str) -> Optional[int]:
        """Pull the binary ``score`` out of a judge reply, or None if unusable.

        Accepts a bare JSON object or one fenced in markdown, and scans candidate
        objects from the end so a worked example quoted mid-reasoning cannot
        outrank the judge's actual verdict. Only integer 0/1 counts — a string
        ``"pass"`` or a float is a parse failure, because silently coercing those
        would let a confused judge vote.
        """
        if not judge_text or not judge_text.strip():
            return None

        for blob in reversed(cls._json_objects(judge_text)):
            try:
                payload = json.loads(blob)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(payload, dict) or "score" not in payload:
                continue
            score = payload["score"]
            # bool is an int subclass in Python; True/False is not a verdict.
            if isinstance(score, int) and not isinstance(score, bool) and score in (0, 1):
                return score
            return None
        return None

    @staticmethod
    def _judge_reply_text(judge_response: NeMoGymResponse) -> str:
        """Concatenate the visible text of a judge response."""
        text = ""
        for item in judge_response.output:
            if getattr(item, "type", None) != "message":
                continue
            for part in getattr(item, "content", []) or []:
                chunk = getattr(part, "text", None)
                if isinstance(chunk, str):
                    text += chunk
        return text

    async def _judge_attempt(self, prompt: str, budget_escalations: int) -> _JudgeAttempt:
        """Make one judge call.

        ``budget_escalations`` counts prior attempts that ran out of output budget;
        each doubles ``max_output_tokens`` and stretches the timeout. Keyed to that
        failure rather than the attempt number, because the ordinary failure is an
        unparseable reply, which a bigger budget does not fix.
        """
        params = (
            self.config.judge_responses_create_params or NeMoGymResponseCreateParamsNonStreaming(input=[])
        ).model_copy(deep=True)
        params.input = [NeMoGymEasyInputMessage(role="user", content=prompt)]

        timeout = self.config.judge_call_timeout
        if budget_escalations and params.max_output_tokens:
            params.max_output_tokens = min(
                params.max_output_tokens * 2**budget_escalations, _JUDGE_MAX_OUTPUT_TOKENS_CAP
            )
            if timeout is not None:
                timeout *= min(2**budget_escalations, 4)

        try:
            response = await asyncio.wait_for(
                self.server_client.post(
                    server_name=self.config.judge_model_server.name,
                    url_path="/v1/responses",
                    json=params,
                ),
                timeout=timeout,
            )
            judge_response = NeMoGymResponse.model_validate(await get_response_json(response))
        except Exception as e:  # noqa: BLE001
            return _JudgeAttempt(None, f"judge call failed: {type(e).__name__}: {e}", "", False)

        reply = self._judge_reply_text(judge_response)
        score = self._extract_score(reply)
        if score is not None:
            return _JudgeAttempt(score, None, reply, False)

        incomplete_reason = getattr(getattr(judge_response, "incomplete_details", None), "reason", None)
        if incomplete_reason:
            error = f"judge reply incomplete (incomplete_details.reason={incomplete_reason})"
        elif not reply.strip():
            error = "judge returned empty output"
        else:
            error = "judge reply had no integer 0/1 score"
        starved = incomplete_reason == "max_output_tokens" or not reply.strip()
        return _JudgeAttempt(None, error, reply, starved)

    async def _judge_criterion(
        self, question: str, generated_answer: str, criterion: Dict[str, Any]
    ) -> RubricJudgement:
        """Judge one criterion, voting over repeated calls.

        Calls until ``judge_required_successes`` replies parse to a 0/1 or
        ``judge_max_attempts`` is spent. Falling short leaves ``score`` None
        (unresolved) — a judge failure, not a "not met".
        """
        prompt = (
            self._rubric_judge_prompt_template.replace("{question}", question)
            .replace("{generated_answer}", generated_answer)
            .replace("{criterion}", criterion["criteria"])
        )

        record = RubricJudgement(
            criteria=criterion["criteria"],
            operator=criterion.get("operator"),
            severity=criterion.get("severity", 1.0),
            must_pass=criterion.get("must_pass", False),
        )
        needed = max(1, self.config.judge_required_successes)
        last_error: Optional[str] = None
        budget_escalations = 0

        max_attempts = max(1, self.config.judge_max_attempts)
        for attempt in range(max_attempts):
            if len(record.votes) >= needed:
                break
            record.attempts_used += 1
            result = await self._judge_attempt(prompt, budget_escalations)

            if result.score is None:
                last_error = result.error
                if result.reply:
                    record.parse_failures += 1
                else:
                    record.api_failures += 1
                if result.starved:
                    budget_escalations += 1
                if len(record.failed_reply_samples) < _MAX_FAILED_REPLY_SAMPLES:
                    record.failed_reply_samples.append(f"{result.error}: {result.reply[:_FAILED_REPLY_SAMPLE_CHARS]}")
                logger.warning(
                    "Rubric judge attempt %d failed (%s) for criterion: %.80s",
                    record.attempts_used,
                    result.error,
                    criterion["criteria"],
                )
                # Between failures only, capped so a flaky judge cannot stretch one
                # criterion past the rollout budget.
                if attempt < max_attempts - 1:
                    await asyncio.sleep(min(2**attempt, 8))
                continue

            record.votes.append(result.score)
            payload = self._extract_json_payload(result.reply)
            record.evidence.append(str(payload.get("extracted_evidence", ""))[:1000])
            record.reasons.append(str(payload.get("reason", ""))[:1000])

        record.votes_for = sum(record.votes)
        record.votes_against = len(record.votes) - record.votes_for

        if len(record.votes) < needed:
            record.error = last_error or f"only {len(record.votes)}/{needed} successful judge verdicts"
            logger.error(
                "Criterion unresolved after %d attempts (%s): %.80s",
                record.attempts_used,
                record.error,
                criterion["criteria"],
            )
            return record

        record.score = 1 if record.votes_for * 2 > len(record.votes) else 0
        record.unanimous = record.votes_for == 0 or record.votes_against == 0
        if not record.unanimous:
            logger.info(
                "Judge disagreed %d-%d on criterion: %.80s",
                record.votes_for,
                record.votes_against,
                criterion["criteria"],
            )
        return record

    @classmethod
    def _extract_json_payload(cls, judge_text: str) -> Dict[str, Any]:
        """Return the scored JSON object from a reply, or {} if not found.

        Mirrors _extract_score's last-object-wins scan so the recorded evidence
        and reason come from the same object the verdict came from.
        """
        for blob in reversed(cls._json_objects(judge_text or "")):
            try:
                payload = json.loads(blob)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(payload, dict) and "score" in payload:
                return payload
        return {}

    async def _verify_rubric(self, body: FinanceAgentVerifyRequest, question: str) -> FinanceAgentRubricVerifyResponse:
        """Score the agent's answer against the row's rubric criteria, as finance_agent_v2's /verify does.

        Each criterion is judged independently by majority vote, then aggregated into
        Partial Credit, which becomes the reward. An unresolved criterion counts as not
        passed and gates like one, but also sets ``judge_error`` so a judge outage never
        reads as a model failure.
        """
        generated_answer = ""
        for output_item in reversed(body.response.output):
            if getattr(output_item, "type", None) == "function_call":
                if getattr(output_item, "name", None) == "submit_final_result":
                    # Arguments are model-authored, so nothing about their shape is
                    # guaranteed. Anything that is not a string answer is treated as
                    # no submission, which scores zero, rather than raising here.
                    try:
                        args = json.loads(getattr(output_item, "arguments", "{}"))
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                    if isinstance(args, dict) and isinstance(args.get("final_result"), str):
                        generated_answer = args["final_result"]
                    break

        criteria = self._parse_rubric(body.rubric)

        # A genuine failure, not a judge failure. Return before spending on the judge.
        # The score fields are written as explicit zeros rather than left unset, so
        # this rollout counts in every mean the way it already counts in mean/reward.
        if not generated_answer:
            return FinanceAgentRubricVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                rubric_fraction=0.0,
                rubric_all_pass=False,
                rubric_partial_credit=0.0,
                rubric_weighted_fraction=0.0,
                rubric_total=len(criteria),
            )

        if not criteria:
            return FinanceAgentRubricVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                judge_error="no usable rubric criteria — unlabeled dry run, not scorable",
            )

        if not self.config.judge_model_server:
            return FinanceAgentRubricVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                rubric_total=len(criteria),
                judge_error="no judge_model_server configured — cannot score the rubric",
            )

        # Criteria are independent; a question is ~9 criteria x 3-10 calls, so judge
        # them concurrently but under a semaphore to spare the judge endpoint.
        semaphore = asyncio.Semaphore(max(1, self.config.judge_max_concurrency))

        async def judge_one(criterion: Dict[str, Any]) -> RubricJudgement:
            async with semaphore:
                return await self._judge_criterion(question, generated_answer, criterion)

        judgements = await asyncio.gather(*(judge_one(c) for c in criteria))

        scores = aggregate_rubric_scores(judgements)

        judge_error = None
        if scores["rubric_unresolved"]:
            first = next(j for j in judgements if j.score is None)
            judge_error = (
                f"{scores['rubric_unresolved']}/{scores['rubric_total']} criteria unresolved (e.g. {first.error})"
            )

        logger.info(
            "Rubric verdict: %d/%d passed, %d unresolved, %d/%d dealbreakers failed, partial_credit=%.3f",
            scores["rubric_passed"],
            scores["rubric_total"],
            scores["rubric_unresolved"],
            scores["rubric_dealbreakers_failed"],
            scores["rubric_dealbreakers_total"],
            scores["rubric_partial_credit"],
        )

        return FinanceAgentRubricVerifyResponse(
            **body.model_dump(),
            reward=scores["rubric_partial_credit"],
            rubric_judgements=list(judgements),
            **scores,
            judge_error=judge_error,
        )

    async def verify(
        self, request: Request, body: FinanceAgentVerifyRequest
    ) -> Union[FinanceAgentVerifyResponse, FinanceAgentRubricVerifyResponse]:
        """Verify the agent's answer.

        Rating scale (reward depends on config.reward_mode):
            [[2]] = fully correct  → binary: 1.0 | scaled: 1.0
            [[1]] = partial        → binary: 0.0 | scaled: 0.5
            [[0]] = incorrect      → binary: 0.0 | scaled: 0.0
        """
        session_id = request.session.get(SESSION_ID_KEY)
        if session_id:
            self._data_storage.pop(session_id, None)
            self._session_start_times.pop(session_id, None)
            self._as_of_dates.pop(session_id, None)

        question = ""
        for msg in body.responses_create_params.input or []:
            if getattr(msg, "role", None) == "user":
                content = getattr(msg, "content", None)
                if isinstance(content, str):
                    question = content

        if self.config.reward_mode == "rubric":
            response = await self._verify_rubric(body, question)
            if not response.rubric_unresolved:
                return response
            # The judge failed, not the policy, so the sample is masked as a binary-mode judge failure is.
            # The reward stays finance_agent_v2's.
            return response.model_copy(
                update={"mask_sample": True, "failure_kind": JUDGE_FAILED, "failure_reason": response.judge_error}
            )

        # Extract the model's answer from the submit_final_result tool call.
        # HARD GATE: we only accept answers submitted via submit_final_result.
        # Previously we fell back to the last assistant text message, which
        # created a reward shortcut: the model could skip all tool use and
        # answer from parametric knowledge. A lenient judge then still gave
        # partial/full credit, and GRPO reinforced the no-tool policy.
        # Now, rollouts without a valid submit_final_result call get reward=0.
        generated_answer = ""
        submit_final_result_called = False
        for output_item in reversed(body.response.output):
            if getattr(output_item, "type", None) == "function_call":
                if getattr(output_item, "name", None) == "submit_final_result":
                    submit_final_result_called = True
                    try:
                        args = json.loads(getattr(output_item, "arguments", "{}"))
                        generated_answer = args.get("final_result", "")
                    except (json.JSONDecodeError, TypeError):
                        pass
                    break

        if not submit_final_result_called or not generated_answer:
            logger.info(
                "Hard gate: reward=0 (submit_final_result_called=%s, final_result_present=%s)",
                submit_final_result_called,
                bool(generated_answer),
            )
            return FinanceAgentVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                judge_rating=0,
                judge_text=(
                    "Hard gate: submit_final_result tool call missing or empty. "
                    "The agent must submit its final answer via submit_final_result to be graded."
                ),
            )

        # No judge model → substring match
        if not self.config.judge_model_server:
            reward = 1.0 if body.expected_answer.lower() in generated_answer.lower() else 0.0
            return FinanceAgentVerifyResponse(**body.model_dump(), reward=reward)

        # Legacy mode: [[0]]/[[1]]/[[2]] judge
        judge_user_prompt = self._judge_prompt_template
        judge_user_prompt = judge_user_prompt.replace("{question}", question)
        judge_user_prompt = judge_user_prompt.replace("{expected_answer}", body.expected_answer)
        # A candidate must not be able to plant a parseable rating for the judge to quote back (#2852).
        judge_user_prompt = judge_user_prompt.replace(
            "{generated_answer}", _JUDGE_RATING_RE.sub(r"[\1]", generated_answer)
        )

        judge_params = (
            self.config.judge_responses_create_params or NeMoGymResponseCreateParamsNonStreaming(input=[])
        ).model_copy(deep=True)
        judge_params.input = [
            NeMoGymEasyInputMessage(role="user", content=judge_user_prompt),
        ]

        max_judge_retries = 3
        judge_text = ""
        rating = None

        for attempt in range(max_judge_retries):
            try:
                judge_response = await self._call_judge(judge_params)
            except JUDGE_FAILURES as judge_error:
                if not is_judge_failure(judge_error):
                    raise
                logger.warning("Judge call attempt %d/%d failed: %s", attempt + 1, max_judge_retries, judge_error)
                if attempt < max_judge_retries - 1:
                    await asyncio.sleep(2**attempt)
                    continue
                logger.error("Judge model call failed after %d attempts", max_judge_retries)
                raise

            try:
                last_output = judge_response.output[-1]
                if getattr(last_output, "type", None) == "message":
                    last_content = last_output.content[-1]
                    judge_text = getattr(last_content, "text", "")
            except Exception:
                pass

            # A reply cut off by max_output_tokens never reached its verdict: retry rather than read a
            # tentative rating out of the unfinished reasoning.
            rating = None if judge_response.incomplete_details else _extract_judge_rating(judge_text)

            if rating is not None:
                break

            logger.warning(
                "Judge returned no [[N]] rating (attempt %d/%d, incomplete_details=%s). Output: %s",
                attempt + 1,
                max_judge_retries,
                judge_response.incomplete_details,
                judge_text[:200],
            )
            if attempt < max_judge_retries - 1:
                await asyncio.sleep(2**attempt)

        if self.config.reward_mode == "scaled":
            _REWARD_MAP = {0: 0.0, 1: 0.5, 2: 1.0}
            reward = _REWARD_MAP.get(rating, 0.0)
        else:
            reward = 1.0 if rating == 2 else 0.0

        return FinanceAgentVerifyResponse(
            **body.model_dump(), reward=reward, judge_rating=rating, judge_text=judge_text
        )

    # ------------------------------------------------------------------
    # Aggregate metrics (reward_mode 'rubric')
    # ------------------------------------------------------------------
    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Rubric-level metrics on top of the RewardProfiler baseline.

        Partial Credit alone cannot separate a genuine miss from a judge that never
        answered, nor broad sloppiness from one tripped dealbreaker. So:

        - ``rubric_weighted_fraction`` is Partial Credit before gating; the pair
          prices what the dealbreakers cost.
        - ``rubric_all_pass`` is the strict rate: every criterion met.
        - ``criterion_pass_rate`` pools criteria, so rubric length does not skew it.
        - ``judge_disagreement_rate`` is the share of resolved criteria the judge
          contradicted itself on; a rise means a less decisive judge, not a worse model.
        - the ``rubric/*`` counters keep judge failures out of the score.
        """
        if self.config.reward_mode != "rubric":
            return super().compute_metrics(tasks)
        rollouts = [r for task in tasks for r in task]
        if not rollouts:
            return {}

        # Gave up: nothing was judged and the judge did not fail either. A
        # pre-weighting rollout also carries no judgements, but it does carry a
        # score, so a nonzero fraction rules it out — and one that scored zero
        # belongs in the means at zero regardless of which bucket it lands in.
        give_ups = [
            r
            for r in rollouts
            if not r.get("rubric_judgements") and not r.get("judge_error") and not r.get("rubric_fraction")
        ]

        def _scores(field: str, zero: Any) -> List[Any]:
            """Values of ``field`` across rollouts, scoring a give-up as ``zero``.

            Scoring writes those zeros itself; rollouts collected before it did are
            filled in here, so a give-up weighs in every mean the way it has always
            weighed in mean/reward. Dropping it would score a run on the subset of
            questions it managed to answer.
            """
            scored = [r[field] for r in rollouts if r.get(field) is not None]
            return scored + [zero] * sum(1 for r in give_ups if r.get(field) is None)

        fractions = _scores("rubric_fraction", 0.0)
        all_pass_flags = [bool(flag) for flag in _scores("rubric_all_pass", False)]
        partial_credits = _scores("rubric_partial_credit", 0.0)
        weighted_fractions = _scores("rubric_weighted_fraction", 0.0)
        # Scored rollouts only: one with no submission was never eligible to trip a
        # dealbreaker, so counting it would dilute the rate.
        gated = [r for r in rollouts if r.get("rubric_dealbreakers_total")]

        criteria: List[Dict[str, Any]] = []
        for r in rollouts:
            for j in r.get("rubric_judgements") or []:
                if isinstance(j, dict):
                    criteria.append(j)

        resolved = [j for j in criteria if j.get("score") is not None]
        unresolved = len(criteria) - len(resolved)
        # Only resolved criteria carry a meaningful unanimity flag.
        contested = sum(1 for j in resolved if j.get("unanimous") is False)
        judge_error_rollouts = sum(1 for r in rollouts if r.get("judge_error"))
        no_submission = len(give_ups)

        metrics: Dict[str, Any] = {
            "rubric/rollouts": len(rollouts),
            "rubric/criteria_total": len(criteria),
            "rubric/criteria_resolved": len(resolved),
            "rubric/criteria_unresolved": unresolved,
            "rubric/rollouts_with_judge_error": judge_error_rollouts,
            "rubric/rollouts_without_submission": no_submission,
            "rubric/parse_failures": sum(int(j.get("parse_failures") or 0) for j in criteria),
            "rubric/api_failures": sum(int(j.get("api_failures") or 0) for j in criteria),
        }
        if partial_credits:
            metrics["mean/rubric_partial_credit"] = sum(partial_credits) / len(partial_credits)
        if weighted_fractions:
            metrics["mean/rubric_weighted_fraction"] = sum(weighted_fractions) / len(weighted_fractions)
        if fractions:
            metrics["mean/rubric_fraction"] = sum(fractions) / len(fractions)
        if all_pass_flags:
            metrics["mean/rubric_all_pass"] = sum(all_pass_flags) / len(all_pass_flags)
        if gated:
            # "Tripped", not "zeroed by": a rollout that failed everything would
            # score 0.0 with or without the gate.
            metrics["mean/rubric_dealbreaker_tripped"] = sum(
                1 for r in gated if r.get("rubric_dealbreakers_failed")
            ) / len(gated)
            metrics["rubric/dealbreakers_total"] = sum(int(r.get("rubric_dealbreakers_total") or 0) for r in gated)
            metrics["rubric/dealbreakers_failed"] = sum(int(r.get("rubric_dealbreakers_failed") or 0) for r in gated)
        if resolved:
            metrics["mean/criterion_pass_rate"] = sum(1 for j in resolved if j.get("score") == 1) / len(resolved)
            metrics["mean/judge_disagreement_rate"] = contested / len(resolved)
        if criteria:
            metrics["rubric/mean_attempts_per_criterion"] = sum(
                int(j.get("attempts_used") or 0) for j in criteria
            ) / len(criteria)
        return metrics

    def compute_repeat_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Apply the full-run rubric scoring rules to each repeat's means."""
        if self.config.reward_mode != "rubric":
            return super().compute_repeat_metrics(tasks)
        return {name: value for name, value in self.compute_metrics(tasks).items() if name.startswith("mean/")}

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Headline metrics: the usual mean/* plus judge-health counters.

        The counters are promoted deliberately — a run whose judge failed on a
        third of its criteria should not be read as a low score, and that is only
        obvious if the failure count sits next to the score.
        """
        if self.config.reward_mode != "rubric":
            return super().get_key_metrics(agent_metrics)
        key = {k: v for k, v in agent_metrics.items() if k.startswith("mean/")}
        for k in (
            "rubric/criteria_unresolved",
            "rubric/rollouts_with_judge_error",
            "rubric/rollouts_without_submission",
            "rubric/dealbreakers_failed",
        ):
            if k in agent_metrics:
                key[k] = agent_metrics[k]
        return key


def _configure_logging() -> None:
    # Root stays at WARNING: at INFO the HTTP client logs a line per model call.
    logging.basicConfig(level=logging.WARNING)
    for _logger_name in (__name__, "resources_servers"):
        logging.getLogger(_logger_name).setLevel(os.environ.get("NEMO_GYM_LOG_LEVEL", "INFO").upper())


if __name__ == "__main__":
    _configure_logging()
    FinanceAgentResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # With num_workers > 1, uvicorn imports this module in each worker and serves its module-level `app`.
    _configure_logging()
    app = FinanceAgentResourcesServer.run_webserver()  # noqa: F401
