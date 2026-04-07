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
import contextlib
import json
import logging
import os
import re
import tempfile
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
import yaml
from bs4 import BeautifulSoup
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator
from starlette.requests import Request
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
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
from nemo_gym.server_utils import SESSION_ID_KEY, get_response_json


logger = logging.getLogger(__name__)


class FinanceAgentResourcesServerConfig(BaseResourcesServerConfig):
    """Configuration for Finance SEC Search resource server."""

    cache_dir: Optional[str] = Field(
        default=None,
        description="Path for caching ticker mappings and filing metadata. Defaults to ~/.cache/nemo_gym/finance_sec_search/ if not set. Relative paths are resolved from cwd.",
    )
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
    reward_mode: str = Field(
        default="binary",
        description="How judge ratings map to rewards. "
        "'binary': only [[2]] → 1.0, else 0.0. "
        "'scaled': [[0]] → 0.0, [[1]] → 0.5, [[2]] → 1.0.",
    )
    retrieval_max_output_tokens: int = Field(
        default=8192,
        description="Max output tokens for retrieve_information LLM calls. Increase for thinking models.",
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


class RetrieveInformationRequest(BaseModel):
    """Request model for retrieve_information tool."""

    prompt: str = Field(description="Prompt with {{key_name}} placeholders for stored documents.")
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


class ParseHtmlPageRequest(BaseModel):
    """Request model for parse_html_page tool."""

    url: str = Field(description="The URL of the HTML page to parse")
    key: str = Field(description="The key to use when saving the result in the conversation's data storage.")


class ParseHtmlPageResponse(BaseModel):
    """Response model for parse_html_page tool."""

    results: str = Field(description="Status message about data storage operation")


class FinanceAgentRunRequest(BaseRunRequest):
    """Run request with question and expected answer."""

    question: str
    expected_answer: str


class FinanceAgentVerifyRequest(FinanceAgentRunRequest, BaseVerifyRequest):
    """Verify request for SEC search tasks."""

    pass


class FinanceAgentVerifyResponse(BaseVerifyResponse):
    """Verify response for SEC search tasks."""

    expected_answer: str
    judge_rating: Optional[int] = None
    judge_text: Optional[str] = None


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


# ============================================================================
# Finance SEC Search Resource Server
# ============================================================================


class FinanceAgentResourcesServer(SimpleResourcesServer):
    """
    SEC EDGAR Filing Search Resource Server.
    - /sec_filing_search: Search for SEC filings by ticker or company name
    - /parse_html_page: Fetch, parse, and store any HTML page (SEC URLs use XBRL-aware parsing + disk cache)
    - /retrieve_information: Query stored documents via LLM prompt with {{key}} syntax
    - /web_search: Tavily web search
    - /submit_final_result: Submit the final answer
    """

    config: FinanceAgentResourcesServerConfig

    def model_post_init(self, context):
        """Initialize after Pydantic model creation."""
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
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._filings_metadata_dir = self._cache_dir / "filings_metadata"
        self._filings_metadata_dir.mkdir(exist_ok=True)
        self._filings_dir = self._cache_dir / "filings"
        self._filings_dir.mkdir(exist_ok=True)
        self._tickers_file = self._cache_dir / "tickers.json"

        self._rate_limiter = RateLimiter(max_requests=self.config.requests_per_second, window_seconds=1.0)

        self._tickers: Dict[str, Dict[str, str]] = {}  # ticker -> {"cik": ..., "name": ...}
        self._filings_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}  # cik -> {acc_nodash -> filing_meta}
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()
        self._filings_locks: Dict[str, asyncio.Lock] = {}
        self._initialized = False

        # session_id -> {key -> parsed text content}; scoped by HTTP session cookie
        self._data_storage: Dict[str, Dict[str, str]] = {}
        self._session_start_times: Dict[str, float] = {}

        # Inline template takes priority over file
        if self.config.judge_prompt_template:
            self._judge_prompt_template = self.config.judge_prompt_template.strip()
        else:
            with open(self.config.judge_prompt_template_fpath, "r") as f:
                data = yaml.safe_load(f)
            self._judge_prompt_template = data["judge_prompt_template"].strip()

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

    async def seed_session(self, request: Request, body: BaseSeedSessionRequest) -> BaseSeedSessionResponse:
        """Reset per-question data storage for this session."""
        session_id = request.session[SESSION_ID_KEY]
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
                        "sec_filing_search, parse_html_page, "
                        "retrieve_information, submit_final_result, web_search"
                    }
                )
            }

        return app

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

        if self._tickers_file.exists():
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
                    with open(self._tickers_file, "w") as f:
                        json.dump(raw, f)
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

        for item in raw.values():
            self._tickers[item["ticker"]] = {"cik": str(item["cik_str"]).zfill(10), "name": item["title"]}
        self._initialized = True
        logger.info("Loaded %d ticker mappings", len(self._tickers))

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
        """Write content to path atomically via temp-file + rename."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding=encoding) as f:
                f.write(content)
            os.replace(tmp, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

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

        if cik_padded not in self._filings_locks:
            self._filings_locks[cik_padded] = asyncio.Lock()
        async with self._filings_locks[cik_padded]:
            if cik_padded in self._filings_cache:
                return self._filings_cache[cik_padded]

            cache_path = self._get_company_cache_path(cik)
            if cache_path.exists():
                with open(cache_path, "r") as f:
                    filings = json.load(f)
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

                if filings:
                    self._atomic_write(cache_path, json.dumps(filings))
                self._filings_cache[cik_padded] = filings
                return filings
            except (json.JSONDecodeError, KeyError):
                logger.warning("Failed to parse SEC submissions for CIK %s (%s)", cik, ticker, exc_info=True)
                return {}

    # ========================================================================
    # Dump Fallback
    # ========================================================================

    async def _lookup_dump(self, url: str) -> Optional[str]:
        """Try to read a filing from the pre-fetched SEC dump (read-only).

        Derives the dump path from in-memory metadata cache:
        {sec_dump_path}/{TICKER}/{FORM}/{YEAR}/{ACCESSION}/primary-document.html

        Uses report_date for year and form.replace("/", "_") for the form folder,
        matching the conventions of the download_filings.py script.
        Returns parsed plain text or None.
        """
        if not self.config.sec_dump_path:
            return None

        parts = self._parse_sec_url(url)
        if not parts:
            return None

        cik_padded = parts["cik"]
        acc_nodash = parts["accession_number"].replace("-", "")

        metadata = self._filings_cache.get(cik_padded)
        if not metadata:
            return None

        filing_meta = metadata.get(acc_nodash)
        if not filing_meta:
            return None

        ticker = filing_meta.get("ticker", "")
        form = filing_meta.get("form", "").replace("/", "_")
        report_date = filing_meta.get("report_date", "")
        year = report_date[:4] if len(report_date) >= 4 else ""
        accession = filing_meta.get("accession_number", "")

        if not all([ticker, form, year, accession]):
            return None

        dump_path = Path(self.config.sec_dump_path) / ticker / form / year / accession / "primary-document.html"
        if not dump_path.exists():
            return None

        def _read_and_parse(p: Path) -> str:
            return self._parse_html_to_text(p.read_text(encoding="utf-8"))

        try:
            return await asyncio.get_running_loop().run_in_executor(None, _read_and_parse, dump_path)
        except OSError:
            logger.warning("Failed to read dump file %s", dump_path)
            return None

    # ========================================================================
    # URL Parsing
    # ========================================================================

    def _parse_sec_url(self, url: str) -> Optional[Dict[str, str]]:
        """Parse SEC URL to extract CIK and accession number."""
        # URL format: https://www.sec.gov/Archives/edgar/data/{CIK}/{ACCESSION_NODASH}/{filename}
        pattern = r"sec\.gov/Archives/edgar/data/(\d+)/(\d+)/"
        match = re.search(pattern, url)
        if match:
            cik = match.group(1).zfill(10)
            acc_nodash = match.group(2)
            # Convert to formatted accession: 0001234567-12-123456
            if len(acc_nodash) == 18:
                accession = f"{acc_nodash[:10]}-{acc_nodash[10:12]}-{acc_nodash[12:]}"
            else:
                accession = acc_nodash
            return {"cik": cik, "accession_number": accession}
        return None

    def _parse_filing_html(self, html_content: str) -> str:
        """Parse HTML content and extract clean text."""
        soup = BeautifulSoup(html_content, "html.parser")

        for tag in soup.find_all(re.compile(r"^ix:")):
            tag.unwrap()
        for tag in soup.find_all(re.compile(r"^(xbrl|xbrli|link|context|unit)")):
            tag.decompose()
        for tag in soup.find_all(["script", "style", "meta"]):
            tag.decompose()
        for tag in soup.find_all(style=re.compile(r"display:\s*none", re.I)):
            tag.decompose()

        text_content = soup.get_text(separator=" ", strip=True)
        text_content = re.sub(r" {2,}", " ", text_content)

        return text_content.strip()

    def _url_to_filing_path(self, url: str) -> Optional[Path]:
        """Convert a SEC EDGAR URL to its local cache file path.

        Returns None if the URL doesn't match the expected SEC format.
        """
        parts = self._parse_sec_url(url)
        if not parts:
            return None
        cik, accession_number = parts["cik"], parts["accession_number"]
        cik_padded = str(cik).zfill(10)
        acc_nodash = accession_number.replace("-", "")
        return self._filings_dir / cik_padded / f"{acc_nodash}.txt"

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
    # parse_html_page Endpoint
    # ========================================================================

    @staticmethod
    def _parse_html_to_text(html_content: str) -> str:
        """Extract plain text from HTML."""
        soup = BeautifulSoup(html_content, "html.parser")
        for script_or_style in soup(["script", "style"]):
            _ = script_or_style.extract()

        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        return "\n".join(chunk for chunk in chunks if chunk)

    @retry(
        retry=retry_if_exception(
            lambda e: (isinstance(e, aiohttp.ClientResponseError) and e.status in (429, 503))
            or any(str(code) in str(e) for code in (429, 503))
        ),
        stop=stop_after_attempt(100),
        wait=wait_exponential_jitter(initial=3, exp_base=2, max=120),
        reraise=True,
    )
    async def _parse_html_page(self, url: str) -> str:
        """Fetch a URL and extract plain text."""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=60),
                    headers={"User-Agent": self.config.user_agent},
                ) as response:
                    response.raise_for_status()
                    html_content = await response.text()
            except Exception as e:
                if len(str(e)) == 0:
                    raise TimeoutError(
                        "Timeout error when parsing HTML page after 60 seconds. "
                        "The URL might be blocked or the server is taking too long to respond."
                    ) from e
                raise

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

        text_content = None
        if file_path.exists():
            text_content = file_path.read_text(encoding="utf-8")

        if text_content is None and self.config.sec_dump_path:
            text_content = await self._lookup_dump(url)
            if text_content:
                self._atomic_write(file_path, text_content)

        if text_content is None:
            html_content = await self._fetch_with_retry(url)
            if not html_content:
                raise RuntimeError(
                    f"Failed to download filing from {url}. The SEC server may be temporarily unavailable."
                )

            text_content = await asyncio.get_running_loop().run_in_executor(
                None, self._parse_html_to_text, html_content
            )
            self._atomic_write(file_path, text_content)

        if not text_content:
            raise ValueError("Filing content was empty after parsing.")

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
        available_keys = ", ".join(storage.keys()) if storage else "(empty)"

        # Extract {{key}} placeholders from prompt
        keys_in_prompt = re.findall(r"\{\{([^{}]+)\}\}", prompt)
        if not keys_in_prompt:
            return RetrieveInformationResponse(
                results="ERROR: Prompt must contain at least one {{key_name}} placeholder. "
                f"Available keys: [{available_keys}]"
            )

        # Validate all keys exist in data storage
        for key in keys_in_prompt:
            if key not in storage:
                return RetrieveInformationResponse(
                    results=f"ERROR: Key '{key}' not in data storage. "
                    f"Available keys: [{available_keys}]. Use parse_html_page first."
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

        max_chars = (self.config.retrieval_model_context_length - self.config.retrieval_max_output_tokens) * 4
        if len(final_prompt) > max_chars:
            sizes = ", ".join(f"{k}: {len(storage[k])} chars" for k in keys_in_prompt)
            return RetrieveInformationResponse(
                results=f"ERROR: Prompt too large ({len(final_prompt)} chars, max {max_chars}). "
                f"Document sizes: [{sizes}]. Use input_character_ranges to select a portion. "
                f"Split the document into sequential chunks of ~{self.config.large_doc_threshold_chars} chars and retry each."
            )

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

            llm_response_json = await get_response_json(llm_response)
            llm_response_obj = NeMoGymResponse.model_validate(llm_response_json)

            result_text = ""
            for output_item in llm_response_obj.output:
                if getattr(output_item, "type", None) == "message":
                    for content_item in getattr(output_item, "content", []):
                        if getattr(content_item, "type", None) == "output_text":
                            result_text += getattr(content_item, "text", "")

            if not result_text:
                return RetrieveInformationResponse(results="ERROR: Retrieval LLM returned no output.")

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
        if end_date:
            if not self._DATE_RE.match(end_date):
                raise ValueError(f"Invalid end_date format: '{end_date}'. Expected YYYY-MM-DD.")

        if start_date:
            if not self._DATE_RE.match(start_date):
                raise ValueError(f"Invalid start_date format: '{start_date}'. Expected YYYY-MM-DD.")
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

    async def verify(self, request: Request, body: FinanceAgentVerifyRequest) -> FinanceAgentVerifyResponse:
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

        question = ""
        for msg in body.responses_create_params.input or []:
            if getattr(msg, "role", None) == "user":
                content = getattr(msg, "content", None)
                if isinstance(content, str):
                    question = content

        generated_answer = ""
        for output_item in reversed(body.response.output):
            if getattr(output_item, "type", None) == "function_call":
                if getattr(output_item, "name", None) == "submit_final_result":
                    try:
                        args = json.loads(getattr(output_item, "arguments", "{}"))
                        generated_answer = args.get("final_result", "")
                    except (json.JSONDecodeError, TypeError):
                        pass
                    break

        if not generated_answer:
            for output_item in reversed(body.response.output):
                if (
                    getattr(output_item, "type", None) == "message"
                    and getattr(output_item, "role", None) == "assistant"
                ):
                    for content_item in getattr(output_item, "content", []):
                        if getattr(content_item, "type", None) == "output_text":
                            generated_answer = getattr(content_item, "text", "")
                            break
                    if generated_answer:
                        break

        # No judge model → substring match
        if not self.config.judge_model_server:
            reward = 1.0 if body.expected_answer.lower() in generated_answer.lower() else 0.0
            return FinanceAgentVerifyResponse(**body.model_dump(), reward=reward)

        # Legacy mode: [[0]]/[[1]]/[[2]] judge
        judge_user_prompt = self._judge_prompt_template
        judge_user_prompt = judge_user_prompt.replace("{question}", question)
        judge_user_prompt = judge_user_prompt.replace("{expected_answer}", body.expected_answer)
        judge_user_prompt = judge_user_prompt.replace("{generated_answer}", generated_answer)

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
                response = await self.server_client.post(
                    server_name=self.config.judge_model_server.name,
                    url_path="/v1/responses",
                    json=judge_params,
                )
                judge_response = NeMoGymResponse.model_validate(await get_response_json(response))
            except Exception as e:
                logger.warning(
                    "Judge call attempt %d/%d failed: %s: %s", attempt + 1, max_judge_retries, type(e).__name__, e
                )
                if attempt < max_judge_retries - 1:
                    await asyncio.sleep(2**attempt)
                    continue
                logger.error("Judge model call failed after %d attempts", max_judge_retries)
                return FinanceAgentVerifyResponse(**body.model_dump(), reward=0.0)

            try:
                last_output = judge_response.output[-1]
                if getattr(last_output, "type", None) == "message":
                    last_content = last_output.content[-1]
                    judge_text = getattr(last_content, "text", "")
            except Exception:
                pass

            rating_match = re.search(r"\[\[(\d+)\]\]", judge_text)
            rating = int(rating_match.group(1)) if rating_match else None

            if rating is not None:
                break

            logger.warning(
                "Judge returned no [[N]] rating (attempt %d/%d). Output: %s",
                attempt + 1,
                max_judge_retries,
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


if __name__ == "__main__":
    FinanceAgentResourcesServer.run_webserver()
