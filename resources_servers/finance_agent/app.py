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
Finance Agent Resource Server.

Provides tools for searching SEC filings by ticker symbol or company name.
Caches ticker mappings and filing metadata locally to minimize SEC API calls.
"""

import asyncio
import json
import logging
import re
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)

import aiohttp
from bs4 import BeautifulSoup
from fastapi import FastAPI
from pydantic import BaseModel, Field

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


class FinanceAgentResourcesServerConfig(BaseResourcesServerConfig):
    """Configuration for SEC Search resource server."""

    cache_dir: str = Field(default="cache", description="Directory for caching ticker mappings and filing metadata")
    user_agent: str = Field(
        default="Gym-SEC-Search/1.0 (research@nvidia.com)", description="User-Agent header for SEC API requests"
    )
    requests_per_second: int = Field(default=10, description="Rate limit for SEC API requests")
    # Optional: Tavily web search (uses tavily package directly)
    tavily_api_key: Optional[str] = Field(default=None, description="Tavily API key for web search")
    retrieval_model_server: Optional[ModelServerRef] = Field(
        default=None, description="Model server for retrieve_information LLM calls"
    )
    # Judge model configuration (LLM-as-judge for answer grading)
    judge_model_server: Optional[ModelServerRef] = Field(default=None, description="Reference to judge model server")
    judge_responses_create_params: Optional[NeMoGymResponseCreateParamsNonStreaming] = Field(
        default=None, description="Parameters for judge model requests"
    )
    judge_prompt_template_fpath: str = Field(
        default="prompt_templates/finance_agent_judge.txt",
        description="Path to the judge prompt template file (supports {question}, {expected_answer}, {generated_answer} placeholders)",
    )


class FinanceAgentSearchRequest(BaseModel):
    """Request model for SEC filing search."""

    ticker: str = Field(description="Stock ticker symbol (e.g., 'AAPL', 'MSFT', 'NVDA')")
    form_types: Optional[List[str]] = Field(
        default=None,
        description="Limits search to specific EDGAR form types (e.g., ['10-K', '10-Q', 'DEF 14A', '8-K']). Default: 10-K, 10-Q, and DEF 14A",
    )


class FinanceAgentSearchResponse(BaseModel):
    """Response model for SEC filing search."""

    results: str = Field(description="JSON string of filing results")


class DownloadAndParseFilingRequest(BaseModel):
    """Request model for download_and_parse_filing tool."""

    url: str = Field(description="The filing URL from sec_filing_search results")
    key: str = Field(description="The key to use when saving the result in the conversation's data storage.")


class DownloadAndParseFilingResponse(BaseModel):
    """Response model for download_and_parse_filing tool."""

    results: str = Field(description="Status message about data storage operation")


class RetrieveInformationRequest(BaseModel):
    """Request model for retrieve_information tool."""

    prompt: str = Field(description="Prompt with {{key_name}} placeholders for stored documents.")
    input_character_ranges: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Optional list of character ranges: [{'key': 'doc', 'start': 0, 'end': 100000}]"
    )


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

    query: str = Field(description="Search query")


class WebSearchResponse(BaseModel):
    """Response model for web_search tool."""

    results: str = Field(description="JSON string with search results")


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
    """Sliding window rate limiter for SEC API compliance."""

    def __init__(self, max_requests: int = 10, window_seconds: float = 1.0):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: deque = deque()
        self.lock = asyncio.Lock()

    async def acquire(self):
        """Wait until a request slot is available."""
        async with self.lock:
            now = time.monotonic()

            # Remove expired timestamps
            while self.requests and (now - self.requests[0]) >= self.window_seconds:
                self.requests.popleft()

            # Wait if at capacity
            if len(self.requests) >= self.max_requests:
                sleep_time = self.window_seconds - (now - self.requests[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    # Clean up again after sleeping
                    now = time.monotonic()
                    while self.requests and (now - self.requests[0]) >= self.window_seconds:
                        self.requests.popleft()

            self.requests.append(time.monotonic())


# ============================================================================
# SEC Search Resource Server
# ============================================================================


class FinanceAgentResourcesServer(SimpleResourcesServer):
    """
    SEC EDGAR Filing Search Resource Server.
    - /sec_filing_search: Search for SEC filings by ticker or company name
    - /download_and_parse_filing: Download, parse filing, store in data storage under a key
    - /retrieve_information: Query stored documents via LLM prompt with {{key}} syntax
    - /web_search: Tavily web search
    - /submit_final_result: Submit the final answer
    """

    config: FinanceAgentResourcesServerConfig

    def model_post_init(self, context):
        """Initialize after Pydantic model creation."""
        # Setup cache directories
        self._cache_dir = Path(self.config.cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._filings_dir = self._cache_dir / "filings"
        self._filings_dir.mkdir(exist_ok=True)
        self._content_dir = self._cache_dir / "content"
        self._content_dir.mkdir(exist_ok=True)
        self._tickers_file = self._cache_dir / "tickers.json"
        self._content_index_file = self._content_dir / "index.json"

        # Rate limiter for SEC API
        self._rate_limiter = RateLimiter(max_requests=self.config.requests_per_second, window_seconds=1.0)

        # In-memory caches (loaded lazily)
        self._tickers: Dict[str, Dict[str, str]] = {}  # ticker -> {"cik": ..., "name": ...}
        self._content_index: Dict[str, str] = {}
        self._session: Optional[aiohttp.ClientSession] = None
        self._initialized = False

        # Data storage: key -> parsed text content
        self._data_storage: Dict[str, str] = {}

        # Load judge prompt template
        with open(self.config.judge_prompt_template_fpath, "r") as f:
            self._judge_prompt_template = f.read().strip()

        # Tavily web search (optional - uses tavily package directly)
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

    def setup_webserver(self) -> FastAPI:
        """Register API routes."""
        app = super().setup_webserver()
        app.post("/sec_filing_search")(self.sec_filing_search)
        app.post("/download_and_parse_filing")(self.download_and_parse_filing)
        app.post("/retrieve_information")(self.retrieve_information)
        app.post("/submit_final_result")(self.submit_final_result)
        app.post("/web_search")(self.web_search)

        # Catch-all for unknown tools - return error to model so it can correct itself
        @app.post("/{tool_name}")
        async def handle_unknown_tool(tool_name: str):
            return {
                "results": json.dumps(
                    {
                        "error": f"Tool '{tool_name}' does not exist. Available tools: sec_filing_search, download_and_parse_filing, retrieve_information, submit_final_result, web_search"
                    }
                )
            }

        return app

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(headers={"User-Agent": self.config.user_agent})
        return self._session

    async def _fetch_with_retry(self, url: str, max_retries: int = 3) -> Optional[str]:
        """Fetch URL with rate limiting and retry logic."""
        session = await self._get_session()

        for attempt in range(max_retries):
            await self._rate_limiter.acquire()
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.text()
                    elif response.status == 429:
                        # Rate limited - wait and retry
                        await asyncio.sleep(2**attempt)
                    else:
                        return None
            except aiohttp.ClientError:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2**attempt)
        return None

    async def _ensure_tickers_loaded(self):
        """Load ticker → {cik, name} mappings from cache or SEC."""
        if self._initialized:
            return

        raw = None
        if self._tickers_file.exists():
            with open(self._tickers_file, "r") as f:
                raw = json.load(f)
        else:
            data = await self._fetch_with_retry("https://www.sec.gov/files/company_tickers.json")
            if data:
                raw = json.loads(data)
                with open(self._tickers_file, "w") as f:
                    json.dump(raw, f)

        if raw:
            for item in raw.values():
                self._tickers[item["ticker"]] = {"cik": str(item["cik_str"]).zfill(10), "name": item["title"]}
            self._initialized = True

    async def _resolve_ticker(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Look up a ticker symbol. Returns company info dict or None."""
        await self._ensure_tickers_loaded()
        query = ticker.strip().upper()
        info = self._tickers.get(query)
        if info is None:
            return None
        return {"cik": info["cik"], "ticker": query, "name": info["name"]}

    # ========================================================================
    # Filing Metadata
    # ========================================================================

    def _get_company_cache_path(self, cik: str) -> Path:
        """Cache file path for a company's filings (CIK zero-padded to 10 digits)."""
        return self._filings_dir / f"{str(cik).zfill(10)}.json"

    async def _get_company_filings(self, cik: str, ticker: str) -> Dict[str, Dict[str, Any]]:
        """Get filings for a company. Loads from disk cache or fetches from SEC."""
        cache_path = self._get_company_cache_path(cik)
        if cache_path.exists():
            with open(cache_path, "r") as f:
                return json.load(f)

        # Fetch from SEC
        data = await self._fetch_with_retry(f"https://data.sec.gov/submissions/CIK{cik}.json")
        if not data:
            return {}

        try:
            recent = json.loads(data).get("filings", {}).get("recent", {})
            acc_numbers = recent.get("accessionNumber", [])
            forms = recent.get("form", [])
            dates = recent.get("filingDate", [])
            report_dates = recent.get("reportDate", [])
            primary_docs = recent.get("primaryDocument", [])

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

            if filings:
                with open(self._get_company_cache_path(cik), "w") as f:
                    json.dump(filings, f)
            return filings
        except (json.JSONDecodeError, KeyError):
            return {}

    # ========================================================================
    # Content Index Management
    # ========================================================================

    def _load_content_index(self):
        """Load content index from disk."""
        if self._content_index_file.exists():
            try:
                with open(self._content_index_file, "r") as f:
                    self._content_index = json.load(f)
            except json.JSONDecodeError:
                self._content_index = {}

    def _save_content_index(self):
        """Save content index to disk."""
        with open(self._content_index_file, "w") as f:
            json.dump(self._content_index, f, indent=2)

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

    def _parse_html_to_text(self, html_content: str) -> str:
        """
        Parse HTML content and extract clean text.
        Uses space separator to keep inline content (like "$3.25 billion") together.
        """
        soup = BeautifulSoup(html_content, "html.parser")

        # Remove iXBRL tags (unwrap to keep content)
        for tag in soup.find_all(re.compile(r"^ix:")):
            tag.unwrap()

        # Remove XBRL-related tags completely
        for tag in soup.find_all(re.compile(r"^(xbrl|xbrli|link|context|unit)")):
            tag.decompose()

        # Remove script and style tags completely
        for tag in soup.find_all(["script", "style", "meta"]):
            tag.decompose()

        # Remove hidden divs that often contain XBRL data
        for tag in soup.find_all(style=re.compile(r"display:\s*none", re.I)):
            tag.decompose()

        # Extract text with space separator - keeps "$3.25 billion" together
        text_content = soup.get_text(separator=" ", strip=True)

        # Collapse multiple spaces
        text_content = re.sub(r" {2,}", " ", text_content)

        return text_content.strip()

    def _get_content_filepath(self, cik: str, accession_number: str) -> Path:
        """Get the file path for storing filing content."""
        cik_padded = str(cik).zfill(10)
        acc_nodash = accession_number.replace("-", "")
        return self._content_dir / cik_padded / f"{acc_nodash}.txt"

    # ========================================================================
    # sec_filing_search Endpoint
    # ========================================================================

    async def sec_filing_search(self, body: FinanceAgentSearchRequest) -> FinanceAgentSearchResponse:
        """Search for SEC filings by ticker symbol.

        Returns at most 30 filing metadata entries (sorted by date, newest first),
        including URLs for the actual filings.
        If form_types is provided, filters to those types. Otherwise returns 10-K, 10-Q, and DEF 14A.
        """
        company = await self._resolve_ticker(body.ticker)

        if not company:
            return FinanceAgentSearchResponse(
                results=json.dumps(
                    {
                        "error": f"No company found for ticker '{body.ticker}'",
                        "suggestion": "Use the exact stock ticker symbol (e.g., 'AAPL' for Apple, 'MSFT' for Microsoft)",
                    }
                )
            )

        filings = await self._get_company_filings(company["cik"], company["ticker"])
        form_types = body.form_types if body.form_types is not None else ["10-K", "10-Q", "DEF 14A"]

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
        all_results = all_results[:30]

        if not all_results:
            filter_msg = f" with form types {form_types}" if form_types else ""
            return FinanceAgentSearchResponse(
                results=json.dumps(
                    {
                        "error": f"No filings found for '{body.ticker}'{filter_msg}",
                        "suggestion": "Try a different form type filter (e.g., ['10-K', '10-Q', '8-K'])",
                    }
                )
            )

        return FinanceAgentSearchResponse(results=json.dumps(all_results, indent=2))

    # ========================================================================
    # download_and_parse_filing Endpoint
    # ========================================================================

    async def download_and_parse_filing(self, body: DownloadAndParseFilingRequest) -> DownloadAndParseFilingResponse:
        """Download and parse an SEC filing, store text in _data_storage[key]."""
        url, key = body.url, body.key

        if not url:
            return DownloadAndParseFilingResponse(
                results="ERROR: url is required. Use the filing_url from sec_filing_search results."
            )
        if not key:
            return DownloadAndParseFilingResponse(
                results="ERROR: key is required. Provide a key to store this filing in data storage."
            )

        # Extract CIK and accession number from URL (needed for disk cache path)
        url_parts = self._parse_sec_url(url)
        if not url_parts:
            return DownloadAndParseFilingResponse(
                results=f"ERROR: Invalid SEC URL format: {url}. Use the filing_url from sec_filing_search results."
            )
        cik = url_parts["cik"]
        accession_number = url_parts["accession_number"]

        # Check disk cache first
        self._load_content_index()
        text_content = None
        if url in self._content_index:
            cached_path = self._content_index[url]
            if Path(cached_path).exists():
                text_content = Path(cached_path).read_text(encoding="utf-8")

        # Download and parse if not cached
        if text_content is None:
            html_content = await self._fetch_with_retry(url)
            if not html_content:
                return DownloadAndParseFilingResponse(
                    results=f"ERROR: Failed to download filing from {url}. "
                    "The SEC server may be temporarily unavailable. Try again later."
                )

            text_content = self._parse_html_to_text(html_content)

            # Cache to disk
            file_path = self._get_content_filepath(cik, accession_number)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text_content)
            self._content_index[url] = str(file_path)
            self._save_content_index()

        if not text_content:
            return DownloadAndParseFilingResponse(results="ERROR: Filing content was empty after parsing.")

        # Store in data storage
        result_msg = ""
        if key in self._data_storage:
            result_msg += "WARNING: Key already exists in data storage. Overwriting.\n"

        self._data_storage[key] = text_content

        result_msg += f"SUCCESS: Filed saved to data storage under key: {key}.\n"
        result_msg += f"Document size: {len(text_content)} characters.\n"

        if len(text_content) > 100_000:
            result_msg += (
                f"Note: This is a large document ({len(text_content)} chars). "
                f"The retrieve_information limit is 500,000 chars, so use "
                f"input_character_ranges to query portions (e.g., [0–250000], then [250000–{len(text_content)}]). "
                f"Financial statements and notes are often in the second half of 10-K/10-Q filings.\n"
            )

        keys_list = ", ".join(self._data_storage.keys())
        result_msg += f"Keys in data_storage: [{keys_list}]\n"

        return DownloadAndParseFilingResponse(results=result_msg)

    # ========================================================================
    # retrieve_information Endpoint (LLM-based document querying)
    # ========================================================================

    async def retrieve_information(self, body: RetrieveInformationRequest) -> RetrieveInformationResponse:
        """Query stored documents using LLM-based prompting."""
        if not self.config.retrieval_model_server:
            return RetrieveInformationResponse(
                results="ERROR: Retrieval model not configured. Set retrieval_model_server in config."
            )

        prompt = body.prompt
        available_keys = ", ".join(self._data_storage.keys()) if self._data_storage else "(empty)"

        # Extract {{key}} placeholders from prompt
        keys_in_prompt = re.findall(r"\{\{([^{}]+)\}\}", prompt)
        if not keys_in_prompt:
            return RetrieveInformationResponse(
                results="ERROR: Prompt must contain at least one {{key_name}} placeholder. "
                f"Available keys: [{available_keys}]"
            )

        # Validate all keys exist in data storage
        for key in keys_in_prompt:
            if key not in self._data_storage:
                return RetrieveInformationResponse(
                    results=f"ERROR: Key '{key}' not in data storage. "
                    f"Available keys: [{available_keys}]. Use download_and_parse_filing first."
                )

        # Parse optional character ranges
        ranges_dict: Dict[str, tuple] = {}
        for r in body.input_character_ranges or []:
            if isinstance(r, dict) and all(k in r for k in ("key", "start", "end")):
                ranges_dict[r["key"]] = (r["start"], r["end"])

        # Substitute {{key}} with document content (optionally sliced by ranges)
        final_prompt = prompt
        for key in keys_in_prompt:
            content = self._data_storage[key]
            if key in ranges_dict:
                start, end = ranges_dict[key]
                content = content[start:end]
            final_prompt = final_prompt.replace("{{" + key + "}}", content)

        # Guard against prompts that exceed model context (~4 chars/token, 131K-262K token limit depending on model)
        max_chars = 500_000
        if len(final_prompt) > max_chars:
            sizes = ", ".join(f"{k}: {len(self._data_storage[k])} chars" for k in keys_in_prompt)
            return RetrieveInformationResponse(
                results=f"ERROR: Prompt too large ({len(final_prompt)} chars, max {max_chars}). "
                f"Document sizes: [{sizes}]. Use input_character_ranges to select a portion. "
                f"Split the document into sequential chunks of ~{max_chars} chars and retry each."
            )

        # Call retrieval LLM
        try:
            llm_response = await self.server_client.post(
                server_name=self.config.retrieval_model_server.name,
                url_path="/v1/responses",
                json={
                    "input": [
                        {
                            "role": "system",
                            "type": "message",
                            "content": "You are a document analysis assistant. Answer based ONLY on "
                            "the document text provided. If the information is not present, state "
                            "that clearly — do NOT guess or fabricate numbers.",
                        },
                        {"role": "user", "content": final_prompt, "type": "message"},
                    ],
                    "temperature": 0.0,
                    "max_output_tokens": 8192,
                },
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

    async def submit_final_result(self, body: SubmitFinalResultRequest) -> SubmitFinalResultResponse:
        """
        Accept the agent's final answer submission.
        """
        final_result = body.final_result
        if not final_result:
            return SubmitFinalResultResponse(results="ERROR: final_result is required. Please provide your answer.")
        return SubmitFinalResultResponse(results=json.dumps({"success": True, "result": final_result}))

    async def web_search(self, body: WebSearchRequest) -> WebSearchResponse:
        """Search the web using Tavily. Returns up to 10 results."""
        if self._tavily is None:
            return WebSearchResponse(
                results=json.dumps(
                    {
                        "error": "web_search is not available. Use sec_filing_search, download_and_parse_filing, and retrieve_information instead.",
                    }
                )
            )

        max_retries = 3
        for attempt in range(max_retries):
            try:
                raw = self._tavily.search(body.query, num_results=10)
                results = [
                    {"url": r.get("url", ""), "title": r.get("title", ""), "content": r.get("content", "")}
                    for r in raw.get("results", [])
                ]
                return WebSearchResponse(results=json.dumps(results))
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"web_search attempt {attempt + 1} failed: {e}. Retrying in {2**attempt}s...")
                    await asyncio.sleep(2**attempt)
                else:
                    logger.error(f"web_search failed after {max_retries} attempts: {e}")
                    return WebSearchResponse(results=json.dumps({"error": str(e)}))

    async def verify(self, body: FinanceAgentVerifyRequest) -> FinanceAgentVerifyResponse:
        """Verify using LLM-as-judge with strict financial grading rubric (0/1/2 scale).

        Rating scale:
            [[2]] = fully correct  → reward 1.0
            [[1]] = partial        → reward 0.0
            [[0]] = incorrect      → reward 0.0
        """
        # Extract question from the last user message
        question = ""
        for msg in body.responses_create_params.input or []:
            if getattr(msg, "role", None) == "user":
                content = getattr(msg, "content", None)
                if isinstance(content, str):
                    question = content

        # Extract generated answer: prefer submit_final_result tool call, fall back to last assistant text message.
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

        # If no judge configured, use simple substring matching
        if not self.config.judge_model_server:
            reward = 1.0 if body.expected_answer.lower() in generated_answer.lower() else 0.0
            return FinanceAgentVerifyResponse(**body.model_dump(), reward=reward)

        # Build judge prompt (use .replace() to avoid KeyError on braces in content)
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

        # Call judge model
        try:
            response = await self.server_client.post(
                server_name=self.config.judge_model_server.name,
                url_path="/v1/responses",
                json=judge_params,
            )
            judge_response = NeMoGymResponse.model_validate(await get_response_json(response))
        except Exception as e:
            logger.error(f"Judge model call failed: {type(e).__name__}: {e}")
            return FinanceAgentVerifyResponse(**body.model_dump(), reward=0.0)

        # Extract verdict text from judge response
        judge_text = ""
        try:
            last_output = judge_response.output[-1]
            if getattr(last_output, "type", None) == "message":
                last_content = last_output.content[-1]
                judge_text = getattr(last_content, "text", "")
        except Exception:
            pass

        # Parse [[N]] rating from judge output
        rating_match = re.search(r"\[\[(\d+)\]\]", judge_text)
        rating = int(rating_match.group(1)) if rating_match else None

        if rating is None:
            logger.warning(f"Judge returned no [[N]] rating. Judge output: {judge_text[:200]}")

        # Only [[2]] (fully correct) gets reward 1.0
        reward = 1.0 if rating == 2 else 0.0

        return FinanceAgentVerifyResponse(
            **body.model_dump(), reward=reward, judge_rating=rating, judge_text=judge_text
        )


if __name__ == "__main__":
    FinanceAgentResourcesServer.run_webserver()
