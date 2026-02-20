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
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

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


# ============================================================================
# Manual Ticker Mappings (for companies missing from SEC's company_tickers.json)
# ============================================================================
# Some companies (e.g., those undergoing mergers) may have empty ticker fields
# in SEC data. Add them here so they're always available regardless of cache state.

MANUAL_TICKER_MAPPINGS = {
    # U.S. Steel - ticker/exchange fields empty in SEC data due to Nippon Steel merger
    "X": {"cik_str": 1163302, "ticker": "X", "title": "UNITED STATES STEEL CORP"},
}

# ============================================================================
# Configuration
# ============================================================================

class FinanceAgentResourcesServerConfig(BaseResourcesServerConfig):
    """Configuration for SEC Search resource server."""
    cache_dir: str = Field(
        default="cache",
        description="Directory for caching ticker mappings and filing metadata"
    )
    user_agent: str = Field(
        default="Gym-SEC-Search/1.0 (research@nvidia.com)",
        description="User-Agent header for SEC API requests"
    )
    requests_per_second: int = Field(
        default=10,
        description="Rate limit for SEC API requests"
    )
    # Optional: Tavily web search (uses tavily package directly)
    tavily_api_key: Optional[str] = Field(
        default=None,
        description="Tavily API key for web search"
    )
    # Retrieval model configuration (for retrieve_information tool)
    # This model is used to query stored documents with LLM prompts.
    # Can be the same as the policy model (agent's model) or a different one.
    retrieval_model_server: Optional[ModelServerRef] = Field(
        default=None,
        description="Model server for retrieve_information LLM calls"
    )
    # Judge model configuration (LLM-as-judge for answer grading)
    judge_model_server: Optional[ModelServerRef] = Field(
        default=None,
        description="Reference to judge model server"
    )
    judge_responses_create_params: Optional[NeMoGymResponseCreateParamsNonStreaming] = Field(
        default=None,
        description="Parameters for judge model requests"
    )


# ============================================================================
# Request/Response Models
# ============================================================================

class FinanceAgentSearchRequest(BaseModel):
    """Request model for SEC filing search."""
    company_or_ticker: str = Field(
        description="Company name or ticker symbol (e.g., 'AAPL', 'Apple', 'Microsoft')"
    )
    form_types: Optional[List[str]] = Field(
        default=None,
        description="Limits search to specific EDGAR form types (e.g., ['10-K', '10-Q', 'DEF 14A', '8-K']). Default: 10-K, 10-Q, and DEF 14A"
    )
    top_n: int = Field(
        default=10,
        description="Maximum number of filings to return"
    )


class FinanceAgentSearchResponse(BaseModel):
    """Response model for SEC filing search."""
    results: str = Field(description="JSON string of filing results")


class PrepareFilingRequest(BaseModel):
    """Request model for prepare_filing tool (analogous to Vals' parse_html_page)."""
    url: str = Field(
        description="The filing URL from sec_filing_search results"
    )
    key: str = Field(
        description="The key to use when saving the result in the conversation's data storage."
    )


class PrepareFilingResponse(BaseModel):
    """Response model for prepare_filing tool."""
    results: str = Field(description="Status message about data storage operation")


class RetrieveInformationRequest(BaseModel):
    """Request model for retrieve_information tool (matches Vals benchmark)."""
    prompt: str = Field(
        description="Prompt with {{key_name}} placeholders for stored documents."
    )
    input_character_ranges: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Optional list of character ranges: [{'key': 'doc', 'start': 0, 'end': 100000}]"
    )


class RetrieveInformationResponse(BaseModel):
    """Response model for retrieve_information tool."""
    results: str = Field(description="LLM response text from querying stored documents")


class SubmitFinalResultRequest(BaseModel):
    """Request model for submit_final_result tool (matches Vals benchmark)."""
    final_result: str = Field(
        description="The final result to submit"
    )


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
    
    Architecture mirrors the Vals Finance Agent benchmark:
    - /sec_filing_search: Search for SEC filings by ticker or company name
      (open-source replacement for Vals' edgar_search which uses paid SEC-API)
    - /prepare_filing: Download, parse filing, store in data storage under a key
      (analogous to Vals' parse_html_page)
    - /retrieve_information: Query stored documents via LLM prompt with {{key}} syntax
      (matches Vals' retrieve_information exactly)
    - /web_search: Tavily web search (same as Vals)
    
    Data Storage (in-memory, per server instance):
    - Agent calls prepare_filing(url, key) to store parsed filing text under a key
    - Agent calls retrieve_information(prompt="... {{key}} ...") to query stored docs
    - The retrieve_information tool substitutes {{key}} with document content and
      sends the assembled prompt to a retrieval LLM, returning the extracted answer
    
    Disk Caching (for efficiency):
    - tickers.json: Ticker-to-CIK mapping with company names
    - filings/{CIK}.jsonl: All filing metadata for each company
    - content/{CIK}/{accession}.txt: Parsed filing content (avoids re-downloading)
    - content/index.json: URL to file path mapping
    """
    config: FinanceAgentResourcesServerConfig
    
    # Judge prompt constants — strict financial grading rubric (0/1/2 scale).
    # Used by verify() for reward calculation.
    JUDGE_SYSTEM_PROMPT: ClassVar[str] = """You are a meticulous financial analyst grader evaluating answers to questions about SEC filings (10-K, 10-Q). Compare a candidate answer to a GOLD reference and rate the response strictly.

Questions may involve:
- Specific line items from financial statements
- Multi-year comparisons and trends
- Qualitative disclosures and risk factors
- Calculated metrics and ratios

Grading priorities (in order):

1) Factual equivalence to GOLD (accept algebraically/formally equivalent formulations for financial ratios and calculations).

2) Completeness on required parts — the candidate must include the same core parts/subclaims as the GOLD.

Rules:

- Treat GOLD as authoritative for what counts as correct.
- If GOLD is a range or set, the candidate is equivalent only if it lies within that range or is a member of that set.
- For financial ratios/calculations, accept mathematically identical transformations (e.g., 0.5 = 50% = 1:2).
- For numerical values, allow strict rounding differences:
  • Percentages: ±0.1 percentage points (e.g., 7.8% ≈ 7.79%)
  • Ratios/multipliers: ±1% relative difference (e.g., 2.5 ≈ 2.525)
  • Dollar amounts: ±1% of the value (e.g., $100M ≈ $101M)
- For units: Accept equivalent representations ($1M = $1,000,000 = $1 million).
- For dates: Accept equivalent representations (Q1 2023 = Jan-Mar 2023 = first quarter 2023).
- If the candidate includes reasoning (e.g., in <think> tags), focus on the final answer.
- If GOLD is a refusal (e.g., "Cannot determine"), accept semantically equivalent refusals from candidate.
- Multi-part answers: all essential parts must match for full credit; missing parts reduce the rating.
- Be concise. Do NOT reveal or rewrite the GOLD.

After your explanation, you must rate the response on a scale of 0 to 2 by strictly following this format: [[rating]], for example: The rating is: [[1]], or: My rating is [[0]].

You should provide a 0 rating when the answer does not match the reference or is factually wrong.
You should provide a 1 rating when the answer is partially correct (correct number but missing explanation, or close but not exact).
You should provide a 2 rating when the answer is fully correct and complete."""

    JUDGE_USER_PROMPT_TEMPLATE: ClassVar[str] = """===== Example 1 (rating 2 - fully correct) =====

QUESTION:

What was Apple's revenue growth rate from 2022 to 2023?

GOLD:

7.8%

CANDIDATE:

The revenue grew by 7.79%.

The candidate provides the same growth rate with minor rounding difference (7.79% vs 7.8%, within ±0.1% tolerance); the meaning and numerical value are equivalent.

The rating is: [[2]]

===== Example 2 (rating 1 - partially correct) =====

QUESTION:

Calculate Microsoft's current ratio and explain what it indicates about the company's liquidity.

GOLD:

Current ratio = 2.5; This indicates strong short-term liquidity as current assets are 2.5x current liabilities, suggesting the company can easily meet short-term obligations.

CANDIDATE:

Current ratio = 2.5

The candidate provides the correct numerical calculation but omits the explanation of what the ratio indicates about liquidity; one core part is missing, so partial credit.

The rating is: [[1]]

===== Example 3 (rating 0 - incorrect) =====

QUESTION:

What was the company's ROE for 2023?

GOLD:

ROE = 15.2%

CANDIDATE:

ROE = 8.5%

The candidate provides a significantly different value (8.5% vs 15.2%), outside the ±0.1% tolerance. This is factually incorrect.

The rating is: [[0]]

===== Example 4 (rating 2 - correct with reasoning) =====

QUESTION:

What was Apple's total revenue in Q1 2023?

GOLD:

$117.2 billion

CANDIDATE:

<think>Looking at the 10-Q filing for Q1 2023, the consolidated statements of operations show net sales of $117,154 million. Converting to billions: $117,154M / 1,000 = $117.154B, which rounds to $117.2B.</think> Apple's total revenue in Q1 2023 was $117.2 billion.

The candidate includes detailed reasoning in <think> tags showing the calculation process. The final answer $117.2 billion matches the GOLD exactly.

The rating is: [[2]]

===== Inputs =====

QUESTION:

{question}

GOLD:

{expected_answer}

CANDIDATE:

{generated_answer}"""

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
        self._rate_limiter = RateLimiter(
            max_requests=self.config.requests_per_second,
            window_seconds=1.0
        )
        
        # In-memory caches (loaded lazily)
        self._ticker_to_cik: Dict[str, str] = {}
        self._cik_to_ticker: Dict[str, str] = {}
        self._companies: List[Dict[str, Any]] = []
        self._content_index: Dict[str, str] = {}
        self._url_metadata: Dict[str, Dict[str, Any]] = {}  # URL -> filing metadata cache
        self._session: Optional[aiohttp.ClientSession] = None
        self._initialized = False
        
        # Data storage: key -> parsed text content (matches Vals benchmark pattern)
        # The agent stores filings here via prepare_filing(key=...) and queries
        # them via retrieve_information(prompt="... {{key}} ...")
        self._data_storage: Dict[str, str] = {}
        
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
        app.post("/prepare_filing")(self.prepare_filing)
        app.post("/retrieve_information")(self.retrieve_information)
        app.post("/submit_final_result")(self.submit_final_result)
        app.post("/web_search")(self.web_search)
        
        # Catch-all for unknown tools - return error to model so it can correct itself
        @app.post("/{tool_name}")
        async def handle_unknown_tool(tool_name: str):
            return {"results": json.dumps({
                "error": f"Tool '{tool_name}' does not exist. Available tools: sec_filing_search, prepare_filing, retrieve_information, submit_final_result, web_search"
            })}
        
        return app
    
    # ========================================================================
    # HTTP Session Management
    # ========================================================================
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"User-Agent": self.config.user_agent}
            )
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
                        await asyncio.sleep(2 ** attempt)
                    else:
                        return None
            except aiohttp.ClientError:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
        return None

    # ========================================================================
    # Ticker/CIK Resolution
    # ========================================================================
    
    def _apply_manual_ticker_mappings(self):
        """Apply manual ticker mappings for companies missing from SEC data."""
        for ticker, item in MANUAL_TICKER_MAPPINGS.items():
            cik = str(item["cik_str"]).zfill(10)
            ticker_upper = ticker.upper()
            
            # Skip if already present
            if ticker_upper in self._ticker_to_cik:
                continue
            
            self._ticker_to_cik[ticker_upper] = cik
            self._cik_to_ticker[cik] = ticker_upper
            self._companies.append({
                "cik": cik,
                "ticker": item["ticker"],
                "name": item["title"]
            })
    
    def _load_tickers_cache(self) -> bool:
        """Load cached ticker data and apply manual mappings."""
        if self._tickers_file.exists():
            try:
                with open(self._tickers_file, "r") as f:
                    data = json.load(f)
                self._ticker_to_cik = {}
                self._cik_to_ticker = {}
                self._companies = []
                for item in data.values():
                    cik = str(item["cik_str"]).zfill(10)
                    ticker = item["ticker"]
                    title = item["title"]
                    self._ticker_to_cik[ticker.upper()] = cik
                    self._cik_to_ticker[cik] = ticker.upper()
                    self._companies.append({
                        "cik": cik,
                        "ticker": ticker,
                        "name": title
                    })
                
                # Always apply manual mappings after loading cache
                self._apply_manual_ticker_mappings()
                return True
            except (json.JSONDecodeError, KeyError):
                pass
        return False
    
    async def _fetch_and_cache_tickers(self) -> bool:
        """Fetch ticker data from SEC and cache it."""
        url = "https://www.sec.gov/files/company_tickers.json"
        data = await self._fetch_with_retry(url)
        if data:
            try:
                parsed = json.loads(data)
                with open(self._tickers_file, "w") as f:
                    json.dump(parsed, f)
                return self._load_tickers_cache()
            except json.JSONDecodeError:
                pass
        return False
    
    async def _ensure_tickers_loaded(self):
        """Ensure ticker data is loaded."""
        if not self._initialized:
            if not self._load_tickers_cache():
                await self._fetch_and_cache_tickers()
            self._initialized = True
    
    def _tokenize(self, text: str, min_length: int = 1) -> set:
        """
        Tokenize text into uppercase word tokens.
        Filters out tokens shorter than min_length.
        """
        tokens = re.findall(r'\w+', text.upper())
        return {t for t in tokens if len(t) >= min_length}
    
    def _search_company_by_tokens(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search companies by token matching.
        ALL query tokens (2+ chars) must appear as exact tokens in company name.
        No partial/substring matching - tokens must match exactly.
        """
        # Tokenize query with min_length=2 to filter single-char tokens
        query_tokens = self._tokenize(query, min_length=2)
        if not query_tokens:
            return []
        
        matches = []
        for company in self._companies:
            # Tokenize company name (no min_length filter for company names)
            name_tokens = self._tokenize(company["name"])
            # All query tokens must exist as exact tokens in company name
            if query_tokens.issubset(name_tokens):
                matches.append(company.copy())
        
        return matches[:limit]
    
    def _match_company_name(self, query: str) -> List[Dict[str, Any]]:
        """
        Match company by name using ALL tokens match logic.
        Query tokens (2+ chars) must all appear as exact tokens in company name.
        """
        return self._search_company_by_tokens(query)
    
    def _resolve_all_matches(self, company_or_ticker: str) -> List[Dict[str, Any]]:
        """
        Resolve company_or_ticker to list of matching companies (sync version).
        First tries exact ticker match, then company name search.
        Deduplicates by CIK, preferring ticker matches.
        """
        query = company_or_ticker.strip().upper()
        seen_ciks = set()
        matches = []
        
        # Try exact ticker match first
        if query in self._ticker_to_cik:
            cik = self._ticker_to_cik[query]
            if cik not in seen_ciks:
                seen_ciks.add(cik)
                matches.append({
                    "cik": cik,
                    "ticker": query,
                    "name": next((c["name"] for c in self._companies if c["ticker"].upper() == query), ""),
                    "match_type": "ticker"
                })
        
        # Try company name matching
        name_matches = self._search_company_by_tokens(company_or_ticker)
        for m in name_matches:
            if m["cik"] not in seen_ciks:
                seen_ciks.add(m["cik"])
                m["match_type"] = "company_name"
                matches.append(m)
        
        return matches
    
    async def _resolve_company(self, company_or_ticker: str) -> List[Dict[str, Any]]:
        """
        Resolve company_or_ticker to list of matching companies.
        First tries exact ticker match, then company name search.
        """
        await self._ensure_tickers_loaded()
        return self._resolve_all_matches(company_or_ticker)
    
    # ========================================================================
    # Filing Metadata
    # ========================================================================
    
    def _get_company_cache_path(self, cik: str) -> Path:
        """Get the cache file path for a company's filings."""
        # Ensure CIK is zero-padded to 10 digits
        cik_padded = str(cik).zfill(10)
        return self._filings_dir / f"{cik_padded}.jsonl"
    
    def _has_company_cache(self, cik: str) -> bool:
        """Check if cache exists for a company."""
        return self._get_company_cache_path(cik).exists()
    
    def _load_company_filings(self, cik: str) -> List[Dict[str, Any]]:
        """Load cached filings for a company."""
        filings_file = self._get_company_cache_path(cik)
        filings = []
        if filings_file.exists():
            with open(filings_file, "r") as f:
                for line in f:
                    if line.strip():
                        filings.append(json.loads(line))
        return filings
    
    def _save_company_filings(self, cik: str, filings: List[Dict[str, Any]]):
        """Save filings to cache."""
        filings_file = self._filings_dir / f"{cik}.jsonl"
        with open(filings_file, "w") as f:
            for filing in filings:
                f.write(json.dumps(filing) + "\n")
    
    async def _fetch_company_filings(self, cik: str, ticker: str) -> List[Dict[str, Any]]:
        """Fetch filings from SEC API."""
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        data = await self._fetch_with_retry(url)
        if not data:
            return []
        
        try:
            parsed = json.loads(data)
            recent = parsed.get("filings", {}).get("recent", {})
            
            filings = []
            accession_numbers = recent.get("accessionNumber", [])
            forms = recent.get("form", [])
            filing_dates = recent.get("filingDate", [])
            report_dates = recent.get("reportDate", [])
            primary_docs = recent.get("primaryDocument", [])
            
            for i in range(len(accession_numbers)):
                acc = accession_numbers[i]
                form = forms[i] if i < len(forms) else ""
                filing_date = filing_dates[i] if i < len(filing_dates) else ""
                report_date = report_dates[i] if i < len(report_dates) else ""
                primary_doc = primary_docs[i] if i < len(primary_docs) else ""
                
                # Build filing URL
                acc_nodash = acc.replace("-", "")
                filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{acc_nodash}/{primary_doc}"
                
                filings.append({
                    "ticker": ticker,
                    "cik": cik,
                    "form": form,
                    "filing_date": filing_date,
                    "report_date": report_date,
                    "accession_number": acc,
                    "primary_document": primary_doc,
                    "filing_url": filing_url
                })
            
            return filings
        except (json.JSONDecodeError, KeyError):
            return []
    
    async def _get_company_filings(self, cik: str, ticker: str) -> List[Dict[str, Any]]:
        """Get filings for a company, using cache or fetching from SEC."""
        filings = self._load_company_filings(cik)
        if not filings:
            filings = await self._fetch_company_filings(cik, ticker)
            if filings:
                self._save_company_filings(cik, filings)
        return filings
    
    def _lookup_filing_by_accession(self, cik: str, accession_number: str) -> Optional[Dict[str, Any]]:
        """Look up filing metadata by CIK and accession number."""
        filings = self._load_company_filings(cik)
        acc_normalized = accession_number.replace("-", "")
        for filing in filings:
            filing_acc = filing.get("accession_number", "").replace("-", "")
            if filing_acc == acc_normalized:
                return filing
        return None
    
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
        """
        Search for SEC filings by company name or ticker.
        
        Returns filing metadata including URLs for the actual filings.
        If form_types is provided, filters to those types. Otherwise returns all form types.
        """
        # Resolve company
        matches = await self._resolve_company(body.company_or_ticker)
        
        if not matches:
            return FinanceAgentSearchResponse(
                results=json.dumps({
                    "error": f"No company found matching '{body.company_or_ticker}'",
                    "suggestion": "Try using the exact ticker symbol (e.g., 'AAPL') or a company name"
                })
            )
        
        # Get filings for all matched companies
        all_results = []
        # If form_types specified, filter to those; otherwise default to common types.
        # NOTE: Vals' edgar_search defaults to ALL form types, but that overwhelms
        # our metadata-only search with too many results. We include DEF 14A (proxy
        # statements) alongside 10-K/10-Q since compensation questions are common.
        form_types = body.form_types if body.form_types is not None else ["10-K", "10-Q", "DEF 14A"]
        
        for company in matches[:3]:  # Limit to top 3 matches
            filings = await self._get_company_filings(company["cik"], company["ticker"])
            
            for filing in filings:
                # Filter by form type if specified
                if form_types and filing["form"] not in form_types:
                    continue
                
                result = {
                    "ticker": company["ticker"],
                    "cik": company["cik"],
                    "form": filing["form"],
                    "filing_date": filing.get("filing_date", ""),
                    "report_date": filing.get("report_date", ""),
                    "accession_number": filing.get("accession_number", ""),
                    "filing_url": filing.get("filing_url", ""),
                    "matched_company": company["name"],
                    "match_type": company["match_type"]
                }
                all_results.append(result)
        
        # Sort by filing date descending
        all_results.sort(key=lambda x: x["filing_date"], reverse=True)
        
        # Limit results
        all_results = all_results[:body.top_n]
        
        # Store URL -> metadata mapping for prepare_filing
        for filing in all_results:
            url = filing.get("filing_url")
            if url:
                self._url_metadata[url] = filing
        
        if not all_results:
            filter_msg = f" with form types {form_types}" if form_types else ""
            return FinanceAgentSearchResponse(
                results=json.dumps({
                    "error": f"No filings found for '{body.company_or_ticker}'{filter_msg}",
                    "suggestion": "Try a different ticker, company name, or form type filter"
                })
            )
        
        return FinanceAgentSearchResponse(results=json.dumps(all_results, indent=2))
    
    # ========================================================================
    # prepare_filing Endpoint
    # ========================================================================
    
    async def prepare_filing(self, body: PrepareFilingRequest) -> PrepareFilingResponse:
        """
        Download, parse an SEC filing, and store it in the data storage.
        
        This mirrors Vals benchmark's parse_html_page tool:
        1. Downloads HTML from the filing URL
        2. Parses HTML to plain text (removes iXBRL, scripts, styles)
        3. Stores the text content in _data_storage[key] for use with retrieve_information
        4. Also caches to disk to avoid re-downloading on future calls
        
        The agent provides a key of their choosing. The key is how they'll
        reference this document later in retrieve_information prompts via {{key}} syntax.
        """
        url = body.url
        key = body.key
        
        if not url:
            return PrepareFilingResponse(
                results="ERROR: url is required. Use the filing_url from sec_filing_search results."
            )
        if not key:
            return PrepareFilingResponse(
                results="ERROR: key is required. Provide a key to store this filing in data storage."
            )
        
        # Load content index (disk cache for avoiding re-downloads)
        self._load_content_index()
        
        text_content = None
        filing_meta = self._url_metadata.get(url)
        
        # Check if already cached on disk (avoids re-downloading)
        if url in self._content_index:
            cached_path = self._content_index[url]
            if Path(cached_path).exists():
                text_content = Path(cached_path).read_text(encoding="utf-8")
                # Load metadata if not in memory
                if not filing_meta:
                    url_parts = self._parse_sec_url(url)
                    if url_parts:
                        filing_meta = self._lookup_filing_by_accession(
                            url_parts["cik"], url_parts["accession_number"]
                        )
        
        # Not cached - need to download and parse
        if text_content is None:
            if filing_meta:
                cik = filing_meta.get("cik", "")
                accession_number = filing_meta.get("accession_number", "")
            else:
                url_parts = self._parse_sec_url(url)
                if not url_parts:
                    return PrepareFilingResponse(
                        results=f"ERROR: Invalid SEC URL format: {url}. "
                        "Use the filing_url from sec_filing_search results."
                    )
                cik = url_parts["cik"]
                accession_number = url_parts["accession_number"]
                filing_meta = self._lookup_filing_by_accession(cik, accession_number)
            
            # Download the filing
            html_content = await self._fetch_with_retry(url)
            if not html_content:
                return PrepareFilingResponse(
                    results=f"ERROR: Failed to download filing from {url}. "
                    "The SEC server may be temporarily unavailable. Try again later."
                )
            
            # Parse HTML and extract text
            text_content = self._parse_html_to_text(html_content)
            
            # Cache to disk for future efficiency
            file_path = self._get_content_filepath(cik, accession_number)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text_content)
            self._content_index[url] = str(file_path)
            self._save_content_index()
        
        if not text_content:
            return PrepareFilingResponse(
                results="ERROR: Filing content was empty after parsing."
            )
        
        # ---- Store in data storage (matches Vals' _save_tool_output pattern) ----
        # This is the key part: the agent's data storage maps key -> full text content.
        # The agent will later reference this key in retrieve_information prompts.
        result_msg = ""
        if key in self._data_storage:
            result_msg += (
                "WARNING: The key already exists in the data storage. "
                "The new result overwrites the old one.\n"
            )
        
        self._data_storage[key] = text_content
        
        result_msg += f"SUCCESS: The result has been saved to the data storage under the key: {key}.\n"
        result_msg += f"Document size: {len(text_content)} characters.\n"
        
        if filing_meta:
            ticker = filing_meta.get("ticker", "")
            form = filing_meta.get("form", "")
            report_date = filing_meta.get("report_date", "")
            if ticker or form or report_date:
                result_msg += f"Filing metadata: ticker={ticker}, form={form}, report_date={report_date}\n"
        
        # Hint about document structure to help with input_character_ranges
        # This addresses the common failure mode where financial statements are in
        # the latter portion of 10-K filings and partial ranges miss them.
        if len(text_content) > 100000:
            result_msg += (
                f"Note: This is a large document. When using retrieve_information with "
                f"input_character_ranges, consider that financial statements and notes "
                f"are often in the second half of 10-K/10-Q filings. If the retrieval tool "
                f"reports data not found, try a different character range or omit "
                f"input_character_ranges to use the full document.\n"
            )
        
        keys_list = "\n".join(self._data_storage.keys())
        result_msg += (
            f"The data_storage currently contains the following keys:\n{keys_list}\n"
        )
        
        return PrepareFilingResponse(results=result_msg)
    
    # ========================================================================
    # retrieve_information Endpoint (LLM-based document querying)
    # ========================================================================
    
    async def retrieve_information(
        self, body: RetrieveInformationRequest
    ) -> RetrieveInformationResponse:
        """
        Query stored documents using LLM-based prompting.
        
        This mirrors Vals benchmark's retrieve_information tool exactly:
        1. Agent provides a prompt containing {{key_name}} placeholders
        2. We validate all referenced keys exist in _data_storage
        3. We substitute each {{key_name}} with the full document text
        4. We send the assembled prompt to the retrieval LLM
        5. We return the LLM's text response
        
        This is the RAG pattern: the document content is injected into a
        separate LLM call (not the agent's main conversation), keeping the
        agent's context window clean while still extracting information.
        """
        if not self.config.retrieval_model_server:
            return RetrieveInformationResponse(
                results="ERROR: Retrieval model not configured. "
                "Set retrieval_model_server in config."
            )
        
        prompt = body.prompt
        input_character_ranges = body.input_character_ranges or []
        
        # --- Validate: prompt must contain at least one {{key}} placeholder ---
        if not re.search(r"\{\{[^{}]+\}\}", prompt):
            # Provide a more helpful error if model passed input_character_ranges
            # but forgot {{key}} in the prompt (common model mistake)
            hint = ""
            if input_character_ranges:
                range_keys = [r.get("key", "") for r in input_character_ranges if isinstance(r, dict)]
                if range_keys:
                    hint = (
                        f" Note: you provided input_character_ranges referencing key(s) "
                        f"{range_keys}, but the prompt text itself must also contain "
                        f"the key in double braces, e.g. '{{{{{range_keys[0]}}}}}'."
                    )
            return RetrieveInformationResponse(
                results="ERROR: Your prompt must include at least one key from data storage "
                "in the format {{key_name}}. Please try again with the correct format. "
                "You can add documents to the data storage with prepare_filing."
                + hint
            )
        
        # --- Parse character ranges into a dict ---
        ranges_dict: Dict[str, tuple] = {}
        for range_spec in input_character_ranges:
            if not isinstance(range_spec, dict):
                return RetrieveInformationResponse(
                    results="ERROR: Each item in input_character_ranges must be an object "
                    "with 'key', 'start', and 'end' fields."
                )
            if "key" not in range_spec or "start" not in range_spec or "end" not in range_spec:
                return RetrieveInformationResponse(
                    results="ERROR: Each range specification must have 'key', 'start', and 'end' fields."
                )
            ranges_dict[range_spec["key"]] = (range_spec["start"], range_spec["end"])
        
        # --- Extract all {{key}} references from the prompt ---
        keys_in_prompt = re.findall(r"\{\{([^{}]+)\}\}", prompt)
        keys_set = set(keys_in_prompt)
        
        # Validate: ranges reference keys that are in the prompt
        for range_key in ranges_dict:
            if range_key not in keys_set:
                return RetrieveInformationResponse(
                    results=f"ERROR: The key '{range_key}' is specified in input_character_ranges "
                    f"but is not referenced in the prompt. "
                    f"Keys in prompt: {', '.join(keys_set) if keys_set else '(none)'}"
                )
        
        # Validate: all referenced keys exist in data storage
        for key in keys_in_prompt:
            if key not in self._data_storage:
                available = ", ".join(self._data_storage.keys()) if self._data_storage else "(empty)"
                return RetrieveInformationResponse(
                    results=f"ERROR: The key '{key}' was not found in the data storage. "
                    f"Available keys are: {available}. "
                    "Use prepare_filing to add documents to the data storage."
                )
        
        # --- Substitute {{key}} placeholders with document content ---
        formatted_data = {}
        for key in keys_in_prompt:
            doc_content = self._data_storage[key]
            if key in ranges_dict:
                start_idx, end_idx = ranges_dict[key]
                formatted_data[key] = doc_content[start_idx:end_idx]
            else:
                formatted_data[key] = doc_content
        
        # Convert {{key}} to {key} for Python .format()
        formatted_prompt = re.sub(r"\{\{([^{}]+)\}\}", r"{\1}", prompt)
        
        try:
            final_prompt = formatted_prompt.format(**formatted_data)
        except KeyError as e:
            available = ", ".join(self._data_storage.keys()) if self._data_storage else "(empty)"
            return RetrieveInformationResponse(
                results=f"ERROR: The key {str(e)} was not found in the data storage. "
                f"Available keys are: {available}"
            )
        
        # --- Send the assembled prompt to the retrieval LLM ---
        # This is a separate, isolated LLM call. The document content is in
        # this request only - it does NOT bloat the agent's conversation context.
        #
        # NOTE: Vals benchmark sends the prompt directly with no system message.
        # We add a minimal system instruction to prevent hallucination when the
        # retrieval LLM receives only a partial document via input_character_ranges.
        # This addresses cases where the LLM fabricates plausible-looking numbers
        # instead of honestly reporting "not found" (e.g., financial statements
        # are typically in the latter half of 10-K filings, so char range 0-200K
        # may not contain them).
        retrieval_system_instruction = (
            "You are a document analysis assistant. Answer the question based ONLY on "
            "the document text provided below. If the requested information is not present "
            "in the provided text, clearly state that the information was not found in the "
            "provided text portion — do NOT guess, estimate, or fabricate numbers."
        )
        try:
            llm_response = await self.server_client.post(
                server_name=self.config.retrieval_model_server.name,
                url_path="/v1/responses",
                json={
                    "input": [
                        {"role": "system", "content": retrieval_system_instruction, "type": "message"},
                        {"role": "user", "content": final_prompt, "type": "message"}
                    ],
                    "temperature": 0.0,
                },
            )
            
            llm_response_json = await get_response_json(llm_response)
            llm_response_obj = NeMoGymResponse.model_validate(llm_response_json)
            
            # Extract text from the LLM response
            result_text = ""
            for output_item in llm_response_obj.output:
                if getattr(output_item, "type", None) == "message":
                    for content_item in getattr(output_item, "content", []):
                        if getattr(content_item, "type", None) == "output_text":
                            result_text += getattr(content_item, "text", "")
            
            if not result_text:
                return RetrieveInformationResponse(
                    results="ERROR: The retrieval LLM returned no text output."
                )
            
            return RetrieveInformationResponse(results=result_text)
            
        except Exception as e:
            return RetrieveInformationResponse(
                results=f"ERROR: Retrieval LLM call failed: {str(e)}"
            )
    
    # ========================================================================
    # submit_final_result Endpoint (matches Vals benchmark)
    # ========================================================================
    
    async def submit_final_result(
        self, body: SubmitFinalResultRequest
    ) -> SubmitFinalResultResponse:
        """
        Accept the agent's final answer submission.
        
        This matches Vals benchmark's submit_final_result tool exactly.
        The tool forces the model to explicitly submit via a tool call rather
        than casually stopping with a text message. This keeps the model in
        tool-calling mode until it deliberately decides it has the answer.
        
        The simple_agent loop will feed this response back to the model,
        which will then output a text message and the loop ends naturally.
        """
        final_result = body.final_result
        if not final_result:
            return SubmitFinalResultResponse(
                results="ERROR: final_result is required. Please provide your answer."
            )
        return SubmitFinalResultResponse(
            results=json.dumps({"success": True, "result": final_result})
        )
    
    # ========================================================================
    # web_search Endpoint (uses tavily package)
    # ========================================================================
    
    async def web_search(self, body: WebSearchRequest) -> WebSearchResponse:
        """Search the web using Tavily. Returns up to 10 results."""
        if self._tavily is None:
            return WebSearchResponse(results=json.dumps({
                "error": "Web search not configured",
                "suggestion": "Set tavily_api_key in config"
            }))
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                raw = self._tavily.search(body.query, num_results=10)
                results = [{"url": r.get("url", ""), "title": r.get("title", ""), "content": r.get("content", "")} 
                           for r in raw.get("results", [])]
                return WebSearchResponse(results=json.dumps(results))
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"web_search attempt {attempt + 1} failed: {e}. Retrying in {2 ** attempt}s...")
                    await asyncio.sleep(2 ** attempt)
                else:
                    logger.error(f"web_search failed after {max_retries} attempts: {e}")
                    return WebSearchResponse(results=json.dumps({"error": str(e)}))
    
    # ========================================================================
    # Verify Endpoint (Required by SimpleResourcesServer)
    # ========================================================================
    
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
        
        # Extract generated answer: prefer submit_final_result tool call,
        # fall back to last assistant text message.
        generated_answer = ""
        
        # First, look for submit_final_result tool call (Vals benchmark pattern)
        for output_item in reversed(body.response.output):
            if getattr(output_item, "type", None) == "function_call":
                if getattr(output_item, "name", None) == "submit_final_result":
                    try:
                        args = json.loads(getattr(output_item, "arguments", "{}"))
                        generated_answer = args.get("final_result", "")
                    except (json.JSONDecodeError, TypeError):
                        pass
                    break
        
        # Fallback: extract from last assistant text message
        if not generated_answer:
            for output_item in reversed(body.response.output):
                if getattr(output_item, "type", None) == "message" and getattr(output_item, "role", None) == "assistant":
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
        
        # Build judge messages (system + user with few-shot examples)
        judge_user_prompt = self.JUDGE_USER_PROMPT_TEMPLATE.format(
            question=question,
            expected_answer=body.expected_answer,
            generated_answer=generated_answer,
        )
        
        judge_params = (self.config.judge_responses_create_params or
                        NeMoGymResponseCreateParamsNonStreaming(input=[])).model_copy(deep=True)
        judge_params.input = [
            NeMoGymEasyInputMessage(role="system", content=self.JUDGE_SYSTEM_PROMPT),
            NeMoGymEasyInputMessage(role="user", content=judge_user_prompt),
        ]
        
        # Call judge model
        response = await self.server_client.post(
            server_name=self.config.judge_model_server.name,
            url_path="/v1/responses",
            json=judge_params,
        )
        judge_response = NeMoGymResponse.model_validate(await get_response_json(response))
        
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
        rating_match = re.search(r'\[\[(\d+)\]\]', judge_text)
        rating = int(rating_match.group(1)) if rating_match else None
        
        # Only [[2]] (fully correct) gets reward 1.0
        reward = 1.0 if rating == 2 else 0.0
        
        return FinanceAgentVerifyResponse(**body.model_dump(), reward=reward)


if __name__ == "__main__":
    FinanceAgentResourcesServer.run_webserver()
