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
"""Tests for Finance Agent Resource Server."""
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nemo_gym.server_utils import ServerClient
from resources_servers.finance_agent.app import (
    PrepareFilingRequest,
    RateLimiter,
    FinanceAgentSearchRequest,
    FinanceAgentResourcesServer,
    FinanceAgentResourcesServerConfig,
)


# ============================================================================
# Mock Data
# ============================================================================

MOCK_HTML = """
<html>
<head>
    <style>body { color: red; }</style>
    <script>alert('hello');</script>
</head>
<body>
    <ix:header>iXBRL Header</ix:header>
    <p>Company Financial Report</p>
    <ix:nonfraction>$1,000,000</ix:nonfraction>
    <p>Revenue Details</p>
</body>
</html>
"""


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def server_config(temp_cache_dir):
    """Create test server configuration."""
    return FinanceAgentResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="finance_agent_test",
        cache_dir=temp_cache_dir,
    )


@pytest.fixture
def server(server_config):
    """Create test server instance."""
    return FinanceAgentResourcesServer(
        config=server_config,
        server_client=MagicMock(spec=ServerClient)
    )


# ============================================================================
# Test: Server Initialization
# ============================================================================

class TestServerInitialization:
    def test_sanity(self, server_config) -> None:
        """Test server can be instantiated."""
        server = FinanceAgentResourcesServer(
            config=server_config,
            server_client=MagicMock(spec=ServerClient)
        )
        assert server is not None
    
    def test_cache_directories_created(self, server, temp_cache_dir) -> None:
        """Test cache directories are created on init."""
        assert Path(temp_cache_dir).exists()
        assert (Path(temp_cache_dir) / "filings").exists()


# ============================================================================
# Test: Rate Limiter
# ============================================================================

class TestRateLimiter:
    @pytest.mark.asyncio
    async def test_rate_limiter_allows_requests(self):
        """Test rate limiter allows requests within limit."""
        limiter = RateLimiter(max_requests=5, window_seconds=1.0)
        
        # Should allow 5 requests immediately
        for _ in range(5):
            await limiter.acquire()
        
        # Requests should be recorded
        assert len(limiter.requests) == 5


# ============================================================================
# Test: Company Search (Helper Methods)
# ============================================================================

class TestCompanySearch:
    def test_tokenize(self, server) -> None:
        """Test tokenization of text."""
        assert server._tokenize("APPLE INC.") == {"APPLE", "INC"}
        assert server._tokenize("shift4 payments") == {"SHIFT4", "PAYMENTS"}
        assert server._tokenize("U.S. Steel Corp") == {"U", "S", "STEEL", "CORP"}
    
    def test_tokenize_min_length(self, server) -> None:
        """Test tokenization with minimum length filter."""
        # With min_length=2, single-char tokens are filtered out
        assert server._tokenize("X", min_length=2) == set()  # Empty - "X" filtered
        assert server._tokenize("U.S. Steel Corp", min_length=2) == {"STEEL", "CORP"}  # U, S filtered
        assert server._tokenize("3M Company", min_length=2) == {"3M", "COMPANY"}  # 3M passes (2 chars)
    
    def test_single_char_query_no_company_match(self, server) -> None:
        """Test that single-char queries don't match via company name search."""
        server._companies = [
            {"ticker": "X", "cik": "0001163302", "name": "UNITED STATES STEEL CORP"},
            {"ticker": "KARX", "cik": "0001729637", "name": "KARBON-X CORP."},
            {"ticker": "LIMX", "cik": "0001803977", "name": "LIMITLESS X HOLDINGS INC."},
        ]
        
        # Single-char "X" should NOT match any company by name
        results = server._search_company_by_tokens("X")
        assert len(results) == 0  # No matches - single char filtered
    
    def test_search_company_token_match(self, server) -> None:
        """Test token-based matching."""
        server._companies = [
            {"ticker": "AAPL", "cik": "0000320193", "name": "APPLE INC."},
            {"ticker": "MSFT", "cik": "0000789019", "name": "MICROSOFT CORP"},
        ]
        
        # "APPLE" token matches "APPLE" in "APPLE INC."
        results = server._search_company_by_tokens("APPLE")
        assert len(results) == 1
        assert results[0]["ticker"] == "AAPL"
    
    def test_search_company_no_partial_match(self, server) -> None:
        """Test that partial tokens don't match (Cat should not match Catalyst)."""
        server._companies = [
            {"ticker": "CPRX", "cik": "0001234567", "name": "CATALYST PHARMACEUTICALS"},
            {"ticker": "CAT", "cik": "0000018230", "name": "CATERPILLAR INC"},
        ]
        
        # "CAT" should NOT match "CATALYST" or "CATERPILLAR" - tokens must match exactly
        results = server._search_company_by_tokens("CAT")
        assert len(results) == 0  # No matches - CAT is not a token in either name
    
    def test_search_company_case_insensitive(self, server) -> None:
        """Test case-insensitive matching."""
        server._companies = [
            {"ticker": "AAPL", "cik": "0000320193", "name": "APPLE INC."},
        ]
        
        results = server._search_company_by_tokens("apple")
        assert len(results) == 1
        assert results[0]["ticker"] == "AAPL"
    
    def test_search_company_multiple_tokens(self, server) -> None:
        """Test matching with multiple query tokens - ALL must match."""
        server._companies = [
            {"ticker": "X", "cik": "0001163302", "name": "UNITED STATES STEEL CORP"},
            {"ticker": "STLD", "cik": "0001022671", "name": "STEEL DYNAMICS INC"},
        ]
        
        # "Steel" matches both companies (STEEL token in both)
        results = server._search_company_by_tokens("Steel")
        assert len(results) == 2
        
        # "United Steel" matches only US Steel (both UNITED and STEEL must be in name)
        results = server._search_company_by_tokens("United Steel")
        assert len(results) == 1
        assert results[0]["ticker"] == "X"
        
        # "US Steel" does NOT match (US is not a token in "UNITED STATES STEEL")
        results = server._search_company_by_tokens("US Steel")
        assert len(results) == 0  # No match - LLM should retry with full name or ticker
    
    def test_search_company_limit(self, server) -> None:
        """Test result limiting."""
        server._companies = [
            {"ticker": "STLA", "cik": "0001", "name": "STEEL COMPANY A"},
            {"ticker": "STLB", "cik": "0002", "name": "STEEL COMPANY B"},
            {"ticker": "STLC", "cik": "0003", "name": "STEEL COMPANY C"},
        ]
        
        results = server._search_company_by_tokens("STEEL", limit=2)
        assert len(results) == 2
    
    def test_search_shift4(self, server) -> None:
        """Test Shift4 company search."""
        server._companies = [
            {"ticker": "FOUR", "cik": "0001639723", "name": "SHIFT4 PAYMENTS INC"},
        ]
        
        # "Shift4" token matches "SHIFT4" in company name
        results = server._search_company_by_tokens("Shift4")
        assert len(results) == 1
        assert results[0]["ticker"] == "FOUR"


# ============================================================================
# Test: Filing Cache
# ============================================================================

class TestFilingCache:
    def test_get_company_cache_path(self, server) -> None:
        """Test cache path generation with CIK padding."""
        path = server._get_company_cache_path("320193")
        assert path.name == "0000320193.jsonl"
    
    def test_has_company_cache_false(self, server) -> None:
        """Test cache check returns False for missing cache."""
        assert server._has_company_cache("0000320193") is False
    
    def test_save_and_load_filings(self, server) -> None:
        """Test saving and loading filings."""
        test_filings = [
            {"ticker": "AAPL", "form": "10-K", "filing_date": "2025-01-15"},
            {"ticker": "AAPL", "form": "10-Q", "filing_date": "2024-11-01"},
        ]
        
        server._save_company_filings("0000320193", test_filings)
        assert server._has_company_cache("0000320193")
        
        loaded = server._load_company_filings("0000320193")
        assert len(loaded) == 2
        assert loaded[0]["form"] == "10-K"


# ============================================================================
# Test: Manual Ticker Mappings
# ============================================================================

class TestManualTickerMappings:
    def test_manual_mappings_applied(self, server) -> None:
        """Test that manual ticker mappings are applied (e.g., US Steel ticker X)."""
        # Simulate loading tickers without X in the cache
        server._ticker_to_cik = {"AAPL": "0000320193"}
        server._cik_to_ticker = {"0000320193": "AAPL"}
        server._companies = [{"ticker": "AAPL", "cik": "0000320193", "name": "APPLE INC."}]
        
        # Apply manual mappings
        server._apply_manual_ticker_mappings()
        
        # X should now be present (US Steel)
        assert "X" in server._ticker_to_cik
        assert server._ticker_to_cik["X"] == "0001163302"
        
        # Company should be in the list
        x_companies = [c for c in server._companies if c["ticker"] == "X"]
        assert len(x_companies) == 1
        assert x_companies[0]["name"] == "UNITED STATES STEEL CORP"
    
    def test_manual_mappings_not_duplicated(self, server) -> None:
        """Test that manual mappings don't create duplicates if already present."""
        # Pre-populate with X
        server._ticker_to_cik = {"X": "0001163302"}
        server._cik_to_ticker = {"0001163302": "X"}
        server._companies = [{"ticker": "X", "cik": "0001163302", "name": "UNITED STATES STEEL CORP"}]
        
        # Apply manual mappings again
        server._apply_manual_ticker_mappings()
        
        # Should still only have one X entry
        x_companies = [c for c in server._companies if c["ticker"] == "X"]
        assert len(x_companies) == 1
    
    def test_us_steel_resolvable_by_ticker(self, server) -> None:
        """Test that US Steel can be resolved by ticker X after manual mapping."""
        server._ticker_to_cik = {}
        server._cik_to_ticker = {}
        server._companies = []
        
        # Apply manual mappings
        server._apply_manual_ticker_mappings()
        
        # Should be able to resolve X
        matches = server._resolve_all_matches("X")
        assert len(matches) == 1
        assert matches[0]["ticker"] == "X"
        assert matches[0]["cik"] == "0001163302"
        assert matches[0]["match_type"] == "ticker"


# ============================================================================
# Test: Resolve All Matches (company_or_ticker logic)
# ============================================================================

class TestResolveAllMatches:
    def test_resolve_ticker_match(self, server) -> None:
        """Test that exact ticker match is found."""
        server._ticker_to_cik = {"AAPL": "0000320193"}
        server._companies = [{"ticker": "AAPL", "cik": "0000320193", "name": "APPLE INC."}]
        
        matches = server._resolve_all_matches("AAPL")
        assert len(matches) == 1
        assert matches[0]["ticker"] == "AAPL"
        assert matches[0]["match_type"] == "ticker"
    
    def test_resolve_company_name_match(self, server) -> None:
        """Test that company name fuzzy match is found."""
        server._ticker_to_cik = {"FOUR": "0001639723"}
        server._companies = [{"ticker": "FOUR", "cik": "0001639723", "name": "SHIFT4 PAYMENTS INC"}]
        
        matches = server._resolve_all_matches("Shift4")
        assert len(matches) == 1
        assert matches[0]["ticker"] == "FOUR"
        assert matches[0]["match_type"] == "company_name"
    
    def test_resolve_deduplicates_by_cik(self, server) -> None:
        """Test that same company matched by both ticker and name is deduplicated."""
        server._ticker_to_cik = {"AAPL": "0000320193"}
        server._companies = [{"ticker": "AAPL", "cik": "0000320193", "name": "APPLE INC."}]
        
        # "AAPL" matches as ticker, "APPLE" also in company name
        matches = server._resolve_all_matches("AAPL")
        assert len(matches) == 1  # Deduplicated
        assert matches[0]["match_type"] == "ticker"  # Ticker match takes priority
    
    def test_resolve_multiple_companies(self, server) -> None:
        """Test that multiple different companies can be returned."""
        server._ticker_to_cik = {
            "AAPL": "0000320193",
            "APP": "0000111111",
        }
        server._companies = [
            {"ticker": "AAPL", "cik": "0000320193", "name": "APPLE INC."},
            {"ticker": "APP", "cik": "0000111111", "name": "APPLOVIN CORP"},
        ]
        
        # "APP" matches ticker exactly
        matches = server._resolve_all_matches("APP")
        assert len(matches) >= 1
        tickers = [m["ticker"] for m in matches]
        assert "APP" in tickers


# ============================================================================
# Test: Main Endpoint (sec_filing_search)
# ============================================================================

class TestSECFilingSearch:
    @pytest.mark.asyncio
    async def test_search_by_ticker(self, server) -> None:
        """Test searching by ticker symbol."""
        server._ticker_to_cik = {"AAPL": "0000320193"}
        server._companies = [{"ticker": "AAPL", "cik": "0000320193", "name": "APPLE INC."}]
        server._initialized = True
        
        # Pre-cache filings
        test_filings = [
            {"ticker": "AAPL", "cik": "0000320193", "form": "10-K",
             "filing_date": "2025-01-15", "report_date": "2024-12-31",
             "accession_number": "0000320193-25-000001", "filing_url": "https://..."},
        ]
        server._save_company_filings("0000320193", test_filings)
        
        request = FinanceAgentSearchRequest(company_or_ticker="AAPL", top_n=5)
        response = await server.sec_filing_search(request)
        
        results = json.loads(response.results)
        assert len(results) == 1
        assert results[0]["ticker"] == "AAPL"
        assert results[0]["match_type"] == "ticker"
    
    @pytest.mark.asyncio
    async def test_search_by_company_name(self, server) -> None:
        """Test searching by company name (fuzzy resolution)."""
        server._ticker_to_cik = {"FOUR": "0001639723"}
        server._companies = [{"ticker": "FOUR", "cik": "0001639723", "name": "SHIFT4 PAYMENTS INC"}]
        server._initialized = True
        
        # Pre-cache filings
        test_filings = [
            {"ticker": "FOUR", "cik": "0001639723", "form": "10-K",
             "filing_date": "2025-01-15", "report_date": "2024-12-31",
             "accession_number": "0001639723-25-000001", "filing_url": "https://..."},
        ]
        server._save_company_filings("0001639723", test_filings)
        
        # Make request by company name
        request = FinanceAgentSearchRequest(company_or_ticker="Shift4", top_n=5)
        response = await server.sec_filing_search(request)
        
        results = json.loads(response.results)
        assert len(results) == 1
        assert results[0]["ticker"] == "FOUR"
        assert results[0]["match_type"] == "company_name"
    
    @pytest.mark.asyncio
    async def test_search_not_found(self, server) -> None:
        """Test error when company not found."""
        server._initialized = True
        
        request = FinanceAgentSearchRequest(company_or_ticker="NOTEXIST")
        response = await server.sec_filing_search(request)
        
        results = json.loads(response.results)
        assert "error" in results
    
    @pytest.mark.asyncio
    async def test_returns_only_default_form_types(self, server) -> None:
        """Test that only 10-K, 10-Q, and DEF 14A filings are returned by default."""
        server._ticker_to_cik = {"AAPL": "0000320193"}
        server._companies = [{"ticker": "AAPL", "cik": "0000320193", "name": "APPLE INC."}]
        server._initialized = True
        
        test_filings = [
            {"ticker": "AAPL", "form": "10-K", "filing_date": "2025-01-15", "report_date": "2024-12-31", "accession_number": "a", "filing_url": ""},
            {"ticker": "AAPL", "form": "10-Q", "filing_date": "2024-11-01", "report_date": "2024-09-30", "accession_number": "b", "filing_url": ""},
            {"ticker": "AAPL", "form": "8-K", "filing_date": "2024-10-01", "report_date": "2024-10-01", "accession_number": "c", "filing_url": ""},
            {"ticker": "AAPL", "form": "DEF 14A", "filing_date": "2024-09-01", "report_date": "2024-09-01", "accession_number": "d", "filing_url": ""},
            {"ticker": "AAPL", "form": "4", "filing_date": "2024-08-01", "report_date": "2024-08-01", "accession_number": "e", "filing_url": ""},
        ]
        server._save_company_filings("0000320193", test_filings)
        
        request = FinanceAgentSearchRequest(company_or_ticker="AAPL")
        response = await server.sec_filing_search(request)
        
        results = json.loads(response.results)
        assert len(results) == 3
        forms = [r["form"] for r in results]
        assert "10-K" in forms
        assert "10-Q" in forms
        assert "DEF 14A" in forms
        assert "8-K" not in forms
        assert "4" not in forms
    
    @pytest.mark.asyncio
    async def test_top_n_limit(self, server) -> None:
        """Test limiting results with top_n."""
        server._ticker_to_cik = {"AAPL": "0000320193"}
        server._companies = [{"ticker": "AAPL", "cik": "0000320193", "name": "APPLE INC."}]
        server._initialized = True
        
        test_filings = [
            {"ticker": "AAPL", "form": "10-Q", "filing_date": f"2024-{i:02d}-01", "report_date": f"2024-{i:02d}-01", "accession_number": f"{i}", "filing_url": ""}
            for i in range(1, 13)
        ]
        server._save_company_filings("0000320193", test_filings)
        
        request = FinanceAgentSearchRequest(company_or_ticker="AAPL", top_n=3)
        response = await server.sec_filing_search(request)
        
        results = json.loads(response.results)
        assert len(results) == 3


# ============================================================================
# Test: Prepare Filing
# ============================================================================

class TestPrepareFiling:
    def test_parse_sec_url(self, server) -> None:
        """Test SEC URL parsing."""
        url = "https://www.sec.gov/Archives/edgar/data/320193/000032019325000008/aapl-20241228.htm"
        result = server._parse_sec_url(url)
        assert result is not None
        assert result["cik"] == "0000320193"
        assert "000032019325000008" in result["accession_number"].replace("-", "")
    
    def test_parse_sec_url_invalid(self, server) -> None:
        """Test parsing invalid URL returns None."""
        result = server._parse_sec_url("https://example.com/file.htm")
        assert result is None
    
    def test_parse_html_to_text(self, server) -> None:
        """Test HTML parsing removes scripts/styles and extracts text."""
        result = server._parse_html_to_text(MOCK_HTML)
        
        # Should have content
        assert "Company Financial Report" in result
        assert "Revenue Details" in result
        assert "$1,000,000" in result
        
        # Should NOT have script/style content
        assert "alert" not in result
        assert "color: red" not in result
        
        # iXBRL tags should be unwrapped (content kept)
        assert "iXBRL Header" in result
    
    def test_get_content_filepath(self, server) -> None:
        """Test filepath generation for content files."""
        path = server._get_content_filepath("0000320193", "0000320193-25-000008")
        assert path.name == "000032019325000008.txt"
        assert "0000320193" in str(path.parent)


