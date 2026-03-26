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
import urllib.error
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.finance_sec_search.app import (
    FinanceAgentResourcesServer,
    FinanceAgentResourcesServerConfig,
    FinanceAgentSearchRequest,
    FinanceAgentVerifyRequest,
    RateLimiter,
    RetrieveInformationRequest,
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
    prompt_fpath = str(Path(__file__).resolve().parents[1] / "prompt_templates/finance_sec_search_judge.yaml")
    return FinanceAgentResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="finance_sec_search_test",
        cache_dir=temp_cache_dir,
        judge_prompt_template_fpath=prompt_fpath,
    )


@pytest.fixture
def server(server_config):
    """Create test server instance."""
    return FinanceAgentResourcesServer(config=server_config, server_client=MagicMock(spec=ServerClient))


# ============================================================================
# Test: Server Initialization
# ============================================================================


class TestServerInitialization:
    def test_sanity(self, server_config) -> None:
        """Test server can be instantiated."""
        server = FinanceAgentResourcesServer(config=server_config, server_client=MagicMock(spec=ServerClient))
        assert server is not None

    def test_cache_directories_created(self, server, temp_cache_dir) -> None:
        """Test cache directories are created on init."""
        assert Path(temp_cache_dir).exists()
        assert (Path(temp_cache_dir) / "filings_metadata").exists()
        assert (Path(temp_cache_dir) / "filings").exists()


# ============================================================================
# Test: Ticker Loading (startup)
# ============================================================================


class TestTickerLoading:
    """Tests for _load_tickers_or_fail startup behavior."""

    MOCK_TICKERS_RAW = {
        "0": {"ticker": "AAPL", "cik_str": "320193", "title": "APPLE INC."},
        "1": {"ticker": "MSFT", "cik_str": "789019", "title": "MICROSOFT CORP"},
    }

    def test_load_from_cache(self, server, temp_cache_dir):
        """Tickers load from disk cache without any network calls."""
        tickers_file = Path(temp_cache_dir) / "tickers.json"
        tickers_file.write_text(json.dumps(self.MOCK_TICKERS_RAW))

        server._load_tickers_or_fail()

        assert server._initialized is True
        assert "AAPL" in server._tickers
        assert "MSFT" in server._tickers
        assert server._tickers["AAPL"]["cik"] == "0000320193"

    @patch("resources_servers.finance_sec_search.app.urllib.request.urlopen")
    def test_load_fetches_from_sec(self, mock_urlopen, server, temp_cache_dir):
        """Downloads tickers from SEC and caches to disk when no cache exists."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(self.MOCK_TICKERS_RAW).encode("utf-8")
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        server._load_tickers_or_fail()

        assert server._initialized is True
        assert "AAPL" in server._tickers
        assert (Path(temp_cache_dir) / "tickers.json").exists()
        mock_urlopen.assert_called_once()

    @patch("time.sleep")
    @patch("resources_servers.finance_sec_search.app.urllib.request.urlopen")
    def test_load_raises_after_retries(self, mock_urlopen, mock_sleep, server):
        """RuntimeError raised when SEC is unreachable after all retries."""
        mock_urlopen.side_effect = urllib.error.URLError("connection refused")

        with pytest.raises(RuntimeError, match="Failed to load SEC ticker data"):
            server._load_tickers_or_fail()

        assert server._initialized is False
        assert mock_urlopen.call_count == 5

    @patch("time.sleep")
    @patch("resources_servers.finance_sec_search.app.urllib.request.urlopen")
    def test_load_succeeds_on_retry(self, mock_urlopen, mock_sleep, server):
        """Recovers after transient failures."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(self.MOCK_TICKERS_RAW).encode("utf-8")
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        mock_urlopen.side_effect = [
            urllib.error.URLError("timeout"),
            urllib.error.URLError("timeout"),
            mock_resp,
        ]

        server._load_tickers_or_fail()

        assert server._initialized is True
        assert "AAPL" in server._tickers
        assert mock_urlopen.call_count == 3

    @patch("time.sleep")
    @patch("resources_servers.finance_sec_search.app.urllib.request.urlopen")
    def test_load_refetches_on_corrupt_cache(self, mock_urlopen, mock_sleep, server, temp_cache_dir):
        """Re-downloads if cached tickers.json contains invalid JSON."""
        tickers_file = Path(temp_cache_dir) / "tickers.json"
        tickers_file.write_text("not valid json{{{")

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(self.MOCK_TICKERS_RAW).encode("utf-8")
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        server._load_tickers_or_fail()

        assert server._initialized is True
        assert "AAPL" in server._tickers
        mock_urlopen.assert_called_once()


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
# Test: Ticker Lookup
# ============================================================================


class TestTickerLookup:
    @pytest.mark.asyncio
    async def test_exact_ticker(self, server) -> None:
        """Exact ticker returns company info."""
        server._tickers = {"AAPL": {"cik": "0000320193", "name": "APPLE INC."}}
        server._initialized = True

        result = await server._resolve_ticker("AAPL")
        assert result is not None
        assert result["ticker"] == "AAPL"
        assert result["cik"] == "0000320193"

    @pytest.mark.asyncio
    async def test_case_insensitive(self, server) -> None:
        """Ticker lookup is case-insensitive."""
        server._tickers = {"AAPL": {"cik": "0000320193", "name": "APPLE INC."}}
        server._initialized = True

        result = await server._resolve_ticker("aapl")
        assert result is not None
        assert result["ticker"] == "AAPL"

    @pytest.mark.asyncio
    async def test_unknown_ticker(self, server) -> None:
        """Unknown ticker returns None."""
        server._tickers = {"AAPL": {"cik": "0000320193", "name": "APPLE INC."}}
        server._initialized = True

        result = await server._resolve_ticker("NOTEXIST")
        assert result is None


def _write_filings_cache(server, cik: str, filings: dict):
    """Test helper: write filings dict to the server's cache directory."""
    with open(server._get_company_cache_path(cik), "w") as f:
        json.dump(filings, f)


# ============================================================================
# Test: Main Endpoint (sec_filing_search)
# ============================================================================


class TestSECFilingSearch:
    @pytest.mark.asyncio
    async def test_search_by_ticker(self, server) -> None:
        """Test searching by ticker symbol."""
        server._tickers = {"AAPL": {"cik": "0000320193", "name": "APPLE INC."}}
        server._initialized = True

        test_filings = {
            "000032019325000001": {
                "ticker": "AAPL",
                "cik": "0000320193",
                "form": "10-K",
                "filing_date": "2025-01-15",
                "report_date": "2024-12-31",
                "accession_number": "0000320193-25-000001",
                "filing_url": "https://...",
            },
        }
        _write_filings_cache(server, "0000320193", test_filings)

        request = FinanceAgentSearchRequest(ticker="AAPL")
        response = await server.sec_filing_search(request)

        results = json.loads(response.results)
        assert len(results) == 1
        assert results[0]["ticker"] == "AAPL"
        assert results[0]["company_name"] == "APPLE INC."

    @pytest.mark.asyncio
    async def test_search_not_found(self, server) -> None:
        """Unknown ticker returns an error with suggestion."""
        server._initialized = True

        request = FinanceAgentSearchRequest(ticker="NOTEXIST")
        response = await server.sec_filing_search(request)

        results = json.loads(response.results)
        assert "error" in results
        assert "NOTEXIST" in results["error"]

    @pytest.mark.asyncio
    async def test_no_default_form_type_filter(self, server) -> None:
        """Without form_types param, all form types are returned."""
        server._tickers = {"AAPL": {"cik": "0000320193", "name": "APPLE INC."}}
        server._initialized = True

        test_filings = {
            "a": {
                "ticker": "AAPL",
                "form": "10-K",
                "filing_date": "2025-01-15",
                "report_date": "2024-12-31",
                "accession_number": "a",
                "filing_url": "",
            },
            "b": {
                "ticker": "AAPL",
                "form": "10-Q",
                "filing_date": "2024-11-01",
                "report_date": "2024-09-30",
                "accession_number": "b",
                "filing_url": "",
            },
            "c": {
                "ticker": "AAPL",
                "form": "8-K",
                "filing_date": "2024-10-01",
                "report_date": "2024-10-01",
                "accession_number": "c",
                "filing_url": "",
            },
            "d": {
                "ticker": "AAPL",
                "form": "DEF 14A",
                "filing_date": "2024-09-01",
                "report_date": "2024-09-01",
                "accession_number": "d",
                "filing_url": "",
            },
            "e": {
                "ticker": "AAPL",
                "form": "4",
                "filing_date": "2024-08-01",
                "report_date": "2024-08-01",
                "accession_number": "e",
                "filing_url": "",
            },
        }
        _write_filings_cache(server, "0000320193", test_filings)

        request = FinanceAgentSearchRequest(ticker="AAPL")
        response = await server.sec_filing_search(request)

        results = json.loads(response.results)
        assert len(results) == 5
        forms = {r["form"] for r in results}
        assert forms == {"10-K", "10-Q", "8-K", "DEF 14A", "4"}

    @pytest.mark.asyncio
    async def test_explicit_form_types_filter(self, server) -> None:
        """Passing form_types filters to only those types."""
        server._tickers = {"AAPL": {"cik": "0000320193", "name": "APPLE INC."}}
        server._initialized = True

        test_filings = {
            "a": {
                "ticker": "AAPL",
                "form": "10-K",
                "filing_date": "2025-01-15",
                "report_date": "2024-12-31",
                "accession_number": "a",
                "filing_url": "",
            },
            "b": {
                "ticker": "AAPL",
                "form": "8-K",
                "filing_date": "2024-10-01",
                "report_date": "2024-10-01",
                "accession_number": "b",
                "filing_url": "",
            },
            "c": {
                "ticker": "AAPL",
                "form": "4",
                "filing_date": "2024-08-01",
                "report_date": "2024-08-01",
                "accession_number": "c",
                "filing_url": "",
            },
        }
        _write_filings_cache(server, "0000320193", test_filings)

        request = FinanceAgentSearchRequest(ticker="AAPL", form_types=["10-K"])
        response = await server.sec_filing_search(request)

        results = json.loads(response.results)
        assert len(results) == 1
        assert results[0]["form"] == "10-K"

    @pytest.mark.asyncio
    async def test_results_capped_at_max_filing_results(self, server) -> None:
        """Results are capped at max_filing_results (default 200)."""
        server._tickers = {"AAPL": {"cik": "0000320193", "name": "APPLE INC."}}
        server._initialized = True

        test_filings = {
            f"{i}": {
                "ticker": "AAPL",
                "form": "10-Q",
                "filing_date": f"20{20 + i // 365:02d}-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
                "report_date": f"20{20 + i // 365:02d}-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
                "accession_number": f"{i}",
                "filing_url": "",
            }
            for i in range(1, 251)
        }
        _write_filings_cache(server, "0000320193", test_filings)

        request = FinanceAgentSearchRequest(ticker="AAPL")
        response = await server.sec_filing_search(request)

        results = json.loads(response.results)
        assert len(results) == 200


# ============================================================================
# Test: Date Filtering (start_date / end_date)
# ============================================================================

MIXED_DATE_FILINGS = {
    "a": {
        "ticker": "AAPL",
        "form": "10-K",
        "filing_date": "2025-01-15",
        "report_date": "2024-12-31",
        "accession_number": "a",
        "filing_url": "",
    },
    "b": {
        "ticker": "AAPL",
        "form": "10-Q",
        "filing_date": "2024-07-15",
        "report_date": "2024-06-30",
        "accession_number": "b",
        "filing_url": "",
    },
    "c": {
        "ticker": "AAPL",
        "form": "10-Q",
        "filing_date": "2024-01-10",
        "report_date": "2023-12-31",
        "accession_number": "c",
        "filing_url": "",
    },
    "d": {
        "ticker": "AAPL",
        "form": "10-K",
        "filing_date": "2023-06-01",
        "report_date": "2023-05-31",
        "accession_number": "d",
        "filing_url": "",
    },
}


class TestDateFiltering:
    @pytest.fixture(autouse=True)
    def _setup(self, server):
        server._tickers = {"AAPL": {"cik": "0000320193", "name": "APPLE INC."}}
        server._initialized = True
        _write_filings_cache(server, "0000320193", MIXED_DATE_FILINGS)

    @pytest.mark.asyncio
    async def test_start_date_only(self, server) -> None:
        """start_date filters out filings before the given date."""
        request = FinanceAgentSearchRequest(ticker="AAPL", start_date="2024-01-01")
        response = await server.sec_filing_search(request)
        results = json.loads(response.results)
        assert len(results) == 3
        assert all(r["filing_date"] >= "2024-01-01" for r in results)

    @pytest.mark.asyncio
    async def test_end_date_only(self, server) -> None:
        """end_date filters out filings after the given date."""
        request = FinanceAgentSearchRequest(ticker="AAPL", end_date="2024-07-15")
        response = await server.sec_filing_search(request)
        results = json.loads(response.results)
        assert len(results) == 3
        assert all(r["filing_date"] <= "2024-07-15" for r in results)

    @pytest.mark.asyncio
    async def test_start_and_end_date(self, server) -> None:
        """Combined date range narrows results."""
        request = FinanceAgentSearchRequest(
            ticker="AAPL", start_date="2024-01-01", end_date="2024-12-31"
        )
        response = await server.sec_filing_search(request)
        results = json.loads(response.results)
        assert len(results) == 2
        for r in results:
            assert "2024-01-01" <= r["filing_date"] <= "2024-12-31"

    @pytest.mark.asyncio
    async def test_date_filter_no_results(self, server) -> None:
        """Date range with no matching filings returns error with filter info."""
        request = FinanceAgentSearchRequest(
            ticker="AAPL", start_date="2030-01-01", end_date="2030-12-31"
        )
        response = await server.sec_filing_search(request)
        results = json.loads(response.results)
        assert "error" in results
        assert "start_date=" in results["error"]
        assert "end_date=" in results["error"]

    @pytest.mark.asyncio
    async def test_results_sorted_newest_first(self, server) -> None:
        """Results are sorted by filing_date descending."""
        request = FinanceAgentSearchRequest(ticker="AAPL")
        response = await server.sec_filing_search(request)
        results = json.loads(response.results)
        dates = [r["filing_date"] for r in results]
        assert dates == sorted(dates, reverse=True)


# ============================================================================
# Test: In-Memory Filings Cache
# ============================================================================


class TestFilingsCache:
    @pytest.mark.asyncio
    async def test_memory_cache_avoids_disk_read(self, server) -> None:
        """Second call for same CIK returns from _filings_cache, not disk."""
        server._tickers = {"AAPL": {"cik": "0000320193", "name": "APPLE INC."}}
        server._initialized = True

        test_filings = {
            "a": {
                "ticker": "AAPL",
                "form": "10-K",
                "filing_date": "2025-01-15",
                "report_date": "2024-12-31",
                "accession_number": "a",
                "primary_document": "doc.htm",
                "filing_url": "",
            },
        }
        _write_filings_cache(server, "0000320193", test_filings)

        result1 = await server._get_company_filings("0000320193", "AAPL")
        assert "a" in result1
        assert "0000320193" in server._filings_cache

        # Delete the disk file -- second call should still work from memory
        server._get_company_cache_path("0000320193").unlink()

        result2 = await server._get_company_filings("0000320193", "AAPL")
        assert result2 == result1


# ============================================================================
# Test: Dump Fallback
# ============================================================================


class TestDumpFallback:
    def test_lookup_dump_returns_none_without_config(self, server) -> None:
        """_lookup_dump returns None when sec_dump_path is not configured."""
        assert server.config.sec_dump_path is None
        result = server._lookup_dump("https://www.sec.gov/Archives/edgar/data/320193/000032019325000001/doc.htm")
        assert result is None

    def test_lookup_dump_returns_none_without_metadata(self, server, tmp_path) -> None:
        """_lookup_dump returns None when metadata is not in _filings_cache."""
        server.config.sec_dump_path = str(tmp_path)
        result = server._lookup_dump("https://www.sec.gov/Archives/edgar/data/320193/000032019325000001/doc.htm")
        assert result is None

    def test_lookup_dump_reads_from_dump(self, server, tmp_path) -> None:
        """_lookup_dump reads HTML from dump, parses to text, returns it."""
        server.config.sec_dump_path = str(tmp_path)

        server._filings_cache["0000320193"] = {
            "000032019325000001": {
                "ticker": "AAPL",
                "form": "10-K",
                "report_date": "2024-12-31",
                "accession_number": "0000320193-25-000001",
            },
        }

        dump_dir = tmp_path / "AAPL" / "10-K" / "2024" / "0000320193-25-000001"
        dump_dir.mkdir(parents=True)
        (dump_dir / "primary-document.html").write_text("<html><body><p>Revenue was $100B</p></body></html>")

        url = "https://www.sec.gov/Archives/edgar/data/320193/000032019325000001/doc.htm"
        result = server._lookup_dump(url)
        assert result is not None
        assert "Revenue was $100B" in result


# ============================================================================
# Test: Download and Parse Filing
# ============================================================================


class TestDownloadAndParseFiling:
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

    def test_url_to_filing_path(self, server) -> None:
        """Test URL-to-filepath conversion for SEC URLs."""
        url = "https://www.sec.gov/Archives/edgar/data/320193/000032019325000008/aapl-20250104.htm"
        path = server._url_to_filing_path(url)
        assert path is not None
        assert path.name == "000032019325000008.txt"
        assert "0000320193" in str(path.parent)

    def test_url_to_filing_path_invalid(self, server) -> None:
        """Invalid URLs return None."""
        assert server._url_to_filing_path("https://example.com/not-sec") is None


# ============================================================================
# Test: Retrieve Information
# ============================================================================


class TestRetrieveInformation:
    @pytest.fixture(autouse=True)
    def _configure_retrieval_model(self, server):
        server.config.retrieval_model_server = ModelServerRef(type="responses_api_models", name="test-model")

    @pytest.mark.asyncio
    async def test_prompt_with_curly_braces_in_content(self, server) -> None:
        """Curly braces in document text must not break placeholder substitution."""
        server._data_storage["doc"] = 'Revenue {"COGS": 500, "net": 1000} end of report'

        import orjson

        payload = orjson.dumps(
            {
                "id": "r1",
                "created_at": 0,
                "model": "m",
                "object": "response",
                "output": [
                    {
                        "id": "msg1",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "COGS is 500", "annotations": []}],
                    }
                ],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
            }
        )
        mock_response = MagicMock()
        mock_response.read = AsyncMock(return_value=payload)
        server.server_client = MagicMock()
        server.server_client.post = AsyncMock(return_value=mock_response)

        request = RetrieveInformationRequest(prompt="What is COGS in {{doc}}?")
        response = await server.retrieve_information(request)
        assert "COGS is 500" in response.results

    @pytest.mark.asyncio
    async def test_missing_key_error(self, server) -> None:
        """Referencing a key not in data storage returns an error."""
        request = RetrieveInformationRequest(prompt="Tell me about {{nonexistent}}")
        response = await server.retrieve_information(request)
        assert "ERROR: Key 'nonexistent' not in data storage." in response.results

    @pytest.mark.asyncio
    async def test_no_placeholder_error(self, server) -> None:
        """Prompt without {{key}} placeholders returns an error."""
        request = RetrieveInformationRequest(prompt="What is the revenue?")
        response = await server.retrieve_information(request)
        assert "ERROR: Prompt must contain at least one {{key_name}} placeholder." in response.results

    @pytest.mark.asyncio
    async def test_prompt_too_large(self, server) -> None:
        """Prompt exceeding max_chars returns a size error."""
        server._data_storage["huge"] = "x" * 600_000
        request = RetrieveInformationRequest(prompt="Summarize {{huge}}")
        response = await server.retrieve_information(request)
        assert "ERROR: Prompt too large" in response.results
        assert "huge: 600000 chars" in response.results


# ============================================================================
# Test: Verify (reward calculation)
# ============================================================================


class TestVerify:
    """Tests for verify() — the reward function used during training."""

    @staticmethod
    def _msg(text: str) -> dict:
        return {
            "id": "msg_1",
            "content": [{"annotations": [], "text": text, "type": "output_text"}],
            "role": "assistant",
            "status": "completed",
            "type": "message",
        }

    @staticmethod
    def _tool_call(name: str, arguments: str) -> dict:
        return {
            "id": "tc_1",
            "call_id": "call_1",
            "name": name,
            "arguments": arguments,
            "type": "function_call",
            "status": "completed",
        }

    def _make_response(self, *output_items) -> NeMoGymResponse:
        return NeMoGymResponse(
            id="resp_test",
            created_at=0.0,
            model="test",
            object="response",
            output=list(output_items),
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
        )

    def _make_verify_request(self, response: NeMoGymResponse, expected_answer: str) -> FinanceAgentVerifyRequest:
        return FinanceAgentVerifyRequest(
            question="What was revenue?",
            expected_answer=expected_answer,
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[{"role": "user", "content": "What was revenue?"}]
            ),
            response=response,
        )

    def _make_judge_response(self, text: str) -> str:
        return NeMoGymResponse(
            id="judge_resp",
            created_at=0.0,
            model="judge",
            object="response",
            output=[
                {
                    "id": "judge_msg",
                    "content": [{"annotations": [], "text": text, "type": "output_text"}],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        ).model_dump_json()

    @staticmethod
    def _prompt_template_fpath() -> str:
        return str(Path(__file__).resolve().parents[1] / "prompt_templates/finance_sec_search_judge.yaml")

    def _create_server_with_judge(self, tmp_path: Path) -> FinanceAgentResourcesServer:
        config = FinanceAgentResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="test",
            cache_dir=str(tmp_path),
            judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            judge_prompt_template_fpath=self._prompt_template_fpath(),
        )
        return FinanceAgentResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    def _create_server_no_judge(self, tmp_path: Path) -> FinanceAgentResourcesServer:
        config = FinanceAgentResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="test",
            cache_dir=str(tmp_path),
            judge_prompt_template_fpath=self._prompt_template_fpath(),
        )
        return FinanceAgentResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    @pytest.mark.asyncio
    async def test_verify_fully_correct(self, tmp_path) -> None:
        """Judge returns [[2]] → reward 1.0 with metadata."""
        server = self._create_server_with_judge(tmp_path)
        post_mock = MagicMock()
        post_mock.read = AsyncMock(
            return_value=self._make_judge_response("The answer matches exactly. The rating is: [[2]]")
        )
        server.server_client.post = AsyncMock(return_value=post_mock)

        response = self._make_response(
            self._tool_call("submit_final_result", json.dumps({"final_result": "$391.0 billion"}))
        )
        req = self._make_verify_request(response, "$391.0 billion")
        res = await server.verify(req)
        assert res.reward == 1.0
        assert res.judge_rating == 2
        assert "[[2]]" in res.judge_text

    @pytest.mark.asyncio
    async def test_verify_partially_correct(self, tmp_path) -> None:
        """Judge returns [[1]] → reward 0.0."""
        server = self._create_server_with_judge(tmp_path)
        post_mock = MagicMock()
        post_mock.read = AsyncMock(
            return_value=self._make_judge_response("Correct number but missing explanation. [[1]]")
        )
        server.server_client.post = AsyncMock(return_value=post_mock)

        response = self._make_response(
            self._tool_call("submit_final_result", json.dumps({"final_result": "$391 billion"}))
        )
        req = self._make_verify_request(response, "$391.0 billion")
        res = await server.verify(req)
        assert res.reward == 0.0

    @pytest.mark.asyncio
    async def test_verify_incorrect(self, tmp_path) -> None:
        """Judge returns [[0]] → reward 0.0."""
        server = self._create_server_with_judge(tmp_path)
        post_mock = MagicMock()
        post_mock.read = AsyncMock(return_value=self._make_judge_response("Completely wrong value. [[0]]"))
        server.server_client.post = AsyncMock(return_value=post_mock)

        response = self._make_response(
            self._tool_call("submit_final_result", json.dumps({"final_result": "$100 million"}))
        )
        req = self._make_verify_request(response, "$391.0 billion")
        res = await server.verify(req)
        assert res.reward == 0.0

    @pytest.mark.asyncio
    async def test_verify_unparseable_judge_output(self, tmp_path) -> None:
        """Judge returns no [[N]] rating → reward 0.0, judge_rating is None."""
        server = self._create_server_with_judge(tmp_path)
        post_mock = MagicMock()
        post_mock.read = AsyncMock(
            return_value=self._make_judge_response("I cannot determine a rating for this response.")
        )
        server.server_client.post = AsyncMock(return_value=post_mock)

        response = self._make_response(
            self._tool_call("submit_final_result", json.dumps({"final_result": "$391.0 billion"}))
        )
        req = self._make_verify_request(response, "$391.0 billion")
        res = await server.verify(req)
        assert res.reward == 0.0
        assert res.judge_rating is None

    @pytest.mark.asyncio
    async def test_verify_extracts_answer_from_submit_tool(self, tmp_path) -> None:
        """Answer is extracted from submit_final_result, not from text messages."""
        server = self._create_server_with_judge(tmp_path)
        post_mock = MagicMock()
        post_mock.read = AsyncMock(return_value=self._make_judge_response("[[2]]"))
        server.server_client.post = AsyncMock(return_value=post_mock)

        response = self._make_response(
            self._msg("I think the answer is maybe $100 million"),
            self._tool_call("submit_final_result", json.dumps({"final_result": "$391.0 billion"})),
        )
        req = self._make_verify_request(response, "$391.0 billion")
        res = await server.verify(req)
        assert res.reward == 1.0

        # Verify the judge was called with the tool call answer, not the text
        call_args = server.server_client.post.call_args
        judge_payload = call_args.kwargs["json"] if "json" in call_args.kwargs else call_args[1].get("json")
        judge_input_text = str(judge_payload)
        assert "$391.0 billion" in judge_input_text

    @pytest.mark.asyncio
    async def test_verify_falls_back_to_text_message(self, tmp_path) -> None:
        """When no submit_final_result tool call, extracts from last assistant message."""
        server = self._create_server_with_judge(tmp_path)
        post_mock = MagicMock()
        post_mock.read = AsyncMock(return_value=self._make_judge_response("[[2]]"))
        server.server_client.post = AsyncMock(return_value=post_mock)

        response = self._make_response(self._msg("The revenue was $391.0 billion."))
        req = self._make_verify_request(response, "$391.0 billion")
        res = await server.verify(req)
        assert res.reward == 1.0

    @pytest.mark.asyncio
    async def test_verify_no_judge_substring_match(self, tmp_path) -> None:
        """Without judge configured, uses substring matching."""
        server = self._create_server_no_judge(tmp_path)

        response = self._make_response(
            self._tool_call("submit_final_result", json.dumps({"final_result": "Revenue was $391.0 billion in 2024"}))
        )
        req = self._make_verify_request(response, "$391.0 billion")
        res = await server.verify(req)
        assert res.reward == 1.0

    @pytest.mark.asyncio
    async def test_verify_no_judge_substring_mismatch(self, tmp_path) -> None:
        """Without judge, substring mismatch → reward 0.0."""
        server = self._create_server_no_judge(tmp_path)

        response = self._make_response(
            self._tool_call("submit_final_result", json.dumps({"final_result": "$100 million"}))
        )
        req = self._make_verify_request(response, "$391.0 billion")
        res = await server.verify(req)
        assert res.reward == 0.0

    @pytest.mark.asyncio
    async def test_verify_judge_call_failure(self, tmp_path) -> None:
        """Judge HTTP call failure → reward 0.0, no crash."""
        server = self._create_server_with_judge(tmp_path)
        server.server_client.post = AsyncMock(side_effect=ConnectionError("judge unavailable"))

        response = self._make_response(
            self._tool_call("submit_final_result", json.dumps({"final_result": "$391.0 billion"}))
        )
        req = self._make_verify_request(response, "$391.0 billion")
        res = await server.verify(req)
        assert res.reward == 0.0

    @pytest.mark.asyncio
    async def test_verify_curly_braces_in_content(self, tmp_path) -> None:
        """Curly braces in answers must not break judge prompt formatting."""
        server = self._create_server_with_judge(tmp_path)
        post_mock = MagicMock()
        post_mock.read = AsyncMock(return_value=self._make_judge_response("[[2]]"))
        server.server_client.post = AsyncMock(return_value=post_mock)

        response = self._make_response(
            self._tool_call("submit_final_result", json.dumps({"final_result": 'Revenue {"net": 1000}'}))
        )
        req = self._make_verify_request(response, '{"net": 1000}')
        res = await server.verify(req)
        assert res.reward == 1.0
