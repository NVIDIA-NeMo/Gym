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
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.finance_agent.app import (
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
    prompt_fpath = str(Path(__file__).resolve().parents[1] / "prompt_templates/finance_agent_judge.txt")
    return FinanceAgentResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="finance_agent_test",
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
    async def test_returns_only_default_form_types(self, server) -> None:
        """Test that only 10-K, 10-Q, and DEF 14A filings are returned by default."""
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
        assert len(results) == 3
        forms = [r["form"] for r in results]
        assert "10-K" in forms
        assert "10-Q" in forms
        assert "DEF 14A" in forms
        assert "8-K" not in forms
        assert "4" not in forms

    @pytest.mark.asyncio
    async def test_results_capped_at_30(self, server) -> None:
        """Results are capped at 30 regardless of how many filings exist."""
        server._tickers = {"AAPL": {"cik": "0000320193", "name": "APPLE INC."}}
        server._initialized = True

        test_filings = {
            f"{i}": {
                "ticker": "AAPL",
                "form": "10-Q",
                "filing_date": f"2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
                "report_date": f"2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
                "accession_number": f"{i}",
                "filing_url": "",
            }
            for i in range(1, 51)
        }
        _write_filings_cache(server, "0000320193", test_filings)

        request = FinanceAgentSearchRequest(ticker="AAPL")
        response = await server.sec_filing_search(request)

        results = json.loads(response.results)
        assert len(results) == 30


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

    def test_get_content_filepath(self, server) -> None:
        """Test filepath generation for content files."""
        path = server._get_content_filepath("0000320193", "0000320193-25-000008")
        assert path.name == "000032019325000008.txt"
        assert "0000320193" in str(path.parent)


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
        return str(Path(__file__).resolve().parents[1] / "prompt_templates/finance_agent_judge.txt")

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
