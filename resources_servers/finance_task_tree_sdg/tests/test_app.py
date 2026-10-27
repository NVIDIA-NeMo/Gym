# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from big_finance_harness.tools import ToolError
from fastapi.testclient import TestClient
from omegaconf import OmegaConf

from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from resources_servers.big_finance.sec_cache import SecHttpCache
from resources_servers.finance_task_tree_sdg.app import (
    FinanceTaskTreeSdgConfig,
    FinanceTaskTreeSdgServer,
    SdgVerifyRequest,
    accepted_submission,
    validate_eligibility,
    validate_leak_proposal,
    validate_revised_question,
    validate_verdict,
)
from resources_servers.finance_task_tree_sdg.local_sec import LocalSecCorpus
from resources_servers.finance_task_tree_sdg.sec_tools import (
    AsOfEdgarSearchTool,
    AsOfFetchUrlTool,
    archive_accession,
)


_REPO_ROOT = Path(__file__).resolve().parents[3]
_STAGES = ("eligibility", "question_gen", "leak_proposal", "leak_removal", "answer_gen", "adjudicate")
_QUESTION = (
    "For Apple Inc., using the fiscal 2023 Form 10-K as originally filed, what was the ratio of "
    "operating income to net sales, in percent rounded to one decimal place?"
)
_JUDGE_SECRET = "the judge's private reasoning must never leave the audit file"


def _server(stage: str, tmp_path: Path | None = None, **overrides) -> FinanceTaskTreeSdgServer:
    values = {
        "host": "0.0.0.0",
        "port": 8080,
        "entrypoint": "",
        "name": f"tt_sdg_{stage}",
        "stage": stage,
        "sec_edgar_user_agent": "Test test@example.com",
        "judge_model_server": ModelServerRef(type="responses_api_models", name="judge"),
        "judge_responses_create_params": NeMoGymResponseCreateParamsNonStreaming(input=[]),
        "audit_dir": str(tmp_path / "audit") if tmp_path else None,
    }
    values.update(overrides)
    return FinanceTaskTreeSdgServer(
        config=FinanceTaskTreeSdgConfig(**values),
        server_client=MagicMock(spec=ServerClient),
    )


def _response(output: list[dict], stop_reason: str = "done_tool") -> NeMoGymResponse:
    return NeMoGymResponse(
        id="response",
        created_at=0,
        model="policy",
        object="response",
        output=output,
        tools=[],
        tool_choice="auto",
        parallel_tool_calls=False,
        metadata={"stop_reason": stop_reason},
    )


def _call(name: str, arguments: dict, call_id: str = "call-1", output: str = "ok") -> list[dict]:
    return [
        {
            "id": f"fc-{call_id}",
            "call_id": call_id,
            "type": "function_call",
            "name": name,
            "arguments": json.dumps(arguments),
            "status": "completed",
        },
        {"type": "function_call_output", "call_id": call_id, "output": output},
    ]


def _request(response: NeMoGymResponse, sdg: dict | None = None) -> SdgVerifyRequest:
    return SdgVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "task"}]),
        response=response,
        sdg={"sample_id": "F01.T01.001:SC_SD#0", "as_of_date": "2023-11-03", **(sdg or {})},
    )


def _judge_reply(verdict_text: str) -> MagicMock:
    payload = {
        "id": "judge",
        "created_at": 0,
        "model": "private-judge",
        "object": "response",
        "output": [
            {"type": "reasoning", "id": "r", "summary": [{"type": "summary_text", "text": _JUDGE_SECRET}]},
            {
                "id": "m",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": verdict_text, "annotations": []}],
            },
        ],
        "tools": [],
        "tool_choice": "auto",
        "parallel_tool_calls": False,
    }
    raw = MagicMock(ok=True)
    raw.read = AsyncMock(return_value=json.dumps(payload).encode())
    return raw


def _json_response(url: str, payload: dict) -> httpx.Response:
    return httpx.Response(200, json=payload, request=httpx.Request("GET", url))


def _submissions() -> dict[str, dict]:
    head = {
        "filings": {
            "recent": {
                "form": ["10-Q", "10-K", "8-K"],
                "filingDate": ["2024-02-02", "2023-11-03", "2023-08-03"],
                "accessionNumber": ["0000320193-24-000006", "0000320193-23-000106", "0000320193-23-000077"],
                "primaryDocument": ["q1.htm", "aapl-20230930.htm", "8k.htm"],
                "reportDate": ["2023-12-30", "2023-09-30", "2023-08-03"],
            },
            "files": [{"name": "CIK0000320193-submissions-001.json"}],
        }
    }
    page = {
        "form": ["10-K"],
        "filingDate": ["2020-10-30"],
        "accessionNumber": ["0000320193-20-000096"],
        "primaryDocument": ["aapl-20200926.htm"],
        "reportDate": ["2020-09-26"],
    }
    return {
        "https://data.sec.gov/submissions/CIK0000320193.json": head,
        "https://data.sec.gov/submissions/CIK0000320193-submissions-001.json": page,
    }


def _edgar_tool() -> AsOfEdgarSearchTool:
    tool = AsOfEdgarSearchTool(cache=SecHttpCache(cache_dir=None, use_cache=False), user_agent="Test test@example.com")
    pages = _submissions()

    async def fake_get(url: str) -> httpx.Response:
        return _json_response(url, pages[url])

    tool._get = fake_get
    tool._ticker_to_cik = AsyncMock(return_value="320193")
    return tool


def test_archive_accession_accepts_folder_and_dashed_forms() -> None:
    assert archive_accession(
        "https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/aapl-20230930.htm"
    ) == (
        "320193",
        "0000320193-23-000106",
    )
    assert archive_accession("https://www.sec.gov/Archives/edgar/data/320193/0000320193-23-000106.txt") == (
        "320193",
        "0000320193-23-000106",
    )
    assert archive_accession("https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany") is None


async def test_edgar_search_hides_later_filings_and_reads_older_pages() -> None:
    tool = _edgar_tool()
    result = json.loads(await tool.run({"ticker": "AAPL", "form_type": "10-K"}, as_of=date(2023, 11, 3)))
    assert result["filed_on_or_before"] == "2023-11-03"
    assert [row["accession"] for row in result["filings"]] == ["0000320193-23-000106", "0000320193-20-000096"]
    assert result["filings"][0]["report_date"] == "2023-09-30"
    assert result["filings"][1]["primary_document_url"].endswith("/000032019320000096/aapl-20200926.htm")

    unclamped = json.loads(await tool.run({"ticker": "AAPL", "limit": 1}))
    assert [row["accession"] for row in unclamped["filings"]] == ["0000320193-24-000006"]
    assert "filed_on_or_before" not in unclamped


async def test_fetch_url_refuses_archive_documents_filed_after_as_of() -> None:
    edgar = _edgar_tool()
    fetch = AsOfFetchUrlTool(
        cache=SecHttpCache(cache_dir=None, use_cache=False), history=edgar.history, sec_user_agent="Test t@e.com"
    )
    as_of = date(2023, 11, 3)
    later = "https://www.sec.gov/Archives/edgar/data/320193/000032019324000006/q1.htm"
    with pytest.raises(ToolError, match="filed on 2024-02-02"):
        await fetch.check_as_of(later, as_of)
    await fetch.check_as_of(
        "https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/aapl-20230930.htm", as_of
    )
    await fetch.check_as_of("https://www.sec.gov/Archives/edgar/data/320193/000032019320000096/a.htm", as_of)
    # Unknown accessions fall back to the year encoded in the accession number.
    with pytest.raises(ToolError, match="postdates"):
        await fetch.check_as_of("https://www.sec.gov/Archives/edgar/data/320193/000099999924000001/x.htm", as_of)
    await fetch.check_as_of("https://www.sec.gov/Archives/edgar/data/320193/000099999923000001/x.htm", as_of)
    await fetch.check_as_of(later, None)
    await fetch.check_as_of("https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json", as_of)


_ARCHIVE = "https://www.sec.gov/Archives/edgar/data/320193"


def _corpus(tmp_path: Path) -> LocalSecCorpus:
    download = tmp_path / "step-0-download"
    container = "/workspace/outputs/step-0-download"
    files = {
        "10-K/aapl.html": "<html><body><p>Net sales were 383,285.</p></body></html>",
        "10-K/exhibits/EX-99.1.html": "<html><body><p>Press release.</p></body></html>",
        "10-K/exhibits/EX-10.1.html": "<html><body><p>Whichever exhibit was written last.</p></body></html>",
        "10-Q/q1.html": "<html><body><p>Quarter.</p></body></html>",
        "4/form4.html": "<html><body><p>Rendered Form 4.</p></body></html>",
    }
    for name, text in files.items():
        (download / "data" / name).parent.mkdir(parents=True, exist_ok=True)
        (download / "data" / name).write_text(text, encoding="utf-8")

    def row(accession: str, form: str, filed: str, url: str, kind: str, name: str) -> dict:
        return {
            "accession_number": accession,
            "form_type": form,
            "report_date": "2023-09-30",
            "filing_date": date.fromisoformat(filed),
            "url": url,
            "file_type": kind,
            "file_location": f"{container}/data/{name}",
            "ticker": "AAPL",
        }

    k, q, f4 = "0000320193-23-000106", "0000320193-24-000006", "0000320193-23-000090"
    rows = [
        row(
            k,
            "10-K",
            "2023-11-03",
            f"{_ARCHIVE}/000032019323000106/aapl-20230930.htm",
            "primary_document",
            "10-K/aapl.html",
        ),
        row(
            k,
            "10-K",
            "2023-11-03",
            f"{_ARCHIVE}/000032019323000106/ex991.htm",
            "exhibit",
            "10-K/exhibits/EX-99.1.html",
        ),
        row(
            k,
            "10-K",
            "2023-11-03",
            f"{_ARCHIVE}/000032019323000106/ex101a.htm",
            "exhibit",
            "10-K/exhibits/EX-10.1.html",
        ),
        row(
            k,
            "10-K",
            "2023-11-03",
            f"{_ARCHIVE}/000032019323000106/ex101b.htm",
            "exhibit",
            "10-K/exhibits/EX-10.1.html",
        ),
        row(q, "10-Q", "2024-02-02", f"{_ARCHIVE}/000032019324000006/q1.htm", "primary_document", "10-Q/q1.html"),
        row(
            f4,
            "4",
            "2023-10-02",
            f"{_ARCHIVE}/000032019323000090/xslF345X05/form4.xml",
            "primary_document",
            "4/form4.html",
        ),
    ]
    pq.write_table(pa.Table.from_pylist(rows), download / "sec_metadata.parquet")
    return LocalSecCorpus(metadata_path=download / "sec_metadata.parquet")


def _offline_tools(tmp_path: Path) -> tuple[AsOfEdgarSearchTool, AsOfFetchUrlTool]:
    corpus = _corpus(tmp_path)
    cache = SecHttpCache(cache_dir=None, use_cache=False)
    edgar = AsOfEdgarSearchTool(cache=cache, user_agent="Test test@example.com", corpus=corpus)
    edgar._get = AsyncMock(side_effect=AssertionError("offline edgar_search requested SEC"))
    fetch = AsOfFetchUrlTool(cache=cache, history=edgar.history, sec_user_agent="Test t@e.com", corpus=corpus)
    return edgar, fetch


async def test_offline_edgar_search_lists_only_readable_filings(tmp_path: Path) -> None:
    edgar, _ = _offline_tools(tmp_path)
    result = json.loads(await edgar.run({"ticker": "aapl"}, as_of=date(2023, 11, 3)))
    assert [row["accession"] for row in result["filings"]] == ["0000320193-23-000106"]
    assert result["filings"][0]["primary_document_url"] == f"{_ARCHIVE}/000032019323000106/aapl-20230930.htm"
    unclamped = json.loads(await edgar.run({"ticker": "AAPL"}))
    assert [row["form"] for row in unclamped["filings"]] == ["10-Q", "10-K"]
    with pytest.raises(ToolError, match="unknown ticker"):
        await edgar.run({"ticker": "NOPE"})


async def test_offline_fetch_url_serves_the_corpus_and_refuses_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("big_finance_harness.tools.fetch_url._check_url_safe", lambda url: None)
    _, fetch = _offline_tools(tmp_path)
    as_of = date(2023, 11, 3)
    folder = f"{_ARCHIVE}/000032019323000106"

    bare_host = "http://sec.gov/Archives/edgar/data/320193/000032019323000106/aapl-20230930.htm#part-ii"
    assert "Net sales were 383,285." in await fetch.run({"url": bare_host}, as_of)
    assert "Press release." in await fetch.run({"url": f"{folder}/ex991.htm"}, as_of)
    for page in (f"{folder}/0000320193-23-000106-index.htm", f"{folder}/", folder):
        index = await fetch.run({"url": page}, as_of)
        assert f"{folder}/aapl-20230930.htm" in index and f"{folder}/ex991.htm" in index and "EX-99.1" in index
        assert "ex101a" not in index
    for url in (
        f"{folder}/ex101a.htm",
        f"{_ARCHIVE}/000032019323000090/xslF345X05/form4.xml",
        f"{folder}/R4.htm",
        "https://efts.sec.gov/LATEST/search-index?q=revenue",
        "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=320193",
    ):
        with pytest.raises(ToolError, match="not in the offline SEC corpus"):
            await fetch.run({"url": url}, as_of)
    with pytest.raises(ToolError, match="filed on 2024-02-02"):
        await fetch.run({"url": f"{_ARCHIVE}/000032019324000006/q1.htm"}, as_of)


def test_server_wires_the_corpus_into_both_sec_tools(tmp_path: Path) -> None:
    _corpus(tmp_path)
    server = _server(
        "question_gen", sec_corpus_metadata_path=str(tmp_path / "step-0-download" / "sec_metadata.parquet")
    )
    assert server._edgar._corpus is server._fetch._corpus is not None


def test_leak_proposal_validation() -> None:
    proposal = validate_leak_proposal(
        {"items": [{"span": "Form 10-K", "action": "generalize", "replacement": "annual report", "reason": "form"}]},
        _QUESTION,
    )
    assert proposal["items"][0]["replacement"] == "annual report"
    with pytest.raises(ToolError, match="verbatim"):
        validate_leak_proposal({"items": [{"span": "Note 12", "action": "delete", "reason": "x"}]}, _QUESTION)
    with pytest.raises(ToolError, match="replacement"):
        validate_leak_proposal({"items": [{"span": "Form 10-K", "action": "generalize", "reason": "x"}]}, _QUESTION)
    with pytest.raises(ToolError, match="no_leak_reason"):
        validate_leak_proposal({"items": []}, _QUESTION)
    assert validate_leak_proposal({"items": [], "no_leak_reason": "clean"}, _QUESTION)["items"] == []


def test_revised_question_must_remove_every_proposed_span() -> None:
    config = FinanceTaskTreeSdgConfig(host="", port=0, entrypoint="", name="x", stage="leak_removal")
    proposal = {"items": [{"span": "Form 10-K", "action": "generalize", "replacement": "annual report"}]}
    with pytest.raises(ToolError, match="still in the question"):
        validate_revised_question({"question": _QUESTION}, _QUESTION, proposal, config)
    revised = _QUESTION.replace("Form 10-K", "annual report")
    assert validate_revised_question({"question": revised}, _QUESTION, proposal, config) == revised


def test_verdict_labels_come_from_candidates() -> None:
    assert validate_verdict({"question_valid": True, "selected": "B"}, ["A", "B"]) == {
        "question_valid": True,
        "selected": "B",
    }
    with pytest.raises(ToolError):
        validate_verdict({"question_valid": True, "selected": "C"}, ["A", "B"])
    with pytest.raises(ToolError):
        validate_verdict({"question_valid": "yes", "selected": "none"}, ["A"])


def test_eligibility_codes_come_from_the_seeded_catalog() -> None:
    catalog = ["1311", "4911", "6021"]
    assert validate_eligibility(
        {"scope": "restricted", "sic_codes": ["6021", "1311", "6021"], "reason": "banks and E&P"}, catalog
    ) == {"scope": "restricted", "sic_codes": ["1311", "6021"], "reason": "banks and E&P"}
    general = validate_eligibility({"scope": "any_industry", "sic_codes": [], "reason": "general"}, catalog)
    assert general["sic_codes"] == []
    with pytest.raises(ToolError, match="not in the provided list"):
        validate_eligibility({"scope": "restricted", "sic_codes": ["9999"], "reason": "x"}, catalog)
    with pytest.raises(ToolError, match="must not be empty"):
        validate_eligibility({"scope": "restricted", "sic_codes": [], "reason": "x"}, catalog)
    with pytest.raises(ToolError, match="must be empty"):
        validate_eligibility({"scope": "unsupported", "sic_codes": ["1311"], "reason": "x"}, catalog)
    with pytest.raises(ToolError, match="reason"):
        validate_eligibility({"scope": "any_industry", "sic_codes": []}, catalog)


async def test_eligibility_verify_has_no_judge() -> None:
    server = _server("eligibility")
    server.server_client.post = AsyncMock()
    submission = {"scope": "restricted", "sic_codes": ["6021"], "reason": "needs loan allowances"}
    result = await server.verify(
        MagicMock(session={}),
        _request(_response(_call("submit_eligibility", submission)), {"sic_codes": ["6021", "4911"]}),
    )
    assert result.reward == 1.0
    assert result.sdg_result == {"stage": "eligibility", "status": "mapped", **submission}
    server.server_client.post.assert_not_awaited()


def test_accepted_submission_skips_failed_and_unfinished_calls() -> None:
    output = [
        *_call("submit_question", {"question": "bad"}, "call-1", "[ERROR] question must be longer"),
        *_call("submit_question", {"question": _QUESTION}, "call-2"),
    ]
    response = _response(output)
    response.metadata["tool_call_error_status"] = json.dumps({"call-1": True, "call-2": False})
    assert accepted_submission(response, ("submit_question",)) == ("submit_question", {"question": _QUESTION})
    assert accepted_submission(_response(output, stop_reason="max_turns"), ("submit_question",)) is None


def test_submit_tools_validate_against_the_seeded_row() -> None:
    server = _server("leak_proposal")
    with TestClient(server.setup_webserver()) as client:
        seed = {
            "responses_create_params": {"input": [{"role": "user", "content": "task"}]},
            "sdg": {"sample_id": "s", "as_of_date": "2023-11-03", "question": _QUESTION},
        }
        assert client.post("/seed_session", json=seed).status_code == 200
        ok = client.post(
            "/submit_leak_proposal",
            json={"items": [{"span": "Form 10-K", "action": "delete", "reason": "names the form"}]},
        )
        assert ok.headers["x-nemo-gym-tool-error"] == "false"
        bad = client.post("/submit_leak_proposal", json={"items": [{"span": "Note 9", "action": "delete"}]})
        assert bad.headers["x-nemo-gym-tool-error"] == "true"
        assert "verbatim" in bad.text
        assert client.post("/edgar_search", json={"ticker": "AAPL"}).status_code == 404


async def test_question_gen_verify_returns_only_the_verdict(tmp_path: Path) -> None:
    server = _server("question_gen", tmp_path)
    server.server_client.post = AsyncMock(return_value=_judge_reply('{"verdict": "pass"}'))
    result = await server.verify(
        MagicMock(session={}), _request(_response(_call("submit_question", {"question": _QUESTION})))
    )

    assert result.reward == 1.0
    assert result.sdg_result["status"] == "judge_pass"
    assert result.sdg_result["question"] == _QUESTION
    assert result.sdg_result["judge"] == {"verdict": "pass", "model": "private-judge", "parsed": True}
    assert _JUDGE_SECRET not in result.model_dump_json()

    audit_files = list((tmp_path / "audit" / "question_gen").rglob("*.json"))
    assert len(audit_files) == 1
    audit = json.loads(audit_files[0].read_text())
    assert audit["verdict"] == "pass"
    assert _JUDGE_SECRET in json.dumps(audit)


async def test_judge_fail_reason_stays_in_the_audit_file(tmp_path: Path) -> None:
    server = _server("question_gen", tmp_path)
    reply = json.dumps({"verdict": "fail", "failed_criteria": [3], "reason": _JUDGE_SECRET})
    server.server_client.post = AsyncMock(return_value=_judge_reply(reply))
    result = await server.verify(
        MagicMock(session={}), _request(_response(_call("submit_question", {"question": _QUESTION})))
    )

    assert result.reward == 0.0
    assert result.sdg_result["judge"] == {"verdict": "fail", "model": "private-judge", "parsed": True}
    assert _JUDGE_SECRET not in result.model_dump_json()

    audit = json.loads(next((tmp_path / "audit" / "question_gen").rglob("*.json")).read_text())
    assert audit["failed_criteria"] == [3]
    assert audit["reason"] == _JUDGE_SECRET


async def test_unparseable_judge_reply_fails_closed(tmp_path: Path) -> None:
    server = _server("leak_removal", tmp_path)
    server.server_client.post = AsyncMock(return_value=_judge_reply("looks fine to me"))
    proposal = {"items": [{"span": "Form 10-K", "action": "delete", "reason": "form"}]}
    revised = _QUESTION.replace("the fiscal 2023 Form 10-K", "fiscal 2023 reporting")
    result = await server.verify(
        MagicMock(session={}),
        _request(
            _response(_call("submit_revised_question", {"question": revised})),
            {"question": _QUESTION, "leak_proposal": proposal},
        ),
    )
    assert result.reward == 0.0
    assert result.sdg_result["status"] == "judge_unparsed"
    assert result.sdg_result["judge"]["parsed"] is False


async def test_rejection_skips_the_judge() -> None:
    server = _server("question_gen")
    server.server_client.post = AsyncMock()
    rejection = {"reason_code": "data_unavailable", "explanation": "No capacity disclosure."}
    result = await server.verify(MagicMock(session={}), _request(_response(_call("submit_rejection", rejection))))
    assert result.reward == 0.0
    assert result.sdg_result["status"] == "rejected"
    assert result.sdg_result["rejection"] == rejection
    server.server_client.post.assert_not_awaited()


async def test_answer_and_adjudicate_stages_pass_through() -> None:
    answer = await _server("answer_gen").verify(
        MagicMock(session={}), _request(_response(_call("final_answer", {"answer": "30.1%"}, output="30.1%")))
    )
    assert answer.reward == 1.0
    assert answer.sdg_result == {"stage": "answer_gen", "status": "answered", "final_answer": "30.1%"}

    candidates = [{"label": "A", "answer": "30.1%"}, {"label": "B", "answer": "29.8%"}]
    verdict = await _server("adjudicate").verify(
        MagicMock(session={}),
        _request(
            _response(_call("submit_verdict", {"question_valid": True, "selected": "A"})), {"candidates": candidates}
        ),
    )
    assert verdict.reward == 1.0
    assert verdict.sdg_result == {
        "stage": "adjudicate",
        "status": "adjudicated",
        "question_valid": True,
        "selected": "A",
    }


@pytest.mark.parametrize("stage", _STAGES)
def test_stage_configs_resolve(stage: str) -> None:
    initial = OmegaConf.merge(
        GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        {
            "config_paths": [
                str(_REPO_ROOT / "responses_api_models/vllm_model/configs/vllm_model.yaml"),
                str(_REPO_ROOT / f"resources_servers/finance_task_tree_sdg/configs/{stage}.yaml"),
            ],
            "policy_base_url": "https://provider.example/v1",
            "policy_api_key": "test-key",
            "policy_model_name": "policy",
        },
    )
    config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=initial,
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    server = config[f"tt_sdg_{stage}_resources_server"].resources_servers.finance_task_tree_sdg
    assert server.stage == stage
    agent = config[f"tt_sdg_{stage}_agent"].responses_api_agents.finance_agent
    assert agent.resources_server.name == f"tt_sdg_{stage}_resources_server"
