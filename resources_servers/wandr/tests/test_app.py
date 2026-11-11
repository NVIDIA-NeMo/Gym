# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from aiohttp import web as web_server
from aiohttp.test_utils import TestServer

from nemo_gym.client import ServerClient
from nemo_gym.core import AGENT_INVALID_OUTPUT, VERIFIER_ERROR, NeMoGymResponse
from nemo_gym.verification import exercise_verifier_fixture
from resources_servers.wandr import app as app_module
from resources_servers.wandr import web
from resources_servers.wandr.app import (
    REWARD_METRICS,
    VERIFIER_FIXTURE,
    FetchPageRequest,
    VerifierRun,
    WandrResourcesServer,
    WandrResourcesServerConfig,
    WandrVerifyRequest,
    WebSearchRequest,
    WriteRecordsRequest,
    read_rewards,
    record_error,
    submitted_records,
)
from resources_servers.wandr.web import (
    DirectFetchClient,
    ExaFetchClient,
    UnsafeURLError,
    extract_page,
    fetch_public,
    gym_get,
    read_body,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
SMOKE_TASK = REPO_ROOT / "benchmarks" / "wandr" / "data" / "tasks" / "smoke"
RECORD = {
    "item": {"topic": "Garbage collection"},
    "url": "https://en.wikipedia.org/wiki/Garbage_collection_(computer_science)",
    "excerpts": ["Garbage collection was invented by American computer scientist John McCarthy around 1959"],
    "answer": {"detail": "Invented by John McCarthy around 1959."},
}


def test_verifier_fixture() -> None:
    asyncio.run(
        exercise_verifier_fixture(
            VERIFIER_FIXTURE,
            reward_range=(0.0, 1.0),
            higher_is_better=True,
            determinism="unknown",
        )
    )


def make_server(**config: object) -> WandrResourcesServer:
    config.setdefault("exa_api_key", "test-exa-key")
    server = WandrResourcesServer(
        config=WandrResourcesServerConfig(host="127.0.0.1", port=0, entrypoint="app.py", name="wandr", **config),
        server_client=MagicMock(spec=ServerClient),
    )
    server._index_files = {
        "results_smoke.jsonl": {"task": "smoke", "item_fields": ["topic"]},
        "results_pairs.jsonl": {"task": "pairs", "item_fields": ["company", "person"]},
    }
    return server


def write_call(call_id: str, file: str, records: list, output: dict | None) -> list[dict]:
    items = [
        {
            "id": f"fc_{call_id}",
            "type": "function_call",
            "call_id": call_id,
            "name": "write_records",
            "arguments": json.dumps({"file": file, "records": records}),
            "status": "completed",
        }
    ]
    if output is not None:
        items.append({"type": "function_call_output", "call_id": call_id, "output": json.dumps(output)})
    return items


def make_response(*items: list[dict]) -> NeMoGymResponse:
    return NeMoGymResponse.model_validate(
        {
            "id": "response",
            "created_at": 0,
            "model": "model",
            "object": "response",
            "output": [item for group in items for item in group],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
        }
    )


def verify_request(response: NeMoGymResponse) -> WandrVerifyRequest:
    return WandrVerifyRequest(
        responses_create_params={"input": "task"},
        response=response,
        task_name="smoke",
        required_output_files=["results_smoke.jsonl"],
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({}, None),
        ({"url": "ftp://example.org/file"}, "url"),
        ({"url": ""}, "url"),
        ({"url": "https://example.org/a page"}, "whitespace"),
        ({"url": "http://[invalid"}, "malformed"),
        ({"url": "https://user:pass@example.org/"}, "credentials"),
        ({"url": "http://10.0.0.1/"}, "non-public"),
        ({"url": "http://example.org:99999/"}, "malformed"),
        ({"item": {"topic": ""}}, "item is missing"),
        ({"item": "Garbage collection"}, "item must be an object"),
        ({"answer": "text"}, "answer must be an object"),
        ({"excerpts": ["ok", 3]}, "excerpts"),
    ],
)
def test_record_error_matches_upstream_row_checks(change: dict, message: str | None) -> None:
    error = record_error({**RECORD, **change}, ["topic"])
    assert error is None if message is None else message in error


def test_submitted_records_keeps_only_rows_the_tool_accepted() -> None:
    bad = {**RECORD, "url": "not a url"}
    response = make_response(
        write_call(
            "a",
            "results_smoke.jsonl",
            [RECORD, bad],
            {"file": "results_smoke.jsonl", "accepted": 1, "rejected": [{"index": 1, "reason": "url"}]},
        ),
        write_call("b", "results_smoke.jsonl", [RECORD], {"file": "results_smoke.jsonl", "error": "unknown file"}),
        write_call("c", "results_smoke.jsonl", [RECORD], None),
        write_call("d", "results_other.jsonl", [RECORD], {"file": "results_other.jsonl", "accepted": 1}),
    )
    assert submitted_records(response, ["results_smoke.jsonl"]) == {"results_smoke.jsonl": [RECORD]}


@pytest.mark.parametrize("arguments", ["null", "[1, 2]", "3", '"text"'])
def test_submitted_records_skips_calls_whose_arguments_are_not_an_object(arguments: str) -> None:
    call = write_call("a", "results_smoke.jsonl", [RECORD], {"file": "results_smoke.jsonl", "accepted": 1})
    call[0]["arguments"] = arguments
    assert submitted_records(make_response(call), ["results_smoke.jsonl"]) == {"results_smoke.jsonl": []}


def test_write_records_reports_each_rejected_row() -> None:
    server = make_server()
    bad = {**RECORD, "item": {"company": "Acme"}}
    result = asyncio.run(server.write_records(WriteRecordsRequest(file="results_pairs.jsonl", records=[bad])))
    assert result.accepted == 0
    assert [entry.index for entry in result.rejected] == [0]
    assert "person" in result.rejected[0].reason

    result = asyncio.run(server.write_records(WriteRecordsRequest(file="results_smoke.jsonl", records=[RECORD])))
    assert (result.accepted, result.rejected, result.error) == (1, [], None)


@pytest.mark.parametrize(("file", "records"), [("results_unknown.jsonl", [RECORD]), ("results_smoke.jsonl", [])])
def test_write_records_refuses_unknown_files_and_empty_batches(file: str, records: list) -> None:
    result = asyncio.run(make_server().write_records(WriteRecordsRequest(file=file, records=records)))
    assert result.error is not None
    assert result.accepted == 0


def test_write_records_names_a_string_records_argument_as_bad_json() -> None:
    result = asyncio.run(
        make_server().write_records(
            WriteRecordsRequest(file="results_smoke.jsonl", records='[{"item": {"topic": "x"}')
        )
    )
    assert result.accepted == 0
    assert "not a JSON array" in result.error


def test_verify_without_records_is_a_valid_zero() -> None:
    server = make_server()
    result = asyncio.run(server.verify(verify_request(make_response())))
    assert (result.reward, result.mask_sample, result.failure_kind) == (0.0, False, AGENT_INVALID_OUTPUT)
    assert result.evaluation_completed is True


def test_verify_scores_zero_when_one_required_file_is_empty() -> None:
    """WANDR rejects a submission that leaves any task-tree file empty, so the verifier is not run."""
    server = make_server()

    async def run_upstream_verifier(task_name: str, files: dict) -> VerifierRun:
        raise AssertionError("the verifier must not run")

    object.__setattr__(server, "run_upstream_verifier", run_upstream_verifier)
    request = verify_request(
        make_response(write_call("a", "results_smoke.jsonl", [RECORD], {"file": "results_smoke.jsonl", "accepted": 1}))
    )
    request.required_output_files.append("results_smoke.subtask.jsonl")
    result = asyncio.run(server.verify(request))
    assert (result.reward, result.submitted_rows, result.failure_kind) == (0.0, 1, AGENT_INVALID_OUTPUT)
    assert "results_smoke.subtask.jsonl" in result.failure_reason


def test_verify_uses_configured_metric_and_reports_all_metrics() -> None:
    server = make_server(reward_metric="hard_f1_full")
    rewards = {name: 0.25 for name in REWARD_METRICS} | {"hard_f1_full": 0.5}

    async def run_upstream_verifier(task_name: str, files: dict) -> VerifierRun:
        assert (task_name, files) == ("smoke", {"results_smoke.jsonl": [RECORD]})
        return VerifierRun(rewards=rewards)

    object.__setattr__(server, "run_upstream_verifier", run_upstream_verifier)
    accepted = {"file": "results_smoke.jsonl", "accepted": 1}
    result = asyncio.run(
        server.verify(verify_request(make_response(write_call("a", "results_smoke.jsonl", [RECORD], accepted))))
    )
    assert result.reward == 0.5
    assert result.soft_f1_full == 0.25
    assert result.submitted_rows == 1
    assert not result.mask_sample
    assert result.evaluation_completed is True


def files_request(output_files: dict[str, str], response: NeMoGymResponse | None = None) -> WandrVerifyRequest:
    request = verify_request(response or make_response())
    request.output_files = output_files
    return request


def test_verify_scores_delivered_output_files_as_written() -> None:
    """An agent that writes files sends their text; the verifier gets it unchanged, invalid lines included."""
    server = make_server()
    text = json.dumps(RECORD) + "\nnot json\n\n"
    seen = []

    async def run_upstream_verifier(task_name: str, files: dict) -> VerifierRun:
        seen.append((task_name, files))
        return VerifierRun(rewards={name: 0.5 for name in REWARD_METRICS})

    object.__setattr__(server, "run_upstream_verifier", run_upstream_verifier)
    # write_records calls in the same response are ignored once files are delivered.
    accepted = {"file": "results_smoke.jsonl", "accepted": 1}
    response = make_response(write_call("a", "results_smoke.jsonl", [RECORD, RECORD], accepted))
    result = asyncio.run(server.verify(files_request({"results_smoke.jsonl": text}, response)))
    assert seen == [("smoke", {"results_smoke.jsonl": text})]
    assert (result.reward, result.submitted_rows, result.evaluation_completed) == (0.5, 2, True)


@pytest.mark.parametrize("output_files", [{}, {"results_smoke.jsonl": ""}, {"results_smoke.jsonl": " \n\n"}])
def test_verify_scores_zero_when_a_delivered_required_file_is_missing_or_blank(output_files: dict) -> None:
    server = make_server()

    async def run_upstream_verifier(task_name: str, files: dict) -> VerifierRun:
        raise AssertionError("the verifier must not run")

    object.__setattr__(server, "run_upstream_verifier", run_upstream_verifier)
    result = asyncio.run(server.verify(files_request(output_files)))
    assert (result.reward, result.submitted_rows, result.failure_kind) == (0.0, 0, AGENT_INVALID_OUTPUT)
    assert "results_smoke.jsonl" in result.failure_reason


def test_verify_passes_only_the_required_delivered_files() -> None:
    server = make_server()
    seen = []

    async def run_upstream_verifier(task_name: str, files: dict) -> VerifierRun:
        seen.append(files)
        return VerifierRun(rewards={name: 0.0 for name in REWARD_METRICS})

    object.__setattr__(server, "run_upstream_verifier", run_upstream_verifier)
    delivered = {"results_smoke.jsonl": "{}\n", "notes.md": "scratch", "../escape.jsonl": "{}\n"}
    asyncio.run(server.verify(files_request(delivered)))
    assert seen == [{"results_smoke.jsonl": "{}\n"}]


def test_run_upstream_verifier_writes_delivered_text_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    server = make_server()
    written = {}

    async def run_subprocess(task_dir: Path, core_dir: Path, workspace: Path, logs: Path) -> VerifierRun:
        written.update({path.name: path.read_text() for path in workspace.iterdir()})
        return VerifierRun(error="stub")

    monkeypatch.setattr(server, "_run_subprocess", run_subprocess)
    text = '{"a": 1}\nnot json\n'
    asyncio.run(server.run_upstream_verifier("smoke", {"results_smoke.jsonl": text}))
    assert written == {"results_smoke.jsonl": text}
    asyncio.run(server.run_upstream_verifier("smoke", {"results_smoke.jsonl": [RECORD]}))
    assert written == {"results_smoke.jsonl": json.dumps(RECORD) + "\n"}


@pytest.mark.parametrize(
    ("task_name", "files"),
    [
        ("../../etc", {"results_smoke.jsonl": [RECORD]}),
        ("smoke", {"../escape.jsonl": [RECORD]}),
        ("pairs", {"results_smoke.jsonl": [RECORD]}),
    ],
)
def test_run_upstream_verifier_accepts_only_indexed_task_files(task_name: str, files: dict) -> None:
    with pytest.raises(ValueError, match="has no output files"):
        asyncio.run(make_server().run_upstream_verifier(task_name, files))


def test_verify_masks_the_sample_when_the_verifier_keeps_failing() -> None:
    server = make_server(inconclusive_verification_retries=2)
    calls = []

    async def run_upstream_verifier(task_name: str, files: dict) -> VerifierRun:
        calls.append(task_name)
        return VerifierRun(error="judge unavailable")

    object.__setattr__(server, "run_upstream_verifier", run_upstream_verifier)
    accepted = {"file": "results_smoke.jsonl", "accepted": 1}
    result = asyncio.run(
        server.verify(verify_request(make_response(write_call("a", "results_smoke.jsonl", [RECORD], accepted))))
    )
    assert len(calls) == 3
    assert (result.reward, result.mask_sample, result.failure_kind) == (0.0, True, VERIFIER_ERROR)
    assert result.evaluation_completed is False
    assert result.failure_reason == "judge unavailable"


def test_extract_page_reads_html_main_text_and_plain_text() -> None:
    html = (
        b"<html><head><title>GC history</title></head><body><nav>Menu</nav><article><p>"
        + b"Garbage collection was invented by John McCarthy around 1959 to simplify memory management in Lisp. " * 3
        + b"</p></article></body></html>"
    )
    page = extract_page(html, "text/html; charset=utf-8", "https://example.org/gc")
    assert page.title == "GC history"
    assert "John McCarthy" in page.text

    assert extract_page(b"plain words", "text/plain", "https://example.org/a.txt").text == "plain words"


def test_extract_page_rejects_unreadable_pdf() -> None:
    with pytest.raises(ValueError, match="unreadable PDF"):
        extract_page(b"%PDF-1.7 broken", "application/pdf", "https://example.org/a.pdf")


@pytest.mark.skipif(not SMOKE_TASK.is_dir(), reason="run benchmarks/wandr/prepare.py first")
def test_upstream_verifier_scores_an_empty_submission_zero_offline() -> None:
    """Runs the pinned verifier through run_verifier.py: an empty file is an invalid submission, scored 0, no calls."""
    server = make_server(
        tasks_dir=str(SMOKE_TASK.parent),
        wandr_core_dir=str(SMOKE_TASK.parents[1] / "wandr_core"),
        inference_api_key="unused",
        judge_base_url="http://127.0.0.1:9",
    )
    run = asyncio.run(server.run_upstream_verifier("smoke", {"results_smoke.jsonl": []}))
    assert run.error is None, run.error
    assert run.rewards["soft_f1_full"] == 0.0
    assert set(REWARD_METRICS) <= set(run.rewards)


GOOD_SMOKE_RECORDS = [
    RECORD,
    {
        "item": {"topic": "Garbage collection"},
        "url": "https://en.wikipedia.org/wiki/Tracing_garbage_collection",
        "excerpts": ["Tracing garbage collection is the most common type of garbage collection"],
        "answer": {"detail": "Tracing is the most common kind of garbage collection."},
    },
    {
        "item": {"topic": "Global interpreter lock"},
        "url": "https://peps.python.org/pep-0703/",
        "excerpts": [
            'CPython\'s global interpreter lock ("GIL") prevents multiple threads from executing Python code '
            "at the same time."
        ],
        "answer": {"detail": "PEP 703 proposes making the GIL optional in CPython."},
    },
]
FABRICATED_RECORD = {
    "item": {"topic": "Type inference"},
    "url": "https://en.wikipedia.org/wiki/Type_inference",
    "excerpts": ["Type inference was invented by Grace Hopper in 1931 for the COBOL compiler."],
    "answer": {"detail": "Invented by Grace Hopper in 1931."},
}


@pytest.mark.skipif(
    not SMOKE_TASK.is_dir() or not os.environ.get("WANDR_LIVE_TEST_INFERENCE_API_KEY"),
    reason="needs prepared data and WANDR_LIVE_TEST_INFERENCE_API_KEY (paid judge calls and live page fetches)",
)
def test_upstream_verifier_credits_supported_records_and_rejects_a_fabricated_one() -> None:
    """Live: pages are fetched and judged, so supported records score and a fabricated excerpt does not."""
    server = make_server(
        tasks_dir=str(SMOKE_TASK.parent),
        wandr_core_dir=str(SMOKE_TASK.parents[1] / "wandr_core"),
        inference_api_key=os.environ["WANDR_LIVE_TEST_INFERENCE_API_KEY"],
        verifier_fetch_backend="direct",
    )
    good = asyncio.run(server.run_upstream_verifier("smoke", {"results_smoke.jsonl": GOOD_SMOKE_RECORDS}))
    assert good.error is None, good.error
    assert good.rewards["soft_precision_full"] > 0.5
    assert good.rewards["soft_f1_full"] > 0.0

    fabricated = asyncio.run(server.run_upstream_verifier("smoke", {"results_smoke.jsonl": [FABRICATED_RECORD]}))
    assert fabricated.error is None, fabricated.error
    assert fabricated.rewards["soft_precision_full"] == 0.0


class FakeStream:
    """An aiohttp body that arrives in several network chunks."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = list(chunks)

    async def read(self, n: int) -> bytes:
        if not self._chunks:
            return b""
        chunk = self._chunks.pop(0)
        if len(chunk) > n:
            self._chunks.insert(0, chunk[n:])
            chunk = chunk[:n]
        return chunk

    def at_eof(self) -> bool:
        return not self._chunks


class FakeResponse:
    def __init__(self, status: int, chunks: list[bytes] = (), headers: dict | None = None) -> None:
        self.status = status
        self.headers = headers or {"Content-Type": "text/plain"}
        self.content = FakeStream(list(chunks))

    def release(self) -> None:
        pass


# A public address literal resolves without DNS, so these tests stay offline.
PUBLIC_URL = "http://93.184.215.14/page"


def fake_get(responses: dict[str, FakeResponse], requested: list[str], headers_seen: list | None = None):
    async def get(url: str, timeout: float, headers: dict, server_hostname: str | None) -> FakeResponse:
        requested.append(url)
        if headers_seen is not None:
            headers_seen.append((headers, server_hostname))
        return responses[url]

    return get


def test_read_body_reads_every_chunk_until_eof_or_the_limit() -> None:
    body, truncated = asyncio.run(read_body(FakeResponse(200, [b"first chunk ", b"second ", b"third"]), 1000))
    assert (body, truncated) == (b"first chunk second third", False)

    body, truncated = asyncio.run(read_body(FakeResponse(200, [b"first chunk ", b"second ", b"third"]), 15))
    assert (body, truncated) == (b"first chunk sec", True)


@pytest.mark.parametrize(
    "url", ["http://127.0.0.1:11000/global_config_dict_yaml", "http://169.254.169.254/latest", "http://localhost/"]
)
def test_fetch_public_refuses_non_public_addresses_before_any_request(url: str) -> None:
    requested: list[str] = []
    with pytest.raises(UnsafeURLError):
        asyncio.run(fetch_public(fake_get({}, requested), url, timeout=5, max_bytes=100))
    assert requested == []


def test_fetch_public_checks_every_redirect_hop() -> None:
    requested: list[str] = []
    responses = {PUBLIC_URL: FakeResponse(302, headers={"Location": "http://10.0.0.5/secret"})}
    with pytest.raises(UnsafeURLError):
        asyncio.run(fetch_public(fake_get(responses, requested), PUBLIC_URL, timeout=5, max_bytes=100))
    assert requested == [PUBLIC_URL]


def test_verifier_fetch_client_answers_the_upstream_call_with_full_pages() -> None:
    """WANDR's fetch_pages calls client.content.fetch(urls=...) and reads .pages by url."""
    responses = {
        PUBLIC_URL: FakeResponse(200, [b"Garbage collection was invented ", b"by John McCarthy around 1959."]),
        "http://93.184.215.14/missing": FakeResponse(404),
    }
    client = DirectFetchClient(fake_get(responses, []), timeout=5, max_bytes=1000)
    result = asyncio.run(client.content.fetch(urls=[PUBLIC_URL, "http://93.184.215.14/missing", "http://10.0.0.1/"]))
    pages = {page.url: page for page in result.pages}
    assert pages[PUBLIC_URL].content == "Garbage collection was invented by John McCarthy around 1959."
    assert pages[PUBLIC_URL].error is None
    assert pages["http://93.184.215.14/missing"].error == "ValueError: HTTP 404"
    assert "non-public" in pages["http://10.0.0.1/"].error


def fake_post(status: int, data: object, sent: list):
    async def post(url: str, headers: dict, payload: dict, timeout: float) -> tuple[int, object]:
        sent.append((url, headers, payload))
        return status, data

    return post


def test_exa_fetch_client_maps_contents_to_the_fields_the_grader_reads() -> None:
    good, gone = "https://example.org/a", "https://example.org/gone"
    data = {
        "results": [
            {
                "id": good,
                "url": good + "/",
                "title": "A page",
                "author": "Ada",
                "publishedDate": "2026-01-02T00:00:00.000Z",
                "text": "Garbage collection was invented by John McCarthy.",
            }
        ],
        "statuses": [
            {"id": good, "status": "success"},
            {"id": gone, "status": "error", "error": {"tag": "CRAWL_NOT_FOUND", "httpStatusCode": 404}},
        ],
    }
    sent: list = []
    client = ExaFetchClient(
        fake_post(200, data, sent), api_key="k", timeout=5, base_url="https://exa.test/", max_age_hours=0
    )
    result = asyncio.run(client.content.fetch(urls=[good, gone, good, "http://10.0.0.1/"]))
    pages = [
        (page.url, page.title, page.author, page.published_date, page.content, page.error) for page in result.pages
    ]
    assert pages[0] == (good, "A page", "Ada", "2026-01-02T00:00:00.000Z", data["results"][0]["text"], None)
    assert pages[1][-1] == "Exa: CRAWL_NOT_FOUND 404"
    assert pages[2] == pages[0]
    assert pages[3][-1].startswith("UnsafeURLError")
    assert sent == [
        ("https://exa.test/contents", {"x-api-key": "k"}, {"urls": [good, gone], "text": True, "maxAgeHours": 0})
    ]


@pytest.mark.parametrize("status, fatal", [(401, True), (500, False)])
def test_exa_fetch_client_raises_on_http_errors_and_records_auth_failures(
    monkeypatch: pytest.MonkeyPatch, status: int, fatal: bool
) -> None:
    monkeypatch.setattr(ExaFetchClient, "fatal_errors", [])
    client = ExaFetchClient(fake_post(status, None, []), api_key="k", timeout=5)
    with pytest.raises(ValueError, match=f"HTTP {status}"):
        asyncio.run(client.content.fetch(urls=["https://example.org/a"]))
    assert ExaFetchClient.fatal_errors == ([f"Exa /contents answered HTTP {status}"] if fatal else [])


@pytest.mark.parametrize(
    "backend, data, expected_url, expected_payload",
    [
        (
            "perplexity",
            {"results": [{"title": "T", "url": "https://a.org", "date": "2026", "snippet": "S"}]},
            "https://inference-api.nvidia.com/v1/search/perplexity-search",
            {"query": "gc", "max_results": 3, "max_tokens_per_page": 1024},
        ),
        (
            "exa",
            {"results": [{"title": "T", "url": "https://a.org", "publishedDate": "2026", "highlights": ["S"]}]},
            "https://api.exa.ai/search",
            {"query": "gc", "numResults": 3, "type": "auto", "contents": {"highlights": {"maxCharacters": 4096}}},
        ),
    ],
)
def test_web_search_calls_the_configured_backend(
    monkeypatch: pytest.MonkeyPatch, backend: str, data: dict, expected_url: str, expected_payload: dict
) -> None:
    sent: list = []
    monkeypatch.setattr(app_module, "gym_post_json", fake_post(200, data, sent))
    server = make_server(search_backend=backend, inference_api_key="inference-key")
    result = asyncio.run(server.web_search(WebSearchRequest(query="gc", max_results=3)))
    assert result.results == [{"title": "T", "url": "https://a.org", "date": "2026", "snippet": "S"}]
    url, headers, payload = sent[0]
    assert (url, payload) == (expected_url, expected_payload)
    assert headers == (
        {"x-api-key": "test-exa-key"} if backend == "exa" else {"Authorization": "Bearer inference-key"}
    )


def test_web_search_reports_only_the_status_of_a_failed_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "gym_post_json", fake_post(401, None, []))
    result = asyncio.run(make_server().web_search(WebSearchRequest(query="gc")))
    assert (result.results, result.error) == ([], "search failed with HTTP 401")


@pytest.mark.parametrize(
    "config, message",
    [
        ({"exa_api_key": None}, "exa_api_key"),
        ({"exa_api_key": None, "search_backend": "perplexity"}, "exa_api_key"),
        ({"verifier_fetch_backend": "perplexity"}, "perplexity_api_key"),
    ],
)
def test_server_refuses_to_start_without_the_keys_its_backends_need(config: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        make_server(**config)


def test_all_four_backend_combinations_start_from_config_only() -> None:
    for search in ("exa", "perplexity"):
        for fetch in ("exa", "perplexity"):
            server = make_server(search_backend=search, verifier_fetch_backend=fetch, perplexity_api_key="p")
            assert (server.config.search_backend, server.config.verifier_fetch_backend) == (search, fetch)


def test_fetch_page_returns_the_page_in_slices_and_reports_errors() -> None:
    server = make_server(fetch_page_chars=10)
    text = b"0123456789abcdefghij"
    responses = {PUBLIC_URL: FakeResponse(200, [text[:7], text[7:]])}
    object.__setattr__(server, "_get", fake_get(responses, []))
    first = asyncio.run(server.fetch_page(FetchPageRequest(url=PUBLIC_URL)))
    assert (first.content, first.end, first.total_chars, first.error) == ("0123456789", 10, 20, None)
    second = asyncio.run(server.fetch_page(FetchPageRequest(url=PUBLIC_URL, start=first.end)))
    assert second.content == "abcdefghij"

    refused = asyncio.run(server.fetch_page(FetchPageRequest(url="http://127.0.0.1:11000/global_config_dict_yaml")))
    assert refused.content == ""
    assert "non-public" in refused.error


def test_extract_page_turns_any_pdf_failure_into_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_reader(_stream: object) -> None:
        raise TypeError("'NullObject' object is not subscriptable")

    monkeypatch.setattr(web, "PdfReader", broken_reader)
    with pytest.raises(ValueError, match="unreadable PDF: TypeError"):
        extract_page(b"%PDF-1.7 ...", "application/pdf", "https://example.org/a.pdf")


def test_fetch_public_refuses_a_hostname_that_resolves_to_a_private_address(monkeypatch: pytest.MonkeyPatch) -> None:
    async def private_dns(hostname: str, port: int, *, timeout: float) -> list[str]:
        return ["93.184.215.14", "10.0.0.5"]

    monkeypatch.setattr(web, "resolve", private_dns)
    requested: list[str] = []
    with pytest.raises(UnsafeURLError, match="non-public"):
        asyncio.run(fetch_public(fake_get({}, requested), "https://rebind.example/x", timeout=5, max_bytes=100))
    assert requested == []


def test_fetch_public_connects_to_the_checked_address(monkeypatch: pytest.MonkeyPatch) -> None:
    async def public_dns(hostname: str, port: int, *, timeout: float) -> list[str]:
        return ["93.184.215.14"]

    monkeypatch.setattr(web, "resolve", public_dns)
    requested: list[str] = []
    seen: list = []
    responses = {"https://93.184.215.14:8443/x?q=1": FakeResponse(200, [b"ok"])}
    page = asyncio.run(
        fetch_public(fake_get(responses, requested, seen), "https://Site.example:8443/x?q=1", timeout=5, max_bytes=100)
    )
    assert (page.body, page.final_url) == (b"ok", "https://Site.example:8443/x?q=1")
    assert seen == [({"Host": "site.example:8443"}, "site.example")]


@pytest.mark.parametrize(
    ("url", "host_header", "server_hostname"),
    [
        ("https://example.com./x", "example.com", "example.com"),
        ("https://[2606:4700::1111]:8443/x", "[2606:4700::1111]:8443", "2606:4700::1111"),
        ("http://a\u00df.example/x", "xn--a-qfa.example", None),
    ],
)
def test_pinned_request_sends_the_host_aiohttp_would(url: str, host_header: str, server_hostname: str | None) -> None:
    _, headers, name = web._pinned_request(url, "93.184.215.14")
    assert (headers, name) == ({"Host": host_header}, server_hostname)


def test_fetch_public_falls_back_to_the_next_checked_address(monkeypatch: pytest.MonkeyPatch) -> None:
    async def dual_stack(hostname: str, port: int, *, timeout: float) -> list[str]:
        return ["2600:1f18::1", "52.1.2.3"]

    monkeypatch.setattr(web, "resolve", dual_stack)
    requested: list[str] = []

    async def get(url: str, timeout: float, headers: dict, server_hostname: str | None) -> FakeResponse:
        requested.append(url)
        if "2600" in url:
            raise OSError("Network is unreachable")
        return FakeResponse(200, [b"ok"])

    page = asyncio.run(fetch_public(get, "http://dual.example/x", timeout=5, max_bytes=100))
    assert page.body == b"ok"
    assert requested == ["http://[2600:1f18::1]/x", "http://52.1.2.3/x"]


def test_gym_get_does_not_follow_a_redirect_to_a_private_address(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real transport, against a local server that redirects to a private address."""
    hits: list[str] = []

    async def redirect(request: web_server.Request) -> web_server.Response:
        hits.append(request.headers["Host"])
        return web_server.Response(status=302, headers={"Location": "http://internal.example/secret"})

    async def run() -> None:
        app = web_server.Application()
        app.router.add_get("/start", redirect)
        server = TestServer(app, host="127.0.0.1")
        await server.start_server()
        try:

            async def local_dns(hostname: str, port: int, *, timeout: float) -> list[str]:
                return {"public.example": ["127.0.0.1"], "internal.example": ["10.0.0.5"]}[hostname]

            monkeypatch.setattr(web, "resolve", local_dns)
            # The local test server is on loopback; only the first hop may use it.
            monkeypatch.setattr(web, "_is_public", lambda address: str(address) == "127.0.0.1")
            with pytest.raises(UnsafeURLError, match="non-public"):
                await fetch_public(gym_get, f"http://public.example:{server.port}/start", timeout=5, max_bytes=100)
        finally:
            await server.close()

    asyncio.run(run())
    assert len(hits) == 1
    assert hits[0].startswith("public.example:")


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("not json", "unreadable"),
        ("[]", "not an object"),
        (json.dumps({name: 0.5 for name in REWARD_METRICS if name != "soft_f1_full"}), "soft_f1_full"),
        (json.dumps({name: "0.5" for name in REWARD_METRICS}), "lacks valid metrics"),
        (json.dumps({name: 2.0 for name in REWARD_METRICS}), "lacks valid metrics"),
    ],
)
def test_read_rewards_treats_a_bad_reward_file_as_inconclusive(tmp_path: Path, content: str, message: str) -> None:
    path = tmp_path / "reward.json"
    path.write_text(content)
    run = read_rewards(path)
    assert run.rewards is None
    assert message in run.error


def test_read_rewards_returns_every_metric(tmp_path: Path) -> None:
    path = tmp_path / "reward.json"
    path.write_text(json.dumps({"reward": 0.5, **{name: 0.25 for name in REWARD_METRICS}}))
    assert read_rewards(path).rewards == {name: 0.25 for name in REWARD_METRICS}


FAKE_VERIFY_PY = """
import asyncio, collections, json, os
from src import components
from src.runtime import clients, llm

Request = collections.namedtuple("Request", "model")


def main():
    model = asyncio.run(llm.llm_completion(None, Request("gpt-5.4")))
    logs = os.environ["LOGS_DIR"]
    os.makedirs(logs, exist_ok=True)
    if "FAKE_JUDGED" in os.environ:
        # As upstream: a failed judge call drops its record, and the run still scores what was judged.
        try:
            asyncio.run(llm.llm_completion(None, Request("judge")))
        except RuntimeError:
            pass
        metadata = {"submitted_record_count": 10, "judged_record_count": int(os.environ["FAKE_JUDGED"])}
        json.dump({"metadata": metadata}, open(os.path.join(logs, "wandr-details.json"), "w"))
    timeouts = [components._TRIAGE_TIMEOUT, components._CANON_TIMEOUT, components._DEDUP_TIMEOUT, components._JUDGE_TIMEOUT]
    if os.environ.get("FAKE_EXPECT_TIMEOUT") and timeouts != [float(os.environ["FAKE_EXPECT_TIMEOUT"])] * 4:
        json.dump({"error": f"timeouts {timeouts}"}, open(os.path.join(logs, "error.json"), "w"))
        return 1
    if os.environ.get("FAKE_VERIFY_FAIL"):
        json.dump({"error": "judge unavailable"}, open(os.path.join(logs, "error.json"), "w"))
        return 1
    fetch = clients.FetchClient
    ok = (
        model == "openai/openai/gpt-5.4"
        and fetch.__name__ == "VerifierFetchClient"
        and fetch.__mro__[1].__name__ == os.environ["FAKE_FETCH_BASE"]
    )
    if os.environ.get("FAKE_EXA_FATAL"):
        fetch.fatal_errors.append("Exa /contents answered HTTP 401")
    metrics = json.loads(os.environ["FAKE_METRICS"])
    json.dump({name: (1.0 if ok else 0.0) for name in metrics}, open(os.path.join(logs, "reward.json"), "w"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""


def fake_verifier_tree(root: Path) -> tuple[Path, Path]:
    """A stand-in wandr_core (llm, clients and components modules only) and a task whose verify.py reports what it saw."""
    core = root / "wandr_core"
    (core / "src" / "runtime").mkdir(parents=True)
    (core / "src" / "__init__.py").write_text("")
    (core / "src" / "runtime" / "__init__.py").write_text("")
    (core / "src" / "runtime" / "llm.py").write_text(
        "import os\n\n\nasync def llm_completion(client, request):\n"
        "    if os.environ.get('FAKE_LLM_FAIL') and 'judge' in request.model:\n"
        "        raise RuntimeError('Budget has been exceeded')\n"
        "    return request.model\n"
    )
    (core / "src" / "components.py").write_text(
        "_TRIAGE_TIMEOUT = 15\n_CANON_TIMEOUT = 60\n_DEDUP_TIMEOUT = 30\n_JUDGE_TIMEOUT = 60\n"
    )
    (core / "src" / "runtime" / "clients.py").write_text("class FetchClient:\n    pass\n")
    task = root / "tasks" / "smoke"
    (task / "tests" / "wandr_task").mkdir(parents=True)
    (task / "tests" / "verify.py").write_text(FAKE_VERIFY_PY)
    (task / "instruction.md").write_text("Find topics.")
    return core, task


@pytest.mark.parametrize(
    "backend, failure",
    [("direct", None), ("exa", None), ("exa", "FAKE_VERIFY_FAIL"), ("exa", "FAKE_EXA_FATAL")],
)
def test_run_subprocess_swaps_the_clients_and_reads_the_result_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, failure: str | None
) -> None:
    core, task = fake_verifier_tree(tmp_path)
    monkeypatch.setenv("FAKE_METRICS", json.dumps(list(REWARD_METRICS)))
    monkeypatch.setenv("FAKE_FETCH_BASE", {"direct": "DirectFetchClient", "exa": "ExaFetchClient"}[backend])
    if failure:
        monkeypatch.setenv(failure, "1")
    server = make_server(
        tasks_dir=str(task.parent),
        wandr_core_dir=str(core),
        inference_api_key="unused",
        verifier_fetch_backend=backend,
    )
    run = asyncio.run(server.run_upstream_verifier("smoke", {"results_smoke.jsonl": [RECORD]}))
    if failure:
        assert run.rewards is None
        assert {"FAKE_VERIFY_FAIL": "judge unavailable", "FAKE_EXA_FATAL": "HTTP 401"}[failure] in run.error
    else:
        assert run.error is None, run.error
        assert run.rewards == {name: 1.0 for name in REWARD_METRICS}


@pytest.mark.parametrize("llm_timeout_s, expected", [(300.0, 300.0), (None, None)])
def test_run_subprocess_sets_the_grader_llm_timeouts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, llm_timeout_s: float | None, expected: float | None
) -> None:
    """The four upstream grader timeouts (triage, canon, dedup, judge) take llm_timeout_s; None keeps upstream's."""
    core, task = fake_verifier_tree(tmp_path)
    monkeypatch.setenv("FAKE_METRICS", json.dumps(list(REWARD_METRICS)))
    monkeypatch.setenv("FAKE_FETCH_BASE", "DirectFetchClient")
    if expected is not None:
        monkeypatch.setenv("FAKE_EXPECT_TIMEOUT", str(expected))
    server = make_server(
        tasks_dir=str(task.parent),
        wandr_core_dir=str(core),
        inference_api_key="unused",
        verifier_fetch_backend="direct",
        llm_timeout_s=llm_timeout_s,
    )
    run = asyncio.run(server.run_upstream_verifier("smoke", {"results_smoke.jsonl": [RECORD]}))
    assert run.error is None, run.error
    assert run.rewards == {name: 1.0 for name in REWARD_METRICS}


@pytest.mark.parametrize(
    "llm_fail, judged, inconclusive",
    [
        (True, 0, True),  # the judge failed for every record: not a measured 0
        (True, 8, True),  # under upstream's 90% coverage, with failed calls
        (True, 9, False),  # failed calls, but upstream's 90% coverage holds
        (False, 0, False),  # no failed call: records left unjudged for task reasons, scored as upstream does
    ],
)
def test_run_subprocess_marks_judging_lost_to_llm_failures_inconclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, llm_fail: bool, judged: int, inconclusive: bool
) -> None:
    core, task = fake_verifier_tree(tmp_path)
    monkeypatch.setenv("FAKE_METRICS", json.dumps(list(REWARD_METRICS)))
    monkeypatch.setenv("FAKE_FETCH_BASE", "DirectFetchClient")
    monkeypatch.setenv("FAKE_JUDGED", str(judged))
    if llm_fail:
        monkeypatch.setenv("FAKE_LLM_FAIL", "1")
    server = make_server(
        tasks_dir=str(task.parent),
        wandr_core_dir=str(core),
        inference_api_key="unused",
        verifier_fetch_backend="direct",
    )
    run = asyncio.run(server.run_upstream_verifier("smoke", {"results_smoke.jsonl": [RECORD]}))
    if inconclusive:
        assert run.rewards is None
        assert f"judged {judged} of 10 records" in run.error
    else:
        assert run.error is None, run.error
        assert run.rewards == {name: 1.0 for name in REWARD_METRICS}


def crash_extract(body: bytes, content_type: str, url: str) -> None:
    """Stands in for a native crash inside trafilatura or lxml."""
    os.abort()


def test_a_crashing_extraction_costs_one_page_not_the_process() -> None:
    async def run() -> tuple[str, str]:
        with pytest.raises(ValueError, match="extraction crashed"):
            await web.extract_page_isolated(b"<p>x</p>", "text/html", "u", extract=crash_extract)
        page = await web.extract_page_isolated(b"plain words", "text/plain", "u")
        return page.text, "alive"

    assert asyncio.run(run()) == ("plain words", "alive")
