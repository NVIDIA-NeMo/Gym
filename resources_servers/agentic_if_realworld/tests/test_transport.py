# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import copy
import json
import multiprocessing
from unittest.mock import patch

import jsonschema
import pytest

from resources_servers.agentic_if_realworld.benchmark.core import (
    HISTORY_MARKER,
    ROOT,
    Dataset,
    bounded_schema,
    digest,
    judge,
    provider_schema,
    read,
    write,
)
from resources_servers.agentic_if_realworld.benchmark.scoring import score
from resources_servers.agentic_if_realworld.benchmark.synthetic import create, review
from resources_servers.agentic_if_realworld.benchmark.transport import (
    RetryableTransportError,
    delivered,
    verified_response,
    wire_call,
)


server_utils = pytest.importorskip("nemo_gym.server_utils")
SYNTHETIC_PROMPT = (ROOT / "synthetic-judge-prompt.md").read_text()


class Response:
    def __init__(self, status, value):
        self.status = status
        self.headers = {"content-type": "application/json"}
        self.content = self
        self.value = value

    async def iter_chunked(self, size):
        yield json.dumps(self.value).encode()

    def release(self):
        pass


@pytest.fixture
def judge_payload(tmp_path):
    create(tmp_path / "data")
    dataset = Dataset(tmp_path / "data", synthetic=True)
    response = {
        "status": "complete",
        "finish_reason": "stop",
        "message": {"role": "assistant", "content": "OK."},
    }
    return dataset.payload(
        "synthetic",
        response,
        {
            "id": "synthetic",
            "input_sha256": judge.sha(dataset.request("synthetic")),
            "response_sha256": judge.sha(response),
            "benchmark_manifest_sha256": dataset.manifest_sha256,
        },
    )


async def no_sleep(seconds):
    pass


@pytest.mark.parametrize("status", [429, 503])
@pytest.mark.parametrize("cancel_at", [2, 3])
def test_interrupted_judge_http_retry_blocks_resume(judge_payload, tmp_path, monkeypatch, status, cancel_at):
    calls = []

    async def request(*args, **kwargs):
        calls.append(1)
        if len(calls) == cancel_at:
            raise asyncio.CancelledError
        return Response(status, {"error": "synthetic temporary outage"})

    monkeypatch.setattr(server_utils, "request", request)
    monkeypatch.setattr("resources_servers.agentic_if_realworld.benchmark.transport.asyncio.sleep", no_sleep)
    monkeypatch.setenv("SYNTHETIC_JUDGE_KEY", "synthetic")
    kwargs = {
        "directory": tmp_path / "receipts",
        "endpoint": "http://synthetic.invalid",
        "policy_prompt": SYNTHETIC_PROMPT,
        "key_env": "SYNTHETIC_JUDGE_KEY",
    }
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(score(judge_payload, **kwargs))
    path = next((kwargs["directory"] / "judges").glob("*/attempt0.json"))
    receipt = read(path)
    assert receipt["status"] == "started"
    assert receipt["http_attempt"] == cancel_at
    assert len(receipt["raw_attempts"]) == cancel_at - 1
    with pytest.raises(ValueError, match="Ambiguous prior dispatch"):
        asyncio.run(score(judge_payload, **kwargs))
    assert len(calls) == cancel_at


@pytest.mark.parametrize("status", [429, 503])
def test_judge_http_retries_remain_bounded_and_completed_review_is_reused(
    judge_payload, tmp_path, monkeypatch, status
):
    calls = []

    async def request(*args, **kwargs):
        calls.append(1)
        if len(calls) < 3:
            return Response(status, {"error": "synthetic temporary outage"})
        return Response(
            200,
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": json.dumps(review(judge_payload))},
                    }
                ]
            },
        )

    monkeypatch.setattr(server_utils, "request", request)
    monkeypatch.setattr("resources_servers.agentic_if_realworld.benchmark.transport.asyncio.sleep", no_sleep)
    monkeypatch.setenv("SYNTHETIC_JUDGE_KEY", "synthetic")
    kwargs = {
        "directory": tmp_path / "receipts",
        "endpoint": "http://synthetic.invalid",
        "policy_prompt": SYNTHETIC_PROMPT,
        "key_env": "SYNTHETIC_JUDGE_KEY",
    }
    result = asyncio.run(score(judge_payload, **kwargs))
    assert result["item_verdict"] == "pass"
    assert asyncio.run(score(judge_payload, **kwargs)) == result
    assert len(calls) == 3
    receipt = read(next((kwargs["directory"] / "judges").glob("*/attempt0.json")))
    assert receipt["status"] == "generated"
    assert len(receipt["raw_attempts"]) == 3
    assert [attempt["http_status"] for attempt in receipt["raw_attempts"]] == [status, status, 200]


def test_concurrent_identical_reviews_share_one_judge_call(judge_payload, tmp_path, monkeypatch):
    calls = []

    async def request(*args, **kwargs):
        calls.append(1)
        await asyncio.sleep(0.05)
        return Response(
            200,
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": json.dumps(review(judge_payload))},
                    }
                ]
            },
        )

    monkeypatch.setattr(server_utils, "request", request)
    monkeypatch.setenv("SYNTHETIC_JUDGE_KEY", "synthetic")

    async def run():
        return await asyncio.gather(
            *(
                score(
                    judge_payload,
                    directory=tmp_path / "receipts",
                    endpoint="http://synthetic.invalid",
                    policy_prompt=SYNTHETIC_PROMPT,
                    key_env="SYNTHETIC_JUDGE_KEY",
                )
                for _ in range(4)
            )
        )

    results = asyncio.run(run())
    assert len(calls) == 1
    assert all(result == results[0] for result in results)
    assert results[0]["item_verdict"] == "pass"


def test_changing_judge_model_preserves_and_separates_cached_grades(judge_payload, tmp_path, monkeypatch):
    calls = []

    async def request(*args, **kwargs):
        calls.append(kwargs["json"]["model"])
        return Response(
            200,
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": json.dumps(review(judge_payload))},
                    }
                ]
            },
        )

    monkeypatch.setattr(server_utils, "request", request)
    monkeypatch.setenv("SYNTHETIC_JUDGE_KEY", "synthetic")
    kwargs = {
        "directory": tmp_path / "receipts",
        "endpoint": "http://synthetic.invalid",
        "policy_prompt": SYNTHETIC_PROMPT,
        "key_env": "SYNTHETIC_JUDGE_KEY",
    }
    models = ["first-judge", "second-judge"]
    astra_result = asyncio.run(score(judge_payload, model=models[0], **kwargs))
    astra_path = next((kwargs["directory"] / "judges").glob("*/result.json"))
    original_astra_bytes = astra_path.read_bytes()
    sol_result = asyncio.run(score(judge_payload, model=models[1], **kwargs))

    assert calls == models
    assert astra_result["judge_model"] == models[0]
    assert sol_result["judge_model"] == models[1]
    assert astra_result["judge_request_sha256"] != sol_result["judge_request_sha256"]
    assert len(list((kwargs["directory"] / "judges").glob("*/result.json"))) == 2
    for model, result in zip(models, [astra_result, sol_result], strict=True):
        assert asyncio.run(score(judge_payload, model=model, **kwargs)) == result
    assert calls == models
    assert astra_path.read_bytes() == original_astra_bytes


@pytest.mark.parametrize(
    "damage", ["raw", "receipt", "reward", "mask", "payload", "schema", "provider_schema", "missing", "policy_prompt"]
)
def test_cached_judge_grade_revalidates_authoritative_receipts(judge_payload, tmp_path, monkeypatch, damage):
    calls = []

    async def request(*args, **kwargs):
        calls.append(1)
        return Response(
            200,
            {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(review(judge_payload))}}]},
        )

    monkeypatch.setattr(server_utils, "request", request)
    monkeypatch.setenv("SYNTHETIC_JUDGE_KEY", "synthetic")
    kwargs = {
        "directory": tmp_path / "receipts",
        "endpoint": "http://synthetic.invalid",
        "policy_prompt": SYNTHETIC_PROMPT,
        "key_env": "SYNTHETIC_JUDGE_KEY",
    }
    result = asyncio.run(score(judge_payload, **kwargs))
    assert asyncio.run(score(judge_payload, **kwargs, cached_only=True)) == result
    work = next((kwargs["directory"] / "judges").iterdir())
    if damage == "policy_prompt":
        kwargs["policy_prompt"] += "\nChanged synthetic policy."
    elif damage in {"reward", "mask"}:
        cached = read(work / "result.json")
        cached["result"]["reward" if damage == "reward" else "mask_sample"] = 99 if damage == "reward" else True
        write(work / "result.json", cached)
    elif damage == "missing":
        (work / "result.json").unlink()
    else:
        names = {
            "raw": "*.http",
            "receipt": "attempt0.json",
            "payload": "payload.json",
            "schema": "transport_schema.json",
            "provider_schema": "provider_schema.json",
        }
        path = next(work.glob(names[damage]))
        path.write_text("changed")
    with pytest.raises(ValueError):
        asyncio.run(score(judge_payload, **kwargs, cached_only=True))
    assert len(calls) == 1


@pytest.mark.parametrize("quote", ['Return exactly "OK."', "Return exactly OK.\nDo not explain."])
def test_provider_grammar_compatibility_preserves_strict_local_support_quotes(
    judge_payload, tmp_path, monkeypatch, quote
):
    payload = copy.deepcopy(judge_payload)
    wrapper = payload["input_request"]["messages"][0]
    prefix = wrapper["content"].split(HISTORY_MARKER, 1)[0]
    wrapper["content"] = prefix + HISTORY_MARKER + json.dumps([{"role": "user", "content": quote}])
    payload["input_sha256"] = judge.sha(payload["input_request"])
    payload["inference_binding"]["input_sha256"] = payload["input_sha256"]
    payload["constraints"][0]["support_quotes"] = [quote]
    payload["constraint_contract_sha256"] = judge.contract_sha(payload["constraints"])
    judge.validate_payload(payload)
    local_schema = bounded_schema(provider_schema(read(ROOT / "policy/judge_output.schema.json")), payload)
    original_schema = copy.deepcopy(local_schema)
    calls = []

    async def request(*args, **kwargs):
        calls.append(kwargs["json"])
        wire_schema = kwargs["json"]["response_format"]["json_schema"]["schema"]
        support = wire_schema["properties"]["constraints"]["items"]["properties"]["support_quote"]
        if any('"' in value or "\n" in value for value in support.get("enum", [])):
            return Response(400, {"error": "Synthetic provider rejects quote/newline enum literals"})
        checked = review(payload)
        if len(calls) == 1:
            checked["constraints"][0]["support_quote"] = "Invented instruction absent from the frozen input"
            jsonschema.validate(checked, wire_schema)
            with pytest.raises(jsonschema.ValidationError):
                jsonschema.validate(checked, local_schema)
        return Response(200, {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(checked)}}]})

    monkeypatch.setattr(server_utils, "request", request)
    monkeypatch.setenv("SYNTHETIC", "synthetic")
    kwargs = {
        "directory": tmp_path / "receipts",
        "endpoint": "http://synthetic.invalid",
        "policy_prompt": SYNTHETIC_PROMPT,
        "key_env": "SYNTHETIC",
    }
    result = asyncio.run(score(payload, **kwargs))
    assert result["item_verdict"] == "pass" and len(calls) == 2
    work = next((kwargs["directory"] / "judges").iterdir())
    assert read(work / "transport_schema.json") == local_schema == original_schema
    expected_wire = copy.deepcopy(local_schema)
    expected_wire["properties"]["constraints"]["items"]["properties"]["support_quote"].pop("enum")
    assert read(work / "provider_schema.json") == expected_wire
    assert calls[0]["response_format"]["json_schema"]["schema"] == expected_wire
    assert read(work / "attempt0.json")["request_sha256"] == judge.sha(calls[0])
    assert asyncio.run(score(payload, **kwargs, cached_only=True)) == result and len(calls) == 2


def test_judge_receipts_reject_git_checkout_before_writing(judge_payload, tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    (checkout / ".git").mkdir(parents=True)
    monkeypatch.setattr(server_utils, "request", lambda *args, **kwargs: pytest.fail("No network call allowed"))
    with pytest.raises(ValueError, match="outside Git"):
        asyncio.run(
            score(
                judge_payload,
                directory=checkout / "receipts",
                endpoint="http://synthetic.invalid",
                policy_prompt=SYNTHETIC_PROMPT,
            )
        )
    assert not (checkout / "receipts").exists()


def test_missing_key_fails_before_any_receipt(tmp_path, monkeypatch):
    monkeypatch.delenv("SYNTHETIC_MISSING_KEY", raising=False)
    monkeypatch.setattr(server_utils, "request", lambda *args, **kwargs: pytest.fail("No network call allowed"))
    path = tmp_path / "receipt.json"
    with pytest.raises(ValueError, match="SYNTHETIC_MISSING_KEY is not set"):
        asyncio.run(
            wire_call(
                {"model": "synthetic"}, endpoint="http://synthetic.invalid", key_env="SYNTHETIC_MISSING_KEY", path=path
            )
        )
    # No receipt and no raw attempt: a later call with the key starts from a clean attempt.
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("failure", ["connect", "connect_timeout", "disconnect", "read_timeout"])
def test_transport_only_marks_proven_connect_failures_as_not_dispatched(tmp_path, monkeypatch, failure):
    from aiohttp import ClientConnectorError, ConnectionTimeoutError, ServerDisconnectedError, SocketTimeoutError

    errors = {
        "connect": ClientConnectorError(None, ConnectionRefusedError()),
        "connect_timeout": ConnectionTimeoutError(),
        "disconnect": ServerDisconnectedError(),
        "read_timeout": SocketTimeoutError(),
    }
    calls = []

    async def request(*args, **kwargs):
        calls.append(1)
        raise errors[failure]

    monkeypatch.setattr(server_utils, "request", request)
    monkeypatch.setenv("SYNTHETIC", "synthetic")
    path = tmp_path / "receipt.json"
    args = {"endpoint": "http://synthetic.invalid", "key_env": "SYNTHETIC", "path": path}
    receipt = asyncio.run(wire_call({"model": "synthetic"}, **args))
    assert receipt["status"] == "delivery_error"
    assert receipt["dispatch_status"] == ("not_dispatched" if failure.startswith("connect") else "unknown")
    assert receipt["raw_attempts"][0]["http_status"] is None
    saved = path.read_bytes()
    assert asyncio.run(wire_call({"model": "synthetic"}, **args)) == receipt
    assert len(calls) == 1 and path.read_bytes() == saved


@pytest.mark.parametrize("done", [False, True])
def test_read_error_preserves_only_a_saved_complete_sse_stream(tmp_path, monkeypatch, done):
    from aiohttp import ClientPayloadError

    event = {"choices": [{"delta": {"content": "OK."}, "finish_reason": "stop"}]}
    raw = ("data: " + json.dumps(event) + "\n\n" + ("data: [DONE]\n\n" if done else "")).encode()

    class InterruptedStream(Response):
        async def iter_chunked(self, size):
            yield raw
            raise ClientPayloadError("stream ended after saved bytes")

    response = InterruptedStream(200, None)
    response.headers = {"content-type": "text/event-stream"}

    async def request(*args, **kwargs):
        return response

    monkeypatch.setattr(server_utils, "request", request)
    monkeypatch.setenv("SYNTHETIC", "synthetic")
    receipt = asyncio.run(
        wire_call({}, endpoint="http://synthetic.invalid", key_env="SYNTHETIC", path=tmp_path / "receipt.json")
    )
    assert receipt["status"] == ("generated" if done else "delivery_error")
    assert len(receipt["raw_attempts"]) == 1
    if done:
        assert delivered(verified_response(receipt))["message"]["content"] == "OK."
    else:
        assert receipt["dispatch_status"] == "unknown"


@pytest.mark.parametrize("status", [401, 404, 429, 503])
@pytest.mark.parametrize("failed_invocations", [1, 2])
def test_judge_transport_failure_recovery_preserves_receipts_and_review_budget(
    judge_payload, tmp_path, monkeypatch, status, failed_invocations
):
    calls = []
    healthy = False

    async def request(*args, **kwargs):
        calls.append(kwargs["headers"]["Authorization"])
        if not healthy:
            return Response(status, {"error": "synthetic service unavailable"})
        content = json.dumps(review(judge_payload))
        return Response(200, {"choices": [{"finish_reason": "stop", "message": {"content": content}}]})

    monkeypatch.setattr(server_utils, "request", request)
    monkeypatch.setattr("resources_servers.agentic_if_realworld.benchmark.transport.asyncio.sleep", no_sleep)
    monkeypatch.setenv("SYNTHETIC", "bad-key")
    kwargs = {
        "directory": tmp_path / "receipts",
        "endpoint": "http://synthetic.invalid",
        "policy_prompt": SYNTHETIC_PROMPT,
        "key_env": "SYNTHETIC",
    }
    for invocation in range(failed_invocations):
        with pytest.raises(RetryableTransportError, match="explicitly resume"):
            asyncio.run(score(judge_payload, **kwargs))
        expected_calls = (invocation + 1) * (3 if status in {429, 503} else 1)
        assert len(calls) == expected_calls
    work = next((kwargs["directory"] / "judges").iterdir())
    original = {path: path.read_bytes() for path in work.glob("attempt*.*")}
    assert len(list(work.glob("attempt*.json"))) == failed_invocations
    assert asyncio.run(score(judge_payload, **kwargs, validate_only=True))["scoring_status"] == "not_scored"
    assert len(calls) == expected_calls
    # A corrected credential does not change the request fingerprint or regenerate the policy answer.
    healthy = True
    monkeypatch.setenv("SYNTHETIC", "corrected-key")
    result = asyncio.run(score(judge_payload, **kwargs))
    assert result["item_verdict"] == "pass" and len(calls) == expected_calls + 1
    assert read(work / "result.json")["receipt_name"] == f"attempt{failed_invocations}.json"
    assert all(path.read_bytes() == content for path, content in original.items())
    assert asyncio.run(score(judge_payload, **kwargs, cached_only=True)) == result
    assert len(calls) == expected_calls + 1


def test_valid_cached_grade_after_legacy_ambiguous_attempt_remains_readable(judge_payload, tmp_path, monkeypatch):
    async def request(*args, **kwargs):
        return Response(
            200, {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(review(judge_payload))}}]}
        )

    monkeypatch.setattr(server_utils, "request", request)
    monkeypatch.setenv("SYNTHETIC", "synthetic")
    kwargs = {
        "directory": tmp_path / "receipts",
        "endpoint": "http://synthetic.invalid",
        "policy_prompt": SYNTHETIC_PROMPT,
        "key_env": "SYNTHETIC",
    }
    result = asyncio.run(score(judge_payload, **kwargs))
    work = next((kwargs["directory"] / "judges").iterdir())
    successful = read(work / "attempt0.json")
    (work / "attempt0.json").rename(work / "attempt1.json")
    raw = work / "legacy-failure.http"
    raw.write_bytes(b"")
    write(
        work / "attempt0.json",
        {
            "status": "delivery_error",
            "reason": "ConnectionError",
            "endpoint": successful["endpoint"],
            "request_body": successful["request_body"],
            "request_sha256": successful["request_sha256"],
            "raw_attempts": [{"path": str(raw), "sha256": digest(raw)}],
        },
    )
    cached = read(work / "result.json")
    cached.update(receipt_name="attempt1.json", receipt_sha256=digest(work / "attempt1.json"))
    write(work / "result.json", cached)
    saved = {path: path.read_bytes() for path in work.iterdir() if path.is_file()}
    monkeypatch.setattr(server_utils, "request", lambda *args, **kwargs: pytest.fail("Cache replay cannot dispatch"))
    assert asyncio.run(score(judge_payload, **kwargs, cached_only=True)) == result
    assert all(path.read_bytes() == content for path, content in saved.items())


def test_two_delivered_judge_reviews_remain_exhausted_after_http_recovery(judge_payload, tmp_path, monkeypatch):
    outcomes = iter([401, 200, 503, 503, 503, 200])
    calls = []

    async def request(*args, **kwargs):
        calls.append(1)
        status = next(outcomes)
        return Response(status, {"choices": [{"finish_reason": "stop", "message": {"content": "not JSON"}}]})

    monkeypatch.setattr(server_utils, "request", request)
    monkeypatch.setattr("resources_servers.agentic_if_realworld.benchmark.transport.asyncio.sleep", no_sleep)
    monkeypatch.setenv("SYNTHETIC", "synthetic")
    kwargs = {
        "directory": tmp_path / "receipts",
        "endpoint": "http://synthetic.invalid",
        "policy_prompt": SYNTHETIC_PROMPT,
        "key_env": "SYNTHETIC",
    }
    for expected_calls in [1, 5]:
        with pytest.raises(RetryableTransportError):
            asyncio.run(score(judge_payload, **kwargs))
        assert len(calls) == expected_calls
    result = asyncio.run(score(judge_payload, **kwargs))
    assert result["scoring_status"] == "not_scored" and result["mask_sample"] is True
    assert len(calls) == 6
    saved = {path: path.read_bytes() for path in kwargs["directory"].rglob("attempt*.*")}
    assert asyncio.run(score(judge_payload, **kwargs)) == result
    assert len(calls) == 6 and all(path.read_bytes() == value for path, value in saved.items())


@pytest.mark.parametrize("content_type", ["application/json", "text/event-stream"])
@pytest.mark.parametrize("body_complete", [False, True])
def test_judge_only_spends_review_budget_on_completed_http_bodies(
    judge_payload, tmp_path, monkeypatch, content_type, body_complete
):
    from aiohttp import ClientPayloadError

    calls = []

    class MalformedResponse(Response):
        async def iter_chunked(self, size):
            yield b'{"choices":' if content_type == "application/json" else b'data: {"choices":\n\n'
            if not body_complete:
                raise ClientPayloadError("Synthetic interrupted response body")

    async def request(*args, **kwargs):
        calls.append(1)
        response = MalformedResponse(200, None)
        response.headers = {"content-type": content_type}
        return response

    monkeypatch.setattr(server_utils, "request", request)
    monkeypatch.setenv("SYNTHETIC", "synthetic")
    kwargs = {
        "directory": tmp_path / "receipts",
        "endpoint": "http://synthetic.invalid",
        "policy_prompt": SYNTHETIC_PROMPT,
        "key_env": "SYNTHETIC",
    }
    if body_complete:
        assert asyncio.run(score(judge_payload, **kwargs))["scoring_status"] == "not_scored"
        assert len(calls) == 2
    else:
        with pytest.raises(ValueError, match="Ambiguous judge dispatch"):
            asyncio.run(score(judge_payload, **kwargs))
        assert len(calls) == 1
    work = next((kwargs["directory"] / "judges").iterdir())
    receipts = list(work.glob("attempt*.json"))
    assert len(receipts) == (2 if body_complete else 1)
    assert all(read(path)["response_body_complete"] is body_complete for path in receipts)
    assert all(
        read(path)["dispatch_status"] == ("response_received" if body_complete else "unknown") for path in receipts
    )
    assert not (work / "result.json").exists()
    saved = {path: path.read_bytes() for path in work.iterdir() if path.is_file()}
    for validate_only in [True, False]:
        if body_complete:
            assert (
                asyncio.run(score(judge_payload, **kwargs, validate_only=validate_only))["scoring_status"]
                == "not_scored"
            )
        else:
            with pytest.raises(ValueError, match="Ambiguous judge dispatch"):
                asyncio.run(score(judge_payload, **kwargs, validate_only=validate_only))
    assert len(calls) == len(receipts) and all(path.read_bytes() == value for path, value in saved.items())


def test_cached_transport_rejects_changed_request_body(tmp_path, monkeypatch):
    body = {"model": "synthetic-judge"}
    endpoint = "http://synthetic.invalid"
    path = tmp_path / "receipt.json"
    write(
        path,
        {
            "status": "generated",
            "request_body": {"model": "changed"},
            "request_sha256": judge.sha(body),
            "endpoint": endpoint,
            "raw_attempts": [],
        },
    )
    monkeypatch.setattr(server_utils, "request", lambda *args, **kwargs: pytest.fail("No network call allowed"))
    with pytest.raises(ValueError, match="fingerprint"):
        asyncio.run(wire_call(body, endpoint=endpoint, key_env="SYNTHETIC", path=path))


def _score_in_worker(payload, directory, start, ready, calls, results):
    import os

    async def request(*args, **kwargs):
        with calls.get_lock():
            calls.value += 1
        await asyncio.sleep(0.1)
        return Response(
            200,
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": json.dumps(review(payload))},
                    }
                ]
            },
        )

    with (
        patch.dict(os.environ, {"SYNTHETIC_JUDGE_KEY": "synthetic"}),
        patch.object(server_utils, "request", request),
    ):
        ready.put(True)
        if not start.wait(20):
            raise TimeoutError("Synthetic parallel test did not start")
        result = asyncio.run(
            score(
                payload,
                directory=directory,
                endpoint="http://synthetic.invalid",
                policy_prompt=SYNTHETIC_PROMPT,
                key_env="SYNTHETIC_JUDGE_KEY",
            )
        )
        results.put(result)


def test_identical_reviews_share_cache_across_workers(judge_payload, tmp_path):
    from resources_servers.agentic_if_realworld.tests.test_transport import _score_in_worker as worker_target

    context = multiprocessing.get_context("spawn")
    start, ready, results = context.Event(), context.Queue(), context.Queue()
    calls = context.Value("i", 0)
    workers = [
        context.Process(
            target=worker_target,
            args=(judge_payload, tmp_path / "receipts", start, ready, calls, results),
        )
        for _ in range(2)
    ]
    try:
        for worker in workers:
            worker.start()
        for _ in workers:
            assert ready.get(timeout=20)
        start.set()
        received = [results.get(timeout=20) for _ in workers]
        for worker in workers:
            worker.join(timeout=20)
            assert worker.exitcode == 0
        assert received[0] == received[1]
        assert calls.value == 1
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
            worker.join(timeout=5)


def test_different_responses_can_be_judged_concurrently(judge_payload, tmp_path, monkeypatch):
    other = copy.deepcopy(judge_payload)
    other["response"]["message"]["content"] = "OK.."
    other["response_sha256"] = judge.sha(other["response"])
    other["inference_binding"]["response_sha256"] = other["response_sha256"]
    calls = []

    async def run():
        both_started = asyncio.Event()

        async def request(*args, **kwargs):
            payload = json.loads(
                kwargs["json"]["messages"][0]["content"].split("UNTRUSTED_EVALUATION_PAYLOAD:\n", 1)[1]
            )
            calls.append(payload["response_sha256"])
            if len(calls) == 2:
                both_started.set()
            await both_started.wait()
            return Response(
                200,
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": json.dumps(review(payload))},
                        }
                    ]
                },
            )

        monkeypatch.setattr(server_utils, "request", request)
        return await asyncio.wait_for(
            asyncio.gather(
                *(
                    score(
                        payload,
                        directory=tmp_path / "receipts",
                        endpoint="http://synthetic.invalid",
                        policy_prompt=SYNTHETIC_PROMPT,
                        key_env="SYNTHETIC_JUDGE_KEY",
                    )
                    for payload in [judge_payload, other]
                )
            ),
            timeout=10,
        )

    monkeypatch.setenv("SYNTHETIC_JUDGE_KEY", "synthetic")
    results = asyncio.run(run())
    assert [result["item_verdict"] for result in results] == ["pass", "fail"]
    assert len(calls) == 2
