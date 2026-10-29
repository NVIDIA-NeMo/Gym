# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import copy
import json
import multiprocessing
from unittest.mock import patch

import pytest

from resources_servers.agentic_if_realworld.benchmark.core import ROOT, Dataset, judge, read, write
from resources_servers.agentic_if_realworld.benchmark.scoring import score
from resources_servers.agentic_if_realworld.benchmark.synthetic import create, review
from resources_servers.agentic_if_realworld.benchmark.transport import wire_call


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
    "damage", ["raw", "receipt", "reward", "mask", "payload", "schema", "missing", "policy_prompt"]
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
        }
        path = next(work.glob(names[damage]))
        path.write_text("changed")
    with pytest.raises(ValueError):
        asyncio.run(score(judge_payload, **kwargs, cached_only=True))
    assert len(calls) == 1


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
