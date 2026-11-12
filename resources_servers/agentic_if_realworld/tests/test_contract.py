# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import copy
import hashlib
import json

import pytest

from resources_servers.agentic_if_realworld.benchmark.core import (
    PHASES,
    ROOT,
    Dataset,
    bounded_schema,
    digest,
    frozen_policy_bytes,
    indexed,
    judge,
    provider_schema,
    read,
    summarize,
    write,
)
from resources_servers.agentic_if_realworld.benchmark.runner import run_cell
from resources_servers.agentic_if_realworld.benchmark.synthetic import create, review
from resources_servers.agentic_if_realworld.benchmark.transport import delivered


@pytest.fixture
def dataset(tmp_path):
    create(tmp_path / "data")
    return Dataset(tmp_path / "data", synthetic=True)


def payload(dataset, text="OK."):
    response = {
        "status": "complete",
        "finish_reason": "stop",
        "message": {"role": "assistant", "content": text},
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


def test_exact_input_private_metadata_and_nested_support(dataset):
    row = dataset.prepared()[0]
    assert row["responses_create_params"]["input"] == dataset.request("synthetic")["messages"]
    assert row["responses_create_params"]["tools"] == []
    actor = json.dumps(row["responses_create_params"])
    for private in [
        "constraint_id",
        "support_quotes",
        "scoring_view",
        "verifier_metadata",
    ]:
        assert private not in actor
    assert judge.validate_review(review(payload(dataset)), payload(dataset))["item_verdict"] == "pass"


def test_minor_violation_fails_and_missing_check_rejected(dataset):
    p = payload(dataset, "OK..")
    assert judge.validate_review(review(p), p)["item_verdict"] == "fail"
    bad = review(p)
    bad["constraints"] = []
    with pytest.raises((ValueError, __import__("jsonschema").ValidationError)):
        judge.validate_review(bad, p)


def test_binding_changes_and_native_calls_rejected(dataset):
    p = payload(dataset)
    p["response"]["message"]["content"] = "changed"
    with pytest.raises(ValueError, match="binding"):
        judge.validate_payload(p)
    for finish, message in [
        ("length", {"role": "assistant", "content": "x"}),
        (
            "stop",
            {"role": "assistant", "content": "", "reasoning_content": "only thinking"},
        ),
        (
            "stop",
            {
                "role": "assistant",
                "content": "x",
                "tool_calls": [{"name": "forbidden"}],
            },
        ),
    ]:
        with pytest.raises(ValueError):
            delivered({"choices": [{"finish_reason": finish, "message": message}]})


def test_full_denominator_unknown_and_subtype_overlap(dataset):
    result = summarize([], dataset)
    assert result["complete_accuracy"] is None and result["not_scored"] == 1
    assert result["views"]["original_failure_focused"]["scoring_coverage"] == 0
    assert result["subtype_denominators_overlap"] is True


def test_unicode_separators_are_not_jsonl_boundaries(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps({"id": "a", "content": "before\u2028after"}, ensure_ascii=False) + "\n")
    assert indexed(path)["a"]["content"] == "before\u2028after"


def test_frozen_assets_are_unchanged():
    for name, expected in read(ROOT / "source_manifest.json")["policy_assets"].items():
        if name in {"judge_prompt.md", "rubric.md"}:
            assert not (ROOT / "policy" / name).exists()
        else:
            assert hashlib.sha256(frozen_policy_bytes(name)).hexdigest() == expected


@pytest.mark.parametrize("name", ["validate_response_judge.py", "test_contract.py"])
def test_frozen_policy_header_must_match_exactly(tmp_path, monkeypatch, name):
    raw = (ROOT / "policy" / name).read_bytes()
    (tmp_path / "policy").mkdir()
    (tmp_path / "policy" / name).write_bytes(
        raw.replace(b"# First tracked package revision:", b"# Changed revision:", 1)
    )
    monkeypatch.setattr("resources_servers.agentic_if_realworld.benchmark.core.ROOT", tmp_path)
    with pytest.raises(ValueError, match="provenance header"):
        frozen_policy_bytes(name)


def test_private_judge_prompt_is_exact_and_rechecked_before_each_use(tmp_path):
    content = b"Synthetic test of a private frozen prompt.\r\n"
    path = tmp_path / "judge_prompt.md"
    path.write_bytes(content)
    dataset = Dataset.__new__(Dataset)
    dataset.directory, dataset.synthetic = tmp_path, False
    dataset.pin = {"policy_assets": {"judge_prompt.md": hashlib.sha256(content).hexdigest()}}
    assert dataset.judge_prompt().encode("utf-8") == content
    path.write_bytes(content + b"changed")
    with pytest.raises(ValueError, match="Frozen private judge prompt hash mismatch"):
        dataset.judge_prompt()


def test_synthetic_prompt_does_not_read_private_policy(dataset):
    (dataset.directory / "judge_prompt.md").write_text("must never be used")
    prompt = dataset.judge_prompt()
    assert "independently authored synthetic literal tasks" in prompt
    assert "must never be used" not in prompt


def test_scoring_requires_explicit_nonempty_policy_prompt(dataset, tmp_path):
    from resources_servers.agentic_if_realworld.benchmark.scoring import score

    kwargs = {"directory": tmp_path / "receipts", "endpoint": "http://synthetic.invalid"}
    with pytest.raises(TypeError, match="policy_prompt"):
        asyncio.run(score(payload(dataset), **kwargs))
    with pytest.raises(ValueError, match="Explicit judge policy prompt"):
        asyncio.run(score(payload(dataset), **kwargs, policy_prompt=" "))


def test_adaptive_first_complete_and_judge_only_resume(dataset, tmp_path):
    calls, judges = [], []

    async def generate(row, seed, cap, receipt_id, attempt_index):
        calls.append((seed, cap))
        return {
            "complete": len(calls) == 2,
            "canonical_response": {"text": "saved even when fail"},
        }

    async def verify(selected, criteria, *, cached_only=False):
        if not cached_only:
            judges.append(copy.deepcopy(selected))
        if len(judges) == 1:
            from resources_servers.agentic_if_realworld.benchmark.core import unscored

            return unscored("synthetic", "judge_unavailable")
        p = payload(dataset, "OK..")
        return judge.validate_review(review(p), p)

    async def run():
        args = {
            "directory": tmp_path / "run",
            "settings": {"model": "test"},
            "generate": generate,
            "verify": verify,
        }
        assert (await run_cell(dataset, "synthetic", **args))["scoring_status"] == "not_scored"
        assert (await run_cell(dataset, "synthetic", **args, resume=True))["item_verdict"] == "fail"
        assert calls == list(PHASES[:2]) and judges[0] == judges[1]
        await run_cell(dataset, "synthetic", **args, resume=True)
        assert len(judges) == 2
        with pytest.raises(ValueError, match="fingerprint"):
            await run_cell(
                dataset,
                "synthetic",
                **{**args, "settings": {"model": "changed"}},
                resume=True,
            )

    asyncio.run(run())


def test_exhaustion_and_ambiguous_dispatch(dataset, tmp_path):
    calls = []

    async def incomplete(row, seed, cap, receipt_id, attempt_index):
        calls.append((seed, cap))
        return {"complete": False}

    async def no_judge(*args):
        pytest.fail("Incomplete delivery must not reach judge")

    args = {
        "directory": tmp_path / "run",
        "settings": {},
        "generate": incomplete,
        "verify": no_judge,
    }
    result = asyncio.run(run_cell(dataset, "synthetic", **args))
    assert calls == list(PHASES) and result["mask_sample"] and result["scoring_status"] == "not_scored"

    async def crash(*args):
        raise RuntimeError("interrupted after dispatch")

    args = {**args, "directory": tmp_path / "ambiguous", "generate": crash}
    with pytest.raises(RuntimeError):
        asyncio.run(run_cell(dataset, "synthetic", **args))
    with pytest.raises(ValueError, match="Ambiguous"):
        asyncio.run(run_cell(dataset, "synthetic", **{**args, "generate": incomplete}, resume=True))


def test_terminal_delivery_stops_profile(dataset, tmp_path):
    calls = []

    async def terminal(*args):
        calls.append(1)
        return {"complete": False, "terminal": True}

    async def no_judge(*args):
        pytest.fail("terminal delivery cannot be judged")

    asyncio.run(
        run_cell(
            dataset,
            "synthetic",
            directory=tmp_path / "run",
            settings={},
            generate=terminal,
            verify=no_judge,
        )
    )
    assert len(calls) == 1


def test_connect_failure_stops_and_explicit_resume_retries_same_phase(dataset, tmp_path, monkeypatch):
    from aiohttp import ClientConnectorError

    from nemo_gym.client import http_client
    from resources_servers.agentic_if_realworld.benchmark.transport import RetryableTransportError, wire_call
    from resources_servers.agentic_if_realworld.tests.test_transport import Response

    calls, receipt_ids = [], []

    async def request(*args, **kwargs):
        calls.append((kwargs["json"]["seed"], kwargs["json"]["max_tokens"]))
        if len(calls) == 1:
            raise ClientConnectorError(None, ConnectionRefusedError())
        return Response(200, {"choices": [{"finish_reason": "stop", "message": {"content": "OK."}}]})

    async def generate(row, seed, cap, receipt_id, attempt_index):
        receipt_ids.append(receipt_id)
        path = tmp_path / "model" / (receipt_id + ".json")
        receipt = await wire_call(
            {**dataset.request("synthetic"), "seed": seed, "max_tokens": cap},
            endpoint="http://synthetic.invalid",
            key_env="SYNTHETIC",
            path=path,
        )
        return {
            "complete": receipt["status"] == "generated",
            "dispatch_status": receipt.get("dispatch_status"),
            "receipt_path": str(path),
            "receipt_sha256": digest(path),
        }

    async def verify(*args, **kwargs):
        p = payload(dataset)
        return judge.validate_review(review(p), p)

    monkeypatch.setattr(http_client, "request", request)
    monkeypatch.setenv("SYNTHETIC", "synthetic")
    args = {"directory": tmp_path / "run", "settings": {}, "generate": generate, "verify": verify}
    with pytest.raises(RetryableTransportError):
        asyncio.run(run_cell(dataset, "synthetic", **args))
    journal = next((tmp_path / "run" / "cells").glob("*.json"))
    state = read(journal)
    assert len(state["attempts"]) == 1 and "selected" not in state and "result" not in state
    old = {path: path.read_bytes() for path in (tmp_path / "model").iterdir()}
    asyncio.run(run_cell(dataset, "synthetic", **args, resume=True, validate_only=True))
    assert len(calls) == 1
    result = asyncio.run(run_cell(dataset, "synthetic", **args, resume=True))
    assert result["item_verdict"] == "pass" and calls == [PHASES[0], PHASES[0]]
    assert receipt_ids[0] != receipt_ids[1]
    state = read(journal)
    assert len(state["attempts"]) == len(state["transport_failures"]) == 1
    assert all(path.read_bytes() == value for path, value in old.items())
    damaged = next(path for path in old if path.suffix == ".http")
    damaged.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Raw transport binding changed"):
        asyncio.run(run_cell(dataset, "synthetic", **args, resume=True))
    assert len(calls) == 2


def test_broken_policy_streams_exhaust_only_the_frozen_schedule(dataset, tmp_path, monkeypatch):
    from aiohttp import SocketTimeoutError

    from nemo_gym.client import http_client
    from resources_servers.agentic_if_realworld.benchmark.transport import wire_call
    from resources_servers.agentic_if_realworld.tests.test_transport import Response

    calls = []

    class InterruptedStream(Response):
        async def iter_chunked(self, size):
            yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
            raise SocketTimeoutError("Synthetic policy read timeout")

    async def request(*args, **kwargs):
        calls.append((kwargs["json"]["seed"], kwargs["json"]["max_tokens"]))
        response = InterruptedStream(200, None)
        response.headers = {"content-type": "text/event-stream"}
        return response

    async def generate(row, seed, cap, receipt_id, attempt_index):
        path = tmp_path / "model" / (receipt_id + ".json")
        receipt = await wire_call(
            {**dataset.request("synthetic"), "seed": seed, "max_tokens": cap, "stream": True},
            endpoint="http://synthetic.invalid",
            key_env="SYNTHETIC",
            path=path,
        )
        return {
            "complete": False,
            "dispatch_status": receipt["dispatch_status"],
            "receipt_path": str(path),
            "receipt_sha256": digest(path),
        }

    async def no_judge(*args, **kwargs):
        raise AssertionError("Partial policy bytes must never be judged")

    monkeypatch.setattr(http_client, "request", request)
    monkeypatch.setenv("SYNTHETIC", "synthetic")
    args = {"directory": tmp_path / "run", "settings": {}, "generate": generate, "verify": no_judge}
    result = asyncio.run(run_cell(dataset, "synthetic", **args))
    assert result["mask_sample"] is True and result["reason"] == "incomplete_delivery_budget_exhausted"
    assert calls == list(PHASES)
    saved = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert asyncio.run(run_cell(dataset, "synthetic", **args, resume=True)) == result
    assert calls == list(PHASES)
    assert all(path.read_bytes() == value for path, value in saved.items())


def test_judge_transport_recovery_never_regenerates_selected_answer(dataset, tmp_path, monkeypatch):
    from nemo_gym.client import http_client
    from resources_servers.agentic_if_realworld.benchmark.scoring import score
    from resources_servers.agentic_if_realworld.benchmark.transport import RetryableTransportError
    from resources_servers.agentic_if_realworld.tests.test_transport import Response

    generations, judges = [], []

    async def generate(*args):
        generations.append(1)
        return {"complete": True, "canonical_response": "OK."}

    async def request(*args, **kwargs):
        judges.append(1)
        if len(judges) == 1:
            return Response(401, {"error": "bad credentials"})
        return Response(
            200, {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(review(payload(dataset)))}}]}
        )

    async def verify(*args, **kwargs):
        return await score(
            payload(dataset),
            directory=tmp_path / "receipts",
            endpoint="http://synthetic.invalid",
            policy_prompt=dataset.judge_prompt(),
            key_env="SYNTHETIC",
            **kwargs,
        )

    monkeypatch.setattr(http_client, "request", request)
    monkeypatch.setenv("SYNTHETIC", "bad")
    args = {"directory": tmp_path / "run", "settings": {}, "generate": generate, "verify": verify}
    with pytest.raises(RetryableTransportError):
        asyncio.run(run_cell(dataset, "synthetic", **args))
    journal = next((tmp_path / "run" / "cells").glob("*.json"))
    selected = read(journal)["selected"]
    assert "result" not in read(journal)
    monkeypatch.setenv("SYNTHETIC", "corrected")
    result = asyncio.run(run_cell(dataset, "synthetic", **args, resume=True))
    assert result["item_verdict"] == "pass" and len(generations) == 1 and len(judges) == 2
    assert read(journal)["selected"] == selected


def test_duplicate_and_invented_checks_rejected(dataset):
    p = payload(dataset)
    for change in ["duplicate", "invented", "wrong_hash"]:
        value = review(p)
        if change == "duplicate":
            value["constraints"].append(copy.deepcopy(value["constraints"][0]))
        elif change == "invented":
            value["constraints"][0]["constraint_id"] = "invented"
        else:
            value["response_sha256"] = "0" * 64
        with pytest.raises((ValueError, __import__("jsonschema").ValidationError)):
            judge.validate_review(value, p)


def test_judge_malformed_and_outage_are_unscored(dataset, tmp_path, monkeypatch):
    from resources_servers.agentic_if_realworld.benchmark import scoring

    calls = []

    async def malformed(body, *, path, **kwargs):
        calls.append(1)
        response = {"choices": [{"finish_reason": "stop", "message": {"content": "not JSON"}}]}
        raw = path.with_suffix(".http")
        write(raw, response)
        receipt = {
            "status": "generated",
            "response": response,
            "content_type": "application/json",
            "raw_attempts": [{"path": str(raw), "sha256": digest(raw)}],
        }
        write(path, receipt)
        return receipt

    monkeypatch.setattr(scoring, "wire_call", malformed)
    result = asyncio.run(
        scoring.score(
            payload(dataset),
            directory=tmp_path,
            endpoint="http://synthetic.invalid",
            policy_prompt=dataset.judge_prompt(),
        )
    )
    assert result["scoring_status"] == "not_scored" and result["mask_sample"] and len(calls) == 2

    async def unavailable(*args, **kwargs):
        return {"status": "delivery_error", "http_status": 401}

    monkeypatch.setattr(scoring, "wire_call", unavailable)
    with pytest.raises(ValueError, match="Judge transport failed"):
        asyncio.run(
            scoring.score(
                payload(dataset),
                directory=tmp_path / "outage",
                endpoint="http://synthetic.invalid",
                policy_prompt=dataset.judge_prompt(),
            )
        )


def test_sse_replay_terminal_and_native_rejection(tmp_path):
    from resources_servers.agentic_if_realworld.benchmark.transport import replay_sse

    path = tmp_path / "raw"

    def event(delta, finish=None):
        return "data: " + json.dumps({"choices": [{"delta": delta, "finish_reason": finish}]}) + "\n\n"

    path.write_text(event({"content": "OK"}) + event({"content": "."}, "stop") + "data: [DONE]\n\n")
    assert delivered(replay_sse(path))["message"]["content"] == "OK."
    path.write_text(event({"content": "partial"}))
    with pytest.raises(ValueError, match="terminal"):
        replay_sse(path)
    path.write_text(event({"tool_calls": [{"name": "forbidden"}]}, "tool_calls"))
    with pytest.raises(ValueError, match="Native"):
        replay_sse(path)


@pytest.mark.parametrize(
    "finish_reason,metadata,complete",
    [
        ("repetition", {}, False),
        ("stop", {"native_finish_reason": "repetition"}, False),
        ("stop", {"stop_reason": "repetition_detected"}, False),
        ("stop", {"stop_reason": 0}, True),
        ("stop", {"stop_reason": 11}, True),
        ("stop", {"stop_reason": "stop_sequence"}, True),
        ("stop", {"native_finish_reason": "provider_stop"}, True),
    ],
)
@pytest.mark.parametrize("consumer", ["policy", "judge"])
def test_completion_checks_reject_only_known_repetition_metadata(
    dataset, tmp_path, finish_reason, metadata, complete, consumer
):
    from nemo_gym.core import NeMoGymChatCompletion
    from resources_servers.agentic_if_realworld.benchmark.scoring import _checked_result
    from resources_servers.agentic_if_realworld.benchmark.transport import verified_response

    p = payload(dataset)
    expected_review = review(p)
    raw_response = {
        "id": "synthetic-judge-response",
        "created": 0,
        "model": "synthetic-judge",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": json.dumps(expected_review)},
                "finish_reason": finish_reason,
                **metadata,
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
    }
    normalized = NeMoGymChatCompletion.model_validate(raw_response)
    [choice] = normalized.choices
    assert choice.finish_reason == "stop"
    if finish_reason == "repetition":
        assert choice.native_finish_reason == "repetition"
    for key, value in metadata.items():
        assert getattr(choice, key) == value
        assert type(getattr(choice, key)) is type(value)
    response = normalized.model_dump(mode="json", exclude_unset=True)
    path = tmp_path / "judge.http"
    write(path, response)
    receipt = {
        "status": "generated",
        "content_type": "application/json",
        "response": response,
        "raw_attempts": [{"path": str(path), "sha256": digest(path)}],
    }
    verified = verified_response(receipt)
    schema = bounded_schema(provider_schema(read(ROOT / "policy/judge_output.schema.json")), p)
    body = {"model": "synthetic-judge", "reasoning_effort": "medium"}
    if complete and consumer == "policy":
        assert delivered(verified) == {
            "status": "complete",
            "finish_reason": "stop",
            "message": response["choices"][0]["message"],
        }
    elif complete:
        checked_review, result = _checked_result(verified, p, schema, body)
        assert checked_review == expected_review
        assert result["item_verdict"] == "pass" and result["reward"] == 1.0
        assert result["mask_sample"] is False
        assert result["judge_response_id"] == response["id"]
        assert result["judge_usage"] == response["usage"]
    else:
        reason = "Incomplete delivery" if consumer == "policy" else "Incomplete judge delivery"
        with pytest.raises(ValueError, match=reason):
            if consumer == "policy":
                delivered(verified)
            else:
                _checked_result(verified, p, schema, body)


@pytest.mark.parametrize("metadata", [{"native_finish_reason": "repetition"}, {"stop_reason": "repetition_detected"}])
@pytest.mark.parametrize("cause_on_terminal", [True, False])
def test_sse_repetition_evidence_survives_raw_binding(tmp_path, metadata, cause_on_terminal):
    from resources_servers.agentic_if_realworld.benchmark.transport import replay_sse, verified_response

    first = {"index": 0, "delta": {"content": "OK"}, "finish_reason": None}
    terminal = {"index": 0, "delta": {"content": "."}, "finish_reason": "stop"}
    (terminal if cause_on_terminal else first).update(metadata)
    usage = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
    chunks = [
        {"choices": [first]},
        {"choices": [terminal]},
        {"choices": [], "usage": usage},
    ]
    raw = ("".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks) + "data: [DONE]\n\n").encode()
    path = tmp_path / "synthetic.http"
    path.write_bytes(raw)
    expected = {
        "object": "chat.completion",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "OK."}, "finish_reason": "stop", **metadata}
        ],
        "usage": usage,
    }
    assert replay_sse(path) == expected
    receipt = {
        "status": "generated",
        "content_type": "text/event-stream",
        "response": expected,
        "raw_attempts": [{"path": str(path), "sha256": digest(path)}],
    }
    assert verified_response(receipt) == expected
    with pytest.raises(ValueError, match="Incomplete delivery"):
        delivered(verified_response(receipt))
    legacy = copy.deepcopy(receipt)
    for key in metadata:
        legacy["response"]["choices"][0].pop(key)
    with pytest.raises(ValueError, match="Saved response differs from raw delivery"):
        verified_response(legacy)
    assert path.read_bytes() == raw


@pytest.mark.parametrize(
    "metadata",
    [
        {"stop_reason": 0},
        {"stop_reason": 11},
        {"stop_reason": "stop_sequence"},
        {"native_finish_reason": "provider_stop"},
        {"unrelated": "value"},
    ],
)
def test_sse_ordinary_metadata_preserves_legacy_receipt_shape(tmp_path, metadata):
    from resources_servers.agentic_if_realworld.benchmark.transport import replay_sse, verified_response

    terminal = {"delta": {"content": "OK."}, "finish_reason": "stop", **metadata}
    raw = ("data: " + json.dumps({"choices": [terminal]}) + "\n\ndata: [DONE]\n\n").encode()
    path = tmp_path / "legacy.http"
    path.write_bytes(raw)
    legacy = {
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK."}, "finish_reason": "stop"}],
    }
    assert replay_sse(path) == legacy
    receipt = {
        "status": "generated",
        "content_type": "text/event-stream",
        "response": legacy,
        "raw_attempts": [{"path": str(path), "sha256": digest(path)}],
    }
    assert delivered(verified_response(receipt)) == {
        "status": "complete",
        "finish_reason": "stop",
        "message": legacy["choices"][0]["message"],
    }
    assert path.read_bytes() == raw


@pytest.mark.parametrize(
    "fields",
    [("reasoning_content",), ("reasoning",), ("reasoning_content", "reasoning")],
)
def test_sse_reasoning_aliases_preserve_raw_bytes_and_judge_fields(tmp_path, fields):
    from resources_servers.agentic_if_realworld.benchmark.transport import replay_sse, verified_response

    chunks = [
        {"choices": [{"index": 0, "delta": dict.fromkeys(fields, text), "finish_reason": None}]}
        for text in ["Synthetic plan: ", "return OK."]
    ]
    chunks.append({"choices": [{"index": 0, "delta": {"content": "OK."}, "finish_reason": "stop"}]})
    raw = ("".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks) + "data: [DONE]\n\n").encode()
    path = tmp_path / "synthetic.http"
    path.write_bytes(raw)
    response = replay_sse(path)
    receipt = {
        "status": "generated",
        "content_type": "text/event-stream",
        "response": response,
        "raw_attempts": [{"path": str(path), "sha256": digest(path)}],
    }
    message = delivered(verified_response(receipt))["message"]
    assert message == {"role": "assistant", "content": "OK.", "reasoning_content": "Synthetic plan: return OK."}
    assert judge.response_fields(message)["reasoning_content"] == "Synthetic plan: return OK."
    assert path.read_bytes() == raw


@pytest.mark.parametrize(
    "delta",
    [
        {"reasoning": "one", "reasoning_content": "two"},
        {"reasoning": "", "reasoning_content": "two"},
        {"reasoning": ["not text"]},
        {"reasoning_content": ["not text"]},
        {"reasoning": 0, "reasoning_content": False},
    ],
)
def test_sse_reasoning_rejects_conflicting_or_nontext_fields(delta):
    from resources_servers.agentic_if_realworld.benchmark.transport import SSE

    with pytest.raises(ValueError):
        SSE().event(json.dumps({"choices": [{"delta": delta}]}))


@pytest.mark.parametrize("field", ["reasoning", "reasoning_content"])
def test_sse_null_reasoning_alias_is_absent(field):
    from resources_servers.agentic_if_realworld.benchmark.transport import SSE

    other = "reasoning_content" if field == "reasoning" else "reasoning"
    parser = SSE()
    parser.event(json.dumps({"choices": [{"delta": {field: None, other: "Synthetic plan."}}]}))
    parser.event(json.dumps({"choices": [{"delta": {"content": "OK."}, "finish_reason": "stop"}]}))
    assert delivered(parser.response())["message"]["reasoning_content"] == "Synthetic plan."
