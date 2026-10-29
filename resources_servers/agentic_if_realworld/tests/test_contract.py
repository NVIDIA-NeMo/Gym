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
    digest,
    frozen_policy_bytes,
    indexed,
    judge,
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

    async def generate(row, seed, cap, receipt_id):
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

    async def incomplete(row, seed, cap, receipt_id):
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
        return {"status": "delivery_error"}

    monkeypatch.setattr(scoring, "wire_call", unavailable)
    result = asyncio.run(
        scoring.score(
            payload(dataset),
            directory=tmp_path / "outage",
            endpoint="http://synthetic.invalid",
            policy_prompt=dataset.judge_prompt(),
        )
    )
    assert result["model_verdict"] is None and result["scoring_status"] == "not_scored"


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
