# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import importlib
import importlib.util
import json

import pytest

from resources_servers.partner_if.engine import EvidenceBundle, GradingContract


def api():
    name = "resources_servers.partner_if.llm"
    assert importlib.util.find_spec(name), "Shared LLM verifier is not implemented"
    return importlib.import_module(name)


def rubric():
    return GradingContract(
        task_id="smoke",
        origin="synthetic",
        source="authored",
        rubric_version="v4",
        audit_signoff=False,
        data_kind="synthetic_smoke",
        rubrics=[
            {
                "id": "one",
                "constraint": "The answer says HELLO.",
                "verifier": {"id": "llm_judge_v1"},
                "taxonomy": ["IF-FORMAT"],
                "analogues": [{"source_id": "private-trace"}],
            }
        ],
    ).rubrics


def verdict(**overrides):
    return {
        "items": [
            {
                "id": "one",
                "passed": True,
                "reason": "Required word present",
                "evidence": "HELLO",
                "references": [{"source": "answer", "start_line": 1, "end_line": 1}],
                **overrides,
            }
        ]
    }


def test_judge_payload_preserves_surfaces_but_excludes_analogue_and_model_labels():
    bundle = EvidenceBundle(
        context=[{"role": "user", "content": "Say HELLO"}],
        answer="HELLO",
        documents={"skill.md": "An actor-visible skill"},
        artifacts={"output.txt": {"status": "present", "text": "written content"}},
        trajectory=[{"kind": "tool_call", "source_index": 7, "tool_name": "write", "arguments": {"text": "x"}}],
    )
    payload = api().payload(rubric(), bundle)
    sources = payload["evidence_sources"]
    assert sources["context:0/role"]["lines"][0]["text"] == "user"
    assert sources["answer"]["lines"][0]["text"] == "HELLO"
    assert sources["artifact:output.txt"]["lines"][0]["text"] == "written content"
    assert sources["event:7/arguments/text"]["lines"][0]["text"] == "x"
    assert "private-trace" not in json.dumps(payload)


@pytest.mark.parametrize(
    "wrap", [lambda x: x, lambda x: "<think>internal</think>\n" + x, lambda x: "```json\n" + x + "\n```"]
)
def test_known_reasoning_and_json_envelopes_are_parsed_without_losing_verdict(wrap):
    result = api().parse_verdict(wrap(json.dumps(verdict())), rubric(), EvidenceBundle(answer="HELLO"))
    assert result[0].passed is True and result[0].evidence == "HELLO"


@pytest.mark.parametrize(
    "change",
    [
        {"references": [{"source": "answer", "start_line": 2, "end_line": 2}]},
        {"references": [{"source": "artifact:/etc/passwd", "start_line": 1, "end_line": 1}]},
        {"references": []},
        {"passed": None, "references": []},
        {"passed": "true"},
        {"passed": "review", "references": []},
        {"passed": 1},
        {"passed": 0},
        {"unexercised": True},
        {"unexercised": False},
        {"id": "unexpected"},
    ],
)
def test_invalid_judge_evidence_or_verdict_rejected(change):
    with pytest.raises(ValueError):
        api().parse_verdict(json.dumps(verdict(**change)), rubric(), EvidenceBundle(answer="HELLO"))


def test_absence_failure_is_a_binary_judgment():
    missing = api().parse_verdict(
        json.dumps(verdict(passed=False, evidence="No requested text", references=[])),
        rubric(),
        EvidenceBundle(answer=""),
    )
    assert missing[0].passed is False


def test_technical_error_judgments_can_be_null_outside_the_judge_boundary():
    from resources_servers.partner_if.grading_contract import RawJudgment

    saved = verdict(passed=None)["items"][0]
    saved.pop("references")
    historical = RawJudgment.model_validate(saved | {"quotes": []})
    assert historical.passed is None


def test_prohibition_pass_is_an_ordinary_evidence_backed_boolean():
    items = rubric()
    items[0].constraint = "The answer contains no emoji."
    result = api().parse_verdict(
        json.dumps(verdict(reason="The answer contains only the required plain text")),
        items,
        EvidenceBundle(answer="HELLO"),
    )
    assert result[0].passed is True


def test_duplicate_json_fields_are_not_accepted_as_a_valid_judgment():
    with pytest.raises(ValueError, match="Duplicate"):
        api().parse_verdict('{"items": [], "items": []}', rubric(), EvidenceBundle(answer="HELLO"))


@pytest.mark.parametrize(
    "source,start,end,quote",
    [
        ("event:2/arguments/filePath", 1, 1, "/workspace/plan.md"),
        ("artifact:output", 1, 2, "first\n  printf '%s\\n' READY"),
    ],
)
def test_references_extract_literal_text_without_model_escape_roundtrip(source, start, end, quote):
    evidence = EvidenceBundle(
        artifacts={"output": {"status": "present", "text": "first\n  printf '%s\\n' READY"}},
        trajectory=[{"source_index": 2, "kind": "tool_call", "arguments": {"filePath": "/workspace/plan.md"}}],
    )
    result = api().parse_verdict(
        json.dumps(verdict(references=[{"source": source, "start_line": start, "end_line": end}])), rubric(), evidence
    )
    assert result[0].passed is True
    assert result[0].quotes[0].quote == quote


def test_live_judge_cannot_supply_an_old_protocol_quote_for_silent_repair():
    evidence = EvidenceBundle(artifacts={"output": {"status": "present", "text": "first\n  second"}})
    with pytest.raises(ValueError, match="quotes"):
        api().parse_verdict(
            json.dumps(verdict(quotes=[{"source": "artifact:output", "quote": r"first\n second"}])),
            rubric(),
            evidence,
        )


def test_judge_requests_object_mode_through_the_real_gym_converter():
    from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
    from nemo_gym.responses_converter import ResponsesConverter

    request = NeMoGymResponseCreateParamsNonStreaming.model_validate(
        api().request_payload(rubric(), EvidenceBundle(answer="HELLO"), max_tokens=2048, temperature=0)
    )
    converted = ResponsesConverter(return_token_id_information=False).responses_to_chat_completion_create_params(
        request
    )
    assert converted.response_format == {"type": "json_object"}


@pytest.mark.parametrize(
    "change",
    [
        {"references": [{"source": "missing", "start_line": 1, "end_line": 1}]},
        {"references": [{"source": "answer", "start_line": 20, "end_line": 20}]},
        {"references": []},
        {"passed": "true"},
        {"passed": None},
    ],
)
def test_invalid_completion_item_does_not_discard_valid_if_judgment(change):
    items = rubric()
    items.append(items[0].model_copy(update={"id": "content", "dimension": "completion"}))
    reply = verdict()
    reply["items"].append(reply["items"][0] | {"id": "content"} | change)
    results = api().parse_verdict(json.dumps(reply), items, EvidenceBundle(answer="HELLO"))
    assert results[0].passed is True
    assert results[0].quotes[0].quote == "HELLO"
    assert results[1].passed is None
    assert "Invalid completion judgment" in results[1].reason


def test_missing_completion_item_is_unavailable_not_an_if_failure():
    items = rubric()
    items.append(items[0].model_copy(update={"id": "content", "dimension": "completion"}))
    results = api().parse_verdict(json.dumps(verdict()), items, EvidenceBundle(answer="HELLO"))
    assert results[0].passed is True and results[1].passed is None


def test_missing_if_item_still_rejects_incomplete_judge_response():
    items = rubric()
    items.append(items[0].model_copy(update={"id": "content", "dimension": "completion"}))
    with pytest.raises(ValueError, match="rubric ID"):
        api().parse_verdict(json.dumps(verdict(id="content")), items, EvidenceBundle(answer="HELLO"))
