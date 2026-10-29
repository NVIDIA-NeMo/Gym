# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from resources_servers.partner_if import llm
from resources_servers.partner_if.evidence import EvidenceBundle
from resources_servers.partner_if.grading_contract import GradingContract


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
    payload = llm.payload(rubric(), bundle)
    sources = payload["evidence_sources"]
    assert sources["context:0/role"]["lines"][0]["text"] == "user"
    assert sources["answer"]["lines"][0]["text"] == "HELLO"
    assert sources["artifact:output.txt"]["lines"][0]["text"] == "written content"
    assert sources["event:7/arguments/text"]["lines"][0]["text"] == "x"
    assert "private-trace" not in json.dumps(payload)


@pytest.mark.parametrize(
    "wrap",
    [
        lambda x: x,
        lambda x: "<think>internal</think>\n" + x,
        lambda x: "```json\n" + x + "\n```",
        lambda x: "```\n" + x + "\n```",
        lambda x: "\n ```JSON " + x + "``` \n",
        lambda x: "<think>internal</think>\n```Json\n\n" + x + "\n\n```",
    ],
)
def test_known_reasoning_and_json_envelopes_are_parsed_without_losing_verdict(wrap):
    result = llm.parse_verdict(wrap(json.dumps(verdict())), rubric(), EvidenceBundle(answer="HELLO"), gating={"one"})
    assert result[0].passed is True and result[0].evidence == "HELLO"


@pytest.mark.parametrize(
    "wrap",
    [
        lambda x: "Verdict:\n```json\n" + x + "\n```",
        lambda x: "```json\n" + x + "\n```\nDone.",
        lambda x: "```json\n" + x + "\n```\n```json\n" + x + "\n```",
        lambda x: "```python\n" + x + "\n```",
    ],
)
def test_only_one_json_fence_around_the_whole_reply_is_unwrapped(wrap):
    with pytest.raises(ValueError):
        llm.parse_verdict(wrap(json.dumps(verdict())), rubric(), EvidenceBundle(answer="HELLO"), gating={"one"})


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
        {"extra": True},
        {"id": "unexpected"},
    ],
)
def test_invalid_judge_evidence_or_verdict_rejected(change):
    with pytest.raises(ValueError):
        llm.parse_verdict(json.dumps(verdict(**change)), rubric(), EvidenceBundle(answer="HELLO"), gating={"one"})


def test_absence_failure_is_a_binary_judgment():
    missing = llm.parse_verdict(
        json.dumps(verdict(passed=False, evidence="No requested text", references=[])),
        rubric(),
        EvidenceBundle(answer=""),
        gating={"one"},
    )
    assert missing[0].passed is False


def test_technical_error_judgments_can_be_null_outside_the_judge_boundary():
    from resources_servers.partner_if.grading_contract import RawJudgment

    saved = verdict(passed=None)["items"][0]
    saved.pop("references")
    judgment = RawJudgment.model_validate(saved | {"quotes": []})
    assert judgment.passed is None


def test_prohibition_pass_is_an_ordinary_evidence_backed_boolean():
    items = rubric()
    items[0].constraint = "The answer contains no emoji."
    result = llm.parse_verdict(
        json.dumps(verdict(reason="The answer contains only the required plain text")),
        items,
        EvidenceBundle(answer="HELLO"),
        gating={"one"},
    )
    assert result[0].passed is True


def test_duplicate_json_fields_are_not_accepted_as_a_valid_judgment():
    with pytest.raises(ValueError, match="Duplicate"):
        llm.parse_verdict('{"items": [], "items": []}', rubric(), EvidenceBundle(answer="HELLO"), gating={"one"})


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
    result = llm.parse_verdict(
        json.dumps(verdict(references=[{"source": source, "start_line": start, "end_line": end}])),
        rubric(),
        evidence,
        gating={"one"},
    )
    assert result[0].passed is True
    assert result[0].quotes[0].quote == quote


def test_a_references_extra_keys_are_dropped_not_fatal():
    # Judges sometimes echo a source property such as supports_candidate onto a reference.
    reference = {"source": "answer", "start_line": 1, "end_line": 1, "supports_candidate": True}
    result = llm.parse_verdict(
        json.dumps(verdict(references=[reference])), rubric(), EvidenceBundle(answer="HELLO"), gating={"one"}
    )
    assert result[0].passed is True and result[0].quotes[0].quote == "HELLO"


def test_judge_supplied_quotes_are_rejected():
    evidence = EvidenceBundle(artifacts={"output": {"status": "present", "text": "first\n  second"}})
    with pytest.raises(ValueError, match="quotes"):
        llm.parse_verdict(
            json.dumps(verdict(quotes=[{"source": "artifact:output", "quote": r"first\n second"}])),
            rubric(),
            evidence,
            gating={"one"},
        )


def test_judge_requests_object_mode_through_the_real_gym_converter():
    from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
    from nemo_gym.models import ResponsesConverter

    configured = NeMoGymResponseCreateParamsNonStreaming(input=[], max_output_tokens=2048, temperature=0, top_p=0.5)
    payload = llm.request_payload(rubric(), EvidenceBundle(answer="HELLO"), configured)
    assert (payload["max_output_tokens"], payload["temperature"], payload["top_p"]) == (2048, 0, 0.5)
    request = NeMoGymResponseCreateParamsNonStreaming.model_validate(payload)
    converted = ResponsesConverter(return_token_id_information=False).responses_to_chat_completion_create_params(
        request
    )
    assert converted.response_format == {"type": "json_object"}


def test_the_judge_user_message_ends_with_a_json_cue_serialized_as_the_offline_regrade_sent_it():
    from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming

    configured = NeMoGymResponseCreateParamsNonStreaming(input=[])
    user = llm.request_payload(rubric(), EvidenceBundle(answer="HÉLLO ✓"), configured)["input"][1]
    # Some hosted routes accept object-mode output only when the user message itself says JSON.
    assert user["role"] == "user" and "JSON" in user["content"]
    sent = json.loads(user["content"])
    assert list(sent) == ["evidence_sources", "rubrics", "_judge_output_format"]
    # An offline regrade added this cue last to earlier requests and serialized them again; requests sent now
    # must match those bytes.
    regraded = {key: value for key, value in sent.items() if key != "_judge_output_format"}
    regraded["_judge_output_format"] = "JSON object with an items array"
    assert json.dumps(regraded, ensure_ascii=False) == user["content"]


def test_the_judge_limit_counts_the_input_as_sent_and_the_artifact_limit_is_separate():
    from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
    from resources_servers.partner_if.evidence import MAX_EVIDENCE_CHARS, EvidenceTooLarge

    configured = NeMoGymResponseCreateParamsNonStreaming(input=[])
    # Many short lines inflate the serialized input far past their text: an artifact within its own limit
    # can still be too large for the judge.
    largest = EvidenceBundle(artifacts={"log": {"status": "present", "text": "x\n" * (MAX_EVIDENCE_CHARS // 2)}})
    assert largest.artifact_text("log")
    with pytest.raises(EvidenceTooLarge, match=f"Judge input exceeds {llm.MAX_JUDGE_INPUT_CHARS:,} characters"):
        llm.request_payload(rubric(), largest, configured)
    # The limit applies to the user message exactly as it is sent.
    small = EvidenceBundle(answer="HELLO")
    sent = len(llm.request_payload(rubric(), small, configured)["input"][1]["content"])
    assert llm.request_payload(rubric(), small, configured, max_input_chars=sent)
    with pytest.raises(EvidenceTooLarge, match=f"Judge input exceeds {sent - 1:,} characters"):
        llm.request_payload(rubric(), small, configured, max_input_chars=sent - 1)
    beyond = EvidenceBundle(artifacts={"log": {"status": "present", "text": "x" * (MAX_EVIDENCE_CHARS + 1)}})
    with pytest.raises(EvidenceTooLarge, match="Artifact log exceeds"):
        beyond.artifact_text("log")
    # An artifact over the limit is not judge evidence for items that do not need it.
    assert "artifact:log" not in llm.payload(rubric(), beyond)["evidence_sources"]


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
    results = llm.parse_verdict(json.dumps(reply), items, EvidenceBundle(answer="HELLO"), gating={"one"})
    assert results[0].passed is True
    assert results[0].quotes[0].quote == "HELLO"
    assert results[1].passed is None
    assert "does not gate reward" in results[1].reason


def test_missing_completion_item_is_unavailable_not_an_if_failure():
    items = rubric()
    items.append(items[0].model_copy(update={"id": "content", "dimension": "completion"}))
    results = llm.parse_verdict(json.dumps(verdict()), items, EvidenceBundle(answer="HELLO"), gating={"one"})
    assert results[0].passed is True and results[1].passed is None


def test_missing_if_item_still_rejects_incomplete_judge_response():
    items = rubric()
    items.append(items[0].model_copy(update={"id": "content", "dimension": "completion"}))
    with pytest.raises(ValueError, match="rubric ID"):
        llm.parse_verdict(json.dumps(verdict(id="content")), items, EvidenceBundle(answer="HELLO"), gating={"one"})


@pytest.mark.parametrize("fault", ["missing", "invalid"])
def test_faulty_prerequisite_verdict_rejects_the_reply_only_when_it_gates_reward(fault):
    # A prerequisite of an IF rubric gates reward (engine.reward_checks), so its verdict is required.
    items = [rubric()[0].model_copy(update={"id": "prerequisite"})] + rubric()
    reply = verdict()
    if fault == "invalid":
        reply["items"].append(verdict(id="prerequisite", passed=None)["items"][0])
    with pytest.raises(ValueError):
        llm.parse_verdict(json.dumps(reply), items, EvidenceBundle(answer="HELLO"), gating={"prerequisite", "one"})
    # The same fault in a check that does not gate reward leaves only that check unscorable.
    results = llm.parse_verdict(json.dumps(reply), items, EvidenceBundle(answer="HELLO"), gating={"one"})
    assert [r.passed for r in results] == [None, True]
