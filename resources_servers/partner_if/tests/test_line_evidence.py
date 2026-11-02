# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for permanent, host-resolved evidence references."""

import json

import pytest

from resources_servers.partner_if.evidence import EvidenceBundle
from resources_servers.partner_if.llm import parse_verdict, payload
from resources_servers.partner_if.tests.test_shared_judge import rubric


def referenced(*refs, passed=True):
    return json.dumps(
        {
            "items": [
                {
                    "id": "one",
                    "passed": passed,
                    "reason": "Grounded decision",
                    "evidence": "Inspected lines",
                    "references": list(refs),
                }
            ]
        }
    )


def ref(source="answer", start=1, end=1):
    return {"source": source, "start_line": start, "end_line": end}


def test_source_context_is_addressable_without_becoming_candidate_behavior():
    bundle = EvidenceBundle(
        context=[{"role": "tool", "content": "[1] Source fact.\n[2] Another fact."}],
        answer="A fact [1].",
    )
    sources = payload(rubric(), bundle)["evidence_sources"]
    assert sources["context:0/content"] == {
        "origin": "context",
        "value_type": "string",
        "container_types": ["object"],
        "supports_candidate": False,
        "lines": [{"number": 1, "text": "[1] Source fact.\n"}, {"number": 2, "text": "[2] Another fact."}],
    }
    assert sources["answer"]["origin"] == "candidate"
    result = parse_verdict(referenced(ref(), ref("context:0/content")), rubric(), bundle, gating={"one"})
    assert [(q.source, q.quote) for q in result[0].quotes] == [
        ("answer", "A fact [1]."),
        ("context:0/content", "[1] Source fact.\n"),
    ]


def test_registry_preserves_nested_tools_documents_and_live_event_arguments():
    bundle = EvidenceBundle(
        instructions="Policy\nKeep it.",
        tools=[{"name": "shell", "parameters": {"type": "object"}}],
        documents={"a/b~c.md": "Read me."},
        context=[{"role": "user", "content": [{"type": "input_text", "text": "Question"}]}],
        answer="Done",
        artifacts={"output": {"status": "present", "text": "FILE"}, "missing": {"status": "absent"}},
        trajectory=[
            {
                "source_index": 7,
                "kind": "tool_call",
                "tool_name": "shell",
                "arguments": {"command": "printf 'HELLO\\n'", "timeout": 10},
            }
        ],
    )
    sources = payload(rubric(), bundle)["evidence_sources"]
    assert sources["context:0/content/0/text"]["lines"][0]["text"] == "Question"
    assert sources["instructions"]["origin"] == "context"
    assert sources["tool:0/name"]["lines"][0]["text"] == "shell"
    assert sources["document:a~1b~0c.md"]["lines"][0]["text"] == "Read me."
    assert sources["event:7/arguments/command"]["origin"] == "candidate"
    assert sources["event:7/arguments/command"]["lines"][0]["text"] == "printf 'HELLO\\n'"
    assert sources["event:7/arguments/timeout"]["lines"][0]["text"] == "10"
    assert sources["event:7/arguments/timeout"]["value_type"] == "number"
    assert sources["event:7/arguments/timeout"]["container_types"] == ["object", "object"]
    assert sources["event:7/arguments/command"]["supports_candidate"] is True
    assert sources["artifact:output"]["lines"][0]["text"] == "FILE"
    assert "artifact:missing" not in sources


def test_reference_resolution_preserves_unicode_markdown_blank_lines_and_crlf():
    text = "**Version\u202f2.0**\r\n\r\n  code\t\nlast"
    results = parse_verdict(referenced(ref(start=1, end=3)), rubric(), EvidenceBundle(answer=text), gating={"one"})
    assert results[0].quotes[0].quote == "**Version\u202f2.0**\r\n\r\n  code\t\n"


def test_nonadjacent_lines_are_resolved_as_separate_original_quotes():
    results = parse_verdict(
        referenced(ref(start=1), ref(start=3, end=3)),
        rubric(),
        EvidenceBundle(answer="First.\nMiddle.\nLast."),
        gating={"one"},
    )
    assert [q.quote for q in results[0].quotes] == ["First.\n", "Last."]


@pytest.mark.parametrize(
    "reference",
    [
        ref("event:1"),
        ref(start=0),
        ref(start=True),
        ref(start="1"),
        ref(end=9),
        ref(start=2, end=1),
        {**ref(), "quote": "rewritten"},
    ],
)
def test_invalid_source_ranges_types_and_invented_quote_fields_are_rejected(reference):
    with pytest.raises(ValueError):
        parse_verdict(referenced(reference), rubric(), EvidenceBundle(answer="HELLO"), gating={"one"})


def test_context_only_references_cannot_support_a_passing_candidate():
    with pytest.raises(ValueError, match="candidate"):
        parse_verdict(
            referenced(ref("context:0/content")),
            rubric(),
            EvidenceBundle(context=[{"content": "HELLO"}], answer="Wrong"),
            gating={"one"},
        )


def test_blank_candidate_line_cannot_support_a_pass():
    with pytest.raises(ValueError, match="candidate"):
        parse_verdict(referenced(ref()), rubric(), EvidenceBundle(answer="\nHELLO"), gating={"one"})


def test_absence_failure_can_reference_an_existing_blank_candidate_line():
    results = parse_verdict(referenced(ref(), passed=False), rubric(), EvidenceBundle(answer="\n"), gating={"one"})
    assert results[0].passed is False
    assert results[0].quotes == []


def test_absence_failure_still_rejects_a_nonexistent_line():
    with pytest.raises(ValueError, match="line range"):
        parse_verdict(referenced(ref(), passed=False), rubric(), EvidenceBundle(answer=""), gating={"one"})


def test_context_event_is_not_new_candidate_behavior():
    bundle = EvidenceBundle(
        answer="Wrong", trajectory=[{"source_index": 2, "kind": "context", "role": "assistant", "text": "HELLO"}]
    )
    assert payload(rubric(), bundle)["evidence_sources"]["event:2/text"]["origin"] == "context"
    with pytest.raises(ValueError, match="candidate"):
        parse_verdict(referenced(ref("event:2/text")), rubric(), bundle, gating={"one"})


def test_duplicate_event_indices_fail_before_any_source_can_be_overwritten():
    bundle = EvidenceBundle(
        trajectory=[
            {"source_index": 3, "kind": "assistant", "text": "A"},
            {"source_index": 3, "kind": "assistant", "text": "B"},
        ]
    )
    with pytest.raises(ValueError, match="Duplicate"):
        payload(rubric(), bundle)


def test_pointer_escaping_keeps_distinct_field_names_distinct():
    bundle = EvidenceBundle(context=[{"a/b": "literal", "a": {"b": "nested"}}], answer="HELLO")
    sources = payload(rubric(), bundle)["evidence_sources"]
    assert sources["context:0/a~1b"]["lines"][0]["text"] == "literal"
    assert sources["context:0/a/b"]["lines"][0]["text"] == "nested"


def test_duplicate_rubric_ids_are_rejected_even_when_all_expected_ids_exist():
    data = json.loads(referenced(ref()))
    data["items"].append(data["items"][0].copy())
    with pytest.raises(ValueError, match="every rubric ID exactly once"):
        parse_verdict(json.dumps(data), rubric(), EvidenceBundle(answer="HELLO"), gating={"one"})


@pytest.mark.parametrize(
    "left,right",
    [
        ({"timeout": 10}, {"timeout": "10"}),
        ({"args": ["x"]}, {"args": {"0": "x"}}),
        ({"flag": False}, {"flag": "false"}),
        ({"args": {}}, {"args": "{}"}),
    ],
)
def test_structurally_distinct_arguments_never_have_identical_judge_payloads(left, right):
    a = EvidenceBundle(trajectory=[{"source_index": 0, "kind": "tool_call", "arguments": left}])
    b = EvidenceBundle(trajectory=[{"source_index": 0, "kind": "tool_call", "arguments": right}])
    assert payload(rubric(), a) != payload(rubric(), b)


@pytest.mark.parametrize("field", ["source_index", "kind", "role", "call_id"])
def test_host_event_metadata_cannot_substitute_for_candidate_behavior(field):
    bundle = EvidenceBundle(
        trajectory=[{"source_index": 0, "kind": "assistant", "role": "assistant", "call_id": "host-1", "text": ""}]
    )
    with pytest.raises(ValueError, match="candidate"):
        parse_verdict(referenced(ref(f"event:0/{field}")), rubric(), bundle, gating={"one"})
