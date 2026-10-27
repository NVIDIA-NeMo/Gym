# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The judge's system prompt, verdict parsing, voting across judges, and the code gates."""

import json
from typing import Any

import pytest

from resources_servers.agentic_turn_simulation.items import MALFORMED_ARGS_KEY
from resources_servers.agentic_turn_simulation.judge import (
    RUBRIC_FILES,
    RUBRIC_SEPARATOR,
    argument_gate,
    build_system_prompt,
    call_in_text,
    code_gate,
    combine,
    correct_ladder,
    evaluation_schema,
    load_prompt,
    parse_verdict,
    usable,
)
from resources_servers.agentic_turn_simulation.tests.support import VERDICT, raw_call, raw_output


GOOD = dict(VERDICT)
BAD = dict(GOOD, correct="no", blocked_by="model")
TOOLS = ["bash", "read_file", "write_file"]
MARKERS = (MALFORMED_ARGS_KEY,)  # the malformed-argument keys the gate is given


def _v(**kw: Any) -> dict[str, Any]:
    return dict(GOOD, **kw)


class TestSystemPrompt:
    def test_prompts_and_schema_load(self) -> None:
        assert load_prompt("scope-turn.md").startswith("The part you are evaluating")
        assert load_prompt("judge-system-sandbox.md") == load_prompt("judge-system-sandbox.md").strip()
        schema = evaluation_schema()
        assert schema["additionalProperties"] is False
        assert {"addresses_concern", "correct", "blocked_by", "discard"} <= set(schema["required"])

    def test_assembled_in_the_fixed_layout(self) -> None:
        tools = [{"name": "bash", "description": "Run a shell command.", "parameters": {"type": "object"}}]
        prompt = build_system_prompt(tools, template="judge-system-sandbox.md")
        assert "$" + "scope_note" not in prompt and "$" + "domain_policy" not in prompt
        assert "$" + "tool_definitions" not in prompt and "$" + "evaluation_schema" not in prompt
        assert "agent's WHOLE TURN in response to the user" in prompt
        assert '<tool name="bash">\nRun a shell command.\nparameters: {"type": "object"}\n</tool>' in prompt
        assert prompt.rstrip().endswith(json.dumps(evaluation_schema()))

    def test_criteria_are_the_four_rubrics_in_the_fixed_order(self) -> None:
        prompt = build_system_prompt(None, template="judge-system-sandbox.md")
        criteria = prompt.split("<criteria>\n", 1)[1].split("\n</criteria>", 1)[0]
        assert criteria == RUBRIC_SEPARATOR.join(load_prompt(f) for f in RUBRIC_FILES)
        assert RUBRIC_FILES == [
            "rubric-steerability.md",
            "rubric-correctness.md",
            "rubric-discard.md",
            "rubric-request.md",
        ]
        assert "(tool definitions omitted)" in prompt

    def test_correctness_rubric_checks_claims_against_evidence(self) -> None:
        prompt = build_system_prompt([], template="judge-system-sandbox.md")
        # The correctness rubric checks the claims against the evidence for whole turns.
        assert "Check the agent's claims against the evidence." in prompt

    def test_correctness_rubric_asks_for_a_retry_after_a_transient_error(self) -> None:
        prompt = build_system_prompt([], template="judge-system-sandbox.md")
        # Quitting at a transient error is the agent's doing, and only a failure that persists is the environment's.
        assert "transient error (a timeout, an HTTP 5xx, a generic try-again error) calls for a retry" in prompt
        assert "a failed call it could retry (unchanged after\n  a transient error, or with corrected input)" in prompt
        assert "`infra` - the environment failed under the agent and kept failing." in prompt
        assert "A transient error the agent never retried is not `infra`: stopping at it is `model`." in prompt

    def test_correctness_rubric_fails_avoidable_stops(self) -> None:
        prompt = build_system_prompt([], template="judge-system-sandbox.md")
        # Stopping is only correct once the cheap routes are used up: a blocker report with an untried way forward,
        # a re-ask after a go-ahead, and a question with an obvious default all fail.
        assert "while the conversation still\n  offered an untried way forward" in prompt
        assert "asked again instead of acting after the user had already given the go-ahead" in prompt
        assert "it stopped to ask when it could have started" in prompt
        assert "it delivers part of what was asked (65 of 300 rows" in prompt
        assert prompt.index("Judge STEERABILITY") < prompt.index("Judge whether the agent took the CORRECT ACTION")
        assert (
            prompt.index("CORRECT ACTION") < prompt.index("FAIRLY JUDGED") < prompt.index("Judge the user's REQUEST")
        )


class TestParseVerdict:
    @pytest.mark.parametrize("text", [None, "", "no json", '{"correct": "yes"}', "[1, 2]"])
    def test_nothing_recoverable(self, text: str | None) -> None:
        assert parse_verdict(text) is None

    def test_plain_and_fenced_json(self) -> None:
        assert parse_verdict(json.dumps(GOOD)) == GOOD
        assert parse_verdict("```json\n" + json.dumps(GOOD) + "\n```") == GOOD
        assert parse_verdict("```\n" + json.dumps(BAD) + "\n```") == BAD

    @pytest.mark.parametrize(
        "text",
        [
            '<think>maybe {"addresses_concern": "no"}</think>' + json.dumps(GOOD),
            '<thinking>{"addresses_concern": "no"}</thinking>\n' + json.dumps(GOOD),
            'stray reasoning {"addresses_concern": "no"} </think>' + json.dumps(GOOD),
        ],
        ids=["think", "thinking", "unopened-think"],
    )
    def test_reasoning_tags_are_dropped(self, text: str) -> None:
        assert parse_verdict(text) == GOOD

    def test_envelope_is_unwrapped(self) -> None:
        assert parse_verdict(json.dumps({"command": GOOD})) == GOOD
        assert parse_verdict(json.dumps({"x": 1, "audit": BAD})) == BAD
        assert parse_verdict(json.dumps({"x": {"y": 1}})) is None

    def test_object_inside_prose(self) -> None:
        assert parse_verdict("Here is my verdict:\n" + json.dumps(GOOD) + "\nThanks.") == GOOD

    def test_verdict_fields_that_are_not_strings_are_dropped(self) -> None:
        verdict = _v(addresses_concern=True, discard=False, request_questionable=["ok"], blocked_by=None, extra=[1])
        dropped = ("addresses_concern", "discard", "request_questionable")
        expected = {k: v for k, v in verdict.items() if k not in dropped}
        assert parse_verdict(json.dumps(verdict)) == expected  # null and non-verdict keys are kept
        # Dropped and null fields leave the verdict short of the schema: it pays nothing.
        assert not usable(expected) and combine([expected])["n_usable"] == 0

    def test_truncated_output_is_salvaged(self) -> None:
        text = (
            '{"analysis": "the agent ran it", "addresses_concern": "yes", "correct": "no", "blocked_by": "infra", "rea'
        )
        salvaged = parse_verdict(text)
        assert salvaged == {
            "analysis": "the agent ran it",
            "addresses_concern": "yes",
            "correct": "no",
            "blocked_by": "infra",
            "truncated": True,
        }
        assert not usable(salvaged)  # fields are missing: the judge is retried

    def test_a_salvage_with_every_field_is_usable(self) -> None:
        # Cut off after the last field, before the closing brace.
        salvaged = parse_verdict(json.dumps(BAD)[:-1])
        assert salvaged == {**BAD, "truncated": True}
        assert usable(salvaged) and correct_ladder(salvaged) == "no"

    def test_salvage_needs_the_required_key(self) -> None:
        assert parse_verdict('{"analysis": "x", "correct": "yes", "blocked_by": "none"') is None


_MISSING = object()


def _invalid_field_cases() -> list[Any]:
    """For each field the verdict schema requires: the field missing, null, a boolean, and a value the schema does
    not allow (outside its enum, or not a string where it has none)."""
    schema = evaluation_schema()
    cases = []
    for key in schema["required"]:
        enum = schema["properties"][key].get("enum")
        for label, value in (
            ("missing", _MISSING),
            ("null", None),
            ("boolean", True),
            ("invalid", "maybe" if enum else 1),
        ):
            cases.append(pytest.param(key, value, id=f"{key}-{label}"))
    return cases


class TestUsable:
    @pytest.mark.parametrize("key, value", _invalid_field_cases())
    def test_a_missing_or_invalid_required_field_is_unusable(self, key: str, value: Any) -> None:
        verdict = {k: v for k, v in GOOD.items() if k != key} if value is _MISSING else _v(**{key: value})
        assert not usable(verdict) and correct_ladder(verdict) is None
        # It never takes part in the vote: alone it leaves no usable verdict, beside a valid one it is ignored.
        assert combine([verdict])["n_usable"] == 0 and combine([verdict])["correct_ladder"] is None
        assert combine([verdict, BAD])["n_usable"] == 1 and combine([verdict, BAD])["correct_ladder"] == "no"

    def test_every_value_the_schema_allows_is_usable(self) -> None:
        schema = evaluation_schema()
        assert set(schema["required"]) == set(GOOD)
        for key in schema["required"]:
            for value in schema["properties"][key].get("enum", ["any text"]):
                assert usable(_v(**{key: value})), (key, value)

    def test_extra_keys_do_not_matter(self) -> None:
        assert usable(_v(truncated=True, note=[1]))

    def test_correct_yes_without_a_valid_steerability_pays_nothing(self) -> None:
        # `correct: yes` alone is not a vote: steerability gates it, so it must be a valid answer too.
        for steer in (None, "", "YES"):
            assert correct_ladder(_v(addresses_concern=steer)) is None


class TestLadder:
    @pytest.mark.parametrize(
        "verdict, ladder",
        [
            (None, None),
            ({}, None),
            (GOOD, "yes"),
            (BAD, "no"),
            (_v(addresses_concern="no"), "no"),  # did not engage: not correct, whatever `correct` says
            (_v(addresses_concern="no", correct="no", blocked_by="model"), "no"),
            (_v(correct="maybe"), None),
            (_v(addresses_concern="no", correct="maybe"), None),  # an invalid field makes the whole vote unusable
            ({"addresses_concern": "yes"}, None),
            ({"addresses_concern": "no"}, None),
        ],
    )
    def test_correct_ladder_and_usable(self, verdict, ladder) -> None:
        assert correct_ladder(verdict) == ladder
        assert usable(verdict) is (ladder is not None)


class TestCombine:
    def test_single_judge_is_its_own_verdict(self) -> None:
        out = combine([BAD])
        assert out == {
            "addresses_concern": "yes",
            "correct": "no",
            "request_questionable": "ok",
            "discard": "no",
            "correct_ladder": "no",
            "blocked_by": "model",
            "discard_reason": "none",
            "n_judges": 1,
            "n_usable": 1,
        }

    def test_majority_wins(self) -> None:
        out = combine([GOOD, BAD, _v(correct="no", blocked_by="infra", request_questionable="questionable")])
        assert out["correct"] == "no" and out["correct_ladder"] == "no"
        assert out["request_questionable"] == "ok"
        assert out["addresses_concern"] == "yes"

    @pytest.mark.parametrize(
        "verdicts, axis, expected",
        [
            ([GOOD, BAD], "correct", "no"),
            ([GOOD, BAD], "correct_ladder", "no"),
            ([GOOD, _v(addresses_concern="no")], "addresses_concern", "no"),
            ([GOOD, _v(request_questionable="questionable")], "request_questionable", "ok"),
            ([_v(discard="yes", discard_reason="harness_ui"), GOOD], "discard", "no"),
        ],
    )
    def test_ties_break_against_credit(self, verdicts: list[dict[str, Any]], axis: str, expected: str) -> None:
        assert combine(verdicts)[axis] == expected
        assert combine(list(reversed(verdicts)))[axis] == expected

    def test_failure_attribution_comes_from_judges_that_saw_a_failure(self) -> None:
        # Two judges say correct (blocked_by none), one says infra: the dissent's "none" does not win.
        assert combine([GOOD, GOOD, _v(correct="no", blocked_by="infra")])["blocked_by"] == "infra"
        reasons = [_v(correct="no", blocked_by=b) for b in ("model", "infra", "infra")]
        assert combine(reasons)["blocked_by"] == "infra"

    def test_attribution_tie_without_a_tie_break_takes_the_first(self) -> None:
        out = combine([_v(correct="no", blocked_by="model"), _v(correct="no", blocked_by="infra")])
        assert out["blocked_by"] == "model"

    def test_blocked_by_when_no_judge_gave_a_reason(self) -> None:
        assert combine([GOOD])["blocked_by"] == "none"
        assert combine([_v(correct="no", blocked_by="none")])["blocked_by"] is None
        assert combine([_v(correct="no", blocked_by=None)])["blocked_by"] is None

    def test_discard_reason(self) -> None:
        discarded = [_v(discard="yes", discard_reason=r) for r in ("outside_trace", "harness_ui", "outside_trace")]
        assert combine(discarded)["discard_reason"] == "outside_trace"
        assert combine([GOOD])["discard_reason"] == "none"

    def test_unusable_verdicts_are_ignored(self) -> None:
        out = combine([None, {"addresses_concern": "yes"}, BAD])
        assert (out["n_judges"], out["n_usable"], out["correct_ladder"]) == (3, 1, "no")

    def test_no_usable_verdict(self) -> None:
        out = combine([None, {}])
        assert (out["n_judges"], out["n_usable"]) == (2, 0)
        assert out["correct"] is None and out["correct_ladder"] is None and out["blocked_by"] is None
        assert out["discard_reason"] == "none"


class TestCodeGate:
    def test_clean_turns_pass(self) -> None:
        assert code_gate([], TOOLS, policy_turn=True, malformed_argument_keys=MARKERS) is None
        items = [
            {"type": "message", "role": "assistant", "content": "hi"},
            raw_call("bash", '{"command": "ls"}'),
            raw_output("c1", "{}"),
            raw_call("read_file", ""),  # no arguments at all reads as {}
        ]
        assert code_gate(items, TOOLS, policy_turn=True, malformed_argument_keys=MARKERS) is None

    def test_nonexistent_tool(self) -> None:
        assert (
            code_gate(
                [raw_call("bash"), raw_call("browser_open")], TOOLS, policy_turn=False, malformed_argument_keys=MARKERS
            )
            == "nonexistent_tool"
        )
        assert code_gate(
            [{"type": "function_call", "arguments": "{}"}], TOOLS, policy_turn=False, malformed_argument_keys=MARKERS
        ) == ("nonexistent_tool")

    @pytest.mark.parametrize(
        "arguments",
        ["{bad json", '["ls"]', '"ls"', json.dumps({MALFORMED_ARGS_KEY: "raw text"})],
        ids=["unparseable", "list", "string", "malformed-marker"],
    )
    def test_malformed_arguments_only_when_checked(self, arguments: str) -> None:
        items = [raw_call("bash", arguments)]
        assert code_gate(items, TOOLS, policy_turn=True, malformed_argument_keys=MARKERS) == "malformed_arguments"
        assert code_gate(items, TOOLS, policy_turn=False, malformed_argument_keys=MARKERS) is None


# Calls the policy wrote into its message instead of calling the tool, in the templates' own markup.
LEAKED_CALLS = {
    "xml-parameter": "Saving the file now.\n<function=write_file>\n<parameter=path>\nnotes.md\n</parameter>\n",
    "xml-close": "Which format do you want?\n</parameter>\n</function>\n</tool_call>",
    "xml-unopened": "# Report\n\nAll done.\n</parameter>\n<parameter=path>\n/home/user/report.md\n</parameter>",
    "hermes-json": 'Let me look.\n<tool_call>\n{"name": "bash", "arguments": {"command": "ls"}}\n</tool_call>',
    "glm": "Checking.<tool_call>bash\n<arg_key>command</arg_key>\n<arg_value>ls</arg_value>\n</tool_call>",
    "code-inside-parameter": "<parameter=content>\n```python\nprint(1)\n```\n</parameter>\n<parameter=path>\na.py",
    # A file's content opened a fence it never closed, then the call's closing markup: an open fence hides nothing.
    "after-an-open-fence": "Here it is:\n```html\n<div>hi</div>\n</parameter>\n<parameter=path>\n/home/user/a.html",
    "after-a-closed-span": "Saved to `a.html`.\n</parameter>\n<parameter=path>\na.html\n</parameter>",
}
# Messages that only show or mention markup.
NOT_CALLS = {
    "fenced-example": (
        "Qwen3-Coder emits calls like this:\n\n```xml\n<tool_call>\n<function=bash>\n<parameter=command>\nls\n"
        "</parameter>\n</function>\n</tool_call>\n```\n\nand Hermes like this:\n\n~~~\n<tool_call>\n"
        '{"name": "bash", "arguments": {}}\n</tool_call>\n~~~'
    ),
    "inline-code": "The parser splits on `<parameter=name>` and stops at `</function></tool_call>`.",
    "xml-config": 'Set it in web.xml:\n<parameter name="timeout">30</parameter>\n</parameters>',
    "lone-token": "Summary of the week<tool_call> so far:\n- Monday: 3 tasks</tool_call>\n- Tuesday: 2 tasks",
    "tag-mention": "Models wrap calls in <tool_call> and </tool_call> tags; I did not need any tool for this.",
    "nested-fences": (
        "Paste this into the docs:\n\n````markdown\nThe model writes:\n```xml\n<tool_call>\n<function=bash>\n"
        "<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>\n```\n````\nDone."
    ),
    "double-backtick-span": "The parser stops at `` </function></tool_call> `` and reads `` <parameter=path> ``.",
    "indented-block": "The template emits:\n\n    <parameter=command>\n    ls\n    </parameter>\n\nthen closes the call.",
}


class TestCallInText:
    @pytest.mark.parametrize("text", LEAKED_CALLS.values(), ids=LEAKED_CALLS)
    def test_markup_outside_code_is_a_call(self, text: str) -> None:
        assert call_in_text(text)

    @pytest.mark.parametrize("text", NOT_CALLS.values(), ids=NOT_CALLS)
    def test_code_and_mentions_are_not(self, text: str) -> None:
        assert not call_in_text(text)

    def test_a_policy_message_with_a_call_fails_native_turns_only(self) -> None:
        message = {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Done."}]}
        leaked = dict(message, content=[{"type": "output_text", "text": LEAKED_CALLS["hermes-json"]}])
        gate = {"malformed_argument_keys": MARKERS}
        assert code_gate([message], TOOLS, policy_turn=True, **gate) is None
        assert code_gate([leaked], TOOLS, policy_turn=True, **gate) == "call_in_text"
        assert code_gate([leaked], TOOLS, policy_turn=True, protocol="native", **gate) == "call_in_text"
        assert code_gate([leaked], TOOLS, policy_turn=True, protocol="cline_xml", **gate) is None  # calls are text
        assert code_gate([leaked], TOOLS, policy_turn=False, **gate) is None  # a reference turn
        user = dict(leaked, role="user")
        assert code_gate([user], TOOLS, policy_turn=True, **gate) is None

    def test_the_first_defect_in_the_turn_decides(self) -> None:
        leaked = {"type": "message", "role": "assistant", "content": LEAKED_CALLS["xml-parameter"]}
        invented = raw_call("browser_open")
        gate = {"malformed_argument_keys": MARKERS}
        assert code_gate([leaked, invented], TOOLS, policy_turn=True, **gate) == "call_in_text"
        assert code_gate([invented, leaked], TOOLS, policy_turn=True, **gate) == "nonexistent_tool"


# A strict schema (additionalProperties: false at every level) with nested objects and arrays.
STRICT = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "content": {"type": "string"},
        "mode": {"type": "string", "enum": ["w", "a"]},
        "limit": {"type": "integer", "minimum": 1},
        "options": {
            "type": "object",
            "properties": {
                "encoding": {"type": "string", "pattern": "^[a-z0-9-]+$"},
                "lines": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"n": {"type": "integer"}, "text": {"type": "string"}},
                        "required": ["n"],
                        "additionalProperties": False,
                    },
                },
            },
            "additionalProperties": False,
        },
    },
    "required": ["path", "content"],
    "additionalProperties": False,
}
VALID = {"path": "a.txt", "content": "hi", "options": {"encoding": "utf-8", "lines": [{"n": 1, "text": "x"}]}}


class TestArgumentGate:
    def test_valid_arguments_pass(self) -> None:
        assert argument_gate(VALID, STRICT, gates_missing_arguments=True) is None

    @pytest.mark.parametrize(
        "arguments, gate",
        [
            ({**VALID, "workdir": "/tmp"}, "unknown_argument"),
            ({**VALID, "options": {"encoding": "utf-8", "indent": 2}}, "unknown_argument"),
            ({**VALID, "options": {"lines": [{"n": 1, "txt": "x"}]}}, "unknown_argument"),
            ({**VALID, "content": 5}, "wrong_argument_type"),
            ({**VALID, "limit": True}, "wrong_argument_type"),
            ({**VALID, "options": {"lines": [{"n": "1"}]}}, "wrong_argument_type"),
            ({**VALID, "options": '{"encoding": "utf-8"}'}, "wrong_argument_type"),  # string-encoded JSON is a string
            ({"path": "a.txt"}, "missing_argument"),
            ({**VALID, "options": {"lines": [{"text": "x"}]}}, "missing_argument"),
        ],
        ids=[
            "unknown-top",
            "unknown-nested",
            "unknown-in-array",
            "type-top",
            "bool-is-not-integer",
            "type-in-array",
            "json-string-not-coerced",
            "missing-top",
            "missing-in-array",
        ],
    )
    def test_each_gate_at_any_depth(self, arguments: dict[str, Any], gate: str) -> None:
        assert argument_gate(arguments, STRICT, gates_missing_arguments=True) == gate

    def test_missing_arguments_only_when_the_harness_gates_them(self) -> None:
        missing = {"path": "a.txt", "options": {"lines": [{"text": "x"}]}}
        assert argument_gate(missing, STRICT, gates_missing_arguments=True) == "missing_argument"
        assert argument_gate(missing, STRICT, gates_missing_arguments=False) is None
        # The other gates do not depend on it.
        assert argument_gate({**missing, "extra": 1}, STRICT, gates_missing_arguments=False) == "unknown_argument"
        assert argument_gate({"path": 1}, STRICT, gates_missing_arguments=False) == "wrong_argument_type"

    def test_one_call_with_several_violations_reports_them_in_gate_order(self) -> None:
        assert argument_gate({"path": 1, "extra": 1}, STRICT, gates_missing_arguments=True) == "unknown_argument"
        assert argument_gate({"path": 1}, STRICT, gates_missing_arguments=True) == "wrong_argument_type"

    @pytest.mark.parametrize(
        "arguments",
        [
            {**VALID, "mode": "x"},  # enum
            {**VALID, "limit": 0},  # minimum
            {**VALID, "options": {"encoding": "UTF 8"}},  # pattern
        ],
        ids=["enum", "minimum", "pattern"],
    )
    def test_other_violations_pass(self, arguments: dict[str, Any]) -> None:
        assert argument_gate(arguments, STRICT, gates_missing_arguments=True) is None

    def test_a_value_that_another_branch_accepts_is_not_a_wrong_type(self) -> None:
        schema = {"type": "object", "properties": {"n": {"anyOf": [{"type": "string"}, {"type": "integer"}]}}}
        assert argument_gate({"n": []}, schema, gates_missing_arguments=True) is None

    @pytest.mark.parametrize("additional", [None, True, {}], ids=["unset", "true", "any-schema"])
    def test_undeclared_keys_in_a_lenient_schema_pass(self, additional: Any) -> None:
        schema = {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
        if additional is not None:
            schema["additionalProperties"] = additional
        assert argument_gate({"path": "a", "workdir": "/tmp"}, schema, gates_missing_arguments=True) is None
        assert (
            argument_gate({"path": 1, "workdir": "/tmp"}, schema, gates_missing_arguments=True)
            == "wrong_argument_type"
        )

    @pytest.mark.parametrize(
        "schema",
        [None, {}, "object", ["object"], {"type": "string", "additionalProperties": False}],
        ids=["none", "empty", "string", "list", "non-object-type"],
    )
    def test_a_tool_without_an_object_schema_is_skipped(self, schema: Any) -> None:
        assert argument_gate({"path": 1, "extra": 1}, schema, gates_missing_arguments=True) is None

    def test_a_schema_without_a_type_is_read_as_an_object_schema(self) -> None:
        schema = {"properties": {"path": {"type": "string"}}, "additionalProperties": False}
        assert argument_gate({"path": "a", "extra": 1}, schema, gates_missing_arguments=True) == "unknown_argument"

    @pytest.mark.parametrize(
        "schema",
        [
            {"properties": {"path": {"type": "text"}}},
            {"properties": {"path": {"type": "string", "pattern": "("}}},
            {"required": "path"},
            {"properties": {"path": {"$ref": "#/$defs/missing"}}},
            {"properties": {"path": {"$ref": "https://example.com/path.json"}}},
            {"$schema": 5},
            {"$schema": {"draft": 7}},
            {"$schema": ["http://json-schema.org/draft-07/schema#"]},
        ],
        ids=[
            "unknown-type",
            "bad-regex",
            "required-not-a-list",
            "dangling-ref",
            "remote-ref",
            "schema-uri-int",
            "schema-uri-object",
            "schema-uri-list",
        ],
    )
    def test_a_schema_jsonschema_cannot_use_is_skipped(self, schema: dict[str, Any]) -> None:
        # Usable, the schema would gate the unknown key.
        schema = {"type": "object", "additionalProperties": False, **schema}
        assert argument_gate({"path": 1, "extra": 1}, schema, gates_missing_arguments=True) is None

    def test_a_remote_ref_is_never_fetched(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fetched: list[Any] = []

        def fetch(request: Any, *args: Any, **kwargs: Any) -> None:
            fetched.append(request)
            raise OSError("no network in this test")

        monkeypatch.setattr("urllib.request.urlopen", fetch)
        schema = {
            "type": "object",
            "properties": {"path": {"$ref": "https://json-schema.example/never-fetched.json"}},
            "additionalProperties": False,
        }
        assert argument_gate({"path": 1, "extra": 1}, schema, gates_missing_arguments=True) is None
        assert fetched == []
        # A local $ref still resolves.
        local = {**schema, "$defs": {"p": {"type": "string"}}, "properties": {"path": {"$ref": "#/$defs/p"}}}
        assert argument_gate({"path": 1}, local, gates_missing_arguments=True) == "wrong_argument_type"

    def test_a_missing_key_with_a_default_is_not_gated(self) -> None:
        # Claude Code's schemas list keys the harness fills in as required, with their default.
        task_output = {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "block": {"type": "boolean", "default": True},
                "timeout": {"type": "number", "default": 30000},
            },
            "required": ["task_id", "block", "timeout"],
            "additionalProperties": False,
        }
        assert argument_gate({"task_id": "t1"}, task_output, gates_missing_arguments=True) is None
        assert argument_gate({}, task_output, gates_missing_arguments=True) == "missing_argument"
        assert argument_gate({"block": "yes"}, task_output, gates_missing_arguments=True) == "wrong_argument_type"
        question = {"type": "object", "properties": {"q": {"type": "string"}, "multi": {"default": False}}}
        ask = {
            "type": "object",
            "properties": {"questions": {"type": "array", "items": {**question, "required": ["q", "multi"]}}},
            "required": ["questions"],
        }
        assert argument_gate({"questions": [{"q": "Which?"}]}, ask, gates_missing_arguments=True) is None
        assert argument_gate({"questions": [{"multi": True}]}, ask, gates_missing_arguments=True) == "missing_argument"

    def test_a_recursive_schema_over_very_deep_arguments_is_skipped(self) -> None:
        schema = {"type": "object", "properties": {"a": {"$ref": "#"}}, "additionalProperties": False}
        deep: dict[str, Any] = {}
        node = deep
        for _ in range(3000):
            node["a"] = {}
            node = node["a"]
        node["b"] = 1
        assert argument_gate(deep, schema, gates_missing_arguments=True) is None
        assert argument_gate({"a": {"b": 1}}, schema, gates_missing_arguments=True) == "unknown_argument"

    def test_the_draft_its_schema_names(self) -> None:
        # Draft 7's array form of `items` is invalid under the default 2020-12, so that schema is skipped.
        schema = {"type": "object", "properties": {"pair": {"type": "array", "items": [{"type": "string"}]}}}
        assert argument_gate({"pair": [1]}, schema, gates_missing_arguments=True) is None
        draft7 = {"$schema": "http://json-schema.org/draft-07/schema#", **schema}
        assert argument_gate({"pair": [1]}, draft7, gates_missing_arguments=True) == "wrong_argument_type"


class TestArgumentGatesInATurn:
    SCHEMAS = {"write_file": STRICT, "bash": {"type": "object", "properties": {"command": {"type": "string"}}}}

    def _gate(self, items: list[dict[str, Any]], *, policy_turn: bool = True, **kw: Any) -> str | None:
        return code_gate(
            items, TOOLS, policy_turn=policy_turn, malformed_argument_keys=MARKERS, schemas=self.SCHEMAS, **kw
        )

    def test_the_first_violation_in_the_turn_decides(self) -> None:
        unknown = raw_call("write_file", json.dumps({**VALID, "extra": 1}), "c1")
        wrong = raw_call("bash", json.dumps({"command": ["ls"]}), "c2")
        invented = raw_call("browser_open", "{}", "c3")
        assert self._gate([unknown, wrong, invented]) == "unknown_argument"
        assert self._gate([wrong, unknown]) == "wrong_argument_type"
        assert self._gate([invented, unknown]) == "nonexistent_tool"
        assert self._gate([raw_call("bash", "{bad", "c4"), wrong]) == "malformed_arguments"

    def test_the_existing_gates_come_first_within_a_call(self) -> None:
        marked = json.dumps({MALFORMED_ARGS_KEY: "raw"})  # also an unknown key without the content STRICT requires
        assert self._gate([raw_call("write_file", marked)]) == "malformed_arguments"
        assert self._gate([raw_call("browser_open", json.dumps({"x": 1}))]) == "nonexistent_tool"

    def test_missing_arguments_follow_the_switch(self) -> None:
        items = [raw_call("write_file", json.dumps({"path": "a.txt"}))]
        assert self._gate(items, gates_missing_arguments=True) == "missing_argument"
        assert self._gate(items) is None  # off by default

    def test_tools_without_a_schema_are_not_checked(self) -> None:
        items = [raw_call("read_file", json.dumps({"path": 1, "extra": True}))]  # read_file has no schema here
        assert self._gate(items, gates_missing_arguments=True) is None
        assert code_gate(items, TOOLS, policy_turn=True, malformed_argument_keys=MARKERS) is None

    def test_a_reference_turn_is_not_checked(self) -> None:
        items = [raw_call("write_file", json.dumps({"path": 1, "extra": True}))]
        assert self._gate(items, gates_missing_arguments=True) == "unknown_argument"
        assert self._gate(items, policy_turn=False, gates_missing_arguments=True) is None
