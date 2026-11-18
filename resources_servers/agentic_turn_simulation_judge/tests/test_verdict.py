# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from resources_servers.agentic_turn_simulation_judge.verdict import (
    correct_ladder,
    parse_lenient,
    parse_strict,
    read_verdict,
    schema_error,
    strip_think,
)


def verdict(**overrides) -> dict:
    base = {
        "analysis": "The user asked for a rename; the agent edited one of three files and claimed it was done.",
        "addresses_concern": "yes",
        "correct": "no",
        "blocked_by": "model",
        "request_questionable": "ok",
        "discard": "no",
        "discard_reason": "none",
        "reason": "Claimed a project-wide rename after editing one file.",
    }
    base.update(overrides)
    return {k: v for k, v in base.items() if v is not ...}


V = verdict()
TEXT = json.dumps(V)


class TestParseStrict:
    @pytest.mark.parametrize(
        "text",
        [
            TEXT,
            f"  \n{TEXT}\n ",
            f"```json\n{TEXT}\n```",
            f"```JSON\n{TEXT}```",
            f"```\n{TEXT}\n```",
            json.dumps(V, indent=2),
            f"<think>Is {{'correct': 'yes'}} right? No.</think>\n{TEXT}",
            f"reasoning that lost its opening tag</think>{TEXT}",
        ],
    )
    def test_one_object_parses(self, text: str) -> None:
        assert parse_strict(text) == (V, None)

    @pytest.mark.parametrize(
        "text,error",
        [
            (None, "empty"),
            ("", "empty"),
            ("   \n", "empty"),
            ("<think>only thinking</think>", "empty"),
            ("```json\n```", "empty"),
            ("The turn is correct.", "no_json"),
            (f"{TEXT}\n{TEXT}", "multiple_objects"),
            (f"{TEXT}{json.dumps(verdict(correct='yes'))}", "multiple_objects"),
            (f"First: {TEXT} and then: {TEXT}", "multiple_objects"),
            (f"Here is my verdict:\n{TEXT}", "bad_json"),
            (f"{TEXT}\nThat is my verdict.", "bad_json"),
            (f"Verdict:\n```json\n{TEXT}\n```", "bad_json"),
            (TEXT[:-25], "bad_json"),  # cut off: never salvaged
            ('{"addresses_concern": "yes", "correct": "yes",}', "bad_json"),
            (json.dumps([V]), "bad_json"),
            ("[" * 100_000 + "]" * 100_000, "no_json"),
            ("{" * 100_000, "bad_json"),
            ('{"a":' * 50_000 + "1" + "}" * 50_000, "bad_json"),  # nesting deeper than the decoder recurses
        ],
    )
    def test_failures_are_classified(self, text, error: str) -> None:
        assert parse_strict(text) == (None, error)

    def test_envelope_is_not_unwrapped(self) -> None:
        parsed, error = parse_strict(json.dumps({"audit": V}))
        assert error is None and parsed == {"audit": V}  # one object; schema validation then rejects it
        assert schema_error(parsed) == "missing_keys"


class TestParseLenient:
    @pytest.mark.parametrize(
        "text",
        [
            TEXT,
            f"```json\n{TEXT}\n```",
            f"Here is my verdict:\n{TEXT}\nThanks.",
            json.dumps({"audit": V}),
            json.dumps({"x": 1, "command": V}),
            f"<think>{json.dumps(verdict(correct='yes'))}</think>{TEXT}",
        ],
    )
    def test_reads_ats_forms(self, text: str) -> None:
        assert parse_lenient(text) == (V, None)

    def test_non_string_verdict_fields_are_dropped(self) -> None:
        parsed, error = parse_lenient(json.dumps(verdict(addresses_concern=True, discard=None, extra=[1])))
        assert error is None
        assert "addresses_concern" not in parsed  # as ATS: a non-string verdict field is dropped
        assert parsed["discard"] is None and parsed["extra"] == [1]

    @pytest.mark.parametrize(
        "text,error",
        [
            ("", "empty"),
            ("```\n```", "empty"),
            ("no verdict at all", "no_json"),
            (TEXT[:-25], "bad_json"),  # ATS would salvage this; never here
            (f"{TEXT}\n{TEXT}", "multiple_objects"),  # ATS would salvage the first verdict's fields
            (json.dumps({"verdict": "yes"}), "missing_keys"),
            ("{" * 100_000, "bad_json"),
        ],
    )
    def test_failures(self, text: str, error: str) -> None:
        assert parse_lenient(text) == (None, error)


class TestSchemaAndLadder:
    def test_valid_verdict(self) -> None:
        assert schema_error(V) is None
        assert schema_error(verdict(extra_key="extras are allowed")) is None

    @pytest.mark.parametrize("missing", ["analysis", "addresses_concern", "correct", "discard_reason", "reason"])
    def test_missing_key(self, missing: str) -> None:
        assert schema_error(verdict(**{missing: ...})) == "missing_keys"

    @pytest.mark.parametrize(
        "overrides",
        [
            {"correct": "partially"},
            {"addresses_concern": "Yes"},
            {"blocked_by": "user"},
            {"request_questionable": "no"},
            {"discard_reason": None},
            {"correct": True},
            {"analysis": ["not", "a", "string"]},
            {"reason": 3},
        ],
    )
    def test_bad_value(self, overrides: dict) -> None:
        assert schema_error(verdict(**overrides)) == "bad_enum"

    @pytest.mark.parametrize(
        "addresses,correct,ladder",
        [("yes", "yes", "yes"), ("yes", "no", "no"), ("no", "yes", "no"), ("no", "no", "no")],
    )
    def test_ladder(self, addresses: str, correct: str, ladder: str) -> None:
        assert correct_ladder({"addresses_concern": addresses, "correct": correct}) == ladder

    def test_ladder_undecided(self) -> None:
        assert correct_ladder(None) is None
        assert correct_ladder({"addresses_concern": "yes"}) is None
        assert correct_ladder({"addresses_concern": "yes", "correct": "maybe"}) is None

    def test_strip_think(self) -> None:
        assert strip_think("<think>a</think>b<thinking>c</thinking>d") == "bd"
        assert strip_think("no tags") == "no tags"


class TestReadVerdict:
    def test_valid(self) -> None:
        read = read_verdict(TEXT)
        assert (read.ladder, read.error, read.verdict) == ("no", None, V)
        assert read.analysis_words == len(V["analysis"].split())

    def test_steerability_no_forces_ladder_no(self) -> None:
        read = read_verdict(json.dumps(verdict(addresses_concern="no", correct="yes", blocked_by="none")))
        assert (read.ladder, read.error) == ("no", None)

    def test_invalid_schema_has_no_ladder(self) -> None:
        read = read_verdict(json.dumps(verdict(blocked_by="nobody")))
        assert (read.ladder, read.error) == (None, "bad_enum")
        assert read.verdict["correct"] == "no"  # still reported

    def test_without_schema_validation_only_the_ladder_counts(self) -> None:
        text = json.dumps({"addresses_concern": "yes", "correct": "yes"})
        assert read_verdict(text).error == "missing_keys"
        read = read_verdict(text, validate_schema=False)
        assert (read.ladder, read.error) == ("yes", None)
        assert read_verdict('{"addresses_concern": "yes"}', validate_schema=False).error == "missing_keys"
        assert read_verdict('{"correct": "perhaps"}', validate_schema=False).error == "bad_enum"

    def test_max_analysis_words(self) -> None:
        words = len(V["analysis"].split())
        assert read_verdict(TEXT, max_analysis_words=words).error is None
        read = read_verdict(TEXT, max_analysis_words=words - 1)
        assert (read.ladder, read.error, read.analysis_words) == (None, "too_long", words)

    def test_too_long_comes_after_schema_errors(self) -> None:
        text = json.dumps(verdict(correct="maybe"))
        assert read_verdict(text, max_analysis_words=1).error == "bad_enum"

    def test_lenient_mode(self) -> None:
        text = f"My verdict follows.\n{TEXT}"
        assert read_verdict(text).error == "bad_json"
        assert read_verdict(text, parse_mode="lenient").ladder == "no"

    def test_parser_exception_is_bad_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import resources_servers.agentic_turn_simulation_judge.verdict as module

        def boom(text: str):
            raise MemoryError("simulated")

        monkeypatch.setattr(module, "parse_strict", boom)
        read = module.read_verdict(TEXT)
        assert (read.verdict, read.ladder, read.error) == (None, None, "bad_json")
