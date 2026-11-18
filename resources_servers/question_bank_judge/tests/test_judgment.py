# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from resources_servers.question_bank_judge.judgment import parse_judgment


def test_structured_judgment_roundtrip() -> None:
    for verdict in ["correct", "incorrect", "unresolved"]:
        parsed = parse_judgment(
            json.dumps({"verdict": verdict, "reason_code": "check", "evidence": "A {quoted} } value"})
        )
        assert parsed is not None
        assert parsed.verdict == verdict
        assert parsed.evidence == "A {quoted} } value"


@pytest.mark.parametrize(
    "text",
    [
        "Judgement: yesterday",
        "Judgement: nobody",
        "Judgement: yes\nFinal Verdict: no",
        '{"verdict":"correct","reason_code":"ok","evidence":"ok","verdict":"incorrect"}',
        '{"verdict":"correct","reason_code":"ok","evidence":"ok"}\n{"verdict":',
        '```json\n{"verdict":"correct","reason_code":"ok","evidence":"ok"}\n```',
        'Explanation\n{"verdict":"correct","reason_code":"ok","evidence":"ok"}',
        '{"verdict":"correct","reason_code":"ok","evidence":"ok"} trailing',
        '{"verdict":"correct","reason_code":"ok","evidence":"ok","extra":true}',
        '{"verdict":"yes","reason_code":"ok","evidence":"ok"}',
        '{"verdict":"correct","evidence":"ok"}',
        '{"verdict":"correct","reason_code":"ok","evidence":""}',
        '{"verdict":"correct","reason_code":" ","evidence":"ok"}',
        '{"verdict":"correct","reason_code":"ok","evidence":NaN}',
        '{"verdict":"correct","reason_code":"ok","evidence":"ok","criteria_met":{}}',
        '"correct"',
        "[]",
        "null",
    ],
)
def test_ambiguous_or_malformed_verdicts_are_unresolved(text: str) -> None:
    assert parse_judgment(text) is None
