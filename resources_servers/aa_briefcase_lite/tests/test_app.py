# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import ExitStack
from pathlib import Path

from resources_servers.aa_briefcase_lite.app import (
    _pairwise_task_prompt,
    _parse_binary_judgement,
    _requested_filenames,
    _stage_submission,
)


def test_parse_binary_judgement_accepts_json_and_fence() -> None:
    assert _parse_binary_judgement('{"passed": true, "reasoning": "visible evidence"}') == {
        "passed": True,
        "reasoning": "visible evidence",
    }
    assert _parse_binary_judgement('```json\n{"passed": false, "reasoning": "missing"}\n```') == {
        "passed": False,
        "reasoning": "missing",
    }


def test_parse_binary_judgement_rejects_non_boolean_and_missing_reasoning_type() -> None:
    assert _parse_binary_judgement('{"passed": 1, "reasoning": "no"}') is None
    assert _parse_binary_judgement('{"passed": true, "reasoning": []}') is None
    assert _parse_binary_judgement("not json") is None


def test_requested_filenames_splits_and_deduplicates() -> None:
    checks = [
        {"taskdoer_output_file": "market_overview.pdf, market_overview.tex"},
        {"taskdoer_output_file": "market_overview.pdf"},
    ]
    assert _requested_filenames(checks) == ["market_overview.pdf", "market_overview.tex"]


def test_stage_submission_is_read_only_view_and_records_missing(tmp_path: Path) -> None:
    artifact = tmp_path / "report.pdf"
    artifact.write_bytes(b"pdf")
    with ExitStack() as stack:
        stage, missing = _stage_submission(str(tmp_path), ["report.pdf", "notes.txt"], stack)
        assert (stage / "report.pdf").is_symlink()
        assert (stage / "report.pdf").read_bytes() == b"pdf"
        assert missing == ["notes.txt"]


def test_pairwise_prompt_is_criterion_specific_and_source_blind() -> None:
    prompt = _pairwise_task_prompt(
        "Make a deck.",
        {
            "check_description": "Prefer stronger analysis.",
            "score_1_criteria": "A wins when its synthesis is stronger.",
            "score_0_criteria": "B wins when its synthesis is stronger.",
        },
    )
    assert "Make a deck." in prompt
    assert "Prefer stronger analysis." in prompt
    assert "A wins" in prompt and "B wins" in prompt
    assert "external source files" in prompt
