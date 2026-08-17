# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run state the agent writes beside its deliverables must never reach the judge."""

from pathlib import Path

from responses_api_agents.stirrup_agent.file_reader import (
    IGNORE_FILES,
    convert_deliverables_to_content_blocks,
    is_deliverable,
    read_deliverable_files,
)


def _run_state(d: Path) -> None:
    (d / "finish_params.json").write_text('{"summary": "did the thing"}')
    (d / "history.json").write_text('[{"role": "assistant", "content": "internal reasoning"}]')
    (d / "history.pkl").write_bytes(b"\x80\x04\x95")
    (d / "inprogress_history.json").write_text("[]")
    (d / "metadata.json").write_text('{"_model_speed": {"num_calls": 42}}')
    (d / "log.txt").write_text("2026-01-01 agent started\n")


def _text_of(blocks) -> str:
    return "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text")


def test_run_state_is_not_a_deliverable(tmp_path: Path):
    _run_state(tmp_path)
    for name in IGNORE_FILES:
        p = tmp_path / name
        if p.exists():
            assert not is_deliverable(p), f"{name} would be graded as work product"


def test_real_deliverable_still_is_one(tmp_path: Path):
    (tmp_path / "report.md").write_text("# findings\n")
    assert is_deliverable(tmp_path / "report.md")


def test_run_state_never_reaches_the_judge_blocks(tmp_path: Path):
    d = tmp_path / "repeat_0"
    d.mkdir()
    _run_state(d)
    (d / "report.md").write_text("# real work\n")

    out = _text_of(convert_deliverables_to_content_blocks(str(d)))
    assert "report.md" in out
    for name in ("finish_params.json", "history.json", "metadata.json", "log.txt"):
        assert name not in out, f"{name} was shown to the judge"
    assert "internal reasoning" not in out
    assert "did the thing" not in out


def test_run_state_never_reaches_the_text_extractor(tmp_path: Path):
    _run_state(tmp_path)
    (tmp_path / "report.md").write_text("# real work\n")

    out = read_deliverable_files(str(tmp_path))
    assert "real work" in out
    assert "internal reasoning" not in out
    assert "finish_params.json" not in out


def test_reference_files_directory_is_not_a_deliverable(tmp_path: Path):
    d = tmp_path / "repeat_0"
    d.mkdir()
    (d / "reference_files").mkdir()
    (d / "reference_files" / "given.pdf").write_bytes(b"%PDF-1.4\n")
    (d / "answer.md").write_text("# answer\n")

    out = _text_of(convert_deliverables_to_content_blocks(str(d)))
    assert "answer.md" in out
    assert "given.pdf" not in out, "an input was graded as the model's output"


def test_a_directory_named_like_run_state_is_still_excluded(tmp_path: Path):
    assert not is_deliverable(tmp_path / "reference_files")


def test_office_sidecar_scan_ignores_run_state(tmp_path: Path):
    """A run-state file must not be treated as an Office source when consuming PDFs."""
    d = tmp_path / "repeat_0"
    d.mkdir()
    _run_state(d)
    (d / "Plan.docx").write_bytes(b"PK\x03\x04")
    (d / "Plan.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")

    out = _text_of(convert_deliverables_to_content_blocks(str(d)))
    assert "Plan.docx" in out
    assert "Plan.pdf:" not in out, "the consumed sibling was emitted standalone"


def test_ignore_list_covers_every_file_the_agent_writes(tmp_path: Path):
    """Guard against the set drifting from what the agent actually produces."""
    for name in ("finish_params.json", "history.json", "history.pkl", "metadata.json", "log.txt"):
        assert name in IGNORE_FILES
