# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from benchmarks.aa_briefcase_lite.prepare import (
    AGENT_MAX_TURNS,
    TOOL_NAMES,
    _dataset_path,
    _render,
    build_prompts,
    task_tools,
)


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    for directory in ("prompts", "tasks", "summary_docs/week_1", "source_files/shared", "source_files/week"):
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
    (tmp_path / "prompts/eval_system.txt").write_text(
        "turns={max_turns}; finish={finish_tool_name}; abandon={abandon_task_finish}", encoding="utf-8"
    )
    (tmp_path / "prompts/eval_submission.txt").write_text(
        "{scenario_overview}\n{week_overview}\n{task}\n{expected_output_filenames}\n"
        "{finish_tool_name}\n{abandon_task_finish}",
        encoding="utf-8",
    )
    (tmp_path / "summary_docs/scenario_overview.md").write_text("scenario", encoding="utf-8")
    (tmp_path / "summary_docs/week_1/week_1_overview.md").write_text("week", encoding="utf-8")
    (tmp_path / "tasks/w1_t1.md").write_text("task", encoding="utf-8")
    return tmp_path


def metadata() -> dict:
    return {
        "task_id": "w1_t1",
        "week": 1,
        "task_md_path": "tasks/w1_t1.md",
        "deliverable_filenames": ["market_overview.tex", "market_overview.pdf"],
        "scenario_overview_path": "summary_docs/scenario_overview.md",
        "week_overview_path": "summary_docs/week_1/week_1_overview.md",
        "shared_files": [],
        "week_files": [],
    }


def test_official_prompts_are_rendered(dataset: Path) -> None:
    system, user = build_prompts(dataset, metadata())

    assert system == f"turns={AGENT_MAX_TURNS}; finish=finish; abandon=abandon_task_finish"
    assert "scenario\nweek\ntask" in user
    assert "- `market_overview.tex`\n- `market_overview.pdf`" in user


@pytest.mark.parametrize(
    ("field", "value"),
    [("shared_files", ["source_files/week/x"]), ("week_files", ["source_files/shared/x"])],
)
def test_input_paths_outside_their_directory_are_rejected(dataset: Path, field: str, value: list) -> None:
    for relative in value:
        target = dataset / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid"):
        build_prompts(dataset, {**metadata(), field: value})


def test_deliverable_names_cannot_leave_the_working_directory(dataset: Path) -> None:
    with pytest.raises(ValueError, match="Invalid deliverable filename"):
        build_prompts(dataset, {**metadata(), "deliverable_filenames": ["../escape.pdf"]})


def test_paths_cannot_escape_the_dataset(dataset: Path) -> None:
    with pytest.raises(ValueError, match="escapes"):
        _dataset_path(dataset, "../outside.md")


def test_unresolved_prompt_placeholders_are_an_error(tmp_path: Path) -> None:
    template = tmp_path / "template.txt"
    template.write_text("{known} {unknown}", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        _render(template, {"known": "x"})


def test_rows_offer_only_the_two_finish_tools() -> None:
    assert [tool["name"] for tool in task_tools()] == list(TOOL_NAMES)
