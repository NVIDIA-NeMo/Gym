# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from benchmarks.gdpval.prepare import reference_files_str
from nemo_gym.prompt import fill_prompt, load_prompt_config


_PROMPT = str(Path(__file__).resolve().parents[3] / "benchmarks" / "gdpval" / "prompts" / "default-stirrup.yaml")


def _render(reference_files: list[str]) -> str:
    row = {
        "sector": "Finance",
        "occupation": "Auditor",
        "prompt": "Audit {this}.",
        "reference_files_str": reference_files_str(reference_files),
    }
    return fill_prompt(load_prompt_config(_PROMPT), row)[0]["content"]


def test_reference_files_are_listed_sorted_and_relative_to_the_working_dir():
    prompt = _render(["reference_files/b/Quota.xlsx", "/reference_files/a/Plan.docx"])

    assert (
        "<reference_files>\n- reference_files/a/Plan.docx\n- reference_files/b/Quota.xlsx\n\n</reference_files>"
        in prompt
    )


def test_no_reference_files_renders_none():
    assert "<reference_files>\nNone\n</reference_files>" in _render([])


def test_task_follows_sector_and_occupation():
    assert "<task>\nSector: Finance\nOccupation: Auditor\n\nAudit {this}.\n</task>" in _render([])


def test_prompt_advertises_the_registered_exec_tool_name():
    prompt = _render([])

    assert "`code_exec` tool" in prompt
    assert "run_shell" not in prompt
