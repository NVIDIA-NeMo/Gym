# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from benchmarks.gdpval import prepare as gdpval_prepare
from benchmarks.gdpval import prepare_harness_neutral
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


_HF_ROW = {
    "task_id": "t1",
    "sector": "Finance",
    "occupation": "Auditor",
    "prompt": "Audit this.",
    "reference_files": ["reference_files/a/Plan.docx"],
    "reference_file_urls": ["https://example.com/Plan.docx"],
    "rubric_json": {"items": []},
    "rubric_pretty": "rubric",
}


def _prepare_rows(monkeypatch, tmp_path: Path, module) -> list[dict]:
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=lambda *args, **kwargs: [_HF_ROW]))
    output = tmp_path / "rows.jsonl"
    monkeypatch.setattr(module, "OUTPUT_FPATH", output)
    assert module.prepare() == output
    return [json.loads(line) for line in output.read_text().splitlines()]


def test_benchmark_rows_carry_the_task_tools(monkeypatch, tmp_path):
    (row,) = _prepare_rows(monkeypatch, tmp_path, gdpval_prepare)

    tools = json.loads(gdpval_prepare.TOOLS_FPATH.read_text())
    assert row["responses_create_params"] == {"tools": tools}
    assert row["reference_files_str"] == "- reference_files/a/Plan.docx\n"


def test_harness_neutral_rows_match_the_benchmark_rows_without_the_tool_list(monkeypatch, tmp_path):
    (benchmark_row,) = _prepare_rows(monkeypatch, tmp_path, gdpval_prepare)
    (row,) = _prepare_rows(monkeypatch, tmp_path, prepare_harness_neutral)

    assert "responses_create_params" not in row
    assert row == {key: value for key, value in benchmark_row.items() if key != "responses_create_params"}
