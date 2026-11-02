# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from benchmarks.gdpval import prepare as prepare_module
from benchmarks.gdpval import prepare_harness_neutral
from benchmarks.gdpval.prepare import reference_files_str
from nemo_gym.prompt import fill_prompt, load_prompt_config


_BENCHMARK = Path(__file__).resolve().parents[3] / "benchmarks" / "gdpval"
_PROMPT = str(_BENCHMARK / "prompts" / "default-stirrup.yaml")


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


_ROW = {
    "task_id": "t-1",
    "sector": "Finance",
    "occupation": "Auditor",
    "prompt": "Audit the ledger.",
    "reference_files": ["reference_files/a/Plan.docx"],
    "reference_file_urls": ["https://hf.test/Plan.docx"],
    "rubric_json": [{"criterion": "clarity", "score": 1}],
    "rubric_pretty": "clarity: 1",
}


@pytest.fixture
def prepared(tmp_path, monkeypatch) -> list[dict]:
    loaded = {}

    def load_dataset(name, split, token):
        loaded.update(name=name, split=split, token=token)
        return [_ROW, {"task_id": "t-2", "prompt": "Bare task."}]

    monkeypatch.setattr("datasets.load_dataset", load_dataset)
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    monkeypatch.setattr(prepare_module, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(prepare_module, "OUTPUT_FPATH", tmp_path / "data" / "out.jsonl")

    output = prepare_module.prepare()

    assert loaded == {"name": "openai/gdpval", "split": "train", "token": "hf_secret"}
    return [json.loads(line) for line in output.read_text().splitlines()]


def test_prepare_writes_one_record_per_task_with_the_verifier_fields(prepared):
    assert prepared[0] == {
        **_ROW,
        "reference_files_str": "- reference_files/a/Plan.docx\n",
        "responses_create_params": {"tools": json.loads((_BENCHMARK / "tools.json").read_text())},
    }


def test_prepare_defaults_the_fields_a_task_lacks(prepared):
    record = prepared[1]
    assert record.pop("responses_create_params")["tools"]
    assert record == {
        "task_id": "t-2",
        "sector": "",
        "occupation": "",
        "prompt": "Bare task.",
        "reference_files": [],
        "reference_files_str": "None",
        "reference_file_urls": [],
        "rubric_json": {},
        "rubric_pretty": "",
    }


@pytest.mark.parametrize("prompt_file", ["default-stirrup.yaml", "harness-neutral.yaml"])
def test_every_benchmark_prompt_renders_a_prepared_record(prepared, prompt_file):
    config = load_prompt_config(str(_BENCHMARK / "prompts" / prompt_file))

    content = fill_prompt(config, prepared[0])[0]["content"]

    assert "- reference_files/a/Plan.docx" in content
    assert "<task>\nSector: Finance\nOccupation: Auditor\n\nAudit the ledger.\n</task>" in content
    assert "{" not in content


def test_harness_neutral_rows_are_the_benchmark_rows_without_the_tool_list(prepared, tmp_path, monkeypatch):
    monkeypatch.setattr(prepare_harness_neutral, "OUTPUT_FPATH", tmp_path / "data" / "harness_neutral.jsonl")

    output = prepare_harness_neutral.prepare()

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows == [{key: value for key, value in row.items() if key != "responses_create_params"} for row in prepared]
