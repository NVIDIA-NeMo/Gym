# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for HelpSteer3 → NeMo Gym JSONL conversion helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


_SCRIPT = (
    Path(__file__).resolve().parent.parent / "resources_servers/genrm_compare/scripts/helpsteer3_to_nemo_gym_jsonl.py"
)


@pytest.fixture(scope="module")
def hs3():
    spec = importlib.util.spec_from_file_location("helpsteer3_to_nemo_gym_jsonl", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.unit
def test_trim_to_last_user(hs3):
    msgs = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
    ]
    assert hs3.trim_to_last_user(msgs) == msgs


@pytest.mark.unit
def test_trim_drops_trailing_assistant(hs3):
    msgs = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
    ]
    assert hs3.trim_to_last_user(msgs) == [{"role": "user", "content": "a"}]


@pytest.mark.unit
def test_trim_no_user_returns_empty(hs3):
    assert hs3.trim_to_last_user([{"role": "assistant", "content": "x"}]) == []


@pytest.mark.unit
def test_html_unescape_in_normalize(hs3):
    ctx = [{"role": "user", "content": "use &lt; here"}]
    out = hs3.parse_context(ctx)
    assert out[0]["content"] == "use < here"


@pytest.mark.unit
def test_row_to_gym_record(hs3):
    row = {
        "id": 42,
        "context": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "next"},
        ],
    }
    rec = hs3.row_to_gym_record(row, 0, agent_ref_name="genrm_simple_agent", dataset_tag="helpsteer3_preference")
    assert rec is not None
    assert rec["id"] == 42
    assert rec["dataset"] == "helpsteer3_preference"
    assert rec["agent_ref"]["name"] == "genrm_simple_agent"
    inp = rec["responses_create_params"]["input"]
    assert inp[-1] == {"role": "user", "content": "next"}
    assert rec["responses_create_params"]["tools"] == []
    assert rec["responses_create_params"]["parallel_tool_calls"] is False


@pytest.mark.unit
def test_context_as_json_string(hs3):
    row = {"context": '[{"role": "user", "content": "from json"}]'}
    rec = hs3.row_to_gym_record(row, 0, agent_ref_name="genrm_simple_agent", dataset_tag="t")
    assert rec["responses_create_params"]["input"] == [{"role": "user", "content": "from json"}]
