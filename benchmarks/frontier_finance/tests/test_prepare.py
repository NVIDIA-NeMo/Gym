# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path

import pytest
import yaml

from benchmarks.frontier_finance import prepare as prepare_module


REPO_ROOT = Path(__file__).resolve().parents[3]


def _rubric(rubric_id: int, text: str, must_have: bool = True) -> dict:
    return {
        "rubric_id": rubric_id,
        "rubric_text": text,
        "must_have": must_have,
        "rubric_type": "Factual Data Extraction",
        "rubric_subtype": "Temporal Information",
        "data_source_type": "sec_filing",
        "data_source_subtype": "10_k",
        "data_source_detailed": ["10_k"],
    }


def _row(query_id: str = "q1", query_date: str = "2025-06-23", rubrics=None) -> dict:
    return {
        "query_id": query_id,
        "query": "What did ACME report for FY2024 revenue?",
        "query_date": query_date,
        "use_cases": ["financial_data_and_modeling"],
        "capabilities": ["simple_retrieval/fact_lookup"],
        "rubrics": rubrics if rubrics is not None else [_rubric(1, "States revenue of $10.0 billion.")],
    }


def _write(path, rows) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _convert(tmp_path, rows):
    src, out = tmp_path / "src.jsonl", tmp_path / "out.jsonl"
    _write(src, rows)
    counts = prepare_module.convert_file(src, out)
    return counts, [json.loads(line) for line in out.open()]


class TestConvert:
    def test_maps_fields_and_counts(self, tmp_path) -> None:
        rubrics = [_rubric(1, "A."), _rubric(2, "B.", must_have=False)]
        (queries, total), rows = _convert(tmp_path, [_row(rubrics=rubrics), _row("q2")])
        assert (queries, total) == (2, 3)
        row = rows[0]
        assert set(row) == {
            "query_id",
            "question",
            "as_of_date",
            "use_cases",
            "capabilities",
            "rubrics",
            "responses_create_params",
        }
        assert row["query_id"] == "q1"
        assert row["as_of_date"] == "2025-06-23"
        assert row["use_cases"] == ["financial_data_and_modeling"]

    def test_rubrics_copied_verbatim(self, tmp_path) -> None:
        rubrics = [_rubric(1, "A."), _rubric(7, "B.", must_have=False)]
        _, rows = _convert(tmp_path, [_row(rubrics=rubrics)])
        assert rows[0]["rubrics"] == rubrics

    def test_prompt_uses_query_date(self, tmp_path) -> None:
        _, rows = _convert(tmp_path, [_row(query_date="2024-10-31")])
        system, user = rows[0]["responses_create_params"]["input"]
        assert "as if the current date is October 31, 2024." in system["content"]
        assert "March 1, 2026" not in system["content"]
        assert "What did ACME report for FY2024 revenue?" in user["content"]

    def test_prompt_states_limits_after_submit_instruction(self, tmp_path) -> None:
        _, rows = _convert(tmp_path, [_row()])
        system = rows[0]["responses_create_params"]["input"][0]["content"]
        assert f"{prepare_module.SUBMIT_SENTENCE}\n\n{prepare_module.TOOL_LIMITS_PARAGRAPH}" in system
        assert "(None for no limit): 200\n" in system
        assert "(None for no limit): 300\n" in system

    def test_prompt_limits_match_server_config(self) -> None:
        config_path = REPO_ROOT / "resources_servers" / "frontierfinance" / "configs" / "frontierfinance.yaml"
        config = yaml.safe_load(config_path.read_text())
        server = config["frontier_finance_resources_server"]["resources_servers"]["frontierfinance"]
        assert server["max_tool_calls"] == prepare_module.MAX_TOOL_CALLS
        assert server["max_rollout_time_seconds"] == prepare_module.MAX_TOOL_CALL_SECONDS

    def test_tools_match_finance_agent_v2(self, tmp_path) -> None:
        _, rows = _convert(tmp_path, [_row()])
        names = [t["name"] for t in rows[0]["responses_create_params"]["tools"]]
        assert "edgar_search" in names and "submit_final_result" in names

    def test_missing_date_sentence_fails(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(prepare_module, "SYSTEM_PROMPT", "You are a financial agent.")
        with pytest.raises(ValueError, match="no longer contains"):
            _convert(tmp_path, [_row()])

    @pytest.mark.parametrize(
        "row",
        [
            _row(rubrics=[]),
            {**_row(), "query_date": ""},
            {**_row(), "query": " "},
            _row(rubrics=[{"rubric_id": 1, "rubric_text": ""}]),
            _row(rubrics=[{"rubric_text": "A."}]),
            _row(query_date="23/06/2025"),
        ],
    )
    def test_unusable_rows_fail(self, tmp_path, row) -> None:
        with pytest.raises(ValueError):
            _convert(tmp_path, [row])

    def test_duplicate_query_id_fails(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="duplicate query_id"):
            _convert(tmp_path, [_row("dup"), _row("dup")])


class TestDownload:
    def test_requires_token(self, tmp_path, monkeypatch) -> None:
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
        with pytest.raises(RuntimeError, match="gated"):
            prepare_module.download_source(tmp_path / "src.jsonl")

    def test_url_is_pinned(self) -> None:
        assert prepare_module.HF_REVISION in prepare_module.SOURCE_URL
        assert "/main/" not in prepare_module.SOURCE_URL
