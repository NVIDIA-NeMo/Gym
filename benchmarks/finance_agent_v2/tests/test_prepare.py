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

# Smoke tests for benchmarks/finance_agent_v2/prepare.py. The real source is the
# upstream Vals public CSV, so these tests convert a small in-repo fixture CSV
# instead and pin the emitted JSONL shape.
#
# The field these lean on hardest is `rubric`: app.py::verify judges each of its
# criteria directly, so a mapping error there does not crash — it produces a
# dataset that still runs, costs a full rollout per question, and scores every row
# 0.0. Requires the upstream `finance_agent` package (prepare builds tool schemas
# from it), so run from the resource server's venv.

import csv
import json

import pytest

from benchmarks.finance_agent_v2 import prepare as prepare_module


_RUBRIC = [
    {"operator": "finance_agent_v2_operator", "criteria": "Revenue was $391.0 billion"},
    {"operator": "finance_agent_v2_operator", "criteria": "Market sentiment was positive"},
]

# Raw Vals CSV column names, which differ from the lowercase JSONL keys.
_CSV_ROWS = [
    {
        "Question": "What was revenue in FY2025 and how did the market react?",
        "Question Type": "Financial Analysis",
        "Expert time (mins)": "45",
        "Rubric": json.dumps(_RUBRIC),
    },
    {
        "Question": "What was gross margin in FY2024?",
        "Question Type": "Data Retrieval",
        "Expert time (mins)": "15",
        "Rubric": json.dumps([{"operator": "finance_agent_v2_operator", "criteria": "Gross margin was 75.0%"}]),
    },
]


def _write_csv(path, rows) -> None:
    fieldnames = sorted({k for row in rows for k in row})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def converted(tmp_path):
    src = tmp_path / "public.csv"
    out = tmp_path / "out.jsonl"
    _write_csv(src, _CSV_ROWS)
    count, labeled = prepare_module.convert_file(src, out)
    rows = [json.loads(line) for line in out.open()]
    return count, labeled, rows


class TestConvert:
    def test_emits_expected_fields(self, converted) -> None:
        count, labeled, rows = converted
        assert (count, labeled) == (2, 2)
        assert len(rows) == 2
        for row in rows:
            assert set(row) == {
                "question",
                "question_type",
                "expert_time_mins",
                "expected_answer",
                "rubric",
                "responses_create_params",
            }

    def test_rubric_criteria_are_copied_verbatim(self, converted) -> None:
        """The scoring input. A reworded criterion is a silently different benchmark."""
        _, _, rows = converted
        assert json.loads(rows[0]["rubric"]) == _RUBRIC

    def test_upper_and_lowercase_source_keys_both_map(self, tmp_path) -> None:
        # Raw CSV headers ("Question"/"Rubric") and JSONL keys ("question"/"rubric")
        # must land in the same output fields.
        src = tmp_path / "labeled.jsonl"
        src.write_text(
            json.dumps({"question": "What was revenue?", "rubric": json.dumps(_RUBRIC)}) + "\n",
            encoding="utf-8",
        )
        count, labeled = prepare_module.convert_file(src, tmp_path / "out.jsonl")
        row = json.loads((tmp_path / "out.jsonl").read_text())
        assert (count, labeled) == (1, 1)
        assert json.loads(row["rubric"]) == _RUBRIC

    def test_prompts_and_tools_come_from_upstream(self, converted) -> None:
        _, _, rows = converted
        params = rows[0]["responses_create_params"]
        # Only input + tools: no sampling params are baked in, so the policy runs at
        # whatever the model config specifies.
        assert set(params) == {"input", "tools"}
        assert [m["role"] for m in params["input"]] == ["system", "user"]
        assert _CSV_ROWS[0]["Question"] in params["input"][1]["content"]
        # Upstream's VALID_TOOLS plus the terminating tool, and nothing else: the
        # tool set is imported from the installed package, not listed here.
        names = {t["name"] for t in params["tools"]}
        assert names == set(prepare_module.VALID_TOOLS) | {"submit_final_result"}

    def test_expected_answer_is_derived_from_rubric(self, converted) -> None:
        # Reference-only now, but still expected to enumerate every criterion.
        _, _, rows = converted
        for criterion in _RUBRIC:
            assert criterion["criteria"] in rows[0]["expected_answer"]


class TestSourceDriftGuard:
    def test_partial_rubric_loss_fails_loudly(self, tmp_path) -> None:
        """A renamed upstream column must not silently yield unscorable rows."""
        rows = [dict(r) for r in _CSV_ROWS]
        rows[0]["RubricRenamed"] = rows[0].pop("Rubric")
        src = tmp_path / "drifted.csv"
        _write_csv(src, rows)

        with pytest.raises(ValueError, match="have no rubric"):
            prepare_module.convert_file(src, tmp_path / "out.jsonl")

    def test_question_only_source_is_allowed(self, tmp_path) -> None:
        # public.txt is a legitimate unlabeled dry-run source: no rubric anywhere,
        # so there is no drift to detect.
        src = tmp_path / "public.txt"
        src.write_text("What was revenue?\nWhat was gross margin?\n", encoding="utf-8")

        count, labeled = prepare_module.convert_file(src, tmp_path / "out.jsonl")
        assert (count, labeled) == (2, 0)

    def test_dataset_csv_is_pinned_to_a_commit(self) -> None:
        """Not a branch: the criteria in this CSV are the scoring input, so `main`
        would let an upstream edit move scores between two runs of identical code."""
        assert "/main/" not in prepare_module.CSV_URL
        assert prepare_module._UPSTREAM_SHA in prepare_module.CSV_URL
        assert len(prepare_module._UPSTREAM_SHA) == 40

    def test_dataset_pin_matches_the_installed_tools(self) -> None:
        # The tool code and the question set come from the same repo; if they came
        # from different commits, a question could reference a tool behavior that
        # is not what is installed.
        requirements = (
            prepare_module.Path("resources_servers/finance_agent_v2/requirements.txt").read_text().splitlines()
        )
        pins = [line for line in requirements if "finance-agent-v2.git@" in line]
        assert len(pins) == 1, f"expected one finance-agent-v2 pin, found {pins}"
        assert prepare_module._UPSTREAM_SHA in pins[0]
