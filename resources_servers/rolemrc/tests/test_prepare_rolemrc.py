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
"""Tests for the dataset integrity guard in ``prepare_rolemrc.py``.

The guard exists because a truncated download or a silently revised upstream
split would shift every score without failing anything — and because the judge
denominators (and therefore ``AvgWeighted``) are derived from this histogram.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


_ROLEMRC_DIR = Path(__file__).resolve().parent.parent
if str(_ROLEMRC_DIR) not in sys.path:
    sys.path.insert(0, str(_ROLEMRC_DIR))

_spec = importlib.util.spec_from_file_location("prepare_rolemrc", _ROLEMRC_DIR / "prepare_rolemrc.py")
prepare_rolemrc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prepare_rolemrc)

_EXPECTED = prepare_rolemrc._EXPECTED_TASK_COUNTS


def _good_rows():
    return [{"task": task} for task, n in _EXPECTED.items() for _ in range(n)]


class TestExpectedHistogram:
    def test_totals_the_published_1400_rows(self) -> None:
        assert sum(_EXPECTED.values()) == 1400

    def test_every_task_has_a_judge_config(self) -> None:
        """A task absent from _EVALUATION_CONFIG would silently score 0 in judge mode."""
        assert set(_EXPECTED) == set(prepare_rolemrc._EVALUATION_CONFIG)

    def test_groups_of_100(self) -> None:
        """8 base tasks of 100 rows, plus a 100-row variant group for 6 of them.

        The variants of a base task (``-refused`` / ``-special-*`` / ``-2nd*``)
        together add up to exactly 100, e.g. 22 + 58 + 20 for
        ``role_related_mrc_answer_with_narration``. 14 groups x 100 = 1400.
        """
        groups: dict[tuple[str, bool], int] = {}
        for task, n in _EXPECTED.items():
            base, _, variant = task.partition("-")
            key = (base, bool(variant))
            groups[key] = groups.get(key, 0) + n
        assert set(groups.values()) == {100}
        assert len(groups) == 14
        assert sum(1 for base, is_variant in groups if not is_variant) == 8


class TestCheckTaskHistogram:
    def test_accepts_the_known_good_split(self) -> None:
        assert prepare_rolemrc.check_task_histogram(_good_rows()) == []

    def test_flags_a_truncated_dataset(self) -> None:
        problems = prepare_rolemrc.check_task_histogram(_good_rows()[:50])
        assert problems
        assert any("expected 100, got" in p for p in problems)

    def test_flags_an_unknown_task(self) -> None:
        problems = prepare_rolemrc.check_task_histogram(_good_rows() + [{"task": "brand_new_task"}])
        assert any("brand_new_task: expected 0, got 1" in p for p in problems)

    def test_flags_a_missing_task(self) -> None:
        rows = [r for r in _good_rows() if r["task"] != "role_related_mrc_answer_with_narration-refused"]
        problems = prepare_rolemrc.check_task_histogram(rows)
        assert any("role_related_mrc_answer_with_narration-refused: expected 22, got 0" in p for p in problems)


class TestJudgeCallCounts:
    def test_matches_the_verified_denominators(self) -> None:
        """These are the denominators the published per-aspect means use.

        Verified against the real ``roleMRC_test.jsonl``. Note ``instruction_priority``
        is 42, not the 84 implied by the reference runs' ``AvgWeighted``.
        """
        counts = prepare_rolemrc.judge_call_counts(_good_rows())
        assert dict(counts) == {
            "knowledge_range": 600,
            "style_compliance": 400,
            "nested_instruction": 158,
            "multi_turn_instruction": 400,
            "instruction_priority": 42,
        }

    def test_more_calls_than_rows(self) -> None:
        """The two knowledge+style tasks fire twice, so 1400 rows -> 1600 calls."""
        rows = _good_rows()
        assert sum(prepare_rolemrc.judge_call_counts(rows).values()) == 1600
        assert len(rows) == 1400

    def test_unknown_task_fires_nothing(self) -> None:
        assert prepare_rolemrc.judge_call_counts([{"task": "nope"}]) == {}


class TestRowTransform:
    def test_normalize_messages_lowercases_roles(self) -> None:
        """The raw dataset ships 'System'/'User'/'Assistant'; upstream lowercases them.

        app.py's judge prompt builder matches on lowercase roles, so a missed
        normalization would silently drop turns from the judge conversation.
        """
        turns = [{"role": "System", "content": "s"}, {"role": "User", "content": "u"}]
        assert prepare_rolemrc._normalize_messages(turns) == [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
        ]

    def test_pick_field_prefers_first_present_and_skips_empty(self) -> None:
        assert prepare_rolemrc._pick_field({"reference": "r", "chosen": "c"}, ("reference", "chosen")) == "r"
        assert prepare_rolemrc._pick_field({"reference": "", "chosen": "c"}, ("reference", "chosen")) == "c"
        assert prepare_rolemrc._pick_field({}, ("reference",)) is None

    def test_row_messages_tolerates_missing_or_bad_field(self) -> None:
        assert prepare_rolemrc._row_messages({}) == []
        assert prepare_rolemrc._row_messages({"question": "not a list"}) == []

    def test_to_task_shape(self) -> None:
        row = {
            "question": [{"role": "System", "content": "sys"}, {"role": "User", "content": "q"}],
            "reference": "gold",
            "task": "role_related_mrc_answer_with_narration-refused",
        }
        task = prepare_rolemrc._to_task(row, "rolemrc_judge_simple_agent")
        assert task["responses_create_params"]["input"][0]["role"] == "system"
        assert task["reference"] == "gold"
        assert task["dimension"] == "instruction_priority"
        assert task["agent_ref"] == {"type": "responses_api_agents", "name": "rolemrc_judge_simple_agent"}

    def test_jsonl_round_trip(self, tmp_path) -> None:
        rows = [{"a": 1}, {"b": "ünicode"}]
        path = tmp_path / "out.jsonl"
        prepare_rolemrc._write_jsonl(path, rows)
        assert prepare_rolemrc._read_jsonl(str(path)) == rows


class TestMain:
    def _fixture(self, tmp_path, rows):
        src = tmp_path / "raw.jsonl"
        src.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        return src

    def _raw_row(self, task):
        return {
            "question": [{"role": "System", "content": "sys"}, {"role": "User", "content": "q"}],
            "reference": "gold",
            "task": task,
        }

    def test_writes_both_splits(self, tmp_path, monkeypatch, capsys) -> None:
        rows = [self._raw_row(t) for t, n in _EXPECTED.items() for _ in range(n)]
        src = self._fixture(tmp_path, rows)
        monkeypatch.setenv("ROLEMRC_LOCAL_JSONL", str(src))
        monkeypatch.setattr(prepare_rolemrc, "_DATA_DIR", tmp_path / "data")
        prepare_rolemrc.main()

        out = capsys.readouterr().out
        assert "judge calls per aspect (1600 over 1400 rows)" in out
        assert "instruction_priority     42" in out
        assert len((tmp_path / "data" / "test.jsonl").read_text().strip().splitlines()) == 1400
        assert len((tmp_path / "data" / "test_judge.jsonl").read_text().strip().splitlines()) == 1400

    def test_aborts_on_dataset_drift(self, tmp_path, monkeypatch) -> None:
        src = self._fixture(tmp_path, [self._raw_row("role_related_mrc_answer_no_narration")])
        monkeypatch.setenv("ROLEMRC_LOCAL_JSONL", str(src))
        monkeypatch.delenv("ROLEMRC_ALLOW_DATASET_DRIFT", raising=False)
        monkeypatch.setattr(prepare_rolemrc, "_DATA_DIR", tmp_path / "data")
        with pytest.raises(SystemExit) as excinfo:
            prepare_rolemrc.main()
        assert "does not match the expected upstream split" in str(excinfo.value)
        assert not (tmp_path / "data").exists()

    def test_drift_override_continues(self, tmp_path, monkeypatch, capsys) -> None:
        src = self._fixture(tmp_path, [self._raw_row("role_related_mrc_answer_no_narration")])
        monkeypatch.setenv("ROLEMRC_LOCAL_JSONL", str(src))
        monkeypatch.setenv("ROLEMRC_ALLOW_DATASET_DRIFT", "1")
        monkeypatch.setattr(prepare_rolemrc, "_DATA_DIR", tmp_path / "data")
        prepare_rolemrc.main()
        assert "continuing: ROLEMRC_ALLOW_DATASET_DRIFT is set" in capsys.readouterr().out
        assert (tmp_path / "data" / "test.jsonl").exists()
