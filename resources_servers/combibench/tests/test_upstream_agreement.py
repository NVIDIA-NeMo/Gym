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

"""The paired-agreement script's keying, which decides how many comparisons a report counts."""

import json
from pathlib import Path

import pytest

from resources_servers.combibench.scripts.upstream_agreement import index_by_key, require_every_key, row_key


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


class TestRowKey:
    def test_repeats_of_one_problem_are_distinct(self) -> None:
        rows = [{"theorem_name": "imo_2000_p4", "_ng_rollout_index": i} for i in range(3)]
        assert [row_key(row, i) for i, row in enumerate(rows)] == [
            "imo_2000_p4#0",
            "imo_2000_p4#1",
            "imo_2000_p4#2",
        ]

    def test_nameless_row_falls_back_to_its_position(self) -> None:
        assert row_key({}, 7) == "row_7#0"


class TestIndexByKey:
    def test_distinct_keys_are_kept(self) -> None:
        keyed = index_by_key([("a#0", {"n": 0}), ("a#1", {"n": 1})], "rollouts")
        assert keyed == {"a#0": {"n": 0}, "a#1": {"n": 1}}

    def test_missing_rollout_index_fails_closed(self) -> None:
        """Without ``_ng_rollout_index`` every repeat keys the same; keeping the last
        would report 100 comparisons of 1600 rollouts as a complete run."""
        rows = [{"theorem_name": "imo_2000_p4"} for _ in range(16)]
        with pytest.raises(SystemExit) as excinfo:
            index_by_key([(row_key(row, i), row) for i, row in enumerate(rows)], "rollouts")
        message = str(excinfo.value)
        assert "16 rollouts collapsed to 1 keys" in message
        assert "imo_2000_p4#0" in message

    def test_rescore_collision_is_named_by_its_source(self) -> None:
        """A collision in --rescore-with mis-pairs verdicts, so the message has to say which file."""
        with pytest.raises(SystemExit, match="--rescore-with verdicts"):
            index_by_key([("a#0", {}), ("a#0", {})], "--rescore-with verdicts")


class TestRequireEveryKey:
    """A rollout with no verdict to compare against is not an agreement."""

    def test_a_complete_mapping_passes(self) -> None:
        require_every_key(["a#0", "a#1"], {"a#0": {}, "a#1": {}, "b#0": {}}, "--rescore-with verdicts")

    def test_a_missing_verdict_fails_closed(self) -> None:
        """``.get(key, {})`` gave gym_status null and gym_success False, which then
        counted as an agreement on every row upstream also rejected."""
        with pytest.raises(SystemExit) as excinfo:
            require_every_key(["a#0", "a#1", "b#0"], {"a#0": {}}, "--rescore-with verdicts")
        message = str(excinfo.value)
        assert "2 of 3 rollouts have no matching entry in the --rescore-with verdicts" in message
        assert "a#1" in message and "b#0" in message


class TestCommittedReports:
    """The committed artifacts must be the shape the script writes, not a hand-trimmed one."""

    @pytest.mark.parametrize(
        "name", ["upstream_agreement_combibench.json", "upstream_agreement_combibench_with_solution.json"]
    )
    def test_report_shape(self, name: str) -> None:
        report = json.loads((DATA_DIR / name).read_text(encoding="utf-8"))
        assert set(report) == {"summary", "rollouts", "upstream_revision", "lean_server_url", "rows", "rows_note"}
        # `rows` holds the disagreements only; the summary counts every row.
        assert report["rows"] == {}
        assert report["summary"]["disagreements"] == 0
        assert report["summary"]["rows"] == 1600
