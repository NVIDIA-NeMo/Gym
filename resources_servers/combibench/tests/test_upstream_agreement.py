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

import pytest

from resources_servers.combibench.scripts.upstream_agreement import (
    build_report,
    index_by_key,
    require_every_key,
    row_key,
)


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


class TestBuildReport:
    """The report shape the README's agreement numbers are read out of.

    The reports themselves are not committed — a resources server's ``data/``
    holds only the example rows, rollouts and metrics — so this asserts the
    shape over a constructed set of per-row records instead of an artifact.
    """

    @staticmethod
    def _rows(agreeing: int, gym_only: int) -> dict[str, dict]:
        rows = {
            f"agree_{i}#0": {
                "upstream_error_type": "SUCCESS",
                "upstream_success": True,
                "gym_status": "success",
                "gym_success": True,
                "agree": True,
            }
            for i in range(agreeing)
        }
        rows.update(
            {
                f"gym_only_{i}#0": {
                    "upstream_error_type": "COMPILE_ERROR",
                    "upstream_success": False,
                    "gym_status": "success",
                    "gym_success": True,
                    "agree": False,
                }
                for i in range(gym_only)
            }
        )
        return rows

    def _build(self, rows: dict[str, dict], *, full_rows: bool = False) -> dict:
        return build_report(rows, rollouts="rollouts.jsonl", lean_server_url="http://lean", full_rows=full_rows)

    def test_a_clean_run_writes_no_rows(self) -> None:
        report = self._build(self._rows(agreeing=3, gym_only=0))
        assert set(report) == {"summary", "rollouts", "upstream_revision", "lean_server_url", "rows", "rows_note"}
        # `rows` holds the disagreements only; the summary counts every row.
        assert report["rows"] == {}
        assert report["summary"]["rows"] == 3
        assert report["summary"]["agreements"] == 3
        assert report["summary"]["disagreements"] == 0

    def test_disagreements_are_kept_and_attributed(self) -> None:
        report = self._build(self._rows(agreeing=2, gym_only=1))
        assert set(report["rows"]) == {"gym_only_0#0"}
        summary = report["summary"]
        assert summary["rows"] == 3
        assert summary["agreements"] == 2
        assert summary["disagreements"] == 1
        assert summary["gym_only"] == ["gym_only_0#0"]
        assert summary["upstream_only"] == []
        assert summary["disagreement_by_gym_status"] == {"success": 1}
        assert summary["gym_status_counts"] == {"success": 3}
        assert summary["upstream_error_type_counts"] == {"SUCCESS": 2, "COMPILE_ERROR": 1}

    def test_full_rows_keeps_every_row_and_drops_the_note(self) -> None:
        rows = self._rows(agreeing=2, gym_only=1)
        report = self._build(rows, full_rows=True)
        assert set(report["rows"]) == set(rows)
        assert "rows_note" not in report
