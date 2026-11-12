# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ensure scorecards preserve failures and never pool incompatible timing evidence."""

import json
import tempfile
import unittest
from pathlib import Path

from scorecard import write_scorecards


class ScorecardTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)

    def report(self, name, *, platform="Linux-x86_64", python="3.13.14", valid=True, error=None, started_at=None):
        path = self.output / "batches" / "batch" / name / "comparison" / "results.json"
        path.parent.mkdir(parents=True)
        result = {
            "started_at": started_at or "2026-11-04T12:00:00+00:00",
            "base_sha": name,
            "overlay_sha256": "overlay-1234567890",
            "platform": platform,
            "python": python,
            "measurements": [{"case": "cold", "valid": valid, "uv": {"seconds": 10}, "pants": {"seconds": 2}}],
        }
        if error:
            result.update(error=error, measurements=[])
        path.write_text(json.dumps(result))
        (path.parent / "summary.md").write_text(f"Report for {name}\n")
        return path

    def read_scorecard(self):
        written = write_scorecards(self.output)
        self.assertEqual(len(written), 1)
        return written[0].read_text()

    def test_platforms_and_python_versions_remain_separate(self):
        self.report("linux", platform="Linux-x86_64")
        self.report("mac", platform="Darwin-arm64")
        self.report("new-python", platform="Linux-x86_64", python="3.14.0")
        content = self.read_scorecard()
        self.assertIn("| Linux-x86_64 / 3.13.14 | cold | 1 | 10.00 | 2.00 | 5.00x |", content)
        self.assertIn("| Darwin-arm64 / 3.13.14 | cold | 1 | 10.00 | 2.00 | 5.00x |", content)
        self.assertIn("| Linux-x86_64 / 3.14.0 | cold | 1 | 10.00 | 2.00 | 5.00x |", content)
        self.assertIn("Comparisons: 3.", content)

    def test_one_invalid_pair_withholds_entire_group_ratio(self):
        self.report("successful")
        self.report("mismatched", valid=False)
        content = self.read_scorecard()
        self.assertIn("| cold | 2 | 10.00 | 2.00 | withheld: outcome mismatch |", content)
        self.assertNotIn("5.00x", content)
        self.assertIn("successful/comparison/summary.md", content)
        self.assertIn("mismatched/comparison/summary.md", content)

    def test_changed_test_scope_and_dependency_pins_are_not_pooled(self):
        for name, manifest, pins in (
            ("small", ["a.py"], "v1"),
            ("larger", ["a.py", "b.py"], "v1"),
            ("new-pins", ["a.py"], "v2"),
        ):
            path = self.report(name)
            result = json.loads(path.read_text())
            result.update(manifest=manifest, dependency_integrity={"requirements_sha256": pins})
            path.write_text(json.dumps(result))
        content = self.read_scorecard()
        self.assertEqual(content.count("| Linux-x86_64 / 3.13.14 | cold | 1 |"), 3)
        self.assertNotIn("| cold | 3 |", content)

    def test_failed_and_incomplete_reports_are_all_linked(self):
        paths = [
            self.report("successful"),
            self.report("failed", valid=False),
            self.report("incomplete", error="crash"),
        ]
        content = self.read_scorecard()
        self.assertIn("Comparisons: 3. Incomplete: 1.", content)
        for path in paths:
            relative = Path("..") / path.parent.relative_to(self.output) / "summary.md"
            self.assertIn(f"]({relative.as_posix()})", content)
        self.assertNotIn("5.00x", content)

    def test_failure_before_comparison_report_is_kept_with_diagnostic_links(self):
        self.report("successful")
        destination = self.output / "batches" / "20261104T120001000001Z" / "failed-preparation"
        destination.mkdir(parents=True)
        (destination / "sample.json").write_text(
            json.dumps(
                {
                    "sha": "failed-preparation",
                    "parent": "parent",
                    "kind": "core",
                    "status": "failed",
                    "error": "overlay did not apply",
                }
            )
        )
        (destination / "prepare.log").write_text("patch does not apply\n")
        content = self.read_scorecard()
        self.assertIn("failed-preparation", content)
        self.assertIn("../batches/20261104T120001000001Z/failed-preparation/sample.json", content)
        self.assertIn("../batches/20261104T120001000001Z/failed-preparation/prepare.log", content)
        self.assertNotIn("../batches/20261104T120001000001Z/failed-preparation/comparison/summary.md", content)

    def test_calendar_dates_get_distinct_reports(self):
        self.report("first-day")
        self.report("second-day", started_at="2026-11-05T00:00:00+00:00")
        written = write_scorecards(self.output)
        self.assertEqual([path.name for path in written], ["2026-11-04.md", "2026-11-05.md"])
        self.assertNotIn("second-day/comparison/summary.md", written[0].read_text())
        self.assertNotIn("first-day/comparison/summary.md", written[1].read_text())

    def test_empty_results_do_not_fabricate_a_scorecard(self):
        self.assertEqual(write_scorecards(self.output), [])
        self.assertFalse((self.output / "daily").exists())


if __name__ == "__main__":
    unittest.main()
