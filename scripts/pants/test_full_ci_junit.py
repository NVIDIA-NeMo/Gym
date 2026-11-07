# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Full CI comparisons must reject omitted work and changed test outcomes."""

import tempfile
import unittest
from pathlib import Path

from full_ci_junit import compare_reports, summarize_reports


class FullCIJUnitTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, lane, xml, name="core.xml"):
        directory = self.root / lane
        directory.mkdir(exist_ok=True)
        (directory / name).write_text(xml)
        return directory

    def test_same_counts_different_cases_are_not_equal(self):
        before = self.write("before", '<testsuite><testcase classname="C" name="a"/></testsuite>')
        after = self.write("after", '<testsuite><testcase classname="C" name="b"/></testsuite>')
        result = compare_reports(summarize_reports(before), summarize_reports(after))
        self.assertFalse(result["equal"])
        self.assertEqual(result["missing"][0]["name"], "a")
        self.assertEqual(result["unexpected"][0]["name"], "b")

    def test_skips_failures_and_duplicate_occurrences_are_preserved(self):
        before = self.write(
            "before",
            '<testsuite><testcase name="a"/><testcase name="a"/><testcase name="b"><skipped/></testcase></testsuite>',
        )
        after = self.write(
            "after", '<testsuite><testcase name="a"/><testcase name="b"><failure/></testcase></testsuite>'
        )
        result = compare_reports(summarize_reports(before), summarize_reports(after))
        self.assertFalse(result["equal"])
        self.assertEqual(sum(row["count"] for row in result["missing"]), 2)
        self.assertEqual(result["unexpected"][0]["outcome"], "failure")

    def test_report_durations_and_order_do_not_change_parity(self):
        before = self.write(
            "before",
            '<testsuites><testsuite><testcase name="a" time="1"/><testcase name="b"/></testsuite></testsuites>',
        )
        after = self.write("after", '<testsuite><testcase name="b"/><testcase name="a" time="10"/></testsuite>')
        self.assertTrue(compare_reports(summarize_reports(before), summarize_reports(after))["equal"])

    def test_expected_failure_and_environment_skip_are_distinct(self):
        before = self.write(
            "before", '<testsuite><testcase name="a"><skipped type="pytest.xfail"/></testcase></testsuite>'
        )
        after = self.write(
            "after", '<testsuite><testcase name="a"><skipped type="pytest.skip"/></testcase></testsuite>'
        )
        result = compare_reports(summarize_reports(before), summarize_reports(after))
        self.assertFalse(result["equal"])
        self.assertEqual(result["missing"][0]["outcome"], "xfailed")

    def test_missing_server_report_fails_even_if_present_reports_match(self):
        directory = self.write("a", '<testsuite><testcase name="a"/></testsuite>', "server_a.xml")
        report = summarize_reports(directory, expected_reports=["server_a.xml", "server_b.xml"])
        self.assertFalse(report["valid"])
        self.assertIn("server_b.xml", report["errors"][0])
        self.assertFalse(compare_reports(report, report)["equal"])

    def test_empty_manifest_is_required_for_routed_away_work(self):
        absent = self.root / "absent"
        self.assertFalse(summarize_reports(absent)["valid"])
        self.assertFalse(summarize_reports(absent, expected_reports=["core.xml"])["valid"])
        self.assertTrue(summarize_reports(absent, expected_reports=[])["valid"])

    def test_unexpected_reports_are_not_silently_pooled(self):
        directory = self.write("a", '<testsuite><testcase name="a"/></testsuite>')
        report = summarize_reports(directory, expected_reports=[])
        self.assertFalse(report["valid"])
        self.assertIn("Unexpected JUnit reports", report["errors"][0])

    def test_malformed_or_empty_reports_preserve_failure(self):
        for value in ("<testsuite", "<testsuite/>", '<testsuite errors="1" tests="0"/>', "<other/>"):
            with self.subTest(xml=value):
                directory = self.write("a", value)
                self.assertFalse(summarize_reports(directory)["valid"])

    def test_module_prefix_separates_identically_named_server_tests(self):
        directory = self.write(
            "a", '<testsuite><testcase classname="resources_servers.a.Test" name="test_app"/></testsuite>', "a.xml"
        )
        self.write(
            "a", '<testsuite><testcase classname="resources_servers.b.Test" name="test_app"/></testsuite>', "b.xml"
        )
        report = summarize_reports(directory)
        self.assertTrue(report["valid"])
        self.assertEqual(len(report["cases"]), 2)


if __name__ == "__main__":
    unittest.main()
