# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Guard selection and actual result verification around upstream batching."""

import json
import tempfile
import unittest
from pathlib import Path

from batching import PANTS_VERSION, single_batch_plan, validate_plan, verify_batch_result


class NativeBatchingTest(unittest.TestCase):
    def test_plan_calls_installed_partition_interface_and_preserves_every_address(self):
        calls = []

        def partition(items, **options):
            calls.append((items, options))
            return [items]

        plan = single_batch_plan(
            [["//b:tests", "//a:tests"]],
            pants_version=PANTS_VERSION,
            hash_prefix=lambda address: 8,
            partition=partition,
        )
        self.assertEqual(validate_plan(plan, ["//b:tests", "//a:tests"]), 512)
        self.assertEqual(calls[0][1]["size_target"], 512)
        self.assertEqual(calls[0][1]["size_max"], 1024)
        with self.assertRaisesRegex(ValueError, "selected native target set"):
            validate_plan(plan, ["//a:tests"])
        with self.assertRaisesRegex(ValueError, "selected native target set"):
            validate_plan(plan, ["//a:tests", "//a:tests"])

    def test_incompatible_empty_duplicate_version_or_unverified_batches_fail_closed(self):
        for partitions, version in (
            ([["a"], ["b"]], PANTS_VERSION),
            ([[]], PANTS_VERSION),
            ([["a", "a"]], PANTS_VERSION),
            ([["a"]], "2.31.0"),
        ):
            with self.subTest(partitions=partitions, version=version), self.assertRaises(ValueError):
                single_batch_plan(
                    partitions, pants_version=version, hash_prefix=lambda address: 0, partition=lambda *a, **k: []
                )
        for batches in ([["a"], ["b"]], [["a"]], [["a", "b", "b"]]):
            with self.subTest(batches=batches), self.assertRaisesRegex(ValueError, "did not confirm"):
                single_batch_plan(
                    [["a", "b"]],
                    pants_version=PANTS_VERSION,
                    hash_prefix=lambda address: 0,
                    partition=lambda *a, **k: batches,
                )

    def test_reports_must_prove_one_actual_batch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reports = root / "reports"
            reports.mkdir()
            (reports / "first.xml").write_text("<testsuites/>")
            provenance = root / "test_result_info_report_runid1.json"
            provenance.write_text(json.dumps({"info": {"//a:tests": {"source": "ran"}}}))
            result = verify_batch_result(root, reports, root, ["//a:tests"])
            self.assertEqual(result["actual_pytest_batches"], 1)
            with self.assertRaisesRegex(ValueError, "representative"):
                verify_batch_result(root, reports, root, ["//wrong:tests"])
            for source in ("hit_locally", "hit_remotely", "memoized", None):
                provenance.write_text(json.dumps({"info": {"//a:tests": {"source": source}}}))
                with self.subTest(source=source), self.assertRaisesRegex(ValueError, "did not execute"):
                    verify_batch_result(root, reports, root, ["//a:tests"])
            provenance.write_text(json.dumps({"info": {"//a:tests": {"source": "ran"}}}))
            (reports / "second.xml").write_text("<testsuites/>")
            with self.assertRaisesRegex(ValueError, "exactly one"):
                verify_batch_result(root, reports, root, ["//a:tests"])
            (reports / "second.xml").unlink()
            provenance.write_text(json.dumps({"info": {"a": {}, "b": {}}}))
            with self.assertRaisesRegex(ValueError, "exactly one"):
                verify_batch_result(root, reports, root, ["//a:tests"])


if __name__ == "__main__":
    unittest.main()
