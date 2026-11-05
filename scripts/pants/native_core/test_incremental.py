# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Guard actual incremental source boundaries and external-runtime invalidation."""

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from incremental import AUDITED_FILES, declared_files, eligibility
from runtime_fingerprint import external_configuration, fingerprint


class IncrementalTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        for name in AUDITED_FILES:
            self.write(name, "# Audited fixture\n")
        self.policy = self.root / "policy.json"
        self.policy.write_text(
            json.dumps(
                {
                    "audited_sha256": {
                        name: hashlib.sha256((self.root / name).read_bytes()).hexdigest() for name in AUDITED_FILES
                    }
                }
            )
        )
        self.stage()

    def write(self, name: str, value: str) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
        return path

    def stage(self):
        subprocess.run(["git", "add", "."], cwd=self.root, check=True)

    def test_source_change_and_new_ancestor_fixture_return_to_forced(self):
        self.assertTrue(eligibility(self.root, self.policy)["eligible"])
        script = self.root / AUDITED_FILES[1]
        before = script.read_text()
        script.write_text(before + "# Boundary changed\n")
        self.assertFalse(eligibility(self.root, self.policy)["eligible"])
        script.write_text(before)
        self.write("tests/unit_tests/tooling/conftest.py", "# New fixture\n")
        self.assertFalse(eligibility(self.root, self.policy)["eligible"])

    def test_scanner_membership_and_internal_link_target_are_declared(self):
        first = self.write("resources_servers/a/config.yaml", "value: 1\n")
        self.stage()
        self.assertIn(str(first.relative_to(self.root)), declared_files(self.root))
        target = self.write("examples/prompt.txt", "Prompt\n")
        link = self.root / "resources_servers/a/prompt.txt"
        link.symlink_to("../../examples/prompt.txt")
        self.stage()
        self.assertIn(str(target.relative_to(self.root)), declared_files(self.root))
        link.unlink()
        link.symlink_to("/outside-declared-checkout")
        with self.assertRaisesRegex(ValueError, "external/directory"):
            declared_files(self.root)

    def test_runtime_contents_and_negative_configuration_invalidate(self):
        runtime = self.root / "runtime"
        runtime.mkdir()
        library = self.write("runtime/lib.so", "first")
        absent = self.root / "external.yaml"
        before = fingerprint([str(runtime)], [str(absent)])
        library.write_text("second")
        self.assertNotEqual(before["sha256"], fingerprint([str(runtime)], [str(absent)])["sha256"])
        library.write_text("first")
        absent.write_text("content: present\n")
        self.assertNotEqual(before["sha256"], fingerprint([str(runtime)], [str(absent)])["sha256"])

    def test_external_yaml_inheritance_records_missing_and_recursive_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            external = Path(temporary).resolve()
            second = external / "missing.yaml"
            first = external / "first.yaml"
            first.write_text(json.dumps({"defaults": [str(second)]}))
            self.write("resources_servers/a/config.yaml", json.dumps({"defaults": [str(first)]}))
            self.stage()
            self.assertEqual(external_configuration(self.root), {str(first), str(second)})


if __name__ == "__main__":
    unittest.main()
