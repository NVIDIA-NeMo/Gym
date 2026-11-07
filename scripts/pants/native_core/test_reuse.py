# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ensure the proposed cache boundary fails closed before it is enabled."""

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import reuse


class ReusePolicyTest(unittest.TestCase):
    def test_changed_transitive_fixture_or_new_conftest_rejects_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            files = [
                *reuse.CANDIDATES,
                *reuse.EXPECTED_CONFTESTS,
                *reuse.SOURCE_FILES,
                "nemo_gym/config/global_config.py",
            ]
            hashes = {}
            for name in files:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# audited\n")
                hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest = root / "audit.json"
            manifest.write_text(
                json.dumps(
                    {"candidate_modules": reuse.CANDIDATES, "observed_sha256": hashes, "source_commit": "sample"}
                )
            )
            with patch.object(reuse, "AUDIT_PATH", manifest):
                self.assertTrue(reuse.audit_source(root)["valid"])
                self.assertFalse(reuse.audit_source(root)["reuse_enabled"])
                changed = root / "nemo_gym/config/global_config.py"
                changed.write_text("# new external read\n")
                self.assertEqual(reuse.audit_source(root)["candidates"], [])
                changed.write_text("# audited\n")
                (root / "tests/unit_tests/models/conftest.py").write_text("# new autouse\n")
                self.assertFalse(reuse.audit_source(root)["valid"])

    def test_inventory_includes_new_data_but_excludes_server_git_and_wheel(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            files = [
                *reuse.SOURCE_FILES,
                "nemo_gym/example.py",
                "tests/fixture.json",
                "scripts/ci/core_test_shards.py",
                "resources_servers/example/app.py",
            ]
            for name in files:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("original")
            subprocess.run(["git", "add", "."], cwd=root, check=True)
            before = reuse.reduced_source_inputs(root)
            self.assertNotIn("resources_servers/example/app.py", {p["path"] for p in before})
            (root / "tests/fixture.json").write_text("changed")
            after = reuse.reduced_source_inputs(root)
            self.assertNotEqual(before, after)
            (root / "tests/new.yaml").write_text("new fixture")
            subprocess.run(["git", "add", "tests/new.yaml"], cwd=root, check=True)
            self.assertIn("tests/new.yaml", {p["path"] for p in reuse.reduced_source_inputs(root)})
            (root / "tests/alias").symlink_to("../resources_servers/example", target_is_directory=True)
            subprocess.run(["git", "add", "tests/alias"], cwd=root, check=True)
            with self.assertRaisesRegex(ValueError, "excluded content"):
                reuse.reduced_source_inputs(root)

    def test_runtime_or_network_capability_changes_invalidate_and_missing_evidence_fails(self):
        evidence = {name: "a" * 64 for name in reuse.RUNTIME_DIGEST_FIELDS}
        evidence.update(
            python_implementation="CPython",
            python_version="3.13.14",
            system="Linux",
            machine="x86_64",
            kernel_release="verified-kernel",
            ipv6_bind_available=True,
        )
        original = reuse.runtime_fingerprint(evidence)
        self.assertNotEqual(original, reuse.runtime_fingerprint({**evidence, "ipv6_bind_available": False}))
        self.assertNotEqual(original, reuse.runtime_fingerprint({**evidence, "native_libraries_sha256": "b" * 64}))
        for name in evidence:
            with self.subTest(missing=name), self.assertRaises(ValueError):
                reuse.runtime_fingerprint({key: value for key, value in evidence.items() if key != name})
        for name in reuse.RUNTIME_DIGEST_FIELDS:
            with self.subTest(unverified_digest=name), self.assertRaises(ValueError):
                reuse.runtime_fingerprint({**evidence, name: "mutable-label"})
            with self.subTest(changed_artifact=name):
                self.assertNotEqual(original, reuse.runtime_fingerprint({**evidence, name: "b" * 64}))
        for name in reuse.RUNTIME_VALUE_FIELDS:
            with self.subTest(empty_identity=name), self.assertRaises(ValueError):
                reuse.runtime_fingerprint({**evidence, name: ""})
        for changed in (
            {"ipv6_bind_available": "true"},
            {"python_implementation": "PyPy"},
            {"python_version": "3.14.0"},
            {"python_version": "3.13"},
            {"unknown_probe": "unkeyed"},
        ):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                reuse.runtime_fingerprint({**evidence, **changed})
        with self.assertRaises(ValueError):
            reuse.runtime_fingerprint({**evidence, "runner_image_sha256": "ubuntu-latest"})

    def test_semantic_environment_does_not_inherit_volatile_or_import_configuration(self):
        with patch.dict(
            "os.environ", {"GITHUB_SHA": "changed", "NEMO_GYM_EXTRA_ROOTS": "/external", "PYTEST_PLUGINS": "external"}
        ):
            self.assertEqual(reuse.candidate_environment(), {"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "TZ": "UTC"})
        self.assertFalse(reuse.RESULT_REUSE_ENABLED)


if __name__ == "__main__":
    unittest.main()
