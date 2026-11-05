# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Guard actual incremental source boundaries and external-runtime invalidation."""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from incremental import AUDITED_FILES, LEGACY_MIGRATION_TEST, declared_files, eligibility, select_policy
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

    def test_historical_policy_requires_its_own_audited_code_and_candidate(self):
        historical = json.loads(self.policy.read_text())
        old = AUDITED_FILES[0]
        self.write(LEGACY_MIGRATION_TEST, (self.root / old).read_text())
        (self.root / old).unlink()
        historical["candidate"] = LEGACY_MIGRATION_TEST
        historical["audited_sha256"][LEGACY_MIGRATION_TEST] = historical["audited_sha256"].pop(old)
        controller = self.root / "controller"
        variant = controller / "incremental-policies" / "historical.json"
        variant.parent.mkdir(parents=True)
        (controller / "incremental-policy.json").write_bytes(self.policy.read_bytes())
        variant.write_text(json.dumps(historical))
        self.stage()
        selected, decision = select_policy(self.root, controller)
        self.assertEqual(selected, variant)
        self.assertEqual(decision["candidate"], LEGACY_MIGRATION_TEST)
        self.assertTrue(decision["eligible"])
        self.assertIn(LEGACY_MIGRATION_TEST, declared_files(self.root))
        (self.root / LEGACY_MIGRATION_TEST).write_text("# Changed actual test code\n")
        selected, decision = select_policy(self.root, controller)
        self.assertIsNone(selected)
        self.assertFalse(decision["eligible"])

    def test_dependency_change_requires_fresh_external_read_audit(self):
        self.write("uv.lock", "audited dependencies")
        policy = json.loads(self.policy.read_text())
        policy["uv_lock_sha256"] = hashlib.sha256((self.root / "uv.lock").read_bytes()).hexdigest()
        self.policy.write_text(json.dumps(policy))
        self.assertTrue(eligibility(self.root, self.policy)["eligible"])
        self.write("uv.lock", "new dependency implementation")
        self.assertFalse(eligibility(self.root, self.policy)["eligible"])

    def test_unread_agent_code_and_docs_do_not_invalidate_yaml_scanner(self):
        self.write("responses_api_agents/example/app.py", "# Not imported by this audited test\n")
        self.write("responses_api_agents/example/tests/test_app.py", "# Not this workload\n")
        self.write("responses_api_models/example/README.md", "Unobserved documentation\n")
        self.write("responses_api_agents/example/config.yaml", "value: 1\n")
        self.write("resources_servers/example/custom.entrypoint", "class Example: pass\n")
        self.stage()
        before = declared_files(self.root)
        self.assertNotIn("responses_api_agents/example/app.py", before)
        self.assertNotIn("responses_api_agents/example/tests/test_app.py", before)
        self.assertNotIn("responses_api_models/example/README.md", before)
        self.assertIn("responses_api_agents/example/config.yaml", before)
        self.assertIn("resources_servers/example/custom.entrypoint", before)
        self.write("responses_api_agents/example/new_module.py", "# Also unobserved\n")
        self.stage()
        self.assertEqual(before, declared_files(self.root))
        self.write("responses_api_agents/example/added.yaml", "value: 2\n")
        self.stage()
        self.assertNotEqual(before, declared_files(self.root))

    def test_resource_ast_entrypoint_cannot_escape_declared_tree(self):
        config = "resources_servers/example/config.yaml"
        for entrypoint in ("/outside/implementation.py", "../../outside.py"):
            with self.subTest(entrypoint=entrypoint):
                self.write(config, f"example:\n  resources_servers:\n    example:\n      entrypoint: {entrypoint}\n")
                self.stage()
                with self.assertRaisesRegex(ValueError, "entrypoint escapes"):
                    external_configuration(self.root)
        self.write(config, "example:\n  resources_servers:\n    ../../outside: {}\n")
        self.stage()
        with self.assertRaisesRegex(ValueError, "entrypoint escapes"):
            external_configuration(self.root)
        self.write(config, "example:\n  resources_servers:\n    example:\n      entrypoint: custom.entrypoint\n")
        self.write("resources_servers/example/custom.entrypoint", "class Example: pass\n")
        with self.assertRaisesRegex(ValueError, "not a tracked"):
            external_configuration(self.root)
        self.stage()
        self.assertEqual(external_configuration(self.root), set())
        self.assertIn("resources_servers/example/custom.entrypoint", declared_files(self.root))

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

    def test_recursive_relative_inheritance_tracks_arbitrary_suffixes_and_absence(self):
        self.write("resources_servers/a/config.yaml", "config_paths: [extras/first]\n")
        self.write("extras/first", "config_paths: [extras/second.conf, extras/future]\n")
        self.write("extras/second.conf", "value: present\n")
        self.stage()
        output = self.root / "inventory.json"
        external_configuration(self.root, inventory_output=output)
        before = json.loads(output.read_text())
        self.assertEqual(before["files"], ["extras/first", "extras/second.conf"])
        self.assertEqual(before["missing"], ["extras/future"])
        self.write("extras/future", "value: newly present\n")
        with self.assertRaisesRegex(ValueError, "untracked"):
            external_configuration(self.root, inventory_output=output)
        self.stage()
        external_configuration(self.root, inventory_output=output)
        self.assertNotEqual(before, json.loads(output.read_text()))
        self.write("extras/first", "config_paths: [../outside.conf]\n")
        with self.assertRaisesRegex(ValueError, "escapes"):
            external_configuration(self.root, inventory_output=output)

    def test_pex_interpreter_path_is_allowed_but_other_tools_and_drift_fail(self):
        from scripts.pants.native_core import incremental_runtime as runtime

        (self.root / "runtime-inputs.json").write_text(
            json.dumps(
                {
                    "roots": [str(Path(sys.base_prefix).resolve())],
                    "paths": [],
                    "fingerprint": {"sha256": "expected"},
                }
            )
        )
        environment = dict(runtime.ENVIRONMENT)
        environment["PATH"] = str(Path(sys.executable).parent) + os.pathsep + environment["PATH"]
        with patch.object(runtime, "NATIVE", self.root), patch.dict(os.environ, environment, clear=True):
            with patch.object(runtime, "fingerprint", return_value={"sha256": "expected"}):
                runtime.verify()
                alias = self.root / "pex-venv"
                alias.symlink_to(sys.prefix, target_is_directory=True)
                os.environ["PATH"] = os.path.relpath(alias / "bin") + os.pathsep + runtime.ENVIRONMENT["PATH"]
                runtime.verify()
                os.environ["PATH"] = "/unexpected/tools:" + runtime.ENVIRONMENT["PATH"]
                with self.assertRaisesRegex(RuntimeError, "PATH"):
                    runtime.verify()
            os.environ["PATH"] = environment["PATH"]
            with patch.object(runtime, "fingerprint", return_value={"sha256": "changed"}):
                with self.assertRaisesRegex(RuntimeError, "runtime changed"):
                    runtime.verify()

    def test_cacheable_batch_keeps_default_external_pytest_temporary_directory(self):
        import run

        commands = []

        def execute(command, **kwargs):
            commands.append(command)
            if "test" in command:
                (self.root / "test_result_info_report_runid-test.json").write_text("{}")
            return Namespace(stdout="{}", returncode=0)

        with (
            patch.object(run.subprocess, "run", side_effect=execute),
            patch.object(run, "validate_plan", return_value=2),
        ):
            run.execute_batch(
                checkout=self.root,
                output=self.root,
                identifier="migration",
                selected=["//candidate:tests"],
                cacheable=True,
                args=Namespace(pants="/pinned/pants", group="core"),
                workers=4,
                environment={"PANTS_LOCAL_STORE_DIR": str(self.root / "store")},
            )
        test_command = commands[-1]
        self.assertFalse(any(value.startswith("--basetemp") for value in test_command))
        self.assertIn("--pex-executable-search-paths=['/usr/bin', '/bin']", test_command)
        self.assertNotIn("--test-force", test_command)


if __name__ == "__main__":
    unittest.main()
