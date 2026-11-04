# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Guard complete native target selection and uv/PEX dependency equivalence."""

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from collect import targets
from prepare import requirement_pins, snapshot_git, validate_lock


class NativeCoreTest(unittest.TestCase):
    def test_declared_git_snapshot_preserves_tree_and_commit_without_parent_checkout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            subprocess.run(["git", "init", "--quiet", str(source)], check=True)
            environment = dict(
                os.environ,
                GIT_AUTHOR_NAME="Fixture",
                GIT_AUTHOR_EMAIL="fixture@example.test",
                GIT_COMMITTER_NAME="Fixture",
                GIT_COMMITTER_EMAIL="fixture@example.test",
            )

            def git(*args, input=None):
                return subprocess.check_output(["git", *args], cwd=source, env=environment, input=input)

            blob = git("hash-object", "-w", "--stdin", input=b"sample\n").strip()
            tree = git("mktree", input=b"100644 blob " + blob + b"\tsource.py\n").strip()
            commit = git("commit-tree", tree.decode(), input=b"fixture\n").strip()
            git("update-ref", "HEAD", commit.decode())
            git("read-tree", "HEAD")
            snapshot_git(source)
            isolated = root / "isolated"
            isolated.mkdir()
            shutil.copytree(source / "scripts/pants/native_core/git-state", isolated / ".git")
            shutil.rmtree(source)
            self.assertEqual(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=isolated).strip(), commit)
            self.assertEqual(subprocess.check_output(["git", "ls-files"], cwd=isolated), b"source.py\n")
            self.assertEqual(subprocess.check_output(["git", "show", "HEAD:source.py"], cwd=isolated), b"sample\n")
            self.assertFalse((isolated / ".git/objects/info/alternates").exists())

    def test_four_shards_cover_each_globally_collected_file_once(self):
        files = {f"tests/unit_tests/test_{index}.py": [f"test_{index}"] for index in range(354)}
        assignments = {path: index % 4 for index, path in enumerate(files)}
        manifest = {"valid": True, "marker": "not sandbox", "files": files, "assignments": assignments}
        selected = [target for shard in range(4) for target in targets(manifest, resolve="core", shard=shard)]
        self.assertEqual(len(selected), 354)
        self.assertEqual(set(selected), {f"//{path}:../../core-tests" for path in files})
        self.assertEqual(len(set(selected)), len(selected))

    def test_wrong_marker_or_partial_collection_cannot_select_targets(self):
        for manifest in ({"valid": False}, {"valid": True, "marker": "sandbox"}):
            with self.subTest(manifest=manifest), self.assertRaises(ValueError):
                targets(manifest, resolve="core", shard=0)

    def test_sandbox_has_no_core_sharding(self):
        manifest = {"valid": True, "marker": "sandbox", "files": {"tests/test_sandbox.py": ["x"]}}
        self.assertEqual(
            targets(manifest, resolve="sandbox", shard=None), ["//tests/test_sandbox.py:../sandbox-tests"]
        )
        with self.assertRaises(ValueError):
            targets(manifest, resolve="sandbox", shard=0)

    def test_mutable_vcs_and_unpinned_registry_requirements_fail(self):
        for line in ("foo>=1", "foo @ git+https://example.test/r.git@main", "foo @ https://example.test/foo.whl"):
            with self.subTest(line=line), self.assertRaises(ValueError):
                requirement_pins(line)

    def lock_check(self, requirements, locked):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            req = root / "requirements.txt"
            lock = root / "lock.json"
            req.write_text(requirements)
            lock.write_text(json.dumps({"locked_resolves": [{"locked_requirements": locked}]}))
            return validate_lock(req, lock)

    def test_excluded_dependency_reintroduced_by_pex_fails(self):
        with self.assertRaisesRegex(ValueError, "absent from uv: unwanted"):
            self.lock_check(
                "foo==1", [{"project_name": "foo", "version": "1"}, {"project_name": "unwanted", "version": "2"}]
            )

    def test_missing_or_drifted_pin_fails(self):
        for locked in ([], [{"project_name": "foo", "version": "2"}]):
            with self.subTest(locked=locked), self.assertRaises(ValueError):
                self.lock_check("foo==1", locked)

    def test_vcs_revision_and_artifact_integrity_are_required(self):
        url = "git+https://example.test/r.git@" + "a" * 40
        item = {"project_name": "foo", "version": "1", "artifacts": [{"url": url, "hash": "sha256"}]}
        self.assertEqual(self.lock_check(f"foo @ {url}", [item])["distributions"], 1)
        item["artifacts"][0]["url"] = url[:-1] + "b"
        with self.assertRaisesRegex(ValueError, "exact VCS source"):
            self.lock_check(f"foo @ {url}", [item])
        item["artifacts"][0] = {"url": url}
        with self.assertRaisesRegex(ValueError, "integrity"):
            self.lock_check(f"foo @ {url}", [item])


if __name__ == "__main__":
    unittest.main()
