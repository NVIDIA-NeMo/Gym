# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise history sampling and disposable comparisons against real Git repositories."""

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import monitor
from monitor import change_kind, select_samples


class SamplingTest(unittest.TestCase):
    def test_configuration_change_in_server_is_not_lost(self):
        self.assertEqual(change_kind(["resources_servers/weather/configs/default.yaml"]), "configuration")
        self.assertEqual(change_kind(["nemo_gym/config/default.json", "nemo_gym/config/load.py"]), "configuration")

    def test_recent_core_changes_do_not_crowd_out_other_kinds(self):
        candidates = [
            {"sha": "a", "kind": "core"},
            {"sha": "b", "kind": "core"},
            {"sha": "c", "kind": "configuration"},
            {"sha": "d", "kind": "server"},
            {"sha": "e", "kind": "other"},
        ]
        selected = select_samples(candidates, limit=3, completed={"a"})
        self.assertEqual([item["sha"] for item in selected], ["b", "d", "c"])

    def test_small_limit_and_exhausted_history(self):
        candidates = [{"sha": "a", "kind": "core"}, {"sha": "b", "kind": "server"}]
        self.assertEqual(len(select_samples(candidates, limit=1, completed=set())), 1)
        self.assertEqual(select_samples(candidates, limit=3, completed={"a", "b"}), [])


class RepositoryTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name).resolve()
        self.root = self.directory / "repo"
        self.root.mkdir()
        self.git("init", "-b", "main")
        self.git("config", "user.name", "Pants monitor test")
        self.git("config", "user.email", "pants-test@example.invalid")
        self.git("config", "commit.gpgsign", "false")
        self.git("config", "core.hooksPath", str(self.directory / "no-hooks"))
        self.initial = self.commit("README.md", "initial\n")
        self.baseline = self.commit("nemo_gym/example.py", "VERSION = 'baseline'\n")
        self.git("update-ref", "refs/remotes/origin/main", self.baseline)
        self.git("switch", "-c", "glovisotto/pants-pilot")

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.root), *args], text=True, stderr=subprocess.PIPE).strip()

    def commit(self, filename, content):
        path = self.root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        self.git("add", filename)
        self.git("commit", "-m", filename)
        return self.git("rev-parse", "HEAD")

    def overlay(self, *, exit_code=0):
        # The real monitor launches this script in the temporary worktree. No Pants or network needed.
        source = """import argparse
import json
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument('--base')
parser.add_argument('--parent')
parser.add_argument('--output', type=Path)
parser.add_argument('--repeats', type=int)
args = parser.parse_args()
args.output.mkdir(parents=True)
(args.output / 'receipt.json').write_text(json.dumps({
    'base': args.base, 'parent': args.parent, 'repeats': args.repeats,
    'source': Path('nemo_gym/example.py').read_text(),
}))
raise SystemExit(EXIT_CODE)
""".replace("EXIT_CODE", str(exit_code))
        self.commit("scripts/pants/compare.py", source)
        return subprocess.check_output(["git", "-C", str(self.root), "diff", "--binary", self.baseline, "HEAD"])

    def sample(self):
        return {"sha": self.baseline, "parent": self.initial, "kind": "core", "paths": ["nemo_gym/example.py"]}

    def assert_checkout_removed(self, destination):
        self.assertFalse((destination / "checkout").exists())
        self.assertNotIn(str(destination / "checkout"), self.git("worktree", "list", "--porcelain"))
        self.assertEqual(self.git("status", "--porcelain"), "")

    def test_history_sampling_uses_main_and_real_parent_changes(self):
        self.git("switch", "main")
        server = self.commit("resources_servers/weather/app.py", "VALUE = 1\n")
        config = self.commit("resources_servers/weather/config.yaml", "enabled: true\n")
        self.commit("README.md", "documentation only\n")
        self.git("update-ref", "refs/remotes/origin/main", "HEAD")
        self.git("switch", "glovisotto/pants-pilot")
        self.commit("nemo_gym/pilot_only.py", "VALUE = 2\n")
        candidates = monitor.candidates_from_git(self.root, lookback=20)
        selected = select_samples(candidates, limit=3, completed=set())
        self.assertEqual([entry["sha"] for entry in selected], [self.baseline, server, config])
        self.assertEqual(selected[2]["parent"], server)
        self.assertEqual(selected[2]["paths"], ["resources_servers/weather/config.yaml"])
        self.assertNotIn(self.git("rev-parse", "HEAD"), {entry["sha"] for entry in candidates})

    def test_overlay_runs_on_historical_source_and_worktree_is_removed(self):
        overlay = self.overlay()
        pilot_head = self.git("rev-parse", "HEAD")
        destination = self.directory / "sample"
        outcome = monitor.run_sample(
            self.root, sample=self.sample(), patch=overlay, destination=destination, repeats=2
        )
        self.assertEqual(outcome["status"], "completed")
        receipt = json.loads((destination / "comparison/receipt.json").read_text())
        self.assertEqual(
            receipt, {"base": self.baseline, "parent": self.initial, "repeats": 2, "source": "VERSION = 'baseline'\n"}
        )
        self.assertEqual(json.loads((destination / "sample.json").read_text()), outcome)
        self.assertEqual(self.git("rev-parse", "HEAD"), pilot_head)
        self.assert_checkout_removed(destination)

    def test_failed_comparison_retains_failure_and_cleans_checkout(self):
        overlay = self.overlay(exit_code=7)
        destination = self.directory / "sample"
        outcome = monitor.run_sample(
            self.root, sample=self.sample(), patch=overlay, destination=destination, repeats=1
        )
        self.assertEqual((outcome["status"], outcome["returncode"]), ("failed", 7))
        self.assertTrue((destination / "comparison/receipt.json").exists())
        self.assertTrue((destination / "compare.log").exists())
        self.assert_checkout_removed(destination)

    def test_inapplicable_overlay_records_failure_and_cleans_checkout(self):
        self.overlay()
        destination = self.directory / "sample"
        outcome = monitor.run_sample(
            self.root, sample=self.sample(), patch=b"invalid patch\n", destination=destination, repeats=1
        )
        self.assertEqual(outcome["status"], "failed")
        self.assertIn("error", outcome)
        self.assertFalse((destination / "compare.log").exists())
        self.assert_checkout_removed(destination)

    def test_dirty_checkout_blocks_refresh_without_overwriting_edits(self):
        self.overlay()
        head = self.git("rev-parse", "HEAD")
        changed = self.root / "nemo_gym/example.py"
        changed.write_text("uncommitted user work\n")
        untracked = self.root / "user-notes.txt"
        untracked.write_text("keep me\n")
        with self.assertRaisesRegex(RuntimeError, "checkout has changes"):
            monitor.refresh(self.root, fetch=True, rebase=True)
        self.assertEqual(changed.read_text(), "uncommitted user work\n")
        self.assertEqual(untracked.read_text(), "keep me\n")
        self.assertEqual(self.git("rev-parse", "HEAD"), head)

    def test_failed_rebase_restores_previous_head_and_preserves_checkpoint(self):
        head = self.commit("nemo_gym/example.py", "VERSION = 'pilot'\n")
        self.git("switch", "main")
        upstream = self.commit("nemo_gym/example.py", "VERSION = 'upstream'\n")
        self.git("update-ref", "refs/remotes/origin/main", upstream)
        self.git("switch", "glovisotto/pants-pilot")
        with self.assertRaisesRegex(RuntimeError, "Rebase failed"):
            monitor.refresh(self.root, fetch=False, rebase=True)
        self.assertEqual(self.git("rev-parse", "HEAD"), head)
        self.assertEqual((self.root / "nemo_gym/example.py").read_text(), "VERSION = 'pilot'\n")
        self.assertEqual(self.git("status", "--porcelain"), "")
        self.assertEqual(self.git("for-each-ref", "--format=%(objectname)", "refs/pants-pilot/checkpoints"), head)
        self.assertFalse((self.root / ".git/rebase-merge").exists())

    def test_successful_batch_records_sample_as_completed(self):
        self.overlay()
        output = self.directory / "results"
        arguments = ["monitor", "--once", "--no-fetch", "--no-rebase", "--output", str(output)]
        with (
            contextlib.chdir(self.root),
            patch.object(sys, "argv", arguments),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            result = monitor.main()
        self.assertEqual(result, 0)
        state = json.loads((output / "state.json").read_text())
        completed = set(next(iter(state["completed"].values())))
        self.assertEqual(completed, {self.baseline})
        self.assertEqual(
            select_samples(monitor.candidates_from_git(self.root, lookback=20), limit=3, completed=completed), []
        )
        self.assertFalse((output / ".monitor.lock").exists())

    def test_existing_batch_lock_prevents_comparison_and_is_preserved(self):
        self.overlay()
        output = self.directory / "results"
        lock = output / ".monitor.lock"
        lock.mkdir(parents=True)
        arguments = ["monitor", "--once", "--no-fetch", "--no-rebase", "--output", str(output)]
        with (
            contextlib.chdir(self.root),
            patch.object(sys, "argv", arguments),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            with self.assertRaises(SystemExit) as error:
                monitor.main()
        self.assertEqual(error.exception.code, 2)
        self.assertTrue(lock.is_dir())
        self.assertFalse((output / "state.json").exists())
        self.assertFalse((output / "batches").exists())

    def test_failed_batch_remains_eligible_and_releases_lock(self):
        self.overlay(exit_code=9)
        output = self.directory / "results"
        arguments = ["monitor", "--once", "--no-fetch", "--no-rebase", "--output", str(output)]
        with (
            contextlib.chdir(self.root),
            patch.object(sys, "argv", arguments),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            result = monitor.main()
        self.assertEqual(result, 1)
        self.assertFalse((output / ".monitor.lock").exists())
        state = json.loads((output / "state.json").read_text())
        self.assertEqual(list(state["completed"].values()), [[]])
        metadata_path = next((output / "batches").glob("*/batch.json"))
        metadata = json.loads(metadata_path.read_text())
        self.assertEqual(metadata["samples"][0]["status"], "failed")
        eligible = select_samples(
            monitor.candidates_from_git(self.root, lookback=20),
            limit=3,
            completed=set(state["completed"][metadata["overlay_sha256"]]),
        )
        self.assertEqual([item["sha"] for item in eligible], [self.baseline])
        self.assert_checkout_removed(metadata_path.parent / self.baseline)


if __name__ == "__main__":
    unittest.main()
