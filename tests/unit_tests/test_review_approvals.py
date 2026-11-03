# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise portable approval decisions without importing the Gym runtime."""

import base64
import importlib.util
import json
import os
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("review_approvals", ROOT / "scripts/ci/review_approvals.py")
approvals = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(approvals)
PUBLISH_SPEC = importlib.util.spec_from_file_location(
    "gitlab_approval_status", ROOT / ".gitlab/ci/check_review_approvals.py"
)
publisher = importlib.util.module_from_spec(PUBLISH_SPEC)
PUBLISH_SPEC.loader.exec_module(publisher)


def encoded_file(text: str) -> dict:
    """Build a provider contents response for an immutable repository file."""
    return {"encoding": "base64", "content": base64.b64encode(text.encode()).decode()}


def review(number: int, user_id: int, state: str, head: str = "head") -> dict:
    """Build one formal GitHub review event."""
    return {"id": number, "user": {"id": user_id}, "state": state, "commit_id": head}


class PolicyTests(unittest.TestCase):
    """Verify path boundaries, precedence, and alternative approval semantics."""

    def test_frontier_approval_satisfies_multiple_packages(self):
        """One Frontier Eval reviewer can approve every changed package."""
        rules = approvals.parse_policy("/a/ @org/a @org/frontier\n/b/ @org/b @org/frontier")
        matches = approvals.match_policy(["a/file.py", "b/nested/file.py"], rules)
        teams = {
            "@org/a": [{"id": 1, "login": "a"}],
            "@org/b": [{"id": 2, "login": "b"}],
            "@org/frontier": [{"id": 3, "login": "fe"}],
        }
        self.assertTrue(approvals.evaluate(matches, teams, {3})["passed"])
        self.assertFalse(approvals.evaluate(matches, teams, {1})["passed"])
        self.assertTrue(approvals.evaluate(matches, teams, {1, 2})["passed"])

    def test_last_match_overrides_model_ownership(self):
        """Checkpoint hooks use runtime reviewers while token capture uses model reviewers."""
        rules = approvals.parse_policy((ROOT / ".github/CODEOWNERS").read_text())
        matches = approvals.match_policy(
            [
                "nemo_gym/models/_checkpoint.py",
                "nemo_gym/models/token_capture/sink.py",
                "nemo_gym/testing/harness_capabilities/checker.py",
            ],
            rules,
        )
        owner_by_path = {path: match["owners"][0] for match in matches for path in match["files"]}
        self.assertEqual(owner_by_path["nemo_gym/models/_checkpoint.py"], "@NVIDIA-NeMo/gym_reviewers_server_runtime")
        self.assertEqual(owner_by_path["nemo_gym/models/token_capture/sink.py"], "@NVIDIA-NeMo/gym_reviewers_model")
        self.assertEqual(
            owner_by_path["nemo_gym/testing/harness_capabilities/checker.py"], "@NVIDIA-NeMo/gym_reviewers_agents"
        )

    def test_github_glob_semantics(self):
        """Directory patterns, root anchors, and double stars preserve boundaries."""
        cases = [
            ("docker/", "nested/docker/Dockerfile", True),
            ("docker/", "nested/docker", False),
            ("/docker/", "nested/docker/Dockerfile", False),
            ("/docker/", "dockerized/file", False),
            ("*.py", "nested/file.py", True),
            ("/tests/*/core/", "tests/unit/core/test.py", True),
            ("/tests/*/core/", "tests/unit/other/core/test.py", False),
            ("/a/**/b.py", "a/b.py", True),
            ("/a/**/b.py", "a/nested/deep/b.py", True),
            ("/a/*/b.py", "a/b.py", False),
        ]
        for pattern, path, expected in cases:
            with self.subTest(pattern=pattern, path=path):
                self.assertEqual(bool(approvals.compile_pattern(pattern).fullmatch(path)), expected)

    def test_advisory_and_empty_overrides_do_not_require_approval(self):
        """Comments are advisory and a last empty rule clears a broader owner."""
        rules = approvals.parse_policy("* @org/a\n# suggest: /docs/ @org/docs\n/docs/\n")
        self.assertEqual(approvals.match_policy(["docs/readme.md"], rules), [])
        self.assertEqual(approvals.match_policy(["source.py"], rules)[0]["owners"], ["@org/a"])

    def test_unsupported_rules_fail_closed(self):
        """Invalid patterns and individual owners cannot silently disable a rule."""
        for text in ("!file @org/a", "/[ab]/ @org/a", "/a\\ b @org/a", "/a/ @person"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                approvals.parse_policy(text)

    def test_all_library_files_have_frontier_alternative(self):
        """Every tracked package file retains the requested shared alternative."""
        paths = [str(path.relative_to(ROOT)) for path in (ROOT / "nemo_gym").rglob("*.py")]
        matches = approvals.match_policy(paths, approvals.parse_policy((ROOT / ".github/CODEOWNERS").read_text()))
        self.assertEqual({path for match in matches for path in match["files"]}, set(paths))
        for match in matches:
            self.assertIn("@NVIDIA-NeMo/frontier-evals", match["owners"])

    def test_benchmark_fallback_and_bionemo_override(self):
        """Preserve coverage outside packages while retaining dedicated BioNeMo owners."""
        policy = approvals.parse_policy((ROOT / ".github/CODEOWNERS").read_text())
        matches = approvals.match_policy(
            ["benchmarks/new-benchmark/config.yaml", "benchmarks/larp/config.yaml"], policy
        )
        by_path = {path: match["owners"] for match in matches for path in match["files"]}
        self.assertEqual(
            by_path["benchmarks/new-benchmark/config.yaml"],
            ["@NVIDIA-NeMo/gym_maintainers", "@NVIDIA-NeMo/frontier-evals"],
        )
        self.assertEqual(
            by_path["benchmarks/larp/config.yaml"], ["@nvidia-nemo/gym_bionemo", "@nvidia-nemo/gym_maintainers"]
        )


class ReviewTests(unittest.TestCase):
    """Reject stale, revoked, author, and unverified cross-platform approvals."""

    def test_latest_substantive_review_wins(self):
        """Comments preserve approval, while changes requested and dismissal revoke it."""
        reviews = [review(1, 10, "APPROVED"), review(2, 10, "COMMENTED")]
        self.assertEqual(approvals.github_approvers(reviews, "head", 99), {10})
        for state in ("CHANGES_REQUESTED", "DISMISSED"):
            with self.subTest(state=state):
                self.assertEqual(approvals.github_approvers(reviews + [review(3, 10, state)], "head", 99), set())

    def test_stale_and_author_reviews_do_not_count(self):
        """Approval must reference this exact head and come from another person."""
        reviews = [review(1, 10, "APPROVED", "previous"), review(2, 99, "APPROVED")]
        self.assertEqual(approvals.github_approvers(reviews, "head", 99), set())

    def test_gitlab_uses_account_ids_instead_of_display_names(self):
        """Only a verified numeric identity link grants GitHub team membership."""
        identities = [{"gitlab_user_id": 100, "github_user_id": 10}]
        approved = [{"user": {"id": 100}}, {"user": {"id": 101}}, {"user": {"id": 99}}]
        self.assertEqual(approvals.gitlab_approvers(approved, identities, 99), ({10}, [101]))
        self.assertEqual(approvals.gitlab_approvers([], identities, 99), (set(), []))

    def test_duplicate_identity_links_fail_closed(self):
        """Ambiguous cross-provider account links cannot authorize a reviewer."""
        identities = [{"gitlab_user_id": 100, "github_user_id": 10}, {"gitlab_user_id": 100, "github_user_id": 11}]
        with self.assertRaises(ValueError):
            approvals.gitlab_approvers([], identities, 99)


class ProviderTests(unittest.TestCase):
    """Verify complete API evidence and immutable target-policy selection."""

    def test_pagination_reads_every_member(self):
        """A reviewer beyond the first page must remain eligible."""
        responses = [[{"id": number} for number in range(100)], [{"id": 100}]]
        self.assertEqual(len(approvals.all_pages(lambda path: responses.pop(0), "members")), 101)

    def test_github_uses_target_policy_and_both_rename_paths(self):
        """A PR cannot choose its own ownership policy or evade an old path's owner."""
        paths_read = []

        def read(path):
            """Return complete fixture data and record target-file requests."""
            paths_read.append(path)
            if path.endswith("/pulls/1"):
                return {"head": {"sha": "head"}, "base": {"sha": "trusted"}, "user": {"id": 99}, "changed_files": 1}
            if "/files?" in path:
                return [{"filename": "new/file.py", "previous_filename": "old/file.py"}]
            if "/reviews?" in path:
                return [review(1, 10, "APPROVED")]
            self.assertTrue(path.endswith("?ref=trusted"))
            return encoded_file("/old/ @org/a")

        snapshot = approvals.github_snapshot(read, "org/repo", 1)
        self.assertEqual(set(snapshot["paths"]), {"new/file.py", "old/file.py"})
        self.assertEqual(snapshot["approved"], {10})
        self.assertTrue(any("?ref=trusted" in path for path in paths_read))

    def test_github_rejects_truncated_changed_files(self):
        """A partial changed-file response must never produce a green decision."""

        def read(path):
            """Return an intentionally truncated provider response."""
            if path.endswith("/pulls/1"):
                return {"changed_files": 2}
            return [{"filename": "one.py"}]

        with self.assertRaisesRegex(ValueError, "incomplete"):
            approvals.github_snapshot(read, "org/repo", 1)

    def test_gitlab_requires_stale_approval_reset(self):
        """GitLab approvals without head-reset protection cannot be trusted."""
        with self.assertRaisesRegex(ValueError, "reset approvals"):
            approvals.gitlab_snapshot(lambda path: {}, 1, 1)

    def test_gitlab_reads_the_shared_policy(self):
        """GitLab evaluates GitHub-format policy and verified account links."""

        def read(path):
            """Serve a merge request with one mapped and one unrelated approver."""
            if path.endswith("/merge_requests/1"):
                return {
                    "sha": "head",
                    "diff_refs": {"head_sha": "head"},
                    "author": {"id": 99},
                    "target_branch": "trusted",
                    "changes_count": "1",
                }
            if path.endswith("/approvals"):
                return (
                    {"approved_by": [{"user": {"id": 100}}]}
                    if "/merge_requests/" in path
                    else {"reset_approvals_on_push": True}
                )
            if "/diffs?" in path:
                return [{"old_path": "old/file.py", "new_path": "new/file.py"}]
            if "/branches/" in path:
                return {"commit": {"id": "target"}}
            self.assertTrue(path.endswith("?ref=target"))
            if "reviewer-identities" in path:
                return encoded_file(json.dumps([{"gitlab_user_id": 100, "github_user_id": 10}]))
            return encoded_file("/old/ @org/a")

        snapshot = approvals.gitlab_snapshot(read, 1, 1)
        self.assertEqual(snapshot["approved"], {10})
        self.assertEqual(snapshot["policy"], "/old/ @org/a")

    def test_api_redirects_do_not_forward_credentials(self):
        """An API response cannot redirect a team token to another origin."""
        with self.assertRaisesRegex(ValueError, "redirected"):
            approvals.NoRedirect().redirect_request(None, None, 302, "", {}, "https://other.example")

    def test_error_is_not_reported_as_approval(self):
        """Transport failures exit distinctly and do not write a passing report."""
        with patch.object(approvals, "run_check", side_effect=ValueError("HTTP 403")):
            self.assertEqual(approvals.main(["github", "--number", "1"]), 2)


class GitLabStatusTests(unittest.TestCase):
    """Verify that only validated results can turn the exact MR pipeline green."""

    def run_publisher(self, *, report_head="head", returncode=0, malformed=False):
        """Execute the real publisher against isolated API and subprocess fixtures."""
        writes = []
        event = {
            "object_kind": "merge_request",
            "project": {"id": 1},
            "object_attributes": {"iid": 2, "target_branch": "nv-internal-main"},
        }
        environment = {
            "CI_PROJECT_ID": "1",
            "TRIGGER_PAYLOAD": "event.json",
            "CI_COMMIT_BRANCH": "nv-internal-main",
            "CI_API_V4_URL": "https://gitlab.example/api/v4",
            "GITLAB_TOKEN": "fixture",
            "CI_JOB_URL": "https://ci.example/job/1",
        }

        def api(base, token, path, **kwargs):
            """Record status writes and provide the current immutable MR identity."""
            if kwargs:
                writes.append((path, dict(kwargs["payload"])))
                return {}
            return {"sha": "head", "state": "opened", "head_pipeline": {"id": 42, "sha": "head"}}

        def run(command, **kwargs):
            """Write the same output contract as the portable shell entry point."""
            output = Path(command[command.index("--output") + 1])
            output.write_text("{" if malformed else json.dumps({"head": report_head}))
            return subprocess.CompletedProcess(command, returncode)

        with (
            patch.dict(os.environ, environment),
            patch.object(publisher.approvals, "api_json", side_effect=api),
            patch.object(publisher.subprocess, "run", side_effect=run),
            patch.object(publisher.Path, "read_text", autospec=True) as read,
        ):
            original_read = Path.read_bytes

            def content(path, *args, **kwargs):
                """Read the webhook fixture or the subprocess's real report file."""
                return json.dumps(event) if str(path) == "event.json" else original_read(path).decode()

            read.side_effect = content
            try:
                result = publisher.run()
            except ValueError:
                result = 2
        return result, writes

    def test_current_head_success_updates_exact_pipeline(self):
        """Attach status to the MR pipeline rather than creating another pipeline."""
        result, writes = self.run_publisher()
        self.assertEqual(result, 0)
        self.assertEqual([item[1]["state"] for item in writes], ["pending", "success"])
        self.assertTrue(all(item[0] == "projects/1/statuses/head" and item[1]["pipeline_id"] == 42 for item in writes))

    def test_missing_approval_updates_failure(self):
        """A failed checker invalidates an earlier successful pipeline verdict."""
        result, writes = self.run_publisher(returncode=1)
        self.assertEqual(result, 1)
        self.assertEqual(writes[-1][1]["state"], "failed")

    def test_head_change_never_publishes_success(self):
        """A report for a different head cannot approve the captured head."""
        result, writes = self.run_publisher(report_head="new-head")
        self.assertEqual(result, 2)
        self.assertEqual(writes[-1][1]["state"], "failed")

    def test_invalid_report_never_publishes_success(self):
        """An unreadable success report leaves the required pipeline failed."""
        result, writes = self.run_publisher(malformed=True)
        self.assertEqual(result, 2)
        self.assertEqual(writes[-1][1]["state"], "failed")

    def test_pipeline_for_another_head_is_rejected(self):
        """An approval status cannot be appended to a stale head's pipeline."""
        with patch.object(publisher.approvals, "api_json", return_value={"sha": "different-head"}):
            with self.assertRaisesRegex(ValueError, "head changed"):
                publisher.head_pipeline(
                    "https://gitlab.example/api/v4", "fixture", "projects/1/merge_requests/2", "head"
                )


if __name__ == "__main__":
    unittest.main()
