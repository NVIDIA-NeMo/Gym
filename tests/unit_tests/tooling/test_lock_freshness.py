# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise advisory drift detection and publication without writing to GitHub."""

import subprocess
from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from scripts import lock_freshness as freshness


REPOSITORY = "NVIDIA-NeMo/Gym"
HEAD = "a" * 40
BASE = "b" * 40
ANCESTOR = "c" * 40
PULL = {"number": 42, "state": "open", "base": {"ref": "main"}, "head": {"sha": HEAD}}


def comparison(*filenames: str, merge_base: str = ANCESTOR) -> dict:
    """Construct a GitHub comparison containing dependency paths."""
    return {"merge_base_commit": {"sha": merge_base}, "files": [{"filename": name} for name in filenames]}


@pytest.mark.parametrize(
    ("pr_paths", "main_paths", "expected"),
    [
        (["pyproject.toml"], ["resources_servers/new/uv.lock"], "failure"),
        (["resources_servers/new/uv.lock"], ["pyproject.toml"], "failure"),
        (["resources_servers/a/uv.lock"], ["resources_servers/a/pyproject.toml"], "failure"),
        (["resources_servers/a/uv.lock"], ["resources_servers/b/uv.lock"], "success"),
        (["nemo_gym/models/converter.py"], ["uv.lock"], "success"),
        (["uv.lock"], ["README.md"], "success"),
        (["scripts/component-lock-policy.json"], ["benchmarks/new/pyproject.toml"], "failure"),
    ],
)
def test_interacting_dependency_changes(pr_paths, main_paths, expected):
    """Flag shared and same-component drift while tolerating independent changes."""
    state, description = freshness.freshness_status(
        freshness.changed_scopes(comparison(*pr_paths)), freshness.changed_scopes(comparison(*main_paths))
    )
    assert state == expected
    assert "advisory" in description


def test_renamed_manifest_and_truncated_diffs_are_conservative():
    """Do not silently miss a removed dependency input or files beyond the API cap."""
    renamed = {"files": [{"filename": "archive/old.txt", "previous_filename": "pyproject.toml"}]}
    assert freshness.changed_scopes(renamed) == {"."}
    truncated = comparison(*(f"docs/{index}.md" for index in range(300)))
    assert freshness.freshness_status(freshness.changed_scopes(truncated), {"resources_servers/new"})[0] == "failure"


def test_status_turns_red_on_unchanged_head_and_clears_after_sync(monkeypatch):
    """Reproduce new locks arriving after a shared dependency PR was checked."""
    responses = {
        f"repos/{REPOSITORY}/compare/{BASE}...{HEAD}?per_page=1": comparison("pyproject.toml", merge_base=BASE),
        f"repos/{REPOSITORY}/compare/{HEAD}...{BASE}?per_page=1": comparison("resources_servers/new/uv.lock"),
    }

    def api(endpoint, **kwargs):
        """Serve comparisons for a fixed PR head and advancing base history."""
        return responses[endpoint]

    monkeypatch.setattr(freshness, "github_api", api)
    assert freshness.evaluate_pr(REPOSITORY, PULL, BASE)["state"] == "success"
    responses[f"repos/{REPOSITORY}/compare/{BASE}...{HEAD}?per_page=1"] = comparison("pyproject.toml")
    stale = freshness.evaluate_pr(REPOSITORY, PULL, BASE)
    assert stale["state"] == "failure"
    assert stale["head_sha"] == HEAD
    assert stale["main_scopes"] == ["resources_servers/new"]
    synced = deepcopy(PULL)
    synced["head"]["sha"] = "d" * 40
    responses[f"repos/{REPOSITORY}/compare/{BASE}...{'d' * 40}?per_page=1"] = comparison(
        "pyproject.toml", merge_base=BASE
    )
    assert freshness.evaluate_pr(REPOSITORY, synced, BASE)["state"] == "success"


@pytest.mark.parametrize("superseded", [None, "head", "base", "closed", "retargeted"])
def test_status_publication_and_revision_guards(monkeypatch, superseded):
    """Post on the PR head, suppress duplicate statuses, and reject stale results."""
    pull = deepcopy(PULL)
    if superseded == "head":
        pull["head"]["sha"] = "d" * 40
    if superseded == "closed":
        pull["state"] = "closed"
    if superseded == "retargeted":
        pull["base"]["ref"] = "release"
    posts = []
    statuses = []

    def api(endpoint, *, payload=None, **kwargs):
        """Record status writes while serving the current PR and base revisions."""
        if payload is not None:
            posts.append((endpoint, payload))
            statuses.append(payload)
            return {}
        if endpoint.endswith("/pulls/42"):
            return pull
        if endpoint.endswith("/git/ref/heads/main"):
            return {"object": {"sha": "e" * 40 if superseded == "base" else BASE}}
        if endpoint.endswith("/status"):
            return {"statuses": statuses}
        raise AssertionError(endpoint)

    monkeypatch.setattr(freshness, "github_api", api)
    result = {"number": 42, "head_sha": HEAD, "base_sha": BASE, "state": "failure", "description": "advisory"}
    publication = freshness.publish_status(REPOSITORY, result, "https://example.com/evidence")
    if superseded:
        assert publication == "superseded"
        assert not posts
    else:
        assert publication == "published"
        assert posts == [
            (
                f"repos/{REPOSITORY}/statuses/{HEAD}",
                {
                    "context": "lock-freshness",
                    "state": "failure",
                    "description": "advisory",
                    "target_url": "https://example.com/evidence",
                },
            )
        ]
        assert freshness.publish_status(REPOSITORY, result, "https://example.com/evidence") == "unchanged"
        assert len(posts) == 1


def test_refresh_paginates_dry_runs_and_continues_after_api_errors(monkeypatch):
    """A failed comparison must not prevent another PR's advisory from updating."""
    second_pull = {**PULL, "number": 43, "head": {"sha": "d" * 40}}
    calls = []

    def api(endpoint, *, payload=None):
        """Reject mutations and simulate paginated PR enumeration and one failure."""
        calls.append(endpoint)
        if endpoint == "graphql":
            first = payload["variables"]["cursor"] is None
            pull = PULL if first else second_pull
            nodes = [
                {
                    "number": pull["number"],
                    "headRefOid": pull["head"]["sha"],
                    "files": {"pageInfo": {"hasNextPage": False}, "nodes": [{"path": "uv.lock"}]},
                }
            ]
            nodes.append(
                {
                    "number": 99,
                    "headRefOid": "f" * 40,
                    "files": {"pageInfo": {"hasNextPage": False}, "nodes": [{"path": "README.md"}]},
                }
            )
            return {
                "data": {
                    "repository": {
                        "pullRequests": {"pageInfo": {"hasNextPage": first, "endCursor": "next"}, "nodes": nodes}
                    }
                }
            }
        assert payload is None
        if endpoint.endswith("/git/ref/heads/main"):
            return {"object": {"sha": BASE}}
        if HEAD in endpoint:
            raise subprocess.CalledProcessError(1, ["gh", "api", endpoint])
        return comparison("README.md", merge_base=BASE)

    monkeypatch.setattr(freshness, "github_api", api)
    results = freshness.refresh(REPOSITORY, pr_number=None, dry_run=True, target_url="https://example.com")
    assert "error" in results[0]
    assert results[1]["state"] == "success"
    assert results[1]["publication"] == "dry-run"
    assert len(calls) == 5


def test_api_sends_status_payload_through_stdin(monkeypatch):
    """Keep structured API content out of executable command text."""
    calls = []

    def run(command, **kwargs):
        """Capture the subprocess boundary without contacting GitHub."""
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="{}")

    monkeypatch.setattr(subprocess, "run", run)
    freshness.github_api("repos/example/repo/statuses/sha", payload={"description": "line1\n$(false)"})
    command, kwargs = calls[0]
    assert command == ["gh", "api", "repos/example/repo/statuses/sha", "--method", "POST", "--input", "-"]
    assert kwargs["input"] == '{"description": "line1\\n$(false)"}'
    assert kwargs["check"] is True


def test_privileged_workflow_only_checks_out_trusted_base_code():
    """PR-controlled code must never run with the status-writing token."""
    root = Path(__file__).resolve().parents[3]
    workflow = yaml.load((root / ".github/workflows/lock-freshness.yml").read_text(), Loader=yaml.BaseLoader)
    assert "pull_request_target" in workflow["on"]
    assert "push" in workflow["on"]
    assert workflow["permissions"] == {"contents": "read", "pull-requests": "read", "statuses": "write"}
    steps = workflow["jobs"]["advisory"]["steps"]
    checkouts = [step for step in steps if step.get("uses", "").startswith("actions/checkout@")]
    assert len(checkouts) == 1
    assert checkouts[0]["with"] == {"ref": "main", "persist-credentials": "false"}
    assert all("github.event.pull_request.head" not in str(step) for step in steps)
