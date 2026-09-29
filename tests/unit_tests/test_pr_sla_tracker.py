# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from datetime import UTC, datetime
from importlib import import_module
from io import StringIO
from pathlib import Path

import pytest
import yaml


def _tracker():
    return import_module("scripts.pr_sla_tracker")


def _at(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=UTC)


def _pull_request(**overrides):
    pull_request = {
        "number": 42,
        "title": "Make rollouts observable",
        "url": "https://github.com/NVIDIA-NeMo/Gym/pull/42",
        "author": "octocat",
        "author_is_bot": False,
        "is_draft": False,
        "head_ref": "feature/observability",
        "created_at": _at("2026-07-10T15:00:00"),
        "ready_at": _at("2026-07-10T15:00:00"),
        "requested_reviewers": (),
        "review_request_times": {},
        "latest_review": None,
        "labels": set(),
    }
    pull_request.update(overrides)
    return pull_request


def test_business_day_deadline_skips_a_weekend() -> None:
    tracker = _tracker()

    deadline = tracker.business_day_deadline(_at("2026-07-10T15:00:00"))

    assert deadline == _at("2026-07-13T15:00:00")


def test_review_request_breaches_after_one_business_day() -> None:
    tracker = _tracker()
    pull_request = _pull_request(
        requested_reviewers=("reviewer-a",),
        review_request_times={"reviewer-a": _at("2026-07-10T15:00:00")},
    )

    assert tracker.classify_pull_request(pull_request, _at("2026-07-13T14:59:59")) is None

    breach = tracker.classify_pull_request(pull_request, _at("2026-07-13T15:00:01"))

    assert breach.owner == "reviewer"
    assert breach.people == ("reviewer-a",)
    assert breach.since == _at("2026-07-10T15:00:00")
    assert breach.deadline == _at("2026-07-13T15:00:00")
    assert breach.reason == "review requested"


def test_only_overdue_reviewers_are_named() -> None:
    tracker = _tracker()
    pull_request = _pull_request(
        requested_reviewers=("reviewer-a", "reviewer-b"),
        review_request_times={
            "reviewer-a": _at("2026-07-10T15:00:00"),
            "reviewer-b": _at("2026-07-13T14:00:00"),
        },
    )

    breach = tracker.classify_pull_request(pull_request, _at("2026-07-13T16:00:00"))

    assert breach.people == ("reviewer-a",)


def test_changes_requested_breaches_against_author() -> None:
    tracker = _tracker()
    pull_request = _pull_request(
        latest_review={"state": "CHANGES_REQUESTED", "submitted_at": _at("2026-07-13T09:00:00")},
    )

    breach = tracker.classify_pull_request(pull_request, _at("2026-07-14T09:00:01"))

    assert breach.owner == "author"
    assert breach.people == ("octocat",)
    assert breach.since == _at("2026-07-13T09:00:00")
    assert breach.reason == "changes requested"


def test_ready_pull_request_without_a_reviewer_needs_maintainer_triage() -> None:
    tracker = _tracker()
    pull_request = _pull_request()

    breach = tracker.classify_pull_request(pull_request, _at("2026-07-13T15:00:01"))

    assert breach.owner == "triage"
    assert breach.people == ()
    assert breach.reason == "reviewer not assigned"


@pytest.mark.parametrize("state", ["APPROVED", "COMMENTED", "DISMISSED"])
def test_non_actionable_review_state_is_not_tracked(state: str) -> None:
    tracker = _tracker()
    pull_request = _pull_request(
        latest_review={"state": state, "submitted_at": _at("2026-07-10T15:00:00")},
    )

    assert tracker.classify_pull_request(pull_request, _at("2026-07-14T15:00:00")) is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"is_draft": True},
        {"author": "dependabot[bot]"},
        {"author_is_bot": True},
        {"head_ref": "release-please--branches--main"},
        {"labels": {"stale"}},
        {"labels": {"PAUSED"}},
        {"labels": {"do-not-track"}},
    ],
)
def test_non_actionable_pull_request_is_excluded(overrides: dict) -> None:
    tracker = _tracker()
    pull_request = _pull_request(**overrides)

    assert tracker.classify_pull_request(pull_request, _at("2026-07-14T15:00:00")) is None


def test_dashboard_groups_reviewer_and_author_breaches() -> None:
    tracker = _tracker()
    generated_at = _at("2026-07-14T16:00:00")
    reviewer_breach = tracker.classify_pull_request(
        _pull_request(
            title="Reviewer | handoff",
            requested_reviewers=("reviewer-a",),
            review_request_times={"reviewer-a": _at("2026-07-13T15:00:00")},
        ),
        generated_at,
    )
    author_breach = tracker.classify_pull_request(
        _pull_request(
            number=43,
            title="Author handoff",
            url="https://github.com/NVIDIA-NeMo/Gym/pull/43",
            latest_review={"state": "CHANGES_REQUESTED", "submitted_at": _at("2026-07-10T10:00:00")},
        ),
        generated_at,
    )
    triage_breach = tracker.classify_pull_request(
        _pull_request(
            number=44,
            title="Review assignment needed",
            url="https://github.com/NVIDIA-NeMo/Gym/pull/44",
            ready_at=_at("2026-07-13T12:00:00"),
        ),
        generated_at,
    )

    dashboard = tracker.render_dashboard([reviewer_breach, author_breach, triage_breach], generated_at)

    assert "<!-- pr-sla-tracker -->" in dashboard
    assert "## Newly breached (2)" in dashboard
    assert "### Waiting on reviewers (1)" in dashboard
    assert "### Needs review assignment (1)" in dashboard
    assert "<summary><strong>Older breach backlog (1)</strong></summary>" in dashboard
    assert "### Waiting on authors (1)" in dashboard
    assert "Reviewer \\| handoff" in dashboard
    assert "[@reviewer-a](https://github.com/reviewer-a)" in dashboard
    assert "Maintainers" in dashboard
    assert "Review during the daily stand-up" in dashboard


def test_dashboard_has_an_explicit_all_clear_state() -> None:
    tracker = _tracker()

    dashboard = tracker.render_dashboard([], _at("2026-07-14T16:00:00"))

    assert "No pull requests have breached the one-business-day handoff SLA." in dashboard


def test_graphql_pull_request_is_normalized_to_latest_handoffs() -> None:
    tracker = _tracker()
    node = {
        "number": 42,
        "title": "Make rollouts observable",
        "url": "https://github.com/NVIDIA-NeMo/Gym/pull/42",
        "isDraft": False,
        "headRefName": "feature/observability",
        "createdAt": "2026-07-10T10:00:00Z",
        "author": {"__typename": "User", "login": "octocat"},
        "labels": {"nodes": [{"name": "bug"}]},
        "reviewRequests": {
            "nodes": [
                {"requestedReviewer": {"__typename": "User", "login": "reviewer-a"}},
                {
                    "requestedReviewer": {
                        "__typename": "Team",
                        "slug": "automation",
                        "organization": {"login": "nvidia-nemo"},
                    }
                },
            ]
        },
        "reviews": {
            "nodes": [
                {"state": "COMMENTED", "submittedAt": "2026-07-10T12:00:00Z"},
                {"state": "CHANGES_REQUESTED", "submittedAt": "2026-07-10T13:00:00Z"},
            ]
        },
        "timelineItems": {
            "nodes": [
                {"__typename": "ReadyForReviewEvent", "createdAt": "2026-07-10T11:00:00Z"},
                {
                    "__typename": "ReviewRequestedEvent",
                    "createdAt": "2026-07-10T11:30:00Z",
                    "requestedReviewer": {"__typename": "User", "login": "reviewer-a"},
                },
                {
                    "__typename": "ReviewRequestedEvent",
                    "createdAt": "2026-07-10T14:00:00Z",
                    "requestedReviewer": {"__typename": "User", "login": "reviewer-a"},
                },
            ]
        },
    }

    pull_request = tracker.normalize_pull_request(node)

    assert pull_request["author"] == "octocat"
    assert pull_request["author_is_bot"] is False
    assert pull_request["ready_at"] == _at("2026-07-10T11:00:00")
    assert pull_request["requested_reviewers"] == ("nvidia-nemo/automation", "reviewer-a")
    assert pull_request["review_request_times"]["reviewer-a"] == _at("2026-07-10T14:00:00")
    assert pull_request["latest_review"] == {
        "state": "CHANGES_REQUESTED",
        "submitted_at": _at("2026-07-10T13:00:00"),
    }
    assert pull_request["labels"] == {"bug"}


class _FakeClient:
    def __init__(self) -> None:
        self.added: list[tuple[int, str]] = []
        self.removed: list[tuple[int, str]] = []
        self.ensured: list[str] = []
        self.dashboard_issue = None
        self.created_body = None
        self.updated = None

    def ensure_label(self, name: str, _color: str, _description: str) -> None:
        self.ensured.append(name)

    def add_label(self, number: int, name: str) -> None:
        self.added.append((number, name))

    def remove_label(self, number: int, name: str) -> None:
        self.removed.append((number, name))

    def find_dashboard_issue(self):
        return self.dashboard_issue

    def create_dashboard_issue(self, body: str) -> None:
        self.created_body = body

    def update_dashboard_issue(self, number: int, body: str, *, reopen: bool) -> None:
        self.updated = (number, body, reopen)


def test_sla_labels_are_reconciled_without_touching_unrelated_labels() -> None:
    tracker = _tracker()
    client = _FakeClient()
    reviewer_pr = _pull_request(labels={"bug", tracker.AUTHOR_LABEL})
    author_pr = _pull_request(number=43, labels=set())
    clear_pr = _pull_request(number=44, labels={tracker.REVIEWER_LABEL})
    triage_pr = _pull_request(number=45, labels=set())
    reviewer_breach = tracker.classify_pull_request(
        _pull_request(
            requested_reviewers=("reviewer-a",),
            review_request_times={"reviewer-a": _at("2026-07-10T15:00:00")},
        ),
        _at("2026-07-14T16:00:00"),
    )
    author_breach = tracker.classify_pull_request(
        _pull_request(
            number=43,
            latest_review={"state": "CHANGES_REQUESTED", "submitted_at": _at("2026-07-10T10:00:00")},
        ),
        _at("2026-07-14T16:00:00"),
    )
    triage_breach = tracker.classify_pull_request(triage_pr, _at("2026-07-14T16:00:00"))

    tracker.sync_sla_labels(
        client,
        [reviewer_pr, author_pr, clear_pr, triage_pr],
        {42: reviewer_breach, 43: author_breach, 45: triage_breach},
    )

    assert set(client.ensured) == {tracker.REVIEWER_LABEL, tracker.AUTHOR_LABEL, tracker.TRIAGE_LABEL}
    assert client.added == [
        (42, tracker.REVIEWER_LABEL),
        (43, tracker.AUTHOR_LABEL),
        (45, tracker.TRIAGE_LABEL),
    ]
    assert client.removed == [(42, tracker.AUTHOR_LABEL), (44, tracker.REVIEWER_LABEL)]


def test_dashboard_issue_is_created_once_then_reopened_and_updated() -> None:
    tracker = _tracker()
    client = _FakeClient()

    tracker.upsert_dashboard(client, "first body")

    assert client.created_body == "first body"

    client.dashboard_issue = {"number": 123, "state": "closed"}
    tracker.upsert_dashboard(client, "second body")

    assert client.updated == (123, "second body", True)


def test_open_pull_requests_are_fetched_across_graphql_pages() -> None:
    tracker = _tracker()
    node = {
        "number": 42,
        "title": "Make rollouts observable",
        "url": "https://github.com/NVIDIA-NeMo/Gym/pull/42",
        "isDraft": False,
        "headRefName": "feature/observability",
        "createdAt": "2026-07-10T10:00:00Z",
        "author": {"login": "octocat"},
        "labels": {"nodes": []},
        "reviewRequests": {"nodes": []},
        "reviews": {"nodes": []},
        "timelineItems": {"nodes": []},
    }

    class FakeGraphQLClient:
        def __init__(self) -> None:
            self.cursors = []

        def graphql(self, _query: str, variables: dict):
            self.cursors.append(variables["cursor"])
            if variables["cursor"] is None:
                return {
                    "repository": {
                        "pullRequests": {
                            "nodes": [node],
                            "pageInfo": {"hasNextPage": True, "endCursor": "page-2"},
                        }
                    }
                }
            return {
                "repository": {
                    "pullRequests": {
                        "nodes": [{**node, "number": 43}],
                        "pageInfo": {"hasNextPage": False, "endCursor": None},
                    }
                }
            }

    client = FakeGraphQLClient()

    pull_requests = tracker.fetch_pull_requests(client, "NVIDIA-NeMo/Gym")

    assert [pull_request["number"] for pull_request in pull_requests] == [42, 43]
    assert client.cursors == [None, "page-2"]


def test_workflow_runs_hourly_on_weekdays_with_minimal_permissions() -> None:
    workflow_path = Path(__file__).parents[2] / ".github/workflows/pr-sla-tracker.yml"

    assert workflow_path.exists()
    workflow = yaml.load(workflow_path.read_text(), Loader=yaml.BaseLoader)

    assert workflow["on"]["schedule"] == [{"cron": "17 * * * 1-5"}]
    assert "workflow_dispatch" in workflow["on"]
    assert workflow["permissions"] == {
        "contents": "read",
        "issues": "write",
        "pull-requests": "write",
    }
    assert workflow["jobs"]["refresh"]["timeout-minutes"] == "10"


def test_dry_run_writes_live_dashboard_without_github_mutations(tmp_path: Path) -> None:
    tracker = _tracker()
    node = {
        "number": 42,
        "title": "Make rollouts observable",
        "url": "https://github.com/NVIDIA-NeMo/Gym/pull/42",
        "isDraft": False,
        "headRefName": "feature/observability",
        "createdAt": "2026-07-10T10:00:00Z",
        "author": {"login": "octocat"},
        "labels": {"nodes": []},
        "reviewRequests": {"nodes": []},
        "reviews": {"nodes": []},
        "timelineItems": {"nodes": []},
    }

    class ReadOnlyClient:
        def graphql(self, _query: str, _variables: dict):
            return {
                "repository": {
                    "pullRequests": {
                        "nodes": [node, {**node, "number": 43, "title": "Draft should stay hidden", "isDraft": True}],
                        "pageInfo": {"hasNextPage": False, "endCursor": None},
                    }
                }
            }

        def __getattr__(self, name: str):
            raise AssertionError(f"dry-run attempted GitHub mutation: {name}")

    preview_path = tmp_path / "tracker-preview.md"
    output = StringIO()

    result = tracker.run(
        "NVIDIA-NeMo/Gym",
        "unused-token",
        now=_at("2026-07-15T09:00:00"),
        output=output,
        dry_run=True,
        preview_path=preview_path,
        client=ReadOnlyClient(),
    )

    assert result == 0
    preview = preview_path.read_text()
    assert "### Needs review assignment (1)" in preview
    assert "Draft should stay hidden" not in preview
    assert "No GitHub changes made" in output.getvalue()


def test_cli_accepts_dry_run_and_output_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    tracker = _tracker()
    preview_path = tmp_path / "live-preview.md"
    received = {}

    def fake_run(repository: str, token: str, **kwargs) -> int:
        received.update(repository=repository, token=token, **kwargs)
        return 0

    monkeypatch.setenv("GH_TOKEN", "test-token")
    monkeypatch.setattr(tracker, "run", fake_run)

    result = tracker.main(
        [
            "--repo",
            "NVIDIA-NeMo/Gym",
            "--dry-run",
            "--output",
            str(preview_path),
        ]
    )

    assert result == 0
    assert received == {
        "repository": "NVIDIA-NeMo/Gym",
        "token": "test-token",
        "dry_run": True,
        "preview_path": preview_path,
    }
