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
"""Track pull requests that exceed the team's review handoff SLA."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, TextIO
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen


SLA_BUSINESS_DAYS = 1
REVIEWER_LABEL = "sla:review-overdue"
AUTHOR_LABEL = "sla:author-overdue"
TRIAGE_LABEL = "sla:triage-overdue"
EXCLUDED_LABELS = frozenset({"stale", "paused", "do-not-track"})
NEW_BREACH_BUSINESS_SECONDS = 24 * 60 * 60
DASHBOARD_MARKER = "<!-- pr-sla-tracker -->"
DASHBOARD_TITLE = "[Tracker] Pull request handoff SLA"
GRAPHQL_QUERY = """
query PullRequestSlaTracker($owner: String!, $name: String!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequests(states: OPEN, first: 50, after: $cursor, orderBy: {field: UPDATED_AT, direction: DESC}) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number
        title
        url
        isDraft
        headRefName
        createdAt
        author { __typename login }
        labels(first: 100) { nodes { name } }
        reviewRequests(first: 100) {
          nodes {
            requestedReviewer {
              __typename
              ... on User { login }
              ... on Mannequin { login }
              ... on Team { slug organization { login } }
            }
          }
        }
        reviews(last: 100) { nodes { state submittedAt } }
        timelineItems(last: 100, itemTypes: [READY_FOR_REVIEW_EVENT, REVIEW_REQUESTED_EVENT]) {
          nodes {
            __typename
            ... on ReadyForReviewEvent { createdAt }
            ... on ReviewRequestedEvent {
              createdAt
              requestedReviewer {
                __typename
                ... on User { login }
                ... on Mannequin { login }
                ... on Team { slug organization { login } }
              }
            }
          }
        }
      }
    }
  }
}
"""


@dataclass(frozen=True)
class Breach:
    """An overdue author, reviewer, or review-assignment handoff."""

    number: int
    title: str
    url: str
    owner: Literal["author", "reviewer", "triage"]
    people: tuple[str, ...]
    since: datetime
    deadline: datetime
    reason: str


class GitHubClient:
    """Small GitHub REST/GraphQL client using only the Python standard library."""

    def __init__(
        self,
        repository: str,
        token: str,
        *,
        api_url: str = "https://api.github.com",
        opener=urlopen,
    ) -> None:
        self.repository = repository
        self.api_url = api_url.rstrip("/")
        self.token = token
        self.opener = opener

    def graphql(self, query: str, variables: dict) -> dict:
        payload = self._request("POST", "/graphql", {"query": query, "variables": variables})
        if payload.get("errors"):
            raise RuntimeError(f"GitHub GraphQL error: {payload['errors']}")
        return payload["data"]

    def ensure_label(self, name: str, color: str, description: str) -> None:
        path = f"/repos/{self.repository}/labels/{quote(name, safe='')}"
        if self._request("GET", path, allow_not_found=True) is None:
            self._request(
                "POST",
                f"/repos/{self.repository}/labels",
                {"name": name, "color": color, "description": description},
            )

    def add_label(self, number: int, name: str) -> None:
        self._request("POST", f"/repos/{self.repository}/issues/{number}/labels", {"labels": [name]})

    def remove_label(self, number: int, name: str) -> None:
        path = f"/repos/{self.repository}/issues/{number}/labels/{quote(name, safe='')}"
        self._request("DELETE", path, allow_not_found=True)

    def find_dashboard_issue(self) -> dict | None:
        page = 1
        while True:
            issues = self._request(
                "GET",
                f"/repos/{self.repository}/issues?state=all&per_page=100&page={page}",
            )
            for issue in issues:
                if "pull_request" not in issue and DASHBOARD_MARKER in (issue.get("body") or ""):
                    return issue
            if len(issues) < 100:
                return None
            page += 1

    def create_dashboard_issue(self, body: str) -> None:
        self._request(
            "POST",
            f"/repos/{self.repository}/issues",
            {"title": DASHBOARD_TITLE, "body": body},
        )

    def update_dashboard_issue(self, number: int, body: str, *, reopen: bool) -> None:
        payload = {"body": body}
        if reopen:
            payload["state"] = "open"
        self._request("PATCH", f"/repos/{self.repository}/issues/{number}", payload)

    def _request(
        self,
        method: str,
        path: str,
        payload: dict | None = None,
        *,
        allow_not_found: bool = False,
    ):
        data = json.dumps(payload).encode() if payload is not None else None
        request = Request(
            f"{self.api_url}{path}",
            data=data,
            method=method,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "nemo-gym-pr-sla-tracker",
            },
        )
        try:
            with self.opener(request) as response:
                body = response.read()
        except HTTPError as error:
            if allow_not_found and error.code == 404:
                return None
            detail = error.read().decode(errors="replace")
            raise RuntimeError(f"GitHub API {method} {path} failed ({error.code}): {detail}") from error
        return json.loads(body) if body else None


def business_day_deadline(start: datetime, business_days: int = SLA_BUSINESS_DAYS) -> datetime:
    """Return the same wall-clock time after ``business_days``, skipping weekends."""
    if start.tzinfo is None:
        raise ValueError("start must be timezone-aware")
    if business_days < 0:
        raise ValueError("business_days must not be negative")

    deadline = start
    remaining = business_days
    while remaining:
        deadline += timedelta(days=1)
        if deadline.weekday() < 5:
            remaining -= 1
    return deadline


def classify_pull_request(pull_request: dict, now: datetime) -> Breach | None:
    """Return the current SLA breach for a normalized pull-request snapshot."""
    if pull_request["is_draft"]:
        return None

    author = pull_request["author"] or "ghost"
    excluded_by_label = EXCLUDED_LABELS.intersection(label.casefold() for label in pull_request["labels"])
    if (
        pull_request.get("author_is_bot", False)
        or author.endswith("[bot]")
        or pull_request["head_ref"].startswith("release-please--")
        or excluded_by_label
    ):
        return None

    requested_reviewers = pull_request["requested_reviewers"]
    if requested_reviewers:
        request_times = pull_request["review_request_times"]
        fallback = pull_request["ready_at"] or pull_request["created_at"]
        overdue = []
        for reviewer in requested_reviewers:
            since = request_times.get(reviewer, fallback)
            deadline = business_day_deadline(since)
            if now > deadline:
                overdue.append((reviewer, since, deadline))

        if not overdue:
            return None

        people = tuple(sorted(reviewer for reviewer, _, _ in overdue))
        since = min(item[1] for item in overdue)
        deadline = min(item[2] for item in overdue)
        return _breach(pull_request, "reviewer", people, since, deadline, "review requested")

    latest_review = pull_request["latest_review"]
    if latest_review is None:
        since = pull_request["ready_at"] or pull_request["created_at"]
        deadline = business_day_deadline(since)
        if now <= deadline:
            return None
        return _breach(pull_request, "triage", (), since, deadline, "reviewer not assigned")
    elif latest_review["state"] == "CHANGES_REQUESTED":
        since = latest_review["submitted_at"]
        reason = "changes requested"
    else:
        return None

    deadline = business_day_deadline(since)
    if now <= deadline:
        return None
    return _breach(pull_request, "author", (author,), since, deadline, reason)


def normalize_pull_request(node: dict) -> dict:
    """Convert a GitHub GraphQL pull-request node into the tracker model."""
    timeline = node["timelineItems"]["nodes"]
    ready_times = [
        _parse_timestamp(event["createdAt"]) for event in timeline if event["__typename"] == "ReadyForReviewEvent"
    ]
    request_times: dict[str, datetime] = {}
    for event in timeline:
        if event["__typename"] != "ReviewRequestedEvent":
            continue
        reviewer = _reviewer_name(event["requestedReviewer"])
        if reviewer:
            event_time = _parse_timestamp(event["createdAt"])
            request_times[reviewer] = max(request_times.get(reviewer, event_time), event_time)

    reviewers = sorted(
        reviewer
        for request in node["reviewRequests"]["nodes"]
        if (reviewer := _reviewer_name(request["requestedReviewer"]))
    )
    reviews = [review for review in node["reviews"]["nodes"] if review.get("submittedAt")]
    latest_review = max(reviews, key=lambda review: review["submittedAt"], default=None)

    return {
        "number": node["number"],
        "title": node["title"],
        "url": node["url"],
        "author": (node.get("author") or {}).get("login"),
        "author_is_bot": (node.get("author") or {}).get("__typename") == "Bot",
        "is_draft": node["isDraft"],
        "head_ref": node["headRefName"],
        "created_at": _parse_timestamp(node["createdAt"]),
        "ready_at": max(ready_times, default=None),
        "requested_reviewers": tuple(reviewers),
        "review_request_times": request_times,
        "latest_review": (
            {
                "state": latest_review["state"],
                "submitted_at": _parse_timestamp(latest_review["submittedAt"]),
            }
            if latest_review
            else None
        ),
        "labels": {label["name"] for label in node["labels"]["nodes"]},
    }


def fetch_pull_requests(client: GitHubClient, repository: str) -> list[dict]:
    """Fetch and normalize all open pull requests, following GraphQL pagination."""
    try:
        owner, name = repository.split("/", maxsplit=1)
    except ValueError as error:
        raise ValueError("repository must use OWNER/NAME format") from error

    pull_requests = []
    cursor = None
    while True:
        data = client.graphql(GRAPHQL_QUERY, {"owner": owner, "name": name, "cursor": cursor})
        connection = data["repository"]["pullRequests"]
        pull_requests.extend(normalize_pull_request(node) for node in connection["nodes"])
        if not connection["pageInfo"]["hasNextPage"]:
            return pull_requests
        cursor = connection["pageInfo"]["endCursor"]


def sync_sla_labels(client, pull_requests: list[dict], breaches_by_number: dict[int, Breach]) -> None:
    """Add and remove only the SLA labels owned by this tracker."""
    client.ensure_label(REVIEWER_LABEL, "B60205", "Review response is over the one-business-day SLA")
    client.ensure_label(AUTHOR_LABEL, "D93F0B", "Author response is over the one-business-day SLA")
    client.ensure_label(TRIAGE_LABEL, "FBCA04", "Review assignment is over the one-business-day SLA")

    for pull_request in pull_requests:
        breach = breaches_by_number.get(pull_request["number"])
        desired = None
        if breach:
            desired = {
                "reviewer": REVIEWER_LABEL,
                "author": AUTHOR_LABEL,
                "triage": TRIAGE_LABEL,
            }[breach.owner]

        current = pull_request["labels"]
        for label in (REVIEWER_LABEL, AUTHOR_LABEL, TRIAGE_LABEL):
            if label == desired and label not in current:
                client.add_label(pull_request["number"], label)
            elif label != desired and label in current:
                client.remove_label(pull_request["number"], label)


def upsert_dashboard(client, body: str) -> None:
    """Create the dashboard issue once, then keep the same issue current."""
    issue = client.find_dashboard_issue()
    if issue is None:
        client.create_dashboard_issue(body)
        return
    client.update_dashboard_issue(issue["number"], body, reopen=issue["state"] == "closed")


def _breach(
    pull_request: dict,
    owner: Literal["author", "reviewer", "triage"],
    people: tuple[str, ...],
    since: datetime,
    deadline: datetime,
    reason: str,
) -> Breach:
    return Breach(
        number=pull_request["number"],
        title=pull_request["title"],
        url=pull_request["url"],
        owner=owner,
        people=people,
        since=since,
        deadline=deadline,
        reason=reason,
    )


def render_dashboard(breaches: list[Breach], generated_at: datetime) -> str:
    """Render the stable GitHub issue body used as the daily dashboard."""
    lines = [
        DASHBOARD_MARKER,
        "# Pull request handoff SLA tracker",
        "",
        "Review during the daily stand-up alongside the architecture/RFC decision tracker.",
        "",
        "The SLA is one business day per handoff; weekends are excluded and public holidays are not modeled.",
        "Drafts, bots, and PRs labeled `stale`, `paused`, or `do-not-track` are excluded.",
        f"Last refreshed: {generated_at.isoformat(timespec='minutes')}",
        "",
    ]

    if not breaches:
        lines.append("No pull requests have breached the one-business-day handoff SLA.")
        return "\n".join(lines) + "\n"

    newly_breached = [
        breach
        for breach in breaches
        if _business_seconds_between(breach.deadline, generated_at) <= NEW_BREACH_BUSINESS_SECONDS
    ]
    backlog = [breach for breach in breaches if breach not in newly_breached]

    lines.extend([f"## Newly breached ({len(newly_breached)})", ""])
    if newly_breached:
        _append_owner_sections(lines, newly_breached, generated_at)
    else:
        lines.extend(["No new breaches in the last business day.", ""])

    if backlog:
        lines.extend(
            [
                "<details>",
                f"<summary><strong>Older breach backlog ({len(backlog)})</strong></summary>",
                "",
            ]
        )
        _append_owner_sections(lines, backlog, generated_at)
        lines.extend(["</details>", ""])
    return "\n".join(lines) + "\n"


def _append_owner_sections(lines: list[str], breaches: list[Breach], generated_at: datetime) -> None:
    sections = (
        ("reviewer", "Waiting on reviewers"),
        ("author", "Waiting on authors"),
        ("triage", "Needs review assignment"),
    )
    for owner, title in sections:
        owned_breaches = [breach for breach in breaches if breach.owner == owner]
        if owned_breaches:
            _append_section(lines, title, owned_breaches, generated_at)


def _append_section(lines: list[str], title: str, breaches: list[Breach], generated_at: datetime) -> None:
    lines.extend(
        [
            f"### {title} ({len(breaches)})",
            "",
            "| PR | Waiting on | Reason | Handoff | SLA deadline | Overdue |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for breach in sorted(breaches, key=lambda item: (item.deadline, item.number)):
        people = "Maintainers" if breach.owner == "triage" else ", ".join(_mention(person) for person in breach.people)
        title_text = _escape_table_cell(breach.title)
        lines.append(
            f"| [#{breach.number} — {title_text}]({breach.url}) | {people} | {breach.reason} | "
            f"{_format_timestamp(breach.since)} | {_format_timestamp(breach.deadline)} | "
            f"{_format_overdue(breach.deadline, generated_at)} |"
        )
    lines.append("")


def _mention(person: str) -> str:
    if "/" in person:
        organization, team = person.split("/", maxsplit=1)
        return f"[@{person}](https://github.com/orgs/{organization}/teams/{team})"
    return f"[@{person}](https://github.com/{person})"


def _escape_table_cell(value: str) -> str:
    return " ".join(value.splitlines()).replace("|", "\\|")


def _format_timestamp(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M UTC")


def _format_overdue(deadline: datetime, now: datetime) -> str:
    total_seconds = max(0, int(_business_seconds_between(deadline, now)))
    days, remainder = divmod(total_seconds, 24 * 60 * 60)
    hours = remainder // (60 * 60)
    if days:
        return f"{days}d {hours}h"
    return f"{hours}h"


def _business_seconds_between(start: datetime, end: datetime) -> float:
    if end <= start:
        return 0

    total = 0.0
    cursor = start
    while cursor.date() < end.date():
        next_day = cursor.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        if cursor.weekday() < 5:
            total += (next_day - cursor).total_seconds()
        cursor = next_day
    if cursor.weekday() < 5:
        total += (end - cursor).total_seconds()
    return total


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _reviewer_name(reviewer: dict | None) -> str | None:
    if not reviewer:
        return None
    if reviewer["__typename"] == "Team":
        return f"{reviewer['organization']['login']}/{reviewer['slug']}"
    return reviewer.get("login")


def run(
    repository: str,
    token: str,
    *,
    now: datetime | None = None,
    output: TextIO = sys.stdout,
    dry_run: bool = False,
    preview_path: Path | None = None,
    client: GitHubClient | None = None,
) -> int:
    """Refresh labels and the dashboard for one repository."""
    generated_at = now or datetime.now(UTC)
    client = client or GitHubClient(repository, token)
    pull_requests = fetch_pull_requests(client, repository)
    breaches = [
        breach
        for pull_request in pull_requests
        if (breach := classify_pull_request(pull_request, generated_at)) is not None
    ]
    breaches_by_number = {breach.number: breach for breach in breaches}
    dashboard = render_dashboard(breaches, generated_at)
    if dry_run:
        if preview_path is None:
            raise ValueError("preview_path is required for a dry run")
        preview_path.write_text(dashboard, encoding="utf-8")
        print(f"Preview written to {preview_path}. No GitHub changes made.", file=output)
        return 0

    sync_sla_labels(client, pull_requests, breaches_by_number)
    upsert_dashboard(client, dashboard)

    reviewer_count = sum(breach.owner == "reviewer" for breach in breaches)
    author_count = sum(breach.owner == "author" for breach in breaches)
    triage_count = sum(breach.owner == "triage" for breach in breaches)
    print(
        f"Tracked {len(pull_requests)} open pull requests: "
        f"{reviewer_count} reviewer breach(es), {author_count} author breach(es), "
        f"{triage_count} review-assignment breach(es).",
        file=output,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", default=os.environ.get("GITHUB_REPOSITORY"), help="GitHub repository in OWNER/NAME form"
    )
    parser.add_argument("--dry-run", action="store_true", help="write a local preview without changing GitHub")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tracker-preview.md"),
        help="dry-run Markdown output path (default: tracker-preview.md)",
    )
    args = parser.parse_args(argv)
    if not args.repo:
        parser.error("--repo or GITHUB_REPOSITORY is required")
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        parser.error("GH_TOKEN or GITHUB_TOKEN is required")
    return run(args.repo, token, dry_run=args.dry_run, preview_path=args.output)


if __name__ == "__main__":
    raise SystemExit(main())
