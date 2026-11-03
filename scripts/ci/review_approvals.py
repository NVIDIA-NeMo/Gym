# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Enforce the GitHub CODEOWNERS policy against GitHub or GitLab approvals.

The executable must run from trusted target-branch code. Policy and identity
links are read from the target commit through the provider API, never the PR.
"""

from __future__ import annotations

import argparse
import base64
import functools
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable


ReadJSON = Callable[[str], object]
CODEOWNERS_PATH = ".github/CODEOWNERS"
IDENTITIES_PATH = ".github/reviewer-identities.json"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """Prevent API redirects from forwarding credentials to another host."""

    def redirect_request(
        self, req: urllib.request.Request, fp: object, code: int, msg: str, headers: object, newurl: str
    ) -> urllib.request.Request | None:
        """Reject redirects; callers must configure the canonical API URL."""
        raise ValueError("API redirected; configure the canonical repository and API URL")


def api_json(base: str, token: str, path: str, *, method: str = "GET", payload: dict | None = None) -> object:
    """Call an authenticated HTTPS API without following redirects."""
    if not token:
        raise ValueError("An API token is required")
    origin = urllib.parse.urlsplit(base)
    if origin.scheme != "https" or origin.username or origin.query or origin.fragment:
        raise ValueError("API URL must be an HTTPS origin or API prefix")
    if path.startswith(("/", "http:", "https:")):
        raise ValueError("API request paths must be relative")
    request = urllib.request.Request(
        f"{base.rstrip('/')}/{path}",
        data=json.dumps(payload).encode() if payload is not None else None,
        method=method,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.build_opener(NoRedirect()).open(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise ValueError(f"API request failed with HTTP {error.code}: {path.split('?')[0]}") from None


def all_pages(read: ReadJSON, path: str) -> list[dict]:
    """Collect complete list responses, failing on unexpected response shapes."""
    result = []
    separator = "&" if "?" in path else "?"
    for page in range(1, 1001):
        items = read(f"{path}{separator}per_page=100&page={page}")
        if not isinstance(items, list):
            raise ValueError(f"Expected a list from {path}")
        result.extend(items)
        if len(items) < 100:
            return result
    raise ValueError(f"Pagination limit exceeded for {path}")


def compile_pattern(pattern: str) -> re.Pattern:
    """Compile GitHub's supported CODEOWNERS glob syntax with path boundaries."""
    if not pattern or pattern.startswith("!") or any(char in pattern for char in "[]\\"):
        raise ValueError(f"Unsupported CODEOWNERS pattern: {pattern!r}")
    anchored = pattern.startswith("/") or "/" in pattern.rstrip("/")
    directory_only = pattern.endswith("/")
    pattern = pattern.lstrip("/").rstrip("/")
    pieces = ["^" if anchored else "^(?:.*/)?"]
    index = 0
    while index < len(pattern):
        if pattern[index : index + 3] == "**/":
            pieces.append("(?:.*/)?")
            index += 3
        elif pattern[index : index + 2] == "**":
            pieces.append(".*")
            index += 2
        else:
            char = pattern[index]
            pieces.append({"*": "[^/]*", "?": "[^/]"}.get(char, re.escape(char)))
            index += 1
    return re.compile("".join(pieces) + ("/.+$" if directory_only else "(?:/.*)?$"))


def parse_policy(text: str) -> list[dict]:
    """Read standard CODEOWNERS rules; advisory comments never gate approvals."""
    rules = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        fields = line.split("#", 1)[0].split()
        if not fields:
            continue
        pattern, *owners = fields
        if any(not re.fullmatch(r"@[A-Za-z0-9-]+/[A-Za-z0-9_.-]+", owner) for owner in owners):
            raise ValueError(f"CODEOWNERS line {line_number}: use GitHub teams, not individual users")
        rules.append({"line": line_number, "pattern": pattern, "owners": owners, "regex": compile_pattern(pattern)})
    if not rules:
        raise ValueError("CODEOWNERS contains no ownership rules")
    return rules


def match_policy(paths: list[str], rules: list[dict]) -> list[dict]:
    """Resolve the last matching rule for every path, including empty overrides."""
    matches = {}
    for path in sorted(set(paths)):
        if path.startswith("/") or ".." in path.split("/"):
            raise ValueError(f"Invalid repository path: {path!r}")
        winner = next((rule for rule in reversed(rules) if rule["regex"].fullmatch(path)), None)
        if winner and winner["owners"]:
            entry = matches.setdefault(
                winner["line"], {key: winner[key] for key in ("line", "pattern", "owners")} | {"files": []}
            )
            entry["files"].append(path)
    return list(matches.values())


def github_approvers(reviews: list[dict], head: str, author_id: int) -> set[int]:
    """Accept only the latest substantive, undismissed approval of this head."""
    latest = {}
    for review in sorted(reviews, key=lambda item: item["id"]):
        if review["state"] in {"APPROVED", "CHANGES_REQUESTED", "DISMISSED"}:
            latest[review["user"]["id"]] = review
    return {
        user_id
        for user_id, review in latest.items()
        if user_id != author_id and review["state"] == "APPROVED" and review["commit_id"] == head
    }


def gitlab_approvers(approvals: list[dict], identities: list[dict], author_id: int) -> tuple[set[int], list[int]]:
    """Link current non-author GitLab approvals using audited numeric account IDs."""
    links = {}
    for identity in identities:
        gitlab_id, github_id = identity["gitlab_user_id"], identity["github_user_id"]
        if type(gitlab_id) is not int or type(github_id) is not int or gitlab_id in links:
            raise ValueError("Reviewer identity links must have unique integer GitLab account IDs")
        links[gitlab_id] = github_id
    approved = {item["user"]["id"] for item in approvals if item["user"]["id"] != author_id}
    return {links[user_id] for user_id in approved if user_id in links}, sorted(approved - links.keys())


def decode_file(data: dict) -> str:
    """Decode a repository file returned by either provider's contents API."""
    if data.get("encoding") != "base64":
        raise ValueError("Expected a base64-encoded repository file")
    return base64.b64decode(data["content"]).decode("utf-8")


def github_snapshot(read: ReadJSON, repository: str, number: int) -> dict:
    """Collect a GitHub head, target policy, complete paths, and current reviews."""
    prefix = f"repos/{repository}"
    pull = read(f"{prefix}/pulls/{number}")
    files = all_pages(read, f"{prefix}/pulls/{number}/files")
    if len(files) != pull["changed_files"] or pull["changed_files"] > 3000:
        raise ValueError("GitHub changed-file response is incomplete")
    paths = [name for file in files for name in (file["filename"], file.get("previous_filename")) if name]
    reviews = all_pages(read, f"{prefix}/pulls/{number}/reviews")
    reference = pull["base"]["sha"]
    policy = decode_file(read(f"{prefix}/contents/{CODEOWNERS_PATH}?ref={reference}"))
    return {
        "head": pull["head"]["sha"],
        "base": reference,
        "paths": paths,
        "policy": policy,
        "approved": github_approvers(reviews, pull["head"]["sha"], pull["user"]["id"]),
        "unlinked": [],
    }


def gitlab_snapshot(read: ReadJSON, project: int, number: int) -> dict:
    """Collect GitLab evidence while using the identical GitHub-format policy."""
    prefix = f"projects/{project}"
    merge = read(f"{prefix}/merge_requests/{number}")
    if not read(f"{prefix}/approvals").get("reset_approvals_on_push"):
        raise ValueError("GitLab must reset approvals on push to reject stale approvals")
    if merge.get("detailed_merge_status") in {"checking", "approvals_syncing"}:
        raise ValueError("GitLab has not finished synchronizing approvals for this head")
    if merge["diff_refs"]["head_sha"] != merge["sha"]:
        raise ValueError("GitLab is still preparing the current head's diff")
    versions = read(f"{prefix}/merge_requests/{number}/versions?per_page=1")
    if not versions or versions[0].get("head_commit_sha") != merge["sha"] or not versions[0].get("patch_id_sha"):
        raise ValueError("GitLab has not finished preparing this head's patch identity")
    diffs = all_pages(read, f"{prefix}/merge_requests/{number}/diffs")
    if not str(merge["changes_count"]).isdigit() or len(diffs) != int(merge["changes_count"]):
        raise ValueError("GitLab changed-file response is incomplete")
    branch = urllib.parse.quote(merge["target_branch"], safe="")
    reference = read(f"{prefix}/repository/branches/{branch}")["commit"]["id"]

    def target_file(path: str) -> str:
        """Read one file from the immutable target commit."""
        encoded = urllib.parse.quote(path, safe="")
        return decode_file(read(f"{prefix}/repository/files/{encoded}?ref={reference}"))

    identities = json.loads(target_file(IDENTITIES_PATH))
    approvals = read(f"{prefix}/merge_requests/{number}/approvals")["approved_by"]
    approved, unlinked = gitlab_approvers(approvals, identities, merge["author"]["id"])
    return {
        "head": merge["sha"],
        "base": reference,
        "paths": [file[key] for file in diffs for key in ("old_path", "new_path")],
        "policy": target_file(CODEOWNERS_PATH),
        "approved": approved,
        "unlinked": unlinked,
    }


def evaluate(matches: list[dict], teams: dict[str, list[dict]], approved: set[int]) -> dict:
    """Require one eligible approval for each winning rule, allowing alternatives."""
    results = []
    for match in matches:
        members = {user["id"]: user["login"] for owner in match["owners"] for user in teams[owner]}
        satisfied = sorted(members[user_id] for user_id in approved & members.keys())
        results.append(match | {"approved_by": satisfied})
    return {"passed": all(result["approved_by"] for result in results), "rules": results}


def run_check(args: argparse.Namespace) -> dict:
    """Fetch live evidence, apply target policy, and reject a changing snapshot."""
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN", "")
    github = functools.partial(api_json, "https://api.github.com", token)
    if args.provider == "github":
        collect = functools.partial(github_snapshot, github, args.repository, args.number)
    else:
        if not args.project_id or not args.gitlab_api:
            raise ValueError("GitLab requires --project-id and --gitlab-api")
        gitlab = functools.partial(api_json, args.gitlab_api, os.environ.get("GITLAB_TOKEN", ""))
        collect = functools.partial(gitlab_snapshot, gitlab, args.project_id, args.number)
    snapshot = collect()
    matches = match_policy(snapshot["paths"], parse_policy(snapshot["policy"]))
    owners = sorted({owner for match in matches for owner in match["owners"]})
    teams = {
        owner: all_pages(github, f"orgs/{owner[1:].split('/')[0]}/teams/{owner.split('/')[1]}/members")
        for owner in owners
    }
    verdict = evaluate(matches, teams, snapshot["approved"])
    current = collect()
    if any(snapshot[key] != current[key] for key in ("head", "base", "paths", "approved", "unlinked")):
        raise ValueError("Request or approvals changed during evaluation; retry the check")
    return verdict | {
        "head": snapshot["head"],
        "base": snapshot["base"],
        "unlinked_gitlab_users": snapshot["unlinked"],
    }


def main(argv: list[str] | None = None) -> int:
    """Run the portable CLI; return 0 for approval, 1 for missing review, 2 for errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("provider", choices=("github", "gitlab"))
    parser.add_argument("--repository", default="NVIDIA-NeMo/Gym")
    parser.add_argument("--number", required=True, type=int)
    parser.add_argument("--project-id", type=int)
    parser.add_argument("--gitlab-api", default=os.environ.get("CI_API_V4_URL"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        report = run_check(args)
    except (ValueError, KeyError, TypeError, OSError, urllib.error.URLError) as error:
        print(f"Approval check error: {error}", file=sys.stderr)
        return 2
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    for rule in report["rules"]:
        state = "PASS" if rule["approved_by"] else "MISSING"
        print(
            f"{state}: {rule['pattern']} ({len(rule['files'])} files): "
            + ", ".join(rule["approved_by"] or rule["owners"])
        )
    if report["unlinked_gitlab_users"]:
        print(f"Ignored GitLab approvals without verified identity links: {report['unlinked_gitlab_users']}")
    print(f"Package approvals: {'passed' if report['passed'] else 'missing'} for {report['head']}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
