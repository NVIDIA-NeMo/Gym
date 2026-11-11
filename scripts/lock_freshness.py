# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Publish advisory lock freshness statuses without resolving or executing PR code.

Dependency changes on both sides of the merge base suggest stale locks when
either side changes shared inputs or both sides change the same component.
This heuristic deliberately permits false positives; resolver CI remains authoritative.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path, PurePosixPath


STATUS_CONTEXT = "lock-freshness"
SHARED = "."
DIFF_FILE_LIMIT = 300
OPEN_PRS_QUERY = """
query($owner: String!, $name: String!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequests(states: OPEN, baseRefName: "main", first: 50, after: $cursor) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number headRefOid
        files(first: 100) {
          pageInfo { hasNextPage }
          nodes { path }
        }
      }
    }
  }
}
"""


def github_api(endpoint: str, *, payload: dict | None = None) -> dict | list:
    """Call GitHub with structured stdin payloads and propagate API failures."""
    command = ["gh", "api", endpoint]
    if payload is not None:
        command.extend(["--method", "POST", "--input", "-"])
    result = subprocess.run(
        command,
        input=json.dumps(payload) if payload is not None else None,
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    )
    return json.loads(result.stdout)


def dependency_scope(filename: str) -> str | None:
    """Map dependency inputs to their component or the shared resolver scope."""
    path = PurePosixPath(filename)
    if filename.startswith("scripts/component"):
        return SHARED
    if path.name in {"pyproject.toml", "uv.lock", "uv.toml", "setup.py", "setup.cfg", ".python-version"}:
        return path.parent.as_posix()
    return None


def changed_scopes(comparison: dict) -> set[str]:
    """Collect both rename paths and conservatively handle truncated API diffs."""
    files = comparison.get("files", [])
    if len(files) >= DIFF_FILE_LIMIT:
        return {SHARED}
    return {
        scope
        for file in files
        for filename in (file["filename"], file.get("previous_filename", file["filename"]))
        if (scope := dependency_scope(filename)) is not None
    }


def freshness_status(pr_scopes: set[str], main_scopes: set[str]) -> tuple[str, str]:
    """Return an advisory verdict for potentially interacting dependency edits."""
    if pr_scopes and main_scopes and (SHARED in pr_scopes | main_scopes or pr_scopes & main_scopes):
        return "failure", "Locks may be outdated; update from main and recheck locks (advisory)."
    return "success", "No potential lock drift detected (advisory; resolver CI still required)."


def open_dependency_prs(repository: str) -> list[dict]:
    """Batch-filter open PRs so main pushes do not spend REST calls on code-only PRs."""
    owner, name = repository.split("/")
    cursor = None
    pulls = []
    while True:
        response = github_api(
            "graphql",
            payload={"query": OPEN_PRS_QUERY, "variables": {"owner": owner, "name": name, "cursor": cursor}},
        )
        if response.get("errors"):
            raise ValueError(f"GitHub GraphQL errors: {response['errors']}")
        connection = response["data"]["repository"]["pullRequests"]
        for pull in connection["nodes"]:
            files = pull["files"]
            if files["pageInfo"]["hasNextPage"] or any(dependency_scope(file["path"]) for file in files["nodes"]):
                pulls.append(
                    {
                        "number": pull["number"],
                        "state": "open",
                        "base": {"ref": "main"},
                        "head": {"sha": pull["headRefOid"]},
                    }
                )
        if not connection["pageInfo"]["hasNextPage"]:
            return pulls
        cursor = connection["pageInfo"]["endCursor"]


def evaluate_pr(repository: str, pull: dict, base_sha: str) -> dict:
    """Compare each branch from their merge base, including newly added locks."""
    head_sha = pull["head"]["sha"]
    pr_diff = github_api(f"repos/{repository}/compare/{base_sha}...{head_sha}?per_page=1")
    pr_scopes = changed_scopes(pr_diff)
    main_scopes = set()
    if pr_scopes and pr_diff["merge_base_commit"]["sha"] != base_sha:
        main_diff = github_api(f"repos/{repository}/compare/{head_sha}...{base_sha}?per_page=1")
        main_scopes = changed_scopes(main_diff)
    state, description = freshness_status(pr_scopes, main_scopes)
    return {
        "number": pull["number"],
        "head_sha": head_sha,
        "base_sha": base_sha,
        "merge_base_sha": pr_diff["merge_base_commit"]["sha"],
        "pr_scopes": sorted(pr_scopes),
        "main_scopes": sorted(main_scopes),
        "state": state,
        "description": description,
    }


def publish_status(repository: str, result: dict, target_url: str) -> str:
    """Update the PR head status only while the evaluated PR revision is current."""
    pull = github_api(f"repos/{repository}/pulls/{result['number']}")
    if pull["state"] != "open" or pull["base"]["ref"] != "main" or pull["head"]["sha"] != result["head_sha"]:
        return "superseded"
    current_base = github_api(f"repos/{repository}/git/ref/heads/main")
    if current_base["object"]["sha"] != result["base_sha"]:
        return "superseded"
    endpoint = f"repos/{repository}/commits/{result['head_sha']}/status"
    existing = github_api(endpoint)["statuses"]
    previous = next((status for status in existing if status["context"] == STATUS_CONTEXT), None)
    if previous and all(previous[key] == result[key] for key in ("state", "description")):
        return "unchanged"
    github_api(
        f"repos/{repository}/statuses/{result['head_sha']}",
        payload={
            "context": STATUS_CONTEXT,
            "state": result["state"],
            "description": result["description"],
            "target_url": target_url,
        },
    )
    return "published"


def refresh(repository: str, *, pr_number: int | None, dry_run: bool, target_url: str) -> list[dict]:
    """Refresh one PR or all open main-targeting PRs, retaining per-PR errors."""
    if pr_number is not None:
        pulls = [github_api(f"repos/{repository}/pulls/{pr_number}")]
    else:
        pulls = open_dependency_prs(repository)
    base_sha = github_api(f"repos/{repository}/git/ref/heads/main")["object"]["sha"]
    results = []
    for pull in pulls:
        if pull["state"] != "open" or pull["base"]["ref"] != "main":
            continue
        try:
            result = evaluate_pr(repository, pull, base_sha)
            result["publication"] = "dry-run" if dry_run else publish_status(repository, result, target_url)
        except (subprocess.CalledProcessError, KeyError, ValueError) as error:
            result = {"number": pull["number"], "error": str(error)}
        print(json.dumps(result), flush=True)
        results.append(result)
    return results


def main() -> None:
    """Run the advisory refresh and record evidence in the workflow summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--pr", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    server_url = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    run_id = os.environ.get("GITHUB_RUN_ID")
    target_url = f"{server_url}/{args.repo}/actions/runs/{run_id}" if run_id else f"{server_url}/{args.repo}/actions"
    results = refresh(args.repo, pr_number=args.pr, dry_run=args.dry_run, target_url=target_url)
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(summary).open("a") as output:
            output.write("### Lock freshness (advisory)\n\n")
            output.write(
                "Possible drift only; this does not validate or regenerate locks. Keep this status optional.\n\n"
            )
            output.write("```json\n" + json.dumps(results, indent=2) + "\n```\n")
    if any("error" in result for result in results):
        raise SystemExit("Some PRs could not be evaluated; see the per-PR diagnostics.")


if __name__ == "__main__":
    main()
