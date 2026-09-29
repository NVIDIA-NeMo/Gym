# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage preview content without importing tooling or configuration from artifacts."""

import argparse
import json
import os
import re
import shutil
from pathlib import Path


FERN_VERSION = "5.50.4"
CONTENT_ROOTS = ("fern",)
FERN_DIRECTORY = "fern"
CONTENT_SUFFIXES = {".md", ".mdx", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico", ".pdf"}


def verify_metadata(artifact: Path, run: dict, pr: dict, repository: str) -> int:
    """Require API-verified same-repository PR provenance and matching metadata."""
    number = pr["number"]
    if type(number) is not int or number <= 0:
        raise ValueError("Invalid PR number")
    if (
        run["event"] != "pull_request"
        or run["conclusion"] != "success"
        or run["status"] != "completed"
        or run["path"] != ".github/workflows/fern-docs-preview-build.yml"
        or not re.fullmatch(r"[0-9a-f]{40}", run["head_sha"])
        or run["repository"]["full_name"] != repository
        or run["head_repository"]["full_name"] != repository
        or pr["head"]["repo"]["full_name"] != repository
        or pr["base"]["repo"]["full_name"] != repository
        or pr["state"] != "open"
        or pr["head"]["sha"] != run["head_sha"]
        or [entry["number"] for entry in run["pull_requests"]] != [number]
    ):
        raise ValueError("Preview run does not identify a current same-repository PR")
    expected = {
        "pr_number": str(number),
        "head_sha": run["head_sha"],
        "run_id": str(run["id"]),
        "run_attempt": str(run["run_attempt"]),
    }
    for name, value in expected.items():
        path = artifact / "preview-metadata" / name
        if path.is_symlink() or path.read_text().strip() != value:
            raise ValueError(f"Preview metadata mismatch: {name}")
    return number


def stage_content(artifact: Path, trusted: Path, destination: Path) -> None:
    """Overlay document content onto trusted docs; never copy executable tooling."""
    # Reject links even when their suffix would otherwise be ignored.
    for path in artifact.rglob("*"):
        if path.is_symlink():
            raise ValueError("Artifact links are not supported")
    for root in CONTENT_ROOTS:
        source = artifact / root
        if not source.is_dir():
            raise ValueError(f"Missing content directory: {root}")
        shutil.copytree(trusted / root, destination / root, symlinks=False)
        # Removed PR pages must not silently reappear from the trusted checkout.
        for path in (destination / root).rglob("*"):
            if path.is_file() and path.suffix.lower() in CONTENT_SUFFIXES:
                path.unlink()
        for path in source.rglob("*"):
            relative = path.relative_to(source)
            if any(part.startswith(".") for part in relative.parts):
                continue
            if path.is_file() and path.suffix.lower() in CONTENT_SUFFIXES:
                target = destination / root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(path.read_bytes())
    config = destination / FERN_DIRECTORY / "fern.config.json"
    config.write_text(json.dumps({"organization": "nvidia", "version": FERN_VERSION}) + "\n")


def preview_url(output: str) -> str:
    """Accept one HTTPS Fern URL for display only; never send it credentials."""
    urls = re.findall(r"Published docs to (https://[^\s<>]+) \(", output)
    if len(urls) != 1:
        raise ValueError("Expected one preview URL")
    from urllib.parse import urlsplit

    parsed = urlsplit(urls[0])
    if (
        parsed.username
        or parsed.password
        or parsed.port
        or not parsed.hostname
        or not parsed.hostname.endswith(".docs.buildwithfern.com")
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Unexpected preview URL")
    return urls[0]


def main() -> None:
    """Validate preview inputs or emit a display-only URL."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["stage", "url"])
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--trusted", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--pr", type=Path)
    parser.add_argument("--log", type=Path)
    args = parser.parse_args()
    if args.command == "url":
        outputs = {"preview_url": preview_url(args.log.read_text())}
    else:
        run = json.loads(args.run.read_text())
        pr = json.loads(args.pr.read_text())
        number = verify_metadata(args.artifact, run, pr, os.environ["GITHUB_REPOSITORY"])
        stage_content(args.artifact, args.trusted, args.destination)
        outputs = {"pr_number": str(number), "preview_id": f"pr-{number}-{run['head_sha'][:12]}"}
    with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
        for key, value in outputs.items():
            stream.write(f"{key}={value}\n")


if __name__ == "__main__":
    main()
