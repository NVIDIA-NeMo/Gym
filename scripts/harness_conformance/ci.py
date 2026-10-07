# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select, install, and report advisory harness probes in GitHub Actions."""

import argparse
import importlib
import json
import os
import subprocess
from pathlib import Path

from .registry import HARNESSES


ROOT = Path(__file__).resolve().parents[2]


def select_harnesses(paths: list[str]) -> list[str]:
    """Narrow adapter-only changes; conservatively cover shared/dynamic dependencies."""
    selected = set()
    for path in paths:
        adapter = next((name for name in HARNESSES if path.startswith(f"responses_api_agents/{name}_agent/")), None)
        if adapter:
            selected.add(adapter)
        elif path.startswith(("fern/", "docs/")) or ("/" not in path and path.endswith(".md")) or path == "LICENSE":
            continue
        else:
            # Includes other servers: adapters/tests can import them dynamically.
            # Unknown code and dependency/configuration changes must not be skipped.
            return list(HARNESSES)
    return [name for name in HARNESSES if name in selected]


def changed_harnesses(base: str, *, root: Path = ROOT) -> list[str]:
    """Compare the entire branch since its merge base, including deleted/renamed paths."""
    if not base:
        return list(HARNESSES)
    try:
        ancestor = subprocess.check_output(
            ["git", "merge-base", "--", base, "HEAD"], cwd=root, text=True, stderr=subprocess.PIPE
        ).strip()
        paths = subprocess.check_output(
            ["git", "diff", "--name-only", "--no-renames", "-z", ancestor, "HEAD", "--"], cwd=root
        ).decode("utf-8", errors="replace")
    except subprocess.CalledProcessError:
        print("::warning::Could not determine changed files; running all registered harnesses.")
        return list(HARNESSES)
    return select_harnesses([path for path in paths.split("\0") if path])


def install_runtime(harness: str) -> None:
    """Install the adapter's pinned runtime into this CI job's isolated environment."""
    import yaml

    adapter = ROOT / "responses_api_agents" / f"{harness}_agent"
    subprocess.run(["uv", "pip", "install", "-r", "requirements.txt"], cwd=adapter, check=True)
    if harness == "hermes":
        return  # The Python requirements contain its pinned runtime.
    defaults = yaml.safe_load((adapter / "configs" / f"{harness}_agent.yaml").read_text())
    config = defaults[f"{harness}_agent"]["responses_api_agents"][f"{harness}_agent"]
    version = config[f"{harness}_version"]
    if not version:
        raise ValueError(f"{harness} needs a pinned runtime version")
    setup = importlib.import_module(f"responses_api_agents.{harness}_agent.setup_{harness}")
    # Explicit install: ensure_* accepts existing binaries, which could bypass the pin.
    setup._npm_install("npm", str(version))


def report(harness: str, *, output: Path, exit_code: int) -> str:
    """Render completed measurements separately from unavailable/incomplete checks."""
    summary_path = output / "conformance_summary.json"
    try:
        summary = json.loads(summary_path.read_text())
        complete = (
            summary["runner_status"] == "completed"
            and summary["full_suite"] is True
            and set(summary["harnesses"]) == {harness}
        )
    except (OSError, ValueError, KeyError, TypeError):
        complete = False
    if exit_code not in (0, 1) or not complete:
        message = f"{harness}: conformance could not be evaluated (setup, tests, execution, or checker error)."
        print(f"::warning title=Harness conformance unavailable::{message}")
    elif exit_code == 1:
        message = f"{harness}: conformance requirements are not fulfilled. This check is advisory."
        print(f"::warning title=Harness conformance::{message}")
    else:
        message = f"{harness}: all selected conformance requirements are fulfilled."
    markdown = f"## Harness conformance: {harness}\n\n{message}\n\n"
    markdown += f"Source: `{os.environ.get('GITHUB_SHA', 'local')}`\n\n"
    report_path = output / "conformance_report.md"
    if report_path.exists():
        markdown += report_path.read_text() + "\n"
    markdown += (
        "Download the harness artifact for runtime versions, scenario details, rollouts, captures, and logs.\n\n"
        "Reproduce from this revision with the pinned runtime installed:\n\n"
        f"```bash\npython scripts/run_harness_conformance.py --harness {harness} --output /tmp/new-conformance-run\n```\n"
    )
    return markdown


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    select = commands.add_parser("select")
    select.add_argument("--base", default="")
    install = commands.add_parser("install")
    install.add_argument("--harness", choices=HARNESSES, required=True)
    publish = commands.add_parser("report")
    publish.add_argument("--harness", choices=HARNESSES, required=True)
    publish.add_argument("--output", type=Path, required=True)
    publish.add_argument("--exit-code", type=int, required=True)
    args = parser.parse_args()
    if args.command == "select":
        harnesses = changed_harnesses(args.base)
        print(f"Selected harnesses: {harnesses}")
        with Path(os.environ["GITHUB_OUTPUT"]).open("a") as handle:
            handle.write(f"harnesses={json.dumps(harnesses)}\n")
    elif args.command == "install":
        install_runtime(args.harness)
    else:
        markdown = report(args.harness, output=args.output, exit_code=args.exit_code)
        with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as handle:
            handle.write(markdown)


if __name__ == "__main__":
    main()
