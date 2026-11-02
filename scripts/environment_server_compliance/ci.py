# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select the components a change touches, inspect them, and report rule violations as workflow warnings."""

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

from scripts.check_suites.github_actions import (
    COMPARE_BRANCH,
    COMPARE_MODES,
    CompareMode,
    annotation_data,
    changed_paths,
    write_ci_result,
)

from .engine import KINDS


ROOT = Path(__file__).resolve().parents[2]
COMPONENT = re.compile(rf"^({'|'.join(KINDS)})/([^/]+)/")
FINDINGS_FILE = "findings.json"


def subject(component: str) -> str:
    """The component's name in result files and artifact names, which cannot contain `/`."""
    return component.replace("/", "__")


def all_components(root: Path = ROOT) -> list[str]:
    return sorted(
        f"{parent}/{path.name}"
        for parent in KINDS
        if (root / parent).is_dir()
        for path in (root / parent).iterdir()
        if path.is_dir() and not path.name.startswith((".", "_"))
    )


def select_components(paths: list[str], *, root: Path = ROOT) -> list[str]:
    """Components with a changed file; a deleted component directory selects nothing."""
    selected = {match.group(0).rstrip("/") for path in paths if (match := COMPONENT.match(path))}
    return sorted(component for component in selected if (root / component).is_dir())


def changed_components(base: str, *, root: Path = ROOT, compare: CompareMode = COMPARE_BRANCH) -> list[str]:
    paths = changed_paths(base, root=root, subject="components", compare=compare)
    return all_components(root) if paths is None else select_components(paths, root=root)


def inspect_component(component: str) -> list[dict] | None:
    """Install the component's venv as Gym does for its tests, then inspect it there; None if either fails."""
    from nemo_gym.config import GlobalConfigDictParserConfig, get_global_config_dict
    from nemo_gym.server_runtime.supervisor import get_venv_path, run_command, setup_env_command

    # Gym's defaults only: this CLI's own arguments are not Hydra overrides.
    config = get_global_config_dict(GlobalConfigDictParserConfig(skip_load_from_cli=True, skip_load_from_dotenv=True))
    directory = ROOT / component
    install = setup_env_command(directory, config, component.replace("/", "\\/"))
    if run_command(install, directory, project_root=ROOT, global_config_dict=config).wait() != 0:
        return None
    # Gym starts a server from its own directory, so the app may import sibling modules by name.
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(ROOT), str(directory)]),
        "NEMO_GYM_EXTRA_ROOTS": str(ROOT),
    }
    with tempfile.TemporaryDirectory() as scratch:
        result = Path(scratch) / "findings.json"
        command = [
            str(get_venv_path(directory, config) / "bin" / "python"),
            "-m",
            "scripts.environment_server_compliance.inspection",
            component,
            "--output",
            str(result),
        ]
        if subprocess.run(command, cwd=ROOT, env=env).returncode != 0:
            return None
        return json.loads(result.read_text())


def check(component: str, *, output: Path | None = None) -> int:
    """Print a warning per violation and return how many there are; with `output`, also write the result there."""
    findings = inspect_component(component)
    if findings is None:
        detail = annotation_data(f"{component}: could not install or import the component.")
        print(f"::warning title=Environment Server compliance unavailable::{detail}")
    else:
        for finding in findings:
            message = annotation_data(finding["message"])
            print(f"::warning file={finding['path']},line={finding['line']},title={finding['rule']}::{message}")
        print(f"{len(findings)} finding(s).")
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        (output / FINDINGS_FILE).write_text(json.dumps(findings or []) + "\n")
        status = "unavailable" if findings is None else "failed" if findings else "passed"
        write_ci_result(output, subject=subject(component), status=status)
    return len(findings or [])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    select = commands.add_parser("select")
    select.add_argument("--base", default="")
    select.add_argument("--compare", choices=COMPARE_MODES, default=COMPARE_BRANCH)
    run = commands.add_parser("check")
    run.add_argument("component", help="component directory, such as responses_api_agents/my_agent")
    run.add_argument("--output", type=Path, help="write ci-result.json and findings.json here")
    args = parser.parse_args()
    if args.command == "select":
        components = changed_components(args.base, compare=args.compare)
        print(f"Selected components: {components}")
        with Path(os.environ["GITHUB_OUTPUT"]).open("a") as handle:
            handle.write(f"components={json.dumps(components)}\n")
    else:
        check(os.path.normpath(args.component), output=args.output)


if __name__ == "__main__":
    main()
