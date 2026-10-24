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
"""Plan and run one model/profile across every original catalog using Gym's CLI.

A plan is immutable and each catalog can be attempted once. A failed or interrupted
attempt is retained for inspection, never automatically rerun or counted as zero.
"""

import argparse
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from benchmarks.skill_trigger.prepare import BENCHMARK_DIR, ROOT, Selection, digest, encode
from environments.skill_trigger.agent import SkillOptions
from environments.skill_trigger.package import HARNESSES, package_tasks
from environments.skill_trigger.scoring import EvidenceError


RUNTIME_KEYS = {
    "gateway_host",
    "gateway_port",
    "node_binary",
    "provider",
    "api",
    "model_catalog_path",
    "model_catalog_sha256",
}


def select_cases(manifest: dict, pilot: bool) -> list[dict]:
    """Reject reduced or changed releases, then optionally select the frozen pilot."""
    selection = Selection.model_validate_json((BENCHMARK_DIR / "selection.json").read_bytes())
    if manifest.get("selection_sha256") != digest(encode(selection.model_dump())):
        raise EvidenceError("Release is not derived from the shipped selection")
    cases = manifest["cases"]
    if len(cases) != len(selection.cases) or len({case["id"] for case in cases}) != len(cases):
        raise EvidenceError("Release must retain every case exactly once")
    originals = {case.id: case for case in selection.cases}
    for case in cases:
        original = originals.get(case["id"])
        if original is None or any(
            case.get(key) != value for key, value in original.model_dump().items() if key != "provenance"
        ):
            raise EvidenceError("Release case differs from the shipped selection")
    if not pilot:
        return cases
    wanted = json.loads((BENCHMARK_DIR / "pilot.json").read_bytes())["cases"]
    by_id = {case["id"]: case for case in cases}
    if any(by_id[row["id"]]["prompt_sha256"] != row["prompt_sha256"] for row in wanted):
        raise EvidenceError("Pilot prompt differs from its frozen selection")
    return [by_id[row["id"]] for row in wanted]


def plan_run(
    *,
    release: Path,
    manifest_sha256: str,
    output: Path,
    image: str,
    harness: str,
    model: str,
    endpoint: str,
    pilot: bool = False,
    runtime: dict | None = None,
) -> Path:
    release, output = release.resolve(), output.absolute()
    raw = (release / "manifest.json").read_bytes()
    if digest(raw) != manifest_sha256:
        raise EvidenceError("Release manifest hash mismatch")
    manifest = json.loads(raw)
    cases = select_cases(manifest, pilot)
    runtime = dict(runtime or {})
    if runtime.keys() - RUNTIME_KEYS:
        raise ValueError("Runtime settings may only select gateway, Node, provider/API, and Codex catalog")
    if runtime.get("model_catalog_path"):
        runtime["model_catalog_path"] = str(Path(runtime["model_catalog_path"]).resolve())
    options = SkillOptions.model_validate(
        {
            "release": str(release),
            "manifest_sha256": manifest_sha256,
            "harness": harness,
            "evidence_root": str(output / "private"),
            "upstream_url": endpoint,
            "api_key_env": "SKILL_MODEL_API_KEY",
            "gateway_host": "127.0.0.1",
            **runtime,
        }
    )
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    # The packager checks release terms and every source file before producing tasks.
    tasks = package_tasks(
        release=release,
        manifest_sha256=manifest_sha256,
        output=output / "tasks",
        image=image,
        harness=harness,
        gateway_host=options.gateway_host,
    )
    repeats = 1 if pilot else 3
    groups = {}
    for case in cases:
        groups.setdefault(case["catalog"], []).append(case)
    records = []
    for catalog, group in groups.items():
        directory = output / "groups" / catalog
        directory.mkdir(parents=True)
        input_file = directory / "input.jsonl"
        input_file.write_text(
            "".join(
                json.dumps({"task_name": case["task_name"], "responses_create_params": {"input": []}}) + "\n"
                for case in group
            )
        )
        config = {
            "config_paths": [str(BENCHMARK_DIR / "configs" / harness / "config.yaml")],
            "policy_model_name": model,
            "policy_base_url": endpoint,
            "skill_trigger_settings": {
                "release": str(release),
                "manifest_sha256": manifest_sha256,
                "tasks": str(tasks),
                "catalog": catalog,
                "image": image,
                "jobs": str(directory / "jobs"),
                "evidence": str(output / "private"),
                **runtime,
            },
            "num_samples_in_parallel": 2,
            "skill_trigger": {
                "responses_api_agents": {
                    "skill_trigger": {
                        "datasets": [
                            {
                                "name": "skill_trigger",
                                "type": "benchmark",
                                "jsonl_fpath": str(input_file),
                                "prepare_script": "environments/skill_trigger/package.py",
                                "num_repeats": repeats,
                            }
                        ]
                    }
                }
            },
        }
        config_file = directory / "config.yaml"
        config_file.write_text(yaml.safe_dump(config, sort_keys=False))
        records.append(
            {
                "catalog": catalog,
                "cases": [case["id"] for case in group],
                "input_sha256": digest(input_file.read_bytes()),
                "config_sha256": digest(config_file.read_bytes()),
            }
        )
    plan = {
        "profile_files": {
            str(path): digest(path.read_bytes())
            for path in (
                BENCHMARK_DIR / "common.yaml",
                BENCHMARK_DIR / "configs" / harness / "config.yaml",
            )
        },
        "release": str(release),
        "manifest_sha256": manifest_sha256,
        "harness": harness,
        "model": model,
        "scope": "frozen-pilot" if pilot else "full",
        "cases": len(cases),
        "repetitions": repeats,
        "planned_trials": len(cases) * repeats,
        "groups": records,
    }
    (output / "plan.json").write_bytes(encode(plan))
    return output / "plan.json"


def execute_run(output: Path) -> None:
    """Use Gym for scheduling and persistence; keep an exclusive per-group attempt record."""
    output = output.resolve()
    with (output / "execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _execute_groups(output)


def _execute_groups(output: Path) -> None:
    if (ROOT / "env.yaml").exists() or any(
        os.environ.get(key)
        for key in (
            "NEMO_GYM_CONFIG_DICT",
            "NEMO_GYM_EXTRA_ROOTS",
        )
    ):
        raise EvidenceError("Use a clean checkout without env.yaml or inherited Gym config overrides")
    plan = json.loads((output / "plan.json").read_bytes())
    if digest((Path(plan["release"]) / "manifest.json").read_bytes()) != plan["manifest_sha256"]:
        raise EvidenceError("Prepared release changed since planning")
    for path, expected in plan["profile_files"].items():
        if digest(Path(path).read_bytes()) != expected:
            raise EvidenceError("Shared profile changed since planning")
    for group in plan["groups"]:
        directory = output / "groups" / group["catalog"]
        for filename, key in (("input.jsonl", "input_sha256"), ("config.yaml", "config_sha256")):
            if digest((directory / filename).read_bytes()) != group[key]:
                raise EvidenceError("Planned input or config changed; create a new plan instead")
    env = {**os.environ, "NEMO_GYM_MAX_ROLLOUT_ATTEMPTS": "1"}
    for group in plan["groups"]:
        directory = output / "groups" / group["catalog"]
        command = [
            str(Path(sys.executable).with_name("gym")),
            "eval",
            "run",
            "--config",
            str(directory / "config.yaml"),
            "--split",
            "benchmark",
            "--output",
            str(directory / "rollouts.jsonl"),
        ]
        try:
            with (directory / "attempt.json").open("x") as record:
                json.dump({"command": command, "max_rollout_attempts": 1}, record, indent=2)
        except FileExistsError:
            continue  # A crash or failure is also an attempt; never infer permission to retry it.
        with (directory / "console.log").open("xb") as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        (directory / "exit.json").write_bytes(encode({"returncode": result.returncode}))
        if result.returncode:
            raise RuntimeError(f"Gym stopped for {group['catalog']}; inspect its retained console log")

    incomplete = []
    for group in plan["groups"]:
        exit_path = output / "groups" / group["catalog"] / "exit.json"
        if not exit_path.exists() or json.loads(exit_path.read_bytes())["returncode"]:
            incomplete.append(group["catalog"])
    if incomplete:
        raise RuntimeError(f"Earlier attempts still need inspection: {', '.join(incomplete)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("plan")
    for name in ("release", "output"):
        prepare.add_argument(f"--{name}", type=Path, required=True)
    for name in ("manifest-sha256", "image", "model", "endpoint"):
        prepare.add_argument(f"--{name}", required=True)
    prepare.add_argument("--harness", choices=HARNESSES, required=True)
    prepare.add_argument("--pilot", action="store_true")
    prepare.add_argument("--runtime-settings", type=Path, help="YAML with permitted runtime settings; no API keys")
    execute = subparsers.add_parser("execute")
    execute.add_argument("--output", type=Path, required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "plan":
        settings = args.pop("runtime_settings")
        args["runtime"] = yaml.safe_load(settings.read_text()) if settings else None
        print(plan_run(**args))
    else:
        execute_run(**args)


if __name__ == "__main__":
    main()
