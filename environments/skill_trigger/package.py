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
"""Package a permitted prepared release as pinned Harbor tasks.

The release manifest and scoring labels remain on the host. Containers receive
only the frozen prompt and their original skill catalog. Packaging performs no
network requests and never resolves distribution blockers by dropping cases.
"""

import argparse
import json
import re
import shutil
import tempfile
from pathlib import Path

from benchmarks.skill_trigger.prepare import digest, source_path
from environments.skill_trigger.scoring import EvidenceError
from environments.skill_trigger.verifier import load_case


HARNESSES = ("pi", "codex", "codex_guidance", "hermes", "hermes_guidance")


def package_tasks(
    *, release: Path, manifest_sha256: str, output: Path, image: str, harness: str, gateway_host: str
) -> Path:
    """Validate every case and atomically create a new Harbor dataset directory."""
    if harness not in HARNESSES:
        raise ValueError("Unknown skill-opening harness configuration")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/:@-]*", image):
        raise ValueError("Expected one Docker image reference")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", gateway_host):
        raise ValueError("Expected an explicit gateway IP address or hostname")
    release, output = release.resolve(), output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    raw = source_path(release, "manifest.json").read_bytes()
    if digest(raw) != manifest_sha256:
        raise EvidenceError("Prepared manifest differs from the configured release hash")
    manifest = json.loads(raw)
    if manifest.get("artifact_kind") != "prepared-release" or manifest.get("release_blockers") != []:
        raise EvidenceError("Packaging requires resolved file-level terms and a prepared release")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise EvidenceError("Prepared release has no cases")
    names = [case["task_name"] for case in cases]
    if len(set(names)) != len(names) or any(not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", name) for name in names):
        raise EvidenceError("Prepared release contains duplicate or unsafe task names")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".skill-tasks-", dir=output.parent) as temporary:
        staged = Path(temporary) / "dataset"
        staged.mkdir()
        rows = []
        for name in names:
            case = load_case(
                release=release, manifest_sha256=manifest_sha256, task_name=name, model="preparation", harness=harness
            )
            task = staged / name
            environment = task / "environment"
            environment.mkdir(parents=True)
            (task / "instruction.md").write_text(case.prompt)
            (task / "tests").mkdir()
            # Fail closed if a runner accidentally omits the host-side verifier.
            (task / "tests/test.sh").write_text("#!/bin/sh\nexit 1\n")
            for directory in case.skill_directories:
                shutil.copytree(directory, environment / "skills" / directory.name, symlinks=False)
            (environment / "Dockerfile").write_text(f"FROM {image}\nCOPY skills /skills\n")
            (environment / "docker-compose.yaml").write_text("""services:
  main:
    build:
      context: .
      dockerfile: Dockerfile
    image: ${MAIN_IMAGE_NAME}
    command: sleep infinity
    user: "1000"
    cpus: 0.5
    mem_limit: 768m
    pids_limit: 256
    cap_drop: [ALL]
    security_opt: ["no-new-privileges:true"]
""")
            (task / "task.toml").write_text(f"""version = "1.0"
[agent]
timeout_sec = 180
[verifier]
timeout_sec = 30
[environment]
build_timeout_sec = 300
cpus = 1
memory_mb = 768
network_mode = {json.dumps("no-network" if harness == "pi" else "allowlist")}
allowed_hosts = {"[]" if harness == "pi" else json.dumps([gateway_host])}
""")
            rows.append({"task_name": name, "responses_create_params": {"input": []}})
        # Labels and source metadata are deliberately absent from task images.
        (staged / "gym.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        (staged / "package.json").write_text(
            json.dumps(
                {"manifest_sha256": manifest_sha256, "harness": harness, "image": image, "tasks": names}, indent=2
            )
            + "\n"
        )
        staged.rename(output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--harness", choices=HARNESSES, required=True)
    parser.add_argument("--gateway-host", required=True)
    print(package_tasks(**vars(parser.parse_args())))
