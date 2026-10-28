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
"""Package cleared source data as standard Harbor tasks; no custom runner."""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from benchmarks.skill_trigger.prepare import digest, source_path
from benchmarks.skill_trigger.scoring import EvidenceError
from benchmarks.skill_trigger.verifier import load_case


DOCKERFILE = """FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash ca-certificates curl git libatomic1 python3 python3-venv sudo \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash agent
WORKDIR /workspace
COPY skills /skills
RUN chmod -R a-w /skills && mkdir -p /workspace && chown agent:agent /workspace
USER agent
"""
TASK_CONFIG = """version = "1.0"
[agent]
timeout_sec = 120
user = "agent"
[verifier]
timeout_sec = 30
[environment]
build_timeout_sec = 600
cpus = 2
memory_mb = 4096
skills_dir = "/skills"
"""


def prepare(*, release: str, manifest_sha256: str, output: str) -> Path:
    """Create all tasks with unchanged prompts/catalogs; refuse uncleared releases."""
    release_path, destination = Path(release).resolve(), Path(output).absolute()
    raw = source_path(release_path, "manifest.json").read_bytes()
    if digest(raw) != manifest_sha256:
        raise EvidenceError("Prepared release hash mismatch")
    manifest = json.loads(raw)
    if manifest.get("artifact_kind") != "prepared-release" or manifest.get("release_blockers") != []:
        raise EvidenceError("Packaging requires resolved source terms")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    names = [case["task_name"] for case in manifest["cases"]]
    if not names or len(set(names)) != len(names):
        raise EvidenceError("Missing or duplicate task names")
    if any(not name.startswith("case-") or any(c not in "0123456789abcdef" for c in name[5:]) for name in names):
        raise EvidenceError("Unsafe task name")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".skill-tasks-", dir=destination.parent) as temporary:
        staged = Path(temporary) / "tasks"
        staged.mkdir()
        rows = []
        for name in names:
            case, _ = load_case(release_path, manifest_sha256, name)
            task = staged / name
            environment = task / "environment"
            environment.mkdir(parents=True)
            (task / "instruction.md").write_text(case["prompt"])
            (task / "task.toml").write_text(TASK_CONFIG)
            (environment / "Dockerfile").write_text(DOCKERFILE)
            (task / "tests").mkdir()
            (task / "tests/test.sh").write_text(
                "#!/bin/sh\n# Requires the host-side skill-opening verifier.\nexit 1\n"
            )
            for member in manifest["catalogs"][case["catalog"]]["members"]:
                source = source_path(release_path, "catalogs/" + case["catalog"] + "/" + member["name"])
                for relative, _ in member["files"]:
                    src = source_path(source, relative)
                    dst = environment / "skills" / member["name"] / relative
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
            rows.append({"task_name": name, "responses_create_params": {"input": []}})
        (staged / "gym.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        (staged / "package.json").write_text(
            json.dumps({"benchmark_version": "0.2.0", "manifest_sha256": manifest_sha256})
        )
        staged.rename(destination)
    return destination / "gym.jsonl"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", required=True)
    print(prepare(**vars(parser.parse_args())))
