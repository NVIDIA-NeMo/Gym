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
"""Create a separate negative-control release from an already cleared release."""

import argparse
import json
import shutil
from pathlib import Path

from benchmarks.skill_trigger.prepare import digest, encode, source_path
from benchmarks.skill_trigger.scoring import EvidenceError
from benchmarks.skill_trigger.verifier import load_case


SESSION_START_SKILLS = {"addy": ("using-agent-skills", "context-engineering")}


PROMPTS = {
    "bionemo": ["What is 7 + 8? Reply with the number only.", "Repeat this word exactly: sunflower"],
    "skillsbench": ["What is 12 minus 5? Reply with the number only.", "Repeat this word exactly: lantern"],
    "addy": ["What is 3 times 4? Reply with the number only.", "Repeat this word exactly: pebble"],
}


def prepare(*, release: str, manifest_sha256: str, output: str) -> Path:
    """Retain full source catalogs and use distinct identities for six controls."""
    root, destination = Path(release).resolve(), Path(output).absolute()
    raw = source_path(root, "manifest.json").read_bytes()
    if digest(raw) != manifest_sha256:
        raise EvidenceError("Prepared release hash mismatch")
    manifest = json.loads(raw)
    if manifest.get("artifact_kind") != "prepared-release" or manifest.get("release_blockers") != []:
        raise EvidenceError("Negative controls require a cleared release")
    cases = []
    for suite, prompts in PROMPTS.items():
        candidates = sorted((c for c in manifest["cases"] if c["suite"] == suite), key=lambda c: c["id"])
        if not candidates:
            raise EvidenceError(f"Missing cleared collection: {suite}")
        template, sources = load_case(root, manifest_sha256, candidates[0]["task_name"])
        for index, prompt in enumerate(prompts):
            identifier = f"negative-{suite}-{index + 1}"
            cases.append(
                {
                    **template,
                    "id": identifier,
                    "task_name": "case-" + digest(identifier.encode())[:12],
                    "case_kind": "negative",
                    "skill": None,
                    "allowed_session_start_skills": [
                        name for name in SESSION_START_SKILLS.get(suite, ()) if name in sources
                    ],
                    "prompt": prompt,
                    "prompt_sha256": digest(prompt.encode()),
                    "provenance": {"kind": "authored-negative-control", "parent_manifest_sha256": manifest_sha256},
                }
            )
    if destination.exists():
        raise FileExistsError(destination)
    # Packaging rechecks member hashes. Source release remains immutable.
    shutil.copytree(root, destination)
    manifest.update({"cases": cases, "control_set_version": "0.2.1", "parent_manifest_sha256": manifest_sha256})
    path = destination / "manifest.json"
    path.write_bytes(encode(manifest))
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", required=True)
    print(prepare(**vars(parser.parse_args())))
