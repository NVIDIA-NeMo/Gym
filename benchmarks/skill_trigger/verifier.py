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
"""Host-side Harbor verifier for the skill-opening benchmark."""

import json
from pathlib import Path

from harbor.models.trajectories import Trajectory
from harbor.models.verifier.result import VerifierResult
from harbor.verifier.base import BaseVerifier

from benchmarks.skill_trigger.prepare import digest, source_path
from benchmarks.skill_trigger.scoring import EvidenceError, score_trajectory


def load_case(release: Path, manifest_sha256: str, task_name: str) -> tuple[dict, dict[str, str]]:
    raw = source_path(release, "manifest.json").read_bytes()
    if digest(raw) != manifest_sha256:
        raise EvidenceError("Prepared release hash mismatch")
    manifest = json.loads(raw)
    if manifest.get("artifact_kind") != "prepared-release" or manifest.get("release_blockers") != []:
        raise EvidenceError("A permitted prepared release is required")
    cases = [case for case in manifest["cases"] if case["task_name"] == task_name]
    if len(cases) != 1:
        raise EvidenceError("Task is missing or duplicated")
    case = cases[0]
    if digest(case["prompt"].encode()) != case["prompt_sha256"]:
        raise EvidenceError("Prompt hash mismatch")
    sources = {}
    for member in manifest["catalogs"][case["catalog"]]["members"]:
        directory = source_path(release, "catalogs/" + case["catalog"] + "/" + member["name"])
        for relative, expected in member["files"]:
            if digest(source_path(directory, relative).read_bytes()) != expected:
                raise EvidenceError("Skill source changed after preparation")
        sources[member["name"]] = source_path(directory, "SKILL.md").read_text()
    if case["skill"] not in sources:
        raise EvidenceError("Target absent from its catalog")
    return case, sources


class SkillOpeningVerifier(BaseVerifier):
    def __init__(self, *, release: str, manifest_sha256: str, harness: str, **kwargs: object) -> None:
        super().__init__(**kwargs)
        if self.step_name:
            raise EvidenceError("Expected a single-step task")
        self.harness = harness
        self.case, self.sources = load_case(Path(release), manifest_sha256, self.task.short_name)
        if self.task.instruction != self.case["prompt"]:
            raise EvidenceError("Task prompt differs from the prepared release")

    async def verify(self) -> VerifierResult:
        # Harbor collects the agent's trajectory before invoking the host verifier.
        trajectory = Trajectory.model_validate_json((self.trial_paths.agent_dir / "trajectory.json").read_bytes())
        reward = score_trajectory(trajectory, sources=self.sources, target=self.case["skill"], harness=self.harness)
        return VerifierResult(rewards={"reward": reward})
