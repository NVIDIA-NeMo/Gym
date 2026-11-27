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
"""Keep authored controls separate from the canonical release."""

import json

import pytest
from harbor.models.task.task import Task
from harbor.models.trial.paths import TrialPaths

from benchmarks.skill_trigger.negative_controls import prepare as prepare_controls
from benchmarks.skill_trigger.package import prepare as package
from benchmarks.skill_trigger.prepare import digest, prepare_release
from benchmarks.skill_trigger.tests.test_prepare import release  # noqa: F401
from benchmarks.skill_trigger.tests.test_scoring import trajectory
from benchmarks.skill_trigger.verifier import SkillOpeningVerifier


@pytest.mark.asyncio
async def test_separate_controls_preserve_catalog_and_score(release, tmp_path):
    selection, sources = release
    path = prepare_release(selection, sources_root=sources, output=tmp_path / "release")
    manifest = json.loads(path.read_text())
    original = manifest["cases"][0]
    manifest["cases"] = [
        {**original, "suite": suite, "id": suite, "task_name": "case-" + digest(suite.encode())[:12]}
        for suite in ("bionemo", "skillsbench", "addy")
    ]
    path.write_text(json.dumps(manifest))
    before = path.read_bytes()
    controls = prepare_controls(
        release=str(path.parent), manifest_sha256=digest(before), output=str(tmp_path / "controls")
    )
    data = json.loads(controls.read_text())
    assert len(data["cases"]) == 6
    assert {c["case_kind"] for c in data["cases"]} == {"negative"}
    assert data["catalogs"] == manifest["catalogs"]
    assert path.read_bytes() == before
    assert not {c["task_name"] for c in data["cases"]} & {c["task_name"] for c in manifest["cases"]}
    checksum = digest(controls.read_bytes())
    rows = package(release=str(controls.parent), manifest_sha256=checksum, output=str(tmp_path / "tasks"))
    first = json.loads(rows.read_text().splitlines()[0])
    task = Task(rows.parent / first["task_name"])
    paths = TrialPaths(tmp_path / "trial")
    paths.mkdir()
    verifier = SkillOpeningVerifier(
        task=task,
        trial_paths=paths,
        environment=None,
        release=str(controls.parent),
        manifest_sha256=checksum,
        harness="pi",
    )
    value = trajectory(content="Read these instructions.")
    (paths.agent_dir / "trajectory.json").write_text(value.model_dump_json())
    assert (await verifier.verify()).rewards == {
        "reward": 0,
        "all_skill_openings": 1,
        "session_start_openings": 0,
        "unnecessary_skill_openings": 1,
    }
    value.steps[1].tool_calls = None
    value.steps[1].observation = None
    (paths.agent_dir / "trajectory.json").write_text(value.model_dump_json())
    assert (await verifier.verify()).rewards == {
        "reward": 1,
        "all_skill_openings": 0,
        "session_start_openings": 0,
        "unnecessary_skill_openings": 0,
    }
