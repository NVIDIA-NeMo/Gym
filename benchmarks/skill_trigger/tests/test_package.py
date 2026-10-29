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
"""Exercise source preparation, Harbor packaging, and host verification together."""

import json

import pytest
from harbor.models.task.task import Task
from harbor.models.trial.paths import TrialPaths

from benchmarks.skill_trigger.package import prepare
from benchmarks.skill_trigger.prepare import digest, prepare_release
from benchmarks.skill_trigger.scoring import EvidenceError
from benchmarks.skill_trigger.tests.test_prepare import release  # noqa: F401
from benchmarks.skill_trigger.tests.test_scoring import trajectory
from benchmarks.skill_trigger.verifier import SkillOpeningVerifier, has_uncollected_delegation


@pytest.fixture
def packaged(release, tmp_path):
    selection, sources = release
    manifest = prepare_release(selection, sources_root=sources, output=tmp_path / "release")
    checksum = digest(manifest.read_bytes())
    rows = prepare(release=str(manifest.parent), manifest_sha256=checksum, output=str(tmp_path / "tasks"))
    name = json.loads(rows.read_text())["task_name"]
    return manifest, checksum, rows, Task(rows.parent / name)


def test_standard_harbor_task_preserves_prompt_and_catalog(packaged):
    manifest, _, rows, task = packaged
    assert task.instruction == "Adapted public prompt"
    assert task.config.environment.skills_dir == "/skills"
    assert task.config.agent.timeout_sec == 120
    environment = rows.parent / task.short_name / "environment"
    assert (environment / "skills/target/SKILL.md").is_file()
    assert not (environment / "skills/target/evals").exists()
    assert not (environment / "manifest.json").exists()
    assert "target" not in (rows.parent / task.short_name / "task.toml").read_text()
    assert json.loads(manifest.read_text())["release_blockers"] == []


@pytest.mark.asyncio
async def test_verifier_uses_host_trajectory_and_rejects_missing_evidence(packaged, tmp_path):
    manifest, checksum, _, task = packaged
    paths = TrialPaths(tmp_path / "trial")
    paths.mkdir()
    verifier = SkillOpeningVerifier(
        task=task,
        trial_paths=paths,
        environment=None,
        release=str(manifest.parent),
        manifest_sha256=checksum,
        harness="pi",
    )
    with pytest.raises(FileNotFoundError):
        await verifier.verify()
    value = trajectory(content="Read these instructions.")
    (paths.agent_dir / "trajectory.json").write_text(value.model_dump_json())
    assert (await verifier.verify()).rewards == {"reward": 1}
    value.steps[1].observation.results[0].content = "Not the instructions"
    (paths.agent_dir / "trajectory.json").write_text(value.model_dump_json())
    assert (await verifier.verify()).rewards == {"reward": 0}
    verifier.harness = "codex"
    value.agent.extra = {"uncollected_subagents": True}
    (paths.agent_dir / "trajectory.json").write_text(value.model_dump_json())
    with pytest.raises(EvidenceError, match="Delegated outcome"):
        await verifier.verify()
    value.steps[1].observation.results[0].content = "Read these instructions."
    (paths.agent_dir / "trajectory.json").write_text(value.model_dump_json())
    assert (await verifier.verify()).rewards == {"reward": 1}


def test_release_changes_and_blockers_fail_before_packaging(packaged, tmp_path):
    manifest, checksum, _, _ = packaged
    data = json.loads(manifest.read_text())
    data["release_blockers"] = [{"reason": "Not cleared"}]
    manifest.write_text(json.dumps(data))
    with pytest.raises(EvidenceError):
        prepare(release=str(manifest.parent), manifest_sha256=checksum, output=str(tmp_path / "bad"))
    with pytest.raises(EvidenceError, match="terms"):
        prepare(
            release=str(manifest.parent), manifest_sha256=digest(manifest.read_bytes()), output=str(tmp_path / "bad")
        )
    assert not (tmp_path / "bad").exists()


def test_changed_skill_cannot_be_scored(packaged, tmp_path):
    manifest, checksum, _, task = packaged
    (manifest.parent / "catalogs/example-all/target/SKILL.md").write_text("Changed instructions")
    with pytest.raises(EvidenceError, match="source changed"):
        SkillOpeningVerifier(
            task=task,
            trial_paths=TrialPaths(tmp_path / "trial"),
            environment=None,
            release=str(manifest.parent),
            manifest_sha256=checksum,
            harness="pi",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool,status,unscored",
    [("delegate_task", "completed", True), ("delegate_task", "failed", False), ("exec_command", "completed", False)],
)
async def test_completed_delegation_without_child_logs_is_unscored(packaged, tmp_path, tool, status, unscored):
    manifest, checksum, _, task = packaged
    paths = TrialPaths(tmp_path / "trial")
    paths.mkdir()
    verifier = SkillOpeningVerifier(
        task=task,
        trial_paths=paths,
        environment=None,
        release=str(manifest.parent),
        manifest_sha256=checksum,
        harness="hermes",
    )
    value = trajectory(
        tool=tool, content=json.dumps({"results": [{"task_index": 0, "status": status, "summary": "Task finished."}]})
    )
    if status is None:
        value.steps[1].observation.results[0].content = json.dumps({"results": None})
    (paths.agent_dir / "trajectory.json").write_text(value.model_dump_json())
    if unscored:
        with pytest.raises(EvidenceError, match="Delegated outcome"):
            await verifier.verify()
    else:
        assert (await verifier.verify()).rewards == {"reward": 0}


def test_missing_delegation_results_list_is_not_completed():
    value = trajectory(tool="delegate_task", content='{"results": null}')
    assert not has_uncollected_delegation(value)
