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
"""Validate actual Harbor task loading, catalog isolation, and fail-closed packaging."""

import copy
import json

import pytest
from harbor.models.job.config import DatasetConfig
from harbor.models.task.task import Task

from benchmarks.skill_trigger.prepare import digest, encode, tree_digest
from environments.skill_trigger.package import HARNESSES, package_tasks, prepare
from environments.skill_trigger.scoring import EvidenceError
from environments.skill_trigger.tests.test_verifier import prepared as prepared


def package(release, output, harness="pi"):
    return package_tasks(
        release=release,
        manifest_sha256=digest((release / "manifest.json").read_bytes()),
        output=output,
        image="skill-harnesses:test",
        harness=harness,
        gateway_host="127.0.0.1",
    )


@pytest.mark.parametrize("harness", HARNESSES)
async def test_packaged_tasks_preserve_separate_catalogs_and_prompts(tmp_path, prepared, harness):
    manifest = json.loads((prepared / "manifest.json").read_bytes())
    # A second case has a different catalog; no union is allowed.
    folder = prepared / "catalogs/second/other"
    folder.mkdir(parents=True)
    text = "---\nname: other\ndescription: Other task\n---\nRead other instructions."
    (folder / "SKILL.md").write_text(text)
    (folder / "reference.txt").write_text("Supporting material")
    files = [(p.name, digest(p.read_bytes())) for p in sorted(folder.iterdir())]
    members = [{"name": "other", "files": files, "mounted_sha256": tree_digest(files)}]
    ch = digest(encode(members))
    manifest["catalogs"]["second"] = {"members": members, "sha256": ch}
    second = copy.deepcopy(manifest["cases"][0])
    second.update(task_name="case-two", id="second", skill="other", catalog="second", catalog_sha256=ch)
    second.update(prompt="Use the other skill\n", prompt_sha256=digest(b"Use the other skill\n"))
    manifest["cases"].append(second)
    (prepared / "manifest.json").write_bytes(encode(manifest))
    output = package(prepared, tmp_path / "tasks", harness)
    tasks = [Task(output / row["task_name"]) for row in manifest["cases"]]
    assert [task.instruction for task in tasks] == [row["prompt"] for row in manifest["cases"]]
    assert {p.name for p in (output / "case-one/environment/skills").iterdir()} == {"target"}
    assert {p.name for p in (output / "case-two/environment/skills").iterdir()} == {"other"}
    assert (output / "case-two/environment/skills/other/reference.txt").read_text() == "Supporting material"
    assert (output / "case-two/environment/skills/other/SKILL.md").read_bytes() == (folder / "SKILL.md").read_bytes()
    assert (output / "case-two/environment/skills/other/SKILL.md").stat().st_ino != (folder / "SKILL.md").stat().st_ino
    assert not list(output.glob("**/manifest.json"))
    assert not list(output.glob("**/evals"))
    assert all(
        (output / row["task_name"] / "tests/test.sh").read_text() == "#!/bin/sh\nexit 1\n" for row in manifest["cases"]
    )
    rows = [json.loads(line) for line in (output / "gym.jsonl").read_text().splitlines()]
    assert rows == [
        {"task_name": row["task_name"], "responses_create_params": {"input": []}} for row in manifest["cases"]
    ]
    grouped = [
        [json.loads(line) for line in p.read_text().splitlines()]
        for p in sorted((output / "catalogs").glob("*.jsonl"))
    ]
    assert grouped == [[rows[0]], [rows[1]]]
    # Harbor's own local-dataset discovery accepts the emitted layout.
    assert len(await DatasetConfig(path=output).get_task_configs()) == 2
    assert tasks[0].config.environment.network_mode == ("no-network" if harness == "pi" else "allowlist")


@pytest.mark.parametrize("problem", ["audit", "license", "changed", "duplicate", "escape"])
def test_invalid_release_never_leaves_partial_dataset(tmp_path, prepared, problem):
    path = prepared / "manifest.json"
    manifest = json.loads(path.read_bytes())
    if problem == "audit":
        manifest["artifact_kind"] = "source-audit"
    elif problem == "license":
        manifest["release_blockers"] = [{"reason": "Unresolved terms"}]
    elif problem == "changed":
        (prepared / "catalogs/catalog/target/SKILL.md").write_text("changed instructions")
    elif problem == "duplicate":
        manifest["cases"].append(copy.deepcopy(manifest["cases"][0]))
    else:
        manifest["cases"][0]["task_name"] = "../escape"
    path.write_bytes(encode(manifest))
    with pytest.raises(EvidenceError):
        package(prepared, tmp_path / "tasks")
    assert not (tmp_path / "tasks").exists()
    assert not list(tmp_path.glob(".skill-tasks-*"))


def test_does_not_replace_existing_package(tmp_path, prepared):
    output = package(prepared, tmp_path / "tasks")
    before = (output / "gym.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        package(prepared, output)
    assert (output / "gym.jsonl").read_bytes() == before


def test_gym_prepare_entrypoint_returns_selected_catalog_rows(tmp_path, prepared):
    path = prepare(
        release=str(prepared),
        manifest_sha256=digest((prepared / "manifest.json").read_bytes()),
        output=str(tmp_path / "tasks"),
        image="skill-harnesses:test",
        harness="pi",
        gateway_host="127.0.0.1",
        catalog="catalog",
    )
    assert path == tmp_path / "tasks/catalogs/catalog.jsonl"
    assert json.loads(path.read_text())["task_name"] == "case-one"
