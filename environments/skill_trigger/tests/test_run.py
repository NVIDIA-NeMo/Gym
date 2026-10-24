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
"""Check complete catalog planning and one-attempt dispatch without model inference."""

import copy
import json
import subprocess
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from benchmarks.skill_trigger.prepare import BENCHMARK_DIR, Selection, digest, encode
from environments.skill_trigger import run
from environments.skill_trigger.scoring import EvidenceError
from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper
from responses_api_agents.harbor_agent_general.app import HarborAgentConfig


@pytest.fixture
def release(tmp_path):
    selection = Selection.model_validate_json((BENCHMARK_DIR / "selection.json").read_bytes())
    manifest = {"selection_sha256": digest(encode(selection.model_dump())), "cases": []}
    for case in selection.cases:
        manifest["cases"].append(case.model_dump() | {"task_name": "case-" + digest(case.id.encode())[:12]})
    release = tmp_path / "release"
    release.mkdir()
    (release / "manifest.json").write_bytes(encode(manifest))
    return release, manifest


@pytest.mark.parametrize("pilot,counts", [(False, (110, 43, 330)), (True, (9, 5, 9))])
def test_all_cases_and_catalogs_appear_once_and_gym_resolves_them(release, tmp_path, monkeypatch, pilot, counts):
    folder, manifest = release

    # No skill bodies or inference: packaging itself has separate real-file tests.
    def package_stub(**kwargs):
        kwargs["output"].mkdir(parents=True)
        return kwargs["output"]

    monkeypatch.setattr(run, "package_tasks", package_stub)
    plan_path = run.plan_run(
        release=folder,
        manifest_sha256=digest((folder / "manifest.json").read_bytes()),
        output=tmp_path / "run",
        image="synthetic:test",
        harness="pi",
        model="test-model",
        endpoint="http://127.0.0.1:9999/v1",
        pilot=pilot,
        runtime={"provider": "openai", "api": "openai-responses"},
    )
    plan = json.loads(plan_path.read_bytes())
    assert (plan["cases"], len(plan["groups"]), plan["planned_trials"]) == counts
    selected = run.select_cases(manifest, pilot)
    assert sorted(case_id for group in plan["groups"] for case_id in group["cases"]) == sorted(
        case["id"] for case in selected
    )
    by_id = {case["id"]: case for case in selected}
    indices = {case["id"]: index for index, case in enumerate(manifest["cases"])}
    seen_indices = set()
    for group in plan["groups"]:
        directory = plan_path.parent / "groups" / group["catalog"]
        parser_options = GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.create({"config_paths": [str(directory / "config.yaml")]}),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
        resolved = GlobalConfigDictParser().parse(parser_options)
        config = HarborAgentConfig.model_validate(
            {"name": "skill_trigger", "host": "127.0.0.1", "port": 8080}
            | OmegaConf.to_container(resolved.skill_trigger.responses_api_agents.skill_trigger, resolve=True)
        )
        assert Path(resolved.skills.path).name == group["catalog"]
        # Gym's E2E collector converts OmegaConf without resolve=True; the input
        # path must already be concrete at this boundary.
        collection = RolloutCollectionConfig.model_validate(
            OmegaConf.to_container(resolved)
            | {
                "input_jsonl_fpath": str(directory / "input.jsonl"),
                "output_jsonl_fpath": str(directory / "rollouts.jsonl"),
            }
        )
        assert collection.skills.path == str(folder / "catalogs" / group["catalog"])
        assert resolved.skill_trigger.responses_api_agents.skill_trigger.datasets[0].num_repeats == plan["repetitions"]
        assert resolved.skill_trigger.responses_api_agents.skill_trigger.datasets[0].jsonl_fpath == str(
            directory / "input.jsonl"
        )
        rows = [json.loads(line) for line in (directory / "input.jsonl").read_text().splitlines()]
        assert [row["task_name"] for row in rows] == [by_id[c]["task_name"] for c in group["cases"]]
        assert all(by_id[c]["catalog"] == group["catalog"] for c in group["cases"])
        assert [row["_ng_task_index"] for row in rows] == [indices[c] for c in group["cases"]]
        assert not seen_indices.intersection(row["_ng_task_index"] for row in rows)
        seen_indices.update(row["_ng_task_index"] for row in rows)
        expanded = RolloutCollectionHelper().preprocess_examples(
            rows,
            agent_map={"_default": "skill_trigger"},
            num_repeats=plan["repetitions"],
        )
        assert [row["_ng_task_index"] for row in expanded] == [
            index for case_id in group["cases"] for index in [indices[case_id]] * plan["repetitions"]
        ]
        assert not resolved.count_missing_rollouts_as_zero and not resolved.count_failure_classes_as_zero
        assert config.job.retry.max_retries == 0
        assert config.job.agents[0].model_name == "test-model"
        assert config.job.agents[0].kwargs["api"] == "openai-responses"


@pytest.mark.parametrize("change", ["drop", "duplicate", "prompt", "catalog", "hash"])
def test_changed_selection_cannot_be_called_full_or_pilot(release, change):
    _, original = release
    manifest = copy.deepcopy(original)
    if change == "drop":
        manifest["cases"].pop()
    elif change == "duplicate":
        manifest["cases"][-1] = manifest["cases"][0]
    elif change == "hash":
        manifest["selection_sha256"] = "changed"
    else:
        manifest["cases"][0][change] = "changed"
    for pilot in (False, True):
        with pytest.raises(EvidenceError):
            run.select_cases(manifest, pilot)


def test_unresolved_terms_do_not_create_run(release, tmp_path):
    folder, manifest = release
    manifest.update(artifact_kind="source-audit", release_blockers=[{"reason": "Unresolved"}])
    (folder / "manifest.json").write_bytes(encode(manifest))
    with pytest.raises(EvidenceError, match="terms"):
        run.plan_run(
            release=folder,
            manifest_sha256=digest((folder / "manifest.json").read_bytes()),
            output=tmp_path / "run",
            image="test:local",
            harness="pi",
            model="test-model",
            endpoint="http://127.0.0.1/v1",
            runtime={"provider": "openai", "api": "openai-responses"},
        )
    assert not (tmp_path / "run").exists()


def test_dispatch_preserves_failures_and_never_retries_attempted_groups(tmp_path, monkeypatch):
    release = tmp_path / "release"
    release.mkdir()
    (release / "manifest.json").write_text("manifest")
    groups = []
    for name in ("one", "two"):
        directory = tmp_path / "groups" / name
        directory.mkdir(parents=True)
        (directory / "input.jsonl").write_text("input")
        (directory / "config.yaml").write_text("config")
        groups.append({"catalog": name, "input_sha256": digest(b"input"), "config_sha256": digest(b"config")})
    (tmp_path / "plan.json").write_bytes(
        encode(
            {
                "release": str(release),
                "profile_files": {},
                "manifest_sha256": digest(b"manifest"),
                "groups": groups,
            }
        )
    )
    calls = []

    def invoke(command, **kwargs):
        assert kwargs["env"]["NEMO_GYM_MAX_ROLLOUT_ATTEMPTS"] == "1"
        assert "--resume" not in command
        calls.append(command)
        kwargs["stdout"].write(b"retained diagnostic")
        return subprocess.CompletedProcess(command, 1 if len(calls) == 1 else 0)

    monkeypatch.setattr(run.subprocess, "run", invoke)
    with pytest.raises(RuntimeError):
        run.execute_run(tmp_path)
    assert len(calls) == 1
    assert (tmp_path / "groups/one/console.log").read_bytes() == b"retained diagnostic"
    assert json.loads((tmp_path / "groups/one/exit.json").read_bytes())["returncode"] == 1
    for _ in range(2):
        # Only the never-attempted second group is dispatched; first failure remains visible.
        with pytest.raises(RuntimeError, match="Earlier attempts"):
            run.execute_run(tmp_path)
    assert len(calls) == 2
    (tmp_path / "groups/two/input.jsonl").write_text("changed")
    with pytest.raises(EvidenceError, match="changed"):
        run.execute_run(tmp_path)
    assert len(calls) == 2


def test_inherited_gym_configuration_cannot_override_plan(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_GYM_CONFIG_DICT", '{"model": "unrelated"}')
    with pytest.raises(EvidenceError, match="inherited"):
        run.execute_run(tmp_path)
    assert not list(tmp_path.glob("groups/*/attempt.json"))
