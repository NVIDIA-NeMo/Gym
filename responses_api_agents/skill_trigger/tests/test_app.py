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
"""Check catalog provenance before the real bridge can dispatch a trial."""

import copy
from unittest.mock import AsyncMock

import pytest

from benchmarks.skill_trigger.prepare import digest
from environments.skill_trigger.tests.test_verifier import prepared as prepared
from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.skills import load_skill_directory
from responses_api_agents.harbor_agent_general import app as bridge
from responses_api_agents.skill_trigger.app import SkillHarborAgent


def inputs(prepared, tmp_path):
    options = dict(
        release=str(prepared),
        manifest_sha256=digest((prepared / "manifest.json").read_bytes()),
        evidence_root=str(tmp_path / "private"),
        harness="pi",
        upstream_url="http://provider/v1",
        api_key_env="TEST_KEY",
        gateway_host="127.0.0.1",
        provider="openai",
        api="openai-responses",
    )
    config = bridge.HarborAgentConfig.model_validate(
        dict(
            name="skill_trigger",
            host="127.0.0.1",
            port=8080,
            entrypoint="app.py",
            job=dict(
                jobs_dir=tmp_path / "jobs",
                datasets=[{"path": str(tmp_path / "tasks")}],
                agents=[
                    {
                        "import_path": "environments.skill_trigger.agent:SkillAgent",
                        "model_name": "test-model",
                        "kwargs": options,
                    }
                ],
            ),
        )
    )
    body = bridge.HarborRunRequest.model_validate(
        {
            "task_name": "case-one",
            TASK_INDEX_KEY_NAME: 0,
            ROLLOUT_INDEX_KEY_NAME: 0,
            "responses_create_params": {"input": []},
            "skills_ref": load_skill_directory(str(prepared / "catalogs/catalog")).model_dump(),
        }
    )
    return config, body


async def test_valid_catalog_is_passed_to_existing_bridge_without_rewriting(prepared, tmp_path, monkeypatch):
    config, body = inputs(prepared, tmp_path)
    monkeypatch.setattr(bridge, "get_global_config_dict", lambda: {})
    parent = AsyncMock(return_value="sentinel")
    monkeypatch.setattr(bridge.HarborAgent, "run", parent)
    agent = SkillHarborAgent.model_construct(config=config)
    before = copy.deepcopy(body.model_dump())
    assert await agent.run(body) == "sentinel"
    parent.assert_awaited_once_with(body)
    assert body.model_dump() == before


@pytest.mark.parametrize("problem", ["missing", "path", "hash", "metadata", "changed_files", "wrong_task"])
async def test_bad_catalog_is_failure_without_dispatch(prepared, tmp_path, monkeypatch, problem):
    config, body = inputs(prepared, tmp_path)
    raw = body.model_dump(by_alias=True)
    if problem == "missing":
        raw.pop("skills_ref")
    elif problem == "path":
        raw["skills_ref"]["path"] = str(tmp_path / "another-catalog")
    elif problem == "hash":
        raw["skills_ref"]["hash"] = "bad"
    elif problem == "metadata":
        raw["skills_ref"]["skills"][0]["description"] = "changed"
    elif problem == "changed_files":
        (prepared / "catalogs/catalog/target/SKILL.md").write_text("changed")
    else:
        raw["task_name"] = "not-in-this-release"
    body = bridge.HarborRunRequest.model_validate(raw)
    monkeypatch.setattr(bridge, "get_global_config_dict", lambda: {})
    parent = AsyncMock()
    monkeypatch.setattr(bridge.HarborAgent, "run", parent)
    agent = SkillHarborAgent.model_construct(config=config)
    response = await agent.run(body)
    parent.assert_not_awaited()
    assert response.response.status == "failed"
    assert response.model_dump()["_ng_failure_class"] == "harbor_failed"
    assert response.model_dump()["error"]
