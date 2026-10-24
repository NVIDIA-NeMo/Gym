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
"""Bind Gym's run-level skill provenance to the frozen Harbor task catalog.

Execution, scheduling, response conversion, and failure handling stay in Gym's
existing Harbor bridge. This entrypoint only rejects mismatched skill catalogs
before any model request can be dispatched.
"""

from pathlib import Path

from pydantic import ValidationError

from environments.skill_trigger.agent import SkillOptions
from environments.skill_trigger.scoring import EvidenceError
from environments.skill_trigger.verifier import load_case
from nemo_gym.skills import SkillsRef, load_skill_directory
from responses_api_agents.harbor_agent_general.app import (
    HarborAgent,
    HarborAgentConfig,
    HarborRunRequest,
    HarborVerifyResponse,
)


def check_catalog(*, config: HarborAgentConfig, body: HarborRunRequest) -> None:
    """Require the core's actual provenance stamp to match this task's release."""
    agent = config.job.agents[0]
    if agent.import_path != "environments.skill_trigger.agent:SkillAgent" or not agent.model_name:
        raise EvidenceError("Skill-opening configuration requires its pinned Harbor adapter and a model")
    options = SkillOptions.model_validate(agent.kwargs)
    case = load_case(
        release=Path(options.release),
        manifest_sha256=options.manifest_sha256,
        task_name=body.task_name,
        model=agent.model_name,
        harness=options.harness,
    )
    raw = body.model_dump().get("skills_ref")
    if not isinstance(raw, dict):
        raise EvidenceError("Run this catalog with skills.path; the request lacks skills_ref")
    supplied = SkillsRef.model_validate(raw)
    catalog = case.skill_directories[0].parent.resolve()
    if Path(supplied.path).resolve() != catalog:
        raise EvidenceError("Run-level skills.path differs from the task's frozen catalog")
    expected = load_skill_directory(str(catalog))
    if supplied.hash != expected.hash or supplied.skills != expected.skills:
        raise EvidenceError("Run-level skills_ref differs from the prepared catalog bytes or metadata")


class SkillHarborAgent(HarborAgent):
    """Use the general Harbor bridge after validating catalog provenance."""

    ray_enabled = True

    async def run(self, body: HarborRunRequest) -> HarborVerifyResponse:
        try:
            check_catalog(config=self.config, body=body)
        except (EvidenceError, ValidationError, OSError, ValueError) as error:
            return self.failure_response(body, error)
        return await super().run(body)


if __name__ == "__main__":
    SkillHarborAgent.run_webserver()
