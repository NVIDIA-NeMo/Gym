# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
from pathlib import Path
from typing import Any, Optional

from minisweagent.config import builtin_config_dir
from minisweagent.run.extra.patch_generation import PatchGenerationRunner
from minisweagent.run.extra.runner_config import RunnerConfig
from pydantic import ConfigDict

from responses_api_agents.mini_swe_agent.app import (
    MiniSWEAgent,
    MiniSWEAgentConfig,
    MiniSWEAgentRunRequest,
    MiniSWEAgentVerifyRequest,
    MiniSWEAgentVerifyResponse,
)


class SWEPatchGenNoToolsAgentConfig(MiniSWEAgentConfig):
    pass


class SWEPatchGenNoToolsAgentRunRequest(MiniSWEAgentRunRequest):
    model_config = ConfigDict(extra="allow")


class SWEPatchGenNoToolsAgentVerifyRequest(MiniSWEAgentVerifyRequest):
    model_config = ConfigDict(extra="allow")


class SWEPatchGenNoToolsAgentVerifyResponse(MiniSWEAgentVerifyResponse):
    model_config = ConfigDict(extra="allow")


class SWEPatchGenNoToolsAgent(MiniSWEAgent):
    config: SWEPatchGenNoToolsAgentConfig

    def get_config_path(self) -> Path:
        return builtin_config_dir / "extra" / "patch_generation.yaml"

    def get_runner_and_params(
        self,
        subset: str,
        split: str,
        workers: int,
        output_file_dir: str,
        model_name: str,
        dummy_key: str,
        base_url: str,
        cache_dir_template: Optional[str],
        env: str,
        instance_id: str,
        body: MiniSWEAgentRunRequest,
        responses_create_params_dict: dict[str, Any],
        step_timeout: int,
        eval_timeout: int,
        step_limit: int,
        collapse_limit: int,
    ) -> tuple[Any, dict[str, Any]]:
        cfg = RunnerConfig(
            subset=subset,
            split=split,
            workers=workers,
            output=output_file_dir,
            model=model_name,
            api_key=dummy_key,
            base_url=base_url,
            cache_dir_template=cache_dir_template,
            env=env,
            instance_id=instance_id,
            instance_dict=body.model_dump_json(),
            responses_create_params=json.dumps(responses_create_params_dict),
            step_timeout=step_timeout,
            eval_timeout=eval_timeout,
            step_limit=step_limit,
            collapse_limit=collapse_limit,
            config=self.get_config_path(),
        )
        return PatchGenerationRunner().run, {"cfg": cfg}


    #TODO: refactor this to use the MiniSWEAgentUtils class
    def get_reward(self,instance_id: str, eval_report: dict[str, Any], partial_reward: bool = False) -> float:
        try:
            if not eval_report:
                return 0.0
            eval_report = eval_report["eval_report"][instance_id]
            resolved = eval_report["resolved"]
            if not eval_report.get("tests_status"):
                return 0.0

            tests_status = eval_report["tests_status"]
            f2f = tests_status.get("FAIL_TO_PASS", {})
            p2p = tests_status.get("PASS_TO_PASS", {})
            f2f_success = len(f2f.get("success", []))
            f2f_failure = len(f2f.get("failure", []))
            p2p_success = len(p2p.get("success", []))
            p2p_failure = len(p2p.get("failure", []))

            if f2f_success == 0 and f2f_failure == 0 and p2p_success == 0 and p2p_failure == 0:
                return 0.0

            if partial_reward:
                return 0.5 if p2p_success > 0 and p2p_failure == 0 else 0.0
            return 1.0 if resolved else 0.0
        except Exception as e:
            print(f"Error in get_reward: {e}")
            return 0.0

    def calculate_reward(self, instance_id: str, result: dict[str, Any], partial_reward: bool = False) -> float:
        return self.get_reward(instance_id, result["eval_report"], partial_reward)


if __name__ == "__main__":
    SWEPatchGenNoToolsAgent.run_webserver()
