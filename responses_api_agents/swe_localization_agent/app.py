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
from minisweagent.run.extra.localization import LocalizationRunner
from minisweagent.run.extra.runner_config import RunnerConfig
from pydantic import ConfigDict

from responses_api_agents.mini_swe_agent.app import (
    MiniSWEAgent,
    MiniSWEAgentConfig,
    MiniSWEAgentRunRequest,
    MiniSWEAgentVerifyRequest,
    MiniSWEAgentVerifyResponse,
)


class SWELocalizationAgentConfig(MiniSWEAgentConfig):
    pass


class SWELocalizationAgentRunRequest(MiniSWEAgentRunRequest):
    model_config = ConfigDict(extra="allow")


class SWELocalizationAgentVerifyRequest(MiniSWEAgentVerifyRequest):
    model_config = ConfigDict(extra="allow")


class SWELocalizationAgentVerifyResponse(MiniSWEAgentVerifyResponse):
    model_config = ConfigDict(extra="allow")


class SWELocalizationAgent(MiniSWEAgent):
    config: SWELocalizationAgentConfig

    def get_config_path(self) -> Path:
        return builtin_config_dir / "extra" / "localization.yaml"

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
        return LocalizationRunner().run, {"cfg": cfg}

    def calculate_reward(self, instance_id: str, result: dict[str, Any]) -> float:
        return result["eval_report"].get("eval_report", {}).get("overlap_score", 0.0)


if __name__ == "__main__":
    SWELocalizationAgent.run_webserver()
