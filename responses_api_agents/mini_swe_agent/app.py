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
import asyncio
import json
import sys
from asyncio import Semaphore
from os import environ, getenv, makedirs
from pathlib import Path
from typing import Any, Callable, Literal, Optional
from uuid import uuid4

import ray
import yaml
from fastapi import Body, FastAPI
from minisweagent.config import builtin_config_dir, get_config_path
from minisweagent.run.extra.swegym_runner import _main as run_swegym
from pydantic import ConfigDict

from nemo_gym.base_resources_server import (
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import (
    ServerClient,
    get_first_server_config_dict,
)
from responses_api_agents.mini_swe_agent.utils import MiniSWEAgentUtils


class MiniSWEAgentConfig(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef
    resources_server: ResourcesServerRef
    env: Literal["docker", "singularity"]
    concurrency: int
    cache_dir_template: Optional[str] = None
    run_golden: bool = False
    step_timeout: int = 600
    eval_timeout: int = 1800
    skip_if_exists: bool = False
    step_limit: int = 250
    collapse_limit: int = 3


class MiniSWEAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class MiniSWEAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class MiniSWEAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


@ray.remote(
    scheduling_strategy="SPREAD",
    runtime_env={
        "py_executable": sys.executable,
    },
)
def runner_ray_remote(runner: Callable, params: dict[str, Any]) -> Any:
    return runner(**params)


class MiniSWEAgent(SimpleResponsesAPIAgent):
    config: MiniSWEAgentConfig
    sem: Semaphore = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)

    def setup_webserver(self) -> FastAPI:
        app = FastAPI()
        app.post("/v1/responses")(self.responses)
        app.post("/run")(self.run)
        return app

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        raise NotImplementedError

    async def run(self, body: MiniSWEAgentRunRequest) -> MiniSWEAgentVerifyResponse:
        async with self.sem:
            model_server_name = self.config.model_server.name
            global_config_dict = ServerClient.load_from_global_config().global_config_dict

            model_server_config = get_first_server_config_dict(
                global_config_dict,
                model_server_name,
            )

            policy_model_name = global_config_dict["policy_model_name"]

            ##### MINI-SWE-AGENT CONFIG #####
            subset = body.subset
            split = body.split
            workers = 1
            cache_dir_template = self.config.cache_dir_template
            run_golden = self.config.run_golden
            base_url = f"http://{model_server_config['host']}:{model_server_config['port']}/v1"
            dummy_key = "dummy_key"
            model_name = f"hosted_vllm/{policy_model_name}"
            step_timeout = self.config.step_timeout
            eval_timeout = self.config.eval_timeout
            env = self.config.env
            step_limit = self.config.step_limit
            collapse_limit = self.config.collapse_limit

            instance_id = body.instance_id

            mini_swe_config_path = builtin_config_dir / "extra" / "swebench.yaml"
            config = yaml.safe_load(get_config_path(mini_swe_config_path).read_text())

            default_model_kwargs = config["model"]["model_kwargs"]
            temperature = body.responses_create_params.temperature or default_model_kwargs["temperature"]
            top_p = body.responses_create_params.top_p or default_model_kwargs["top_p"]

            output_file_dir = f"{Path.cwd()}/results/{subset}/{policy_model_name}"

            if self.config.skip_if_exists:
                if Path(f"{output_file_dir}/{instance_id}/{instance_id}.json").exists():
                    with open(f"{output_file_dir}/{instance_id}/{instance_id}.json", "r") as f:
                        print(f"Skipping {instance_id} because it already exists")
                        verify_response = MiniSWEAgentVerifyResponse.model_validate_json(f.read())
                    return verify_response

            env_vars = environ.copy()
            if env == "singularity":
                slurm_job_id = getenv("SLURM_JOB_ID", str(uuid4()))
                env_vars.update(
                    {
                        "SINGULARITY_CACHEDIR": f"/tmp/singularity_cache_${slurm_job_id}_$$",
                        "APPTAINER_CACHEDIR": f"/tmp/apptainer_cache_${slurm_job_id}_$$",
                        "SINGULARITY_TMPDIR": f"/tmp/singularity_tmp_${slurm_job_id}_$$",
                        "APPTAINER_TMPDIR": f"/tmp/apptainer_tmp_${slurm_job_id}_$$",
                    }
                )
                for var in [
                    "SINGULARITY_CACHEDIR",
                    "APPTAINER_CACHEDIR",
                    "SINGULARITY_TMPDIR",
                    "APPTAINER_TMPDIR",
                ]:
                    makedirs(env_vars[var], exist_ok=True)

            #### RUN MINI-SWE-AGENT #####
            reseponses_create_params_dict = body.responses_create_params.model_dump()
            try:
                params = dict(
                    subset=subset,
                    split=split,
                    workers=workers,
                    output=output_file_dir,
                    model=model_name,
                    api_key=dummy_key,
                    base_url=base_url,
                    cache_dir_template=cache_dir_template,
                    env=env,
                    run_golden=run_golden,
                    instance_id=instance_id,
                    # TODO: add this later
                    instance_dict=body.model_dump(),
                    responses_create_params=json.dumps(reseponses_create_params_dict),
                    step_timeout=step_timeout,
                    eval_timeout=eval_timeout,
                    step_limit=step_limit,
                    collapse_limit=collapse_limit,
                )
                future = runner_ray_remote.remote(run_swegym, params)
                result = await asyncio.to_thread(ray.get, future)
                result = result[instance_id]
                messages = result["messages"]
                responses = result["responses"]
                reward = 1.0 if MiniSWEAgentUtils.is_resolved(instance_id, result["eval_report"]) else 0.0

            except Exception as e:
                print(f"Error running swegym: {e}")
                result = None
                messages = []
                responses = []
                reward = 0.0

            # The first two messages are the system and user message generated by the harness
            # TODO(sugam): what if the user only provides the system/user message
            body.responses_create_params.input = messages[:2]

            response = MiniSWEAgentUtils.get_default_response_object()
            response["model"] = policy_model_name
            response["temperature"] = temperature
            response["top_p"] = top_p

            # Wrap output messages in responses format
            response["output"] = MiniSWEAgentUtils.chat_cmp_to_responses(messages[2:], responses)

            verify_response = MiniSWEAgentVerifyResponse(
                responses_create_params=body.responses_create_params,
                reward=reward,
                response=response,
                instance_id=instance_id,
                metadata=result.get("eval_report", {}) if result else {},
            )

            output_path = Path(f"{output_file_dir}/{instance_id}")
            output_path.mkdir(parents=True, exist_ok=True)

            with open(f"{output_file_dir}/{instance_id}/{instance_id}.json", "w") as f:
                json.dump(verify_response.model_dump(), f)

            return verify_response


if __name__ == "__main__":
    MiniSWEAgent.run_webserver()
