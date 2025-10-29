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

"""
Run example
```bash
python scripts/run_customer_evals.py \
    ++model_short_name_for_upload=nemotron-nano-9b-v2 \
    ++policy_api_key={endpoint API key}
```
"""

from asyncio import run
from copy import deepcopy
from typing import Any, Dict, List

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel
from tqdm.auto import tqdm

from nemo_gym.cli import RunHelper
from nemo_gym.gitlab_utils import UploadJsonlDatasetGitlabConfig, upload_jsonl_dataset
from nemo_gym.global_config import GlobalConfigDictParserConfig, get_global_config_dict, set_global_config_dict
from nemo_gym.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper


class RunCustomerEvalConfig(BaseModel):
    model_short_name_for_upload: str


class CustomerEval(BaseModel):
    eval_name: str
    config_path: str
    rollout_collection_config: RolloutCollectionConfig


CUSTOMER_EVALS: List[CustomerEval] = [
    CustomerEval(
        eval_name="cohesity_netbackup_rag",
        config_path="resources_servers/cohesity_netbackup_rag/configs/cohesity_netbackup_rag.yaml",
        rollout_collection_config=RolloutCollectionConfig(
            agent_name="cohesity_netbackup_rag_simple_agent",
            input_jsonl_fpath="data/cohesity_netbackup_rag/validation.jsonl",
            output_jsonl_fpath="resources_servers/cohesity_netbackup_rag/data/{model_short_name_for_upload}_validation_rollouts.jsonl",
        ),
    ),
    CustomerEval(
        eval_name="crowdstrike_logscale_syntax",
        config_path="resources_servers/crowdstrike_logscale_syntax/configs/crowdstrike_logscale_syntax.yaml",
        rollout_collection_config=RolloutCollectionConfig(
            agent_name="crowdstrike_logscale_syntax_simple_agent",
            input_jsonl_fpath="data/crowdstrike_logscale_syntax/validation.jsonl",
            output_jsonl_fpath="resources_servers/crowdstrike_logscale_syntax/data/{model_short_name_for_upload}_validation_rollouts.jsonl",
        ),
    ),
    CustomerEval(
        eval_name="servicenow_document_reasoning",
        config_path="resources_servers/servicenow_document_reasoning/configs/servicenow_document_reasoning.yaml",
        rollout_collection_config=RolloutCollectionConfig(
            agent_name="servicenow_document_reasoning_simple_agent",
            input_jsonl_fpath="data/servicenow_document_reasoning/validation.jsonl",
            output_jsonl_fpath="resources_servers/servicenow_document_reasoning/data/{model_short_name_for_upload}_validation_rollouts.jsonl",
        ),
    ),
]


class ModelEvalConfig(BaseModel):
    model_short_name_for_upload: str
    initial_global_config_dict: Dict[str, Any]


MODEL_EVAL_CONFIGS: List[ModelEvalConfig] = [
    ModelEvalConfig(
        model_short_name_for_upload="nemotron-nano-9b-v2",
        initial_global_config_dict={
            "policy_base_url": "https://integrate.api.nvidia.com/v1",
            "policy_api_key": "???",
            "policy_model_name": "nvidia/nvidia-nemotron-nano-9b-v2",
            "config_paths": [
                "responses_api_models/vllm_model/configs/vllm_model.yaml",
            ],
            "num_samples_in_parallel": 8,
            "responses_create_params": {
                "temperature": 0.6,
                "max_output_tokens": 32768,
                "top_p": 0.95,
            },
        },
    ),
    ModelEvalConfig(
        model_short_name_for_upload="llama-3.3-nemotron-super-49b-v1.5",
        initial_global_config_dict={
            "policy_base_url": "https://integrate.api.nvidia.com/v1",
            "policy_api_key": "???",
            "policy_model_name": "nvidia/llama-3.3-nemotron-super-49b-v1.5",
            "config_paths": [
                "responses_api_models/vllm_model/configs/vllm_model.yaml",
            ],
            "num_samples_in_parallel": 8,
            "responses_create_params": {
                "temperature": 0.6,
                "max_output_tokens": 65536,
                "top_p": 0.95,
            },
        },
    ),
    ModelEvalConfig(
        model_short_name_for_upload="qwen3-235b-a22b",
        initial_global_config_dict={
            "policy_base_url": "https://integrate.api.nvidia.com/v1",
            "policy_api_key": "???",
            "policy_model_name": "qwen/qwen3-235b-a22b",
            "config_paths": [
                "responses_api_models/vllm_model/configs/vllm_model.yaml",
            ],
            "policy_model": {
                "responses_api_models": {
                    "vllm_model": {
                        "replace_developer_role_with_system": True,
                    },
                },
            },
            "num_samples_in_parallel": 8,
            "responses_create_params": {
                "temperature": 0.2,
                # Omit max output tokens since this endpoint has max len 32k anyways
                # "max_output_tokens": 32768,
                "top_p": 0.7,
            },
        },
    ),
    ModelEvalConfig(
        model_short_name_for_upload="qwen3-next-80b-a3b-thinking",
        initial_global_config_dict={
            "policy_base_url": "https://integrate.api.nvidia.com/v1",
            "policy_api_key": "???",
            "policy_model_name": "qwen/qwen3-next-80b-a3b-thinking",
            "config_paths": [
                "responses_api_models/vllm_model/configs/vllm_model.yaml",
            ],
            "policy_model": {
                "responses_api_models": {
                    "vllm_model": {
                        "replace_developer_role_with_system": True,
                    },
                },
            },
            "num_samples_in_parallel": 8,
            "responses_create_params": {
                "temperature": 0.6,
                "max_output_tokens": 32768,
                "top_p": 0.7,
            },
        },
    ),
    ModelEvalConfig(
        model_short_name_for_upload="deepseek-v3.1",
        initial_global_config_dict={
            "config_paths": [
                "responses_api_models/vllm_model/configs/vllm_model.yaml",
            ],
            "policy_model": {
                "responses_api_models": {
                    "vllm_model": {
                        "replace_developer_role_with_system": True,
                    },
                },
            },
            "policy_base_url": "https://integrate.api.nvidia.com/v1",
            "policy_api_key": "???",
            "policy_model_name": "deepseek-ai/deepseek-v3.1",
            "num_samples_in_parallel": 8,
            "responses_create_params": {
                "temperature": 0.2,
                "max_output_tokens": 32768,
                "top_p": 0.7,
            },
        },
    ),
]


async def main():
    global_config_dict = get_global_config_dict()
    customer_eval_config = RunCustomerEvalConfig.model_validate(global_config_dict)

    model_eval_config = [
        c
        for c in MODEL_EVAL_CONFIGS
        if c.model_short_name_for_upload == customer_eval_config.model_short_name_for_upload
    ]
    assert len(model_eval_config) == 1
    model_eval_config = model_eval_config[0]

    customer_eval_config_paths = [ce.config_path for ce in CUSTOMER_EVALS]
    initial_global_config_dict = deepcopy(model_eval_config.initial_global_config_dict)
    initial_global_config_dict["config_paths"] += customer_eval_config_paths
    global_config_dict_parser_config = GlobalConfigDictParserConfig(
        initial_global_config_dict=DictConfig(initial_global_config_dict),
    )
    set_global_config_dict(global_config_dict_parser_config=global_config_dict_parser_config)
    global_config_dict = get_global_config_dict()

    rh = RunHelper()
    rh.start(None)

    global_config_dict_dictionary = OmegaConf.to_container(global_config_dict)

    rch = RolloutCollectionHelper()

    try:
        for customer_eval in tqdm(CUSTOMER_EVALS, desc="Running customer evals"):
            print(f"Running `{customer_eval.eval_name}`")

            rollout_collection_config_dict = OmegaConf.merge(
                global_config_dict_dictionary,
                customer_eval.rollout_collection_config.model_dump(exclude_unset=True),
                {
                    "output_jsonl_fpath": customer_eval.rollout_collection_config.output_jsonl_fpath.format(
                        model_short_name_for_upload=customer_eval_config.model_short_name_for_upload,
                    )
                },
            )
            rollout_collection_config = RolloutCollectionConfig.model_validate(rollout_collection_config_dict)
            await rch.run_from_config(rollout_collection_config)

            upload_jsonl_dataset_config = UploadJsonlDatasetGitlabConfig(
                dataset_name=customer_eval.eval_name,
                version="0.0.1",
                input_jsonl_fpath=rollout_collection_config.output_jsonl_fpath,
            )
            upload_jsonl_dataset(config=upload_jsonl_dataset_config)
    finally:
        rh.shutdown()


if __name__ == "__main__":
    run(main())
