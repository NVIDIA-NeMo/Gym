# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from asyncio import run
from typing import List

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel
from tqdm.auto import tqdm

from nemo_gym.cli import RunHelper
from nemo_gym.gitlab_utils import UploadJsonlDatasetGitlabConfig, upload_jsonl_dataset
from nemo_gym.global_config import GlobalConfigDictParserConfig, get_global_config_dict, set_global_config_dict
from nemo_gym.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper


class RunEvalConfig(BaseModel):
    model_short_name_for_upload: str
    upload: bool = False


class Eval(BaseModel):
    eval_name: str
    config_path: str
    rollout_collection_config: RolloutCollectionConfig


EVALS: List[Eval] = []


async def main():
    global_config_dict = get_global_config_dict()
    eval_config = RunEvalConfig.model_validate(global_config_dict)

    eval_config_paths = [e.config_path for e in EVALS]
    initial_global_config_dict = dict()
    initial_global_config_dict["config_paths"] += eval_config_paths
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
        for eval_ in tqdm(EVALS, desc="Running evals"):
            print(f"Running `{eval_.eval_name}`")

            rollout_collection_config_dict = OmegaConf.merge(
                global_config_dict_dictionary,
                eval_.rollout_collection_config.model_dump(exclude_unset=True),
                {
                    "output_jsonl_fpath": eval_.rollout_collection_config.output_jsonl_fpath.format(
                        model_short_name_for_upload=eval_config.model_short_name_for_upload,
                    )
                },
            )
            rollout_collection_config_dict = OmegaConf.to_container(rollout_collection_config_dict)
            rollout_collection_config = RolloutCollectionConfig.model_validate(rollout_collection_config_dict)
            await rch.run_from_config(rollout_collection_config)

            if eval_config.upload:
                upload_jsonl_dataset_config = UploadJsonlDatasetGitlabConfig(
                    dataset_name=eval_.eval_name,
                    version="0.0.1",
                    input_jsonl_fpath=rollout_collection_config.output_jsonl_fpath,
                )
                upload_jsonl_dataset(config=upload_jsonl_dataset_config)
            else:
                print("Skipping dataset upload!")
    except KeyboardInterrupt:
        pass
    finally:
        rh.shutdown()


if __name__ == "__main__":
    run(main())
