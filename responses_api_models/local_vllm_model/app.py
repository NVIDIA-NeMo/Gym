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
from pathlib import Path
from typing import Any, Dict, Optional

import uvloop
from huggingface_hub import snapshot_download
from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import validate_parsed_serve_args

from nemo_gym.global_config import HF_TOKEN_KEY_NAME, get_global_config_dict
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


class LocalVLLMModelConfig(VLLMModelConfig):
    hf_home: Optional[str] = None
    vllm_serve_kwargs: Dict[str, Any]

    # TODO eventually we may need to support these env vars
    # vllm_serve_env_vars: Dict[str, str]

    def model_post_init(self, context):
        # base_url and api_key are set later in the model spinup
        self.base_url = ""
        self.api_key = ""

        # Default to the .cache/huggingface in this directory.
        if not self.hf_home:
            current_directory = Path.cwd()
            self.hf_home = str(current_directory / ".cache" / "huggingface")

        return super().model_post_init(context)


class LocalVLLMModel(VLLMModel):
    config: LocalVLLMModelConfig

    def model_post_init(self, context):
        self.download_model()
        self.start_vllm_server()

        return super().model_post_init(context)

    def download_model(self) -> None:
        maybe_hf_token = get_global_config_dict().get(HF_TOKEN_KEY_NAME)

        # We need to reconstruct the cache dir as HF does it given HF_HOME. See https://github.com/huggingface/huggingface_hub/blob/b2723cad81f530e197d6e826f194c110bf92248e/src/huggingface_hub/constants.py#L146
        cache_dir = Path(self.config.hf_home) / "hub"

        snapshot_download(repo_id=self.config.model, token=maybe_hf_token, cache_dir=cache_dir)

    def start_vllm_server(self) -> None:
        server_args = self.config.vllm_serve_kwargs
        server_args.update(
            {
                "model": self.config.model,
                "host": None,
                "port": None,
                "distributed_executor_backend": "ray",
                "data-parallel-backend": "ray",
                "dtype": "auto",
            }
        )

        validate_parsed_serve_args(server_args)

        # The main vllm server will be run on the name node as this Gym model server, but the engines can be scheduled as seen fit by Ray.
        uvloop.run(run_server(server_args))


if __name__ == "__main__":
    LocalVLLMModel.run_webserver()
