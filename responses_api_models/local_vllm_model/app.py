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
from argparse import Namespace
from multiprocessing import Process
from os import environ
from pathlib import Path
from time import sleep
from typing import Any, Dict, List, Optional, Union

import requests
import uvloop
from huggingface_hub import snapshot_download
from vllm.entrypoints.openai.api_server import (
    FlexibleArgumentParser,
    cli_env_setup,
    make_arg_parser,
    run_server,
    validate_parsed_serve_args,
)

from nemo_gym.global_config import DISALLOWED_PORTS_KEY_NAME, HF_TOKEN_KEY_NAME, find_open_port, get_global_config_dict
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


class LocalVLLMModelConfig(VLLMModelConfig):
    # We inherit these configs from VLLMModelConfig, but they are set to optional since they will be set later on after we spin up a model endpoint.
    base_url: Optional[Union[str, List[str]]] = None
    api_key: Optional[str] = None

    hf_home: Optional[str] = None
    vllm_serve_kwargs: Dict[str, Any]

    # TODO eventually we may need to support these env vars
    # vllm_serve_env_vars: Dict[str, str]

    def model_post_init(self, context):
        # Default to the .cache/huggingface in this directory.
        if not self.hf_home:
            current_directory = Path.cwd()
            self.hf_home = str(current_directory / ".cache" / "huggingface")

        return super().model_post_init(context)


def vllm_server_proc_target(server_args):
    uvloop.run(run_server(server_args))


class LocalVLLMModel(VLLMModel):
    config: LocalVLLMModelConfig

    def model_post_init(self, context):
        print(
            f"Downloading {self.config.model}. If the model has been downloaded previously, the cached version will be used."
        )
        self.download_model()

        print("Starting vLLM server. This will take a couple of minutes...")
        self.start_vllm_server()

        return super().model_post_init(context)

    def get_hf_token(self) -> Optional[str]:
        return get_global_config_dict().get(HF_TOKEN_KEY_NAME)

    def get_cache_dir(self) -> str:
        # We need to reconstruct the cache dir as HF does it given HF_HOME. See https://github.com/huggingface/huggingface_hub/blob/b2723cad81f530e197d6e826f194c110bf92248e/src/huggingface_hub/constants.py#L146
        return str(Path(self.config.hf_home) / "hub")

    def download_model(self) -> None:
        maybe_hf_token = self.get_hf_token()
        cache_dir = self.get_cache_dir()

        snapshot_download(repo_id=self.config.model, token=maybe_hf_token, cache_dir=cache_dir)

    def start_vllm_server(self) -> None:
        server_args = self.config.vllm_serve_kwargs

        port = find_open_port(disallowed_ports=get_global_config_dict()[DISALLOWED_PORTS_KEY_NAME])
        cache_dir = self.get_cache_dir()
        server_args = server_args | {
            "model": self.config.model,
            "host": "127.0.0.1",
            "port": port,
            "distributed_executor_backend": "ray",
            "data_parallel_backend": "ray",
            "download_dir": cache_dir,
        }

        cli_env_setup()
        parser = FlexibleArgumentParser(description="vLLM OpenAI-Compatible RESTful API server.")
        parser = make_arg_parser(parser)
        args = parser.parse_args()
        validate_parsed_serve_args(args)

        server_args = Namespace(**(vars(args) | server_args))

        # vLLM accepts a `hf_token` parameter but it's not used everywhere. We need to set HF_TOKEN environment variable here.
        maybe_hf_token = self.get_hf_token()
        if maybe_hf_token:
            environ["HF_TOKEN"] = maybe_hf_token

        # The main vllm server will be run on the name node as this Gym model server, but the engines can be scheduled as seen fit by Ray.
        proc = Process(target=vllm_server_proc_target, args=(server_args,), daemon=True)
        proc.start()

        while True:
            assert proc.is_alive(), "Server process died! See the error trace above."

            try:
                response = requests.get(f"http://{server_args.host}:{server_args.port}/v1/models")
                assert response.ok, (response.status_code, response.content)
                break
            except requests.exceptions.ConnectionError:
                print(
                    f"Polling for {self.config.name} LocalVLLMModel server to spinup. Received a ConnectionError since the server isn't up yet. Sleeping for 3s..."
                )
                sleep(3)


if __name__ == "__main__":
    LocalVLLMModel.run_webserver()
