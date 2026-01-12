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
import asyncio
import signal
import sys
from argparse import Namespace
from os import environ
from pathlib import Path
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple, Union

import ray
from huggingface_hub import snapshot_download
from ray import available_resources, cluster_resources, runtime_env
from vllm.entrypoints.openai.api_server import (
    FlexibleArgumentParser,
    cli_env_setup,
    make_arg_parser,
    run_server,
    validate_parsed_serve_args,
)

from nemo_gym.global_config import (
    DISALLOWED_PORTS_KEY_NAME,
    HF_TOKEN_KEY_NAME,
    find_open_port,
    get_global_config_dict,
)
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


class LocalVLLMModelConfig(VLLMModelConfig):
    # We inherit these configs from VLLMModelConfig, but they are set to optional since they will be set later on after we spin up a model endpoint.
    base_url: Optional[Union[str, List[str]]] = None
    api_key: Optional[str] = None

    hf_home: Optional[str] = None
    vllm_serve_kwargs: Dict[str, Any]
    vllm_serve_env_vars: Dict[str, str]

    debug: bool = False

    def model_post_init(self, context):
        # Default to the .cache/huggingface in this directory.
        if not self.hf_home:
            current_directory = Path.cwd()
            self.hf_home = str(current_directory / ".cache" / "huggingface")

        return super().model_post_init(context)


def _vllm_asyncio_task(server_args: Namespace):
    asyncio.run(run_server(server_args))


@ray.remote
class LocalVLLMModelActor:
    def __init__(self, server_args: Namespace, env_vars: Dict[str, str]) -> None:
        self.server_args = server_args
        self.env_vars = env_vars

        node_ip = ray._private.services.get_node_ip_address()
        self._base_url = f"http://{node_ip}:{self.server_args.port}/v1"

        # vLLM doesn't expose a config for this yet, so we need to pass via environment variable.
        self.env_vars["VLLM_DP_MASTER_IP"] = node_ip  # This is the master node.

        # Pass through signal setting not allowed in threads.
        signal.signal = lambda *args, **kwargs: None

        for k, v in env_vars.items():
            environ[k] = v

        self.server_thread = Thread(target=_vllm_asyncio_task, args=(server_args,), daemon=True)

    def base_url(self) -> str:
        return self._base_url


class LocalVLLMModel(VLLMModel):
    config: LocalVLLMModelConfig

    _local_vllm_model_actor: LocalVLLMModelActor

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

    def _configure_vllm_serve(self) -> Tuple[Namespace, Dict[str, str]]:
        server_args = self.config.vllm_serve_kwargs

        port = find_open_port(disallowed_ports=get_global_config_dict()[DISALLOWED_PORTS_KEY_NAME])
        cache_dir = self.get_cache_dir()
        server_args = server_args | {
            "model": self.config.model,
            "host": "0.0.0.0",  # Must be 0.0.0.0 for cross-node communication.
            "port": port,
            "distributed_executor_backend": "ray",
            "data_parallel_backend": "ray",
            "download_dir": cache_dir,
        }

        env_vars = dict()
        # vLLM accepts a `hf_token` parameter but it's not used everywhere. We need to set HF_TOKEN environment variable here.
        maybe_hf_token = self.get_hf_token()
        if maybe_hf_token:
            env_vars["HF_TOKEN"] = maybe_hf_token

        env_vars.update(self.config.vllm_serve_env_vars)

        # Ray backend only works if dp_size > 1
        assert server_args.get("data_parallel_size") is None or server_args.get("data_parallel_size") > 1, (
            "Ray backend only works with data parallel size > 1!"
        )

        cli_env_setup()
        parser = FlexibleArgumentParser(description="vLLM OpenAI-Compatible RESTful API server.")
        parser = make_arg_parser(parser)
        final_args = parser.parse_args(namespace=Namespace(**server_args))
        validate_parsed_serve_args(final_args)

        if self.config.debug:
            env_vars_to_print = env_vars.copy()
            if "HF_TOKEN" in env_vars_to_print:
                env_vars_to_print["HF_TOKEN"] = "****"
            print(f"""Final vLLM serve arguments: {final_args}
Environment variables: {env_vars_to_print}""")

        return final_args, env_vars

    def _patch_vllm_ray_runtime_env(self) -> None:
        # TODO this may not be necessary anymore
        # This patch may be sensitive to vLLM version! See https://github.com/vllm-project/vllm/blob/275de34170654274616082721348b7edd9741d32/vllm/v1/engine/utils.py#L651
        original_RuntimeEnv = runtime_env.RuntimeEnv

        def new_RuntimeEnv(*args, **kwargs):
            kwargs = kwargs or dict()
            kwargs["py_executable"] = sys.executable
            if self.config.debug:
                print(f"Patched RuntimeEnv py_executable with {sys.executable}")

            # Necessary for downstream vLLM ray actor spinup otherwise we get CUDA device ordinal out of range errors.
            env_vars = kwargs.get("env_vars") or dict()
            env_vars.pop("CUDA_VISIBLE_DEVICES", None)
            env_vars["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"

            return original_RuntimeEnv(*args, **kwargs)

        runtime_env.RuntimeEnv = new_RuntimeEnv

    def start_vllm_server(self) -> None:
        if self.config.debug:
            print(f"""Currently available Ray cluster resources: {available_resources()}
Total Ray cluster resources: {cluster_resources()}""")

        server_args, env_vars = self._configure_vllm_serve()

        self._local_vllm_model_actor = LocalVLLMModelActor.remote(server_args, env_vars)

        self.config.base_url = [self._local_vllm_model_actor.base_url.remote()]
        self.config.api_key = "dummy_key"  # dummy key


if __name__ == "__main__":
    LocalVLLMModel.run_webserver()
