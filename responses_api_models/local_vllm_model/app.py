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
from typing import Any, Coroutine, Dict, List, Optional, Tuple, Union

import ray
from aiohttp.client_exceptions import ClientConnectorError
from huggingface_hub import snapshot_download
from ray import available_resources, cluster_resources, runtime_env
from vllm.entrypoints.openai.api_server import (
    FlexibleArgumentParser,
    cli_env_setup,
    make_arg_parser,
    run_server,
    validate_parsed_serve_args,
)

from nemo_gym.global_config import DISALLOWED_PORTS_KEY_NAME, HF_TOKEN_KEY_NAME, find_open_port, get_global_config_dict
from nemo_gym.server_utils import get_global_aiohttp_client
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


class LocalVLLMModel(VLLMModel):
    config: LocalVLLMModelConfig

    _server_thread: Thread  # Set later on

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
        node_ip = ray._private.services.get_node_ip_address()
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

        # vLLM doesn't expose a config for this yet, so we need to pass via environment variable.
        env_vars["VLLM_DP_MASTER_IP"] = node_ip  # This is the master node.

        cli_env_setup()
        parser = FlexibleArgumentParser(description="vLLM OpenAI-Compatible RESTful API server.")
        parser = make_arg_parser(parser)
        final_args = parser.parse_args(namespace=Namespace(**server_args))
        validate_parsed_serve_args(final_args)

        base_url = f"http://{node_ip}:{final_args.port}/v1"
        self.config.base_url = base_url
        self.config.api_key = "dummy_key"  # dummy key

        return final_args, env_vars

    def _patch_uvicorn_server(self, vllm_server_coroutine: Coroutine) -> None:
        # We need to run two servers in the same process, so we override the normal Gym server spinup to also perform vLLM server spinup logic.
        # This patch may be sensitive to uvicorn version!
        from uvicorn import server as uvicorn_server

        original_asyncio_run = uvicorn_server.asyncio_run

        def new_asyncio_run(uvicorn_server_coroutine, *args, **kwargs):
            async def wait_for_vllm_server() -> None:
                poll_count = 0
                client = get_global_aiohttp_client()
                while True:
                    try:
                        await client.request(method="GET", url=f"{self.config.base_url}/models")
                        return
                    except ClientConnectorError:
                        if poll_count % 10 == 0:  # Print every 30s
                            print(f"Waiting for {self.config.name} LocalVLLMModel server to spinup...")

                        poll_count += 1
                        await asyncio.sleep(3)

            async def wrapper_fn() -> None:
                vllm_server_task = asyncio.create_task(vllm_server_coroutine)
                uvicorn_server_task = asyncio.create_task(uvicorn_server_coroutine)

                done, pending = await asyncio.wait(
                    (vllm_server_task, asyncio.create_task(wait_for_vllm_server())),
                    return_when="FIRST_COMPLETED",
                )

                # If the vllm task finishes first
                if list(done)[0] == vllm_server_task:
                    list(pending)[0].cancel()  # Cancel the waiting task.
                    uvicorn_server_task.cancel()
                    raise vllm_server_task.exception()

                print(f"{self.config.name} finished vLLM server spinup!")

                _, pending = await asyncio.wait(
                    (vllm_server_task, uvicorn_server_task),
                    return_when="FIRST_COMPLETED",
                )
                for task in pending:
                    task.cancel()

            return original_asyncio_run(wrapper_fn(), *args, **kwargs)

        uvicorn_server.asyncio_run = new_asyncio_run

    def _patch_vllm_ray_runtime_env(self) -> None:
        # This patch may be sensitive to vLLM version! See https://github.com/vllm-project/vllm/blob/275de34170654274616082721348b7edd9741d32/vllm/v1/engine/utils.py#L651
        original_RuntimeEnv = runtime_env.RuntimeEnv

        def new_RuntimeEnv(*args, **kwargs):
            kwargs = kwargs or dict()
            kwargs["py_executable"] = sys.executable

            # Necessary for downstream vLLM ray actor spinup otherwise we get CUDA device ordinal out of range errors.
            env_vars = kwargs.get("env_vars") or dict()
            env_vars.pop("CUDA_VISIBLE_DEVICES", None)
            env_vars["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"

            return original_RuntimeEnv(*args, **kwargs)

        runtime_env.RuntimeEnv = new_RuntimeEnv

    def start_vllm_server(self) -> None:
        # TODO remove
        print(f"{available_resources()=} {cluster_resources()=}")

        server_args, env_vars = self._configure_vllm_serve()

        for k, v in env_vars.items():
            environ[k] = v

        # Pass through signal setting not allowed in threads.
        signal.signal = lambda *args, **kwargs: None

        self._patch_vllm_ray_runtime_env()

        vllm_server_coroutine = run_server(server_args)

        self._patch_uvicorn_server(vllm_server_coroutine)


if __name__ == "__main__":
    LocalVLLMModel.run_webserver()
