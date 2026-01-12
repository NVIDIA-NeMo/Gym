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
import sys
from argparse import Namespace
from pathlib import Path
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple, Union

import ray
from aiohttp import ClientConnectorError
from huggingface_hub import snapshot_download
from ray import available_resources, cluster_resources
from ray._private.state import available_resources_per_node
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from ray.util.state import list_nodes
from vllm.entrypoints.openai.api_server import (
    FlexibleArgumentParser,
    cli_env_setup,
    make_arg_parser,
    validate_parsed_serve_args,
)

from nemo_gym.global_config import (
    DISALLOWED_PORTS_KEY_NAME,
    HF_TOKEN_KEY_NAME,
    find_open_port,
    get_global_config_dict,
)
from nemo_gym.server_utils import get_global_aiohttp_client
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
    from vllm.entrypoints.openai.api_server import run_server

    asyncio.run(run_server(server_args))


@ray.remote
class LocalVLLMModelActor:
    def __init__(self, server_args: Namespace, env_vars: Dict[str, str], server_name: str) -> None:
        from os import environ

        self.server_args = server_args
        self.env_vars = env_vars
        self.server_name = server_name

        self.env_vars.pop("CUDA_VISIBLE_DEVICES", None)

        node_ip = ray._private.services.get_node_ip_address()
        self._base_url = f"http://{node_ip}:{self.server_args.port}/v1"

        # vLLM doesn't expose a config for this yet, so we need to pass via environment variable.
        self.env_vars["VLLM_DP_MASTER_IP"] = node_ip  # This is the master node.

        self._patch_signal_handler()

        for k, v in self.env_vars.items():
            environ[k] = v

        self.server_thread = Thread(target=_vllm_asyncio_task, args=(server_args,), daemon=True)
        self.server_thread.start()

    def _patch_signal_handler(self) -> None:
        # Pass through signal setting not allowed in threads.
        # See https://github.com/vllm-project/vllm/blob/275de34170654274616082721348b7edd9741d32/vllm/entrypoints/launcher.py#L94
        # This may be vLLM version specific!

        import signal
        from asyncio import get_running_loop

        from vllm.entrypoints import launcher

        original_serve_http = launcher.serve_http

        def new_serve_http(*args, **kwargs):
            loop = get_running_loop()
            loop.add_signal_handler = lambda *args, **kwargs: None

            return original_serve_http(*args, **kwargs)

        launcher.serve_http = new_serve_http

        # Patch signal as well.
        signal.signal = lambda *args, **kwargs: None

    def base_url(self) -> str:
        return self._base_url

    def is_alive(self) -> bool:
        return self.server_thread.is_alive()


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

    def _select_vllm_server_head_node(self) -> NodeAffinitySchedulingStrategy:
        """
        There are a few params vLLM has:
        - data parallel size
        - data parallel size local
        - tensor parallel size
        - pipeline parallel size
        - vllm ray dp pack strategy

        As of vLLM 0.11.2, the way vLLM + Ray works is:
        1. allocate (tensor parallel size * pipeline parallel size)-sized placement groups
        2. for vllm ray dp pack strategy
            - span (not relevant for my tp * pp within one node)
            - fill: basically as many as possible
                - this will clash if there are > 1 endpoints or the compute necessary is less than what is available (mismatch throws an error in vllm)
            - strict: data parallel size local * num nodes placement groups

        Now the problem is that for `strict`, if we spin up the head server on the same node, we need to set data parallel size local to 0. So `fill` and `strict` don't work out of the box.

        Here, we fix `strict` by spinning things up on not the head server node. We find a currently available GPU node and star the vLLM server there so the head node address is propagated properly.
        """
        alive_gpu_nodes = [n for n in list_nodes() if n.state == "ALIVE" and n.resources_total.get("GPU", 0) > 0]
        assert alive_gpu_nodes

        node_id_to_available_resources = available_resources_per_node()

        selected_node = None
        partial_node = None
        for node in alive_gpu_nodes:
            total_gpus = node.resources_total.get("GPU", 0) > 0
            available_gpus = node_id_to_available_resources[node.node_id]["GPU"]

            if total_gpus == available_gpus:
                selected_node = node
                break

            if available_gpus != 0:
                partial_node = node

        selected_node = selected_node or partial_node
        return NodeAffinitySchedulingStrategy(
            node_id=selected_node.node_id,
            soft=False,  # Hard constraint - must run on this node
        )

    def start_vllm_server(self) -> None:
        if self.config.debug:
            print(f"""Currently available Ray cluster resources: {available_resources()}
Total Ray cluster resources: {cluster_resources()}""")

        server_args, env_vars = self._configure_vllm_serve()

        self._local_vllm_model_actor = LocalVLLMModelActor.options(
            scheduling_strategy=self._select_vllm_server_head_node(),
            runtime_env=dict(
                py_executable=sys.executable,
                env_vars={
                    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                    **env_vars,
                },
            ),
        ).remote(server_args, env_vars, self.config.name)

        self.config.base_url = [ray.get(self._local_vllm_model_actor.base_url.remote())]
        self.config.api_key = "dummy_key"  # dummy key

        asyncio.run(self.await_server_ready())

    async def await_server_ready(self) -> None:
        poll_count = 0
        client = get_global_aiohttp_client()
        while True:
            is_alive = await self._local_vllm_model_actor.is_alive.remote()
            assert is_alive, f"{self.config.name} LocalVLLMModel server spinup failed, see the error logs above!"

            try:
                await client.request(method="GET", url=f"{self.config.base_url[0]}/models")
                return
            except ClientConnectorError:
                if poll_count % 10 == 0:  # Print every 30s
                    print(f"Waiting for {self.config.name} LocalVLLMModel server to spinup...")

                poll_count += 1
                await asyncio.sleep(3)


if __name__ == "__main__":
    LocalVLLMModel.run_webserver()
