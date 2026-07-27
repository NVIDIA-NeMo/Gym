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
import sys
from argparse import Namespace
from pathlib import Path
from time import sleep
from typing import Any, Dict, List, Optional, Tuple, Union

import ray
import requests
from pydantic import BaseModel, Field
from ray import available_resources, cluster_resources
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from requests.exceptions import ConnectionError
from vllm.entrypoints.openai.api_server import (
    FlexibleArgumentParser,
    cli_env_setup,
    make_arg_parser,
    validate_parsed_serve_args,
)

from nemo_gym.global_config import (
    DISALLOWED_PORTS_KEY_NAME,
    find_open_port,
    get_global_config_dict,
    get_hf_token,
)
from responses_api_models.local_vllm_model.local_vllm_model_actor import LocalVLLMModelActor
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


class LocalVLLMModelConfig(VLLMModelConfig):
    # We inherit these configs from VLLMModelConfig, but they are set to optional since they will be set later on after we spin up a model endpoint.
    base_url: Union[str, List[str]] = Field(default_factory=list)
    # Not used on local deployments
    api_key: str = "dummy"  # pragma: allowlist secret

    hf_home: Optional[str] = None
    vllm_serve_kwargs: Dict[str, Any]
    vllm_serve_env_vars: Dict[str, str]

    ray_worker_py_executable: str = sys.executable

    show_vllm_engine_stats: bool = False
    debug: bool = False

    def model_post_init(self, context):
        # Default to the .cache/huggingface in this directory.
        if not self.hf_home:
            current_directory = Path.cwd()
            self.hf_home = str(current_directory / ".cache" / "huggingface")

        return super().model_post_init(context)


class GetInnerVLLMConfigResponse(BaseModel):
    base_url: List[str]
    api_key: str
    model: str


class LocalVLLMModel(VLLMModel):
    config: LocalVLLMModelConfig

    _local_vllm_model_actor: LocalVLLMModelActor

    def setup_webserver(self):
        print("Starting vLLM server. This will take a few minutes...")
        self.start_vllm_server()

        app = super().setup_webserver()

        # This route is only used to support LocalVLLMModelProxy
        app.get("/get_inner_vllm_config")(self.get_inner_vllm_config)

        return app

    async def get_inner_vllm_config(self) -> GetInnerVLLMConfigResponse:
        return GetInnerVLLMConfigResponse(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            model=self.config.model,
        )

    def get_cache_dir(self) -> str:
        # We need to reconstruct the cache dir as HF does it given HF_HOME. See https://github.com/huggingface/huggingface_hub/blob/b2723cad81f530e197d6e826f194c110bf92248e/src/huggingface_hub/constants.py#L146
        return str(Path(self.config.hf_home) / "hub")

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

        env_vars = {"HF_HUB_ENABLE_HF_TRANSFER": "1"}
        # vLLM accepts a `hf_token` parameter but it's not used everywhere. We need to set HF_TOKEN environment variable here.
        maybe_hf_token = get_hf_token()
        if maybe_hf_token:
            env_vars["HF_TOKEN"] = maybe_hf_token

        env_vars.update(self.config.vllm_serve_env_vars)

        assert "VLLM_RAY_DP_PACK_STRATEGY" in env_vars, (
            f"Please provide a value for `VLLM_RAY_DP_PACK_STRATEGY` for `{self.config.name}`"
        )
        assert server_args.get("data_parallel_size")
        assert server_args.get("tensor_parallel_size")
        assert server_args.get("pipeline_parallel_size")

        # With our vLLM patches, this assert is no longer necessary
        # Ray backend only works if dp_size > 1
        # assert server_args.get("data_parallel_size") is None or server_args.get("data_parallel_size") > 1, (
        #     "Ray backend only works with data parallel size > 1!"
        # )

        # With our vLLM patches, this is no longer necessary for people to set.
        server_args["data_parallel_size_local"] = 1

        # TODO multi-node model instances still need to be properly supported
        # We get a vLLM error: Exception: Error setting CUDA_VISIBLE_DEVICES: local range: [0, 16) base value: "0,1,2,3,4,5,6,7"
        if env_vars.get("VLLM_RAY_DP_PACK_STRATEGY") == "span":
            # Unset this flag since it's set by default using span
            server_args.pop("data_parallel_size_local", None)

        cli_env_setup()
        parser = FlexibleArgumentParser(description="vLLM OpenAI-Compatible RESTful API server.")
        parser = make_arg_parser(parser)
        final_args = parser.parse_args(namespace=Namespace(**server_args))
        validate_parsed_serve_args(final_args)

        # @bxyu-nvidia: TODO remove, specific to Nemotron 3 Ultra vLLM version.
        # Upstream vLLM only exposes `enable_return_routed_experts`, so alias it across.
        final_args.return_routed_experts = final_args.enable_return_routed_experts

        if self.config.debug:
            env_vars_to_print = env_vars.copy()
            if "HF_TOKEN" in env_vars_to_print:
                env_vars_to_print["HF_TOKEN"] = "****"
            print(f"""Final vLLM serve arguments: {final_args}
Environment variables: {env_vars_to_print}""")

        return final_args, env_vars

    def _select_vllm_server_head_node(self, server_args: Namespace, env_vars: Dict[str, str]) -> PlacementGroup:
        """
        Our LocalVLLMModelActor Ray actor scheduling strategy is as follows:
        1. We estimate the size of a single placement group vLLM will make using TP * PP
        2. We pre-maturely create one placement group of this size which will server as the master node for the vLLM instance
        3. This placement group is also provided on input to the LocalVLLMModelActor, which will schedule (DP - 1) additional placement groups of size TP * PP
        """
        # This mirrors the placement group logic above
        pack_strategy = env_vars["VLLM_RAY_DP_PACK_STRATEGY"]
        if pack_strategy in ("strict", "fill"):
            placement_strategy = "STRICT_PACK"
        else:
            placement_strategy = "PACK"

        device_str = "GPU"
        device_bundle = [{device_str: 1.0}]
        world_size = server_args.pipeline_parallel_size * server_args.tensor_parallel_size
        bundles = device_bundle * world_size + [{"CPU": 1.0}]
        head_node_placement_group = ray.util.placement_group(
            name=f"{self.config.name}_dp_rank_0",
            strategy=placement_strategy,
            bundles=bundles,
        )
        ray.get(head_node_placement_group.ready())

        return head_node_placement_group

    def start_vllm_server(self) -> None:
        # If base_url is already set, skip local launch — connect to external server.
        if self.config.base_url:
            print(f"External base_url configured: {self.config.base_url}. Skipping local vLLM launch.")
            self._post_init()
            return

        if self.config.debug:
            print(f"""Currently available Ray cluster resources: {available_resources()}
Total Ray cluster resources: {cluster_resources()}""")

        server_args, env_vars = self._configure_vllm_serve()
        head_node_placement_group = self._select_vllm_server_head_node(server_args, env_vars)

        pythonpath = str(Path(__file__).parent.parent.parent)
        if self.config.debug:
            print(f"Using PYTHONPATH={pythonpath}")

        self._local_vllm_model_actor = LocalVLLMModelActor.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=head_node_placement_group,
            ),
            runtime_env=dict(
                py_executable=self.config.ray_worker_py_executable,
                env_vars={
                    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                    "PYTHONPATH": pythonpath,
                    **env_vars,
                },
            ),
        ).remote(
            head_node_placement_group=head_node_placement_group,
            server_args=server_args,
            env_vars=env_vars,
            server_name=self.config.name,
            debug=self.config.debug,
            show_vllm_engine_stats=self.config.show_vllm_engine_stats,
        )

        self.config.base_url = [ray.get(self._local_vllm_model_actor.base_url.remote())]

        # Reset clients after base_url config
        self._post_init()

        self.await_server_ready()

    def await_server_ready(self) -> None:
        poll_count = 0
        while True:
            is_alive = ray.get(self._local_vllm_model_actor.is_alive.remote())
            assert is_alive, f"{self.config.name} LocalVLLMModel server spinup failed, see the error logs above!"

            try:
                requests.get(url=f"{self.config.base_url[0]}/models")
                return
            except ConnectionError:
                if poll_count % 10 == 0:  # Print every 30s
                    print(f"Waiting for {self.config.name} LocalVLLMModel server to spinup...")

                poll_count += 1
                sleep(3)


if __name__ == "__main__":
    LocalVLLMModel.run_webserver()
