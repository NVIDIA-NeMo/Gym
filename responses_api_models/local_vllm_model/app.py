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
from time import sleep
from typing import Any, Dict, List, Optional, Tuple, Union

import ray
import requests
from pydantic import Field
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
    HF_TOKEN_KEY_NAME,
    find_open_port,
    get_global_config_dict,
)
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


class LocalVLLMModelConfig(VLLMModelConfig):
    # We inherit these configs from VLLMModelConfig, but they are set to optional since they will be set later on after we spin up a model endpoint.
    base_url: Union[str, List[str]] = Field(default_factory=list)
    # Not used on local deployments
    api_key: str = "dummy"  # pragma: allowlist secret

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
    def __init__(
        self,
        head_node_placement_group: PlacementGroup,
        server_args: Namespace,
        env_vars: Dict[str, str],
        server_name: str,
        debug: bool,
    ) -> None:
        from os import environ

        self.head_node_placement_group = head_node_placement_group
        self.server_args = server_args
        self.env_vars = env_vars
        self.server_name = server_name
        self.debug = debug

        self.env_vars.pop("CUDA_VISIBLE_DEVICES", None)

        node_ip = ray._private.services.get_node_ip_address()
        self._base_url = f"http://{node_ip}:{self.server_args.port}/v1"
        print(f"Spinning up local vLLM server at {self._base_url}", file=sys.stderr)

        # vLLM doesn't expose a config for this yet, so we need to pass via environment variable.
        self.env_vars["VLLM_DP_MASTER_IP"] = node_ip  # This is the master node.

        self._patch_signal_handler()
        self._patch_uvicorn_logger()
        self._maybe_patch_engine_stats()
        self._patch_create_dp_placement_groups()
        self._patch_init_data_parallel()
        self._patch_DPEngineCoreActor_set_cuda_visible_devices()

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

    def _patch_uvicorn_logger(self) -> None:
        from logging import Filter as LoggingFilter
        from logging import LogRecord, getLogger

        print(
            "Adding a uvicorn logging filter so that the logs aren't spammed with 200 OK messages. This is to help errors pop up better and filter out noise."
        )

        class No200Filter(LoggingFilter):
            def filter(self, record: LogRecord) -> bool:
                msg = record.getMessage()
                return not msg.strip().endswith("200")

        uvicorn_logger = getLogger("uvicorn.access")
        uvicorn_logger.addFilter(No200Filter())

    def _maybe_patch_engine_stats(self) -> None:
        from logging import ERROR

        from vllm.v1.metrics.loggers import logger as metrics_logger

        if self.debug:
            print("vLLM metrics logger will display engine stats.")
        else:
            print(
                f"Setting vLLM metrics logger for {self.server_name} to ERROR which will not print engine stats. This helps declutter the logs. Use `debug` for LocalVLLMModel to see them."
            )
            metrics_logger.setLevel(ERROR)

    def _patch_init_data_parallel(self) -> None:
        from vllm.v1.engine.core import DPEngineCoreProc, logger

        def new_init_data_parallel(self, vllm_config):
            # Configure GPUs and stateless process group for data parallel.
            dp_rank = vllm_config.parallel_config.data_parallel_rank
            dp_size = vllm_config.parallel_config.data_parallel_size
            local_dp_rank = vllm_config.parallel_config.data_parallel_rank_local

            # This allows the vLLM DP Ray flow to be run even with a single DP instance.
            # assert dp_size > 1

            assert local_dp_rank is not None
            assert 0 <= local_dp_rank <= dp_rank < dp_size

            if vllm_config.kv_transfer_config is not None:
                # modify the engine_id and append the local_dp_rank to it to ensure
                # that the kv_transfer_config is unique for each DP rank.
                vllm_config.kv_transfer_config.engine_id = (
                    f"{vllm_config.kv_transfer_config.engine_id}_dp{local_dp_rank}"
                )
                logger.debug(
                    "Setting kv_transfer_config.engine_id to %s",
                    vllm_config.kv_transfer_config.engine_id,
                )

            self.dp_rank = dp_rank
            self.dp_group = vllm_config.parallel_config.stateless_init_dp_group()

        DPEngineCoreProc._init_data_parallel = new_init_data_parallel

    def _patch_create_dp_placement_groups(self) -> None:
        head_node_placement_group = self.head_node_placement_group

        from ray.util.placement_group import PlacementGroup
        from vllm.v1.engine.utils import (
            CoreEngineActorManager,
            current_platform,
            envs,
            logger,
        )

        ########################################
        # The logic below is an exact copy of CoreEngineActorManager.create_dp_placement_groups
        # Except in places where we specify it differs
        ########################################
        def new_create_dp_placement_groups(vllm_config):
            """
            Create placement groups for data parallel.
            """

            import ray
            from ray._private.state import available_resources_per_node, total_resources_per_node

            logger.info("Creating placement groups for data parallel")
            dp_master_ip = vllm_config.parallel_config.data_parallel_master_ip
            dp_size = vllm_config.parallel_config.data_parallel_size
            dp_size_local = vllm_config.parallel_config.data_parallel_size_local

            available_resources = available_resources_per_node()

            """
            START Patch colocated placement group logic

            When running multiple local vLLM model instances on the same node, the placement group logic will error with the following since multiple placement groups are now on the same node.

            (LocalVLLMModelActor pid=504531) (APIServer pid=504531)   File "responses_api_models/local_vllm_model/.venv/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 858, in launch_core_engines
            (LocalVLLMModelActor pid=504531) (APIServer pid=504531)     engine_actor_manager = CoreEngineActorManager(
            (LocalVLLMModelActor pid=504531) (APIServer pid=504531)                            ^^^^^^^^^^^^^^^^^^^^^^^
            (LocalVLLMModelActor pid=504531) (APIServer pid=504531)   File "responses_api_models/local_vllm_model/.venv/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 300, in __init__
            (LocalVLLMModelActor pid=504531) (APIServer pid=504531)     CoreEngineActorManager.create_dp_placement_groups(vllm_config)
            (LocalVLLMModelActor pid=504531) (APIServer pid=504531)   File "responses_api_models/local_vllm_model/.venv/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 467, in create_dp_placement_groups
            (LocalVLLMModelActor pid=504531) (APIServer pid=504531)     assert len(node_ip_keys) == 1, (
            (LocalVLLMModelActor pid=504531) (APIServer pid=504531)            ^^^^^^^^^^^^^^^^^^^^^^
            (LocalVLLMModelActor pid=504531) (APIServer pid=504531) AssertionError: Zero or multiple node IP keys found in node resources: ['node:10.65.9.15_group_a036a448bf98d155cd0d6a8991f902000000', 'node:10.65.9.15_group_1_8786b4bfb840f7ba7af007e7e41602000000', 'node:10.65.9.15', 'node:10.65.9.15_group_8786b4bfb840f7ba7af007e7e41602000000', 'node:10.65.9.15_group_1_a036a448bf98d155cd0d6a8991f902000000', 'node:10.65.9.15_group_0_8786b4bfb840f7ba7af007e7e41602000000', 'node:10.65.9.15_group_0_a036a448bf98d155cd0d6a8991f902000000']
            """

            for node_hex_id, node_resources in list(available_resources.items()):
                available_resources[node_hex_id] = {
                    resource_id: resource
                    for resource_id, resource in node_resources.items()
                    if "_group_" not in resource_id
                }

            """
            END Patch colocated placement group logic
            """

            world_size = vllm_config.parallel_config.world_size
            """
            START Use our initial placement group
            """
            placement_groups: list[PlacementGroup] = [head_node_placement_group]
            local_dp_ranks: list[int] = [0]
            """
            END Use our initial placement group
            """

            dp_master_ip_key = f"node:{dp_master_ip}"
            nodes = sorted(available_resources.values(), key=lambda x: dp_master_ip_key not in x)
            assert len(nodes) > 0, "No nodes with resources found in Ray cluster."
            assert dp_master_ip_key in nodes[0], (
                "The DP master node (ip: %s) is missing or dead",
                dp_master_ip,
            )
            device_str = current_platform.ray_device_key

            n_node_devices: list[int] = [
                int(node_resources[device_str]) for node_resources in nodes if device_str in node_resources
            ]
            """
            START Account for cases when the initial placement groups we create i.e. DP == 1 are already sufficient
            """
            # Original code:
            # assert n_node_devices, f"No {device_str} found in Ray cluster."

            # Modified code:
            if dp_size == 1:
                total_nodes = total_resources_per_node().values()
                total_n_node_devices: list[int] = [
                    int(node_resources[device_str]) for node_resources in total_nodes if device_str in node_resources
                ]
                max_device_per_node = max(total_n_node_devices)
            else:
                assert n_node_devices, f"No {device_str} found in Ray cluster."
                max_device_per_node = max(n_node_devices)
            """
            END Account for cases when the initial placement groups we create i.e. DP == 1 are already sufficient
            """

            pack_strategy = envs.VLLM_RAY_DP_PACK_STRATEGY
            _supported_pack_strategies = ("strict", "fill", "span")
            if pack_strategy not in _supported_pack_strategies:
                raise ValueError(
                    f"{envs.VLLM_RAY_DP_PACK_STRATEGY} is not supported. "
                    "Make sure to set `VLLM_RAY_DP_PACK_STRATEGY` "
                    f"to one of {_supported_pack_strategies}"
                )

            all2all_backend = vllm_config.parallel_config.all2all_backend
            if pack_strategy == "fill" and (
                all2all_backend == "deepep_high_throughput" or all2all_backend == "deepep_low_latency"
            ):
                raise ValueError(
                    "DeepEP kernels require EP ranks [0,7] (same for [8,15], ...) "
                    "to be on the same node, but VLLM_RAY_DP_PACK_STRATEGY=fill "
                    "does not guarantee that. "
                    "Please use VLLM_RAY_DP_PACK_STRATEGY=strict instead."
                )

            if pack_strategy in ("strict", "fill"):
                placement_strategy = "STRICT_PACK"
            else:
                placement_strategy = "PACK"
                assert world_size > max_device_per_node, (
                    f"World size {world_size} is smaller than the "
                    "maximum number of devices per node "
                    f"{max_device_per_node}. Make sure to set "
                    "`VLLM_RAY_DP_PACK_STRATEGY` to `strict` or `fill`"
                )

                # if we need multiple nodes per dp group, we require for now that
                # available nodes are homogenous
                if dp_size == 1:
                    assert set(total_n_node_devices) == {max_device_per_node}, f"Nodes are not homogenous, {nodes}"
                else:
                    assert set(n_node_devices) == {max_device_per_node}, f"Nodes are not homogenous, {nodes}"
                assert world_size % max_device_per_node == 0, (
                    f"For multi-node data parallel groups, world_size ({world_size}) must "
                    f"be a multiple of number of devices per node ({max_device_per_node})."
                )
                """
                START Fix required GPU compute necessary calculation given we already reserve one placement group
                """
                # Original code:
                # assert len(n_node_devices) * max_device_per_node >= world_size * dp_size, (

                # Modified code:
                assert len(n_node_devices) * max_device_per_node >= world_size * (dp_size - 1), (
                    f"Not enough total available nodes ({len(n_node_devices)}) "
                    f"and devices per node ({max_device_per_node}) "
                    f"to satisfy required world size {world_size} and data parallel size "
                    f"{dp_size}"
                )
                """
                END Fix required GPU compute necessary calculation given we already reserve one placement group
                """
                assert dp_size_local == 1, (
                    f"data-parallel-size-local {dp_size_local} should be set as the "
                    "default (1) for VLLM_RAY_DP_PACK_STRATEGY=span. "
                    "The actual data-parallel-size-local will be auto determined."
                )

            for _ in range(dp_size - 1):
                bundles = [{device_str: 1.0}] * world_size + [{"CPU": 1.0}]

                pg_name = f"{self.server_name}_dp_rank_{len(placement_groups)}"
                pg = ray.util.placement_group(
                    name=pg_name,
                    strategy=placement_strategy,
                    bundles=bundles,
                )

                placement_groups.append(pg)
                local_dp_ranks.append(0)

            if len(placement_groups) < dp_size:
                raise ValueError(
                    f"Not enough resources to allocate {dp_size} "
                    "placement groups, only created "
                    f"{len(placement_groups)} placement groups. "
                    "Available resources: "
                    f"{available_resources}"
                )
            assert len(placement_groups) == dp_size, (
                f"Created {len(placement_groups)} DP placement groups, expected {dp_size}"
            )
            assert len(local_dp_ranks) == dp_size, (
                f"local_dp_ranks length {len(local_dp_ranks)} does not match expected {dp_size}"
            )

            return placement_groups, local_dp_ranks

        CoreEngineActorManager.create_dp_placement_groups = new_create_dp_placement_groups

    def _patch_DPEngineCoreActor_set_cuda_visible_devices(self) -> None:
        from vllm.v1.engine.utils import (
            CoreEngineActorManager,
        )

        original__init__ = CoreEngineActorManager.__init__

        def new__init__(self, *args, **kwargs):
            from vllm.v1.engine.core import (
                DPEngineCoreActor,
                VllmConfig,
                get_device_indices,
                os,
            )

            def new_DPEngineCoreActor_set_cuda_visible_devices(
                self, vllm_config: VllmConfig, local_dp_rank: int, device_control_env_var: str
            ):
                print(
                    f"CURRENT DPEngineCoreActor_set_cuda_visible_devices: {os.environ[device_control_env_var]}. RAY SET CUDA_VISIBLE_DEVICES: {ray.get_gpu_ids()}",
                    file=sys.stderr,
                )
                world_size = vllm_config.parallel_config.world_size
                # Set CUDA_VISIBLE_DEVICES or equivalent.
                try:
                    value = get_device_indices(device_control_env_var, local_dp_rank, world_size)
                    os.environ[device_control_env_var] = value
                    print(
                        f"NEW DPEngineCoreActor_set_cuda_visible_devices: {os.environ[device_control_env_var]}",
                        file=sys.stderr,
                    )
                except IndexError as e:
                    raise Exception(
                        f"Error setting {device_control_env_var}: "
                        f"local range: [{local_dp_rank * world_size}, "
                        f"{(local_dp_rank + 1) * world_size}) "
                        f'base value: "{os.getenv(device_control_env_var)}"'
                    ) from e

            DPEngineCoreActor._set_cuda_visible_devices = new_DPEngineCoreActor_set_cuda_visible_devices

            return original__init__(self, *args, **kwargs)

        CoreEngineActorManager.__init__ = new__init__

    def base_url(self) -> str:
        return self._base_url

    def is_alive(self) -> bool:
        return self.server_thread.is_alive()


class LocalVLLMModel(VLLMModel):
    config: LocalVLLMModelConfig

    _local_vllm_model_actor: LocalVLLMModelActor

    def setup_webserver(self):
        print("Starting vLLM server. This will take a few minutes...")
        self.start_vllm_server()

        return super().setup_webserver()

    def get_hf_token(self) -> Optional[str]:
        return get_global_config_dict().get(HF_TOKEN_KEY_NAME)

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
        maybe_hf_token = self.get_hf_token()
        if maybe_hf_token:
            env_vars["HF_TOKEN"] = maybe_hf_token

        env_vars.update(self.config.vllm_serve_env_vars)

        assert "VLLM_RAY_DP_PACK_STRATEGY" in env_vars, (
            f"Please provide a value for `VLLM_RAY_DP_PACK_STRATEGY` for `{self.config.name}`"
        )
        assert "data_parallel_size" in server_args
        assert "tensor_parallel_size" in server_args
        assert "pipeline_parallel_size" in server_args

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
        if self.config.debug:
            print(f"""Currently available Ray cluster resources: {available_resources()}
Total Ray cluster resources: {cluster_resources()}""")

        server_args, env_vars = self._configure_vllm_serve()
        head_node_placement_group = self._select_vllm_server_head_node(server_args, env_vars)

        self._local_vllm_model_actor = LocalVLLMModelActor.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=head_node_placement_group,
            ),
            runtime_env=dict(
                py_executable=sys.executable,
                env_vars={
                    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                    **env_vars,
                },
            ),
        ).remote(
            head_node_placement_group=head_node_placement_group,
            server_args=server_args,
            env_vars=env_vars,
            server_name=self.config.name,
            debug=self.config.debug,
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
