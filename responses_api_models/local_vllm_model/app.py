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
from huggingface_hub import snapshot_download
from ray import available_resources, cluster_resources
from ray._private.state import available_resources_per_node
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from ray.util.state import list_nodes, list_placement_groups
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
    def __init__(self, server_args: Namespace, env_vars: Dict[str, str], server_name: str, debug: bool) -> None:
        from os import environ

        self.server_args = server_args
        self.env_vars = env_vars
        self.server_name = server_name
        self.debug = debug

        self.env_vars.pop("CUDA_VISIBLE_DEVICES", None)

        node_ip = ray._private.services.get_node_ip_address()
        self._base_url = f"http://{node_ip}:{self.server_args.port}/v1"

        # vLLM doesn't expose a config for this yet, so we need to pass via environment variable.
        self.env_vars["VLLM_DP_MASTER_IP"] = node_ip  # This is the master node.

        self._patch_signal_handler()
        self._patch_uvicorn_logger()
        self._maybe_patch_engine_stats()
        self._patch_vllm_placement_group_filter()

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

    def _patch_vllm_placement_group_filter(self) -> None:
        """
        Patch vLLM's v1 engine to handle multiple node resource keys from placement groups.

        vLLM's v1 engine expects exactly one node resource key per node, but when multiple
        vLLM actors or other Ray actors use placement groups, additional node resource keys
        are created (e.g., 'node:IP_group_N_hash'). This patch filters out placement group
        keys to only use the base node IP key.

        Based on vLLM v0.11.2's CoreEngineActorManager.create_dp_placement_groups:
        https://github.com/vllm-project/vllm/blob/v0.11.2/vllm/v1/engine/utils.py#L329-L528
        """
        try:
            from vllm.v1.engine import utils as vllm_utils

            @staticmethod
            def patched_create_dp_placement_groups(vllm_config):
                """
                Create placement groups for data parallel.

                This is a patched version of vLLM's create_dp_placement_groups that handles
                multiple node resource keys from existing placement groups.
                """
                import ray
                from ray._private.state import available_resources_per_node
                from vllm import envs
                from vllm.logger import init_logger
                from vllm.platforms import current_platform

                logger = init_logger(__name__)

                ########################################
                # START Patch: Enhanced logging for debugging
                ########################################
                # Original code:
                # logger.info("Creating placement groups for data parallel")

                # Modified code:
                logger.info("=== PATCHED create_dp_placement_groups called ===")
                logger.info("Creating placement groups for data parallel")
                ########################################
                # END Patch: Enhanced logging for debugging
                ########################################

                ########################################
                # START Patch: Unique placement group names
                ########################################
                # Original code:
                # (no unique suffix generation)

                # Modified code:
                import time

                unique_suffix = int(time.time() * 1000) % 1000000  # Use timestamp for uniqueness
                ########################################
                # END Patch: Unique placement group names
                ########################################

                dp_master_ip = vllm_config.parallel_config.data_parallel_master_ip
                dp_size = vllm_config.parallel_config.data_parallel_size
                dp_size_local = vllm_config.parallel_config.data_parallel_size_local

                available_resources = available_resources_per_node()
                world_size = vllm_config.parallel_config.world_size
                placement_groups = []
                local_dp_ranks = []

                dp_master_ip_key = f"node:{dp_master_ip}"
                nodes = sorted(available_resources.values(), key=lambda x: dp_master_ip_key not in x)
                assert len(nodes) > 0, "No nodes with resources found in Ray cluster."
                assert dp_master_ip_key in nodes[0], (
                    "The DP master node (ip: %s) is missing or dead",
                    dp_master_ip,
                )
                device_str = current_platform.ray_device_key
                n_node_devices = [
                    int(node_resources[device_str]) for node_resources in nodes if device_str in node_resources
                ]
                assert n_node_devices, f"No {device_str} found in Ray cluster."
                max_device_per_node = max(n_node_devices)

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

                    assert set(n_node_devices) == {max_device_per_node}, f"Nodes are not homogenous, {nodes}"
                    assert world_size % max_device_per_node == 0, (
                        f"For multi-node data parallel groups, world_size ({world_size}) must "
                        f"be a multiple of number of devices per node ({max_device_per_node})."
                    )
                    assert len(n_node_devices) * max_device_per_node >= world_size * dp_size, (
                        f"Not enough total available nodes ({len(n_node_devices)}) "
                        f"and devices per node ({max_device_per_node}) "
                        f"to satisfy required world size {world_size} and data parallel size "
                        f"{dp_size}"
                    )
                    assert dp_size_local == 1, (
                        f"data-parallel-size-local {dp_size_local} should be set as the "
                        "default (1) for VLLM_RAY_DP_PACK_STRATEGY=span. "
                        "The actual data-parallel-size-local will be auto determined."
                    )

                collected_bundles = []
                for node_resources in nodes:
                    ########################################
                    # START Patch: Filter placement group node keys
                    ########################################
                    # Original code:
                    # node_ip_keys = [
                    #     key
                    #     for key in node_resources
                    #     if key != "node:__internal_head__" and key.startswith("node:")
                    # ]

                    # Modified code:
                    node_ip_keys = [
                        key
                        for key in node_resources
                        if key != "node:__internal_head__"
                        and key.startswith("node:")
                        and "_group_" not in key  # Filter out placement group keys
                    ]
                    ########################################
                    # END Patch: Filter placement group node keys
                    ########################################

                    assert len(node_ip_keys) == 1, (
                        "Zero or multiple node IP keys found in node resources: %s",
                        node_ip_keys,
                    )
                    node_ip_key = node_ip_keys[0]
                    node_ip = node_ip_key.split(":")[1]

                    n_device_on_node = int(node_resources.get(device_str, 0))
                    if pack_strategy == "span" and n_device_on_node != 0:
                        dp_size_available = 1
                    else:
                        dp_size_available = n_device_on_node // world_size

                    if node_ip == dp_master_ip:
                        if dp_size_available < dp_size_local:
                            raise ValueError(
                                "Not enough resources to allocate %s DP ranks "
                                "on DP master node %s, possible to fit %s DP ranks",
                                dp_size_local,
                                dp_master_ip,
                                dp_size_available,
                            )
                        dp_size_to_allocate = dp_size_local
                    elif pack_strategy == "strict":
                        if dp_size_available < dp_size_local:
                            logger.info(
                                "Skipping node %s as %s DP ranks could not fit, possible to fit %s DP ranks",
                                node_ip,
                                dp_size_local,
                                dp_size_available,
                            )
                            continue
                        dp_size_to_allocate = dp_size_local
                    else:
                        dp_size_to_allocate = dp_size_available

                    for i in range(dp_size_to_allocate):
                        device_bundle = [{device_str: 1.0, "node:" + node_ip: 0.001}]
                        if pack_strategy == "span":
                            collected_bundles += device_bundle * n_device_on_node
                            assert len(collected_bundles) <= world_size, (
                                "collected_bundles should be <= world_size, "
                                f"but got {len(collected_bundles)=} and {world_size=}"
                            )

                            if len(collected_bundles) < world_size:
                                continue

                            bundles = collected_bundles + [{"CPU": 1.0}]
                            collected_bundles = []
                        else:
                            bundles = device_bundle * world_size + [{"CPU": 1.0}]

                        ########################################
                        # START Patch: Unique placement group names and enhanced logging
                        ########################################
                        # Original code:
                        # pg = ray.util.placement_group(
                        #     name=f"dp_rank_{len(placement_groups)}",
                        #     strategy=placement_strategy,
                        #     bundles=bundles,
                        # )

                        # Modified code:
                        pg_name = f"dp_rank_{len(placement_groups)}_{unique_suffix}"
                        logger.info(
                            f"Creating placement group {pg_name} with {len(bundles)} bundles (world_size={world_size})"
                        )

                        pg = ray.util.placement_group(
                            name=pg_name,
                            strategy=placement_strategy,
                            bundles=bundles,
                        )
                        ray.get(pg.ready(), timeout=1800)

                        # Verify the placement group was created with the right number of bundles
                        pg_table = ray.util.placement_group_table(pg)
                        actual_bundle_count = len(pg_table["bundles"])
                        logger.info(
                            f"Placement group {pg_name} created successfully with {actual_bundle_count} bundles"
                        )
                        ########################################
                        # END Patch: Unique placement group names and enhanced logging
                        ########################################

                        placement_groups.append(pg)
                        local_dp_ranks.append(i)
                        if len(placement_groups) == dp_size:
                            break

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

            # Apply the patch
            vllm_utils.CoreEngineActorManager.create_dp_placement_groups = patched_create_dp_placement_groups

            if self.debug:
                print("Applied patch to vLLM v1 engine to handle multiple node resource keys from placement groups.")
        except Exception as e:
            print(
                f"Warning: Could not patch vLLM placement group filtering: {e}. This may cause issues if there are existing placement groups."
            )

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
        # Check if model is a local path
        if Path(self.config.model).exists():
            print(f"Model path {self.config.model} exists locally, skipping download.")
            return

        # Otherwise, download from HuggingFace
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

    def _cleanup_stale_placement_groups(self) -> None:
        """
        Log information about existing placement groups for debugging.

        With the vLLM patch in place, existing placement groups shouldn't cause issues,
        but we log them for debugging purposes.
        """
        if not self.config.debug:
            return

        try:
            placement_groups = list_placement_groups(filters=[("state", "=", "CREATED")])

            if not placement_groups:
                print("No existing placement groups found.")
            else:
                print(f"Found {len(placement_groups)} existing placement group(s):")
                for pg in placement_groups:
                    print(f"  - {pg.name or pg.placement_group_id} (state: {pg.state})")
        except Exception as e:
            print(f"Could not list placement groups: {e}")

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
            total_gpus = node.resources_total["GPU"]
            # We use .get("GPU") here since if there are no available GPUs, the property won't be set.
            available_gpus = node_id_to_available_resources[node.node_id].get("GPU", 0)

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

        # Log existing placement groups for debugging (vLLM patch handles multiple PGs)
        self._cleanup_stale_placement_groups()

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
        ).remote(server_args, env_vars, self.config.name, self.config.debug)

        self.config.base_url = [ray.get(self._local_vllm_model_actor.base_url.remote())]
        self.config.api_key = "dummy_key"  # pragma: allowlist secret

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
