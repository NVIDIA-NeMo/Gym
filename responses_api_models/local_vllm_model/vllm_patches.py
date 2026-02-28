# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Patches for vLLM when used inside nemo_gym (e.g. multi-node Ray).

Apply before starting vLLM server with data parallel on a multi-node cluster.
"""


def apply_vllm_dp_placement_groups_patch() -> None:
    """
    Patch vLLM's create_dp_placement_groups to stop after creating dp_size
    placement groups on multi-node clusters.

    In vLLM v1 engine, the inner 'break' when len(placement_groups) == dp_size
    only exits the inner loop; the outer loop over nodes keeps running and
    creates one PG per node, causing AssertionError: Created N DP placement
    groups, expected M. This patch replaces the method with a version that
    also breaks out of the outer node loop. Idempotent; safe to call multiple
    times.
    """
    import vllm.v1.engine.utils as vllm_utils
    from vllm.config import VllmConfig

    if getattr(vllm_utils.CoreEngineActorManager, "_nemo_gym_dp_patch_applied", False):
        return

    @staticmethod
    def patched_create_dp_placement_groups(vllm_config: VllmConfig):
        import ray
        from ray._private.state import available_resources_per_node
        from vllm import envs
        from vllm.logger import init_logger
        from vllm.platforms import current_platform

        logger = init_logger(__name__)
        logger.info("Creating placement groups for data parallel")
        placement_groups = []
        local_dp_ranks = []
        dp_master_ip = vllm_config.parallel_config.data_parallel_master_ip
        dp_size = vllm_config.parallel_config.data_parallel_size
        dp_size_local = vllm_config.parallel_config.data_parallel_size_local
        available_resources = available_resources_per_node()
        world_size = vllm_config.parallel_config.world_size
        dp_master_ip_key = f"node:{dp_master_ip}"
        nodes = sorted(available_resources.values(), key=lambda x: dp_master_ip_key not in x)
        assert len(nodes) > 0, "No nodes with resources found in Ray cluster."
        assert dp_master_ip_key in nodes[0], (
            "The DP master node (ip: %s) is missing or dead",
            dp_master_ip,
        )
        device_str = current_platform.ray_device_key
        n_node_devices = [int(node_resources[device_str]) for node_resources in nodes if device_str in node_resources]
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
            node_ip_keys = [
                key for key in node_resources if key != "node:__internal_head__" and key.startswith("node:")
            ]
            # Ray may expose additional synthetic keys like
            # "node:<ip>_group_*". Prefer the canonical "node:<ip>" key.
            canonical_node_ip_keys = [key for key in node_ip_keys if "_group_" not in key]
            if len(canonical_node_ip_keys) == 1:
                node_ip_key = canonical_node_ip_keys[0]
            elif len(node_ip_keys) == 1:
                node_ip_key = node_ip_keys[0]
            else:
                raise ValueError(
                    f"Could not infer canonical node IP key from node resources. node_ip_keys={node_ip_keys}"
                )
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
                pg = ray.util.placement_group(
                    name=f"dp_rank_{len(placement_groups)}",
                    strategy=placement_strategy,
                    bundles=bundles,
                )
                placement_groups.append(pg)
                local_dp_ranks.append(i)
                if len(placement_groups) == dp_size:
                    break
            # Fix: also break outer loop once we have enough placement groups.
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

    vllm_utils.CoreEngineActorManager.create_dp_placement_groups = patched_create_dp_placement_groups
    vllm_utils.CoreEngineActorManager._nemo_gym_dp_patch_applied = True
