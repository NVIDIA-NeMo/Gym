# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest
from pydantic import ValidationError

from nemo_gym.orchestration.api import SubmitConfig


COMPUTE = {"cluster": {"type": "slurm", "account": "my-account", "hostname": "foo"}}
COMPUTE_TWO = {
    "cluster_a": {"type": "slurm", "account": "my-account", "hostname": "foo"},
    "cluster_b": {"type": "slurm", "account": "my-account", "hostname": "bar"},
}

SERVICE = {"container": "gym:latest", "type": "vllm", "model": "org/model"}
DRIVER = {"container": "gym:latest", "benchmarks": {"gsm8k": {}}}
# The Ray Serve gateway script is fetched from driver.gym_install's repo/ref into the vLLM
# service's own container - required whenever effective_ray_serve is true for a service.
DRIVER_WITH_GYM_INSTALL = {**DRIVER, "gym_install": {"ref": "main"}}
JOB = {"output_path": "/tmp/gym-jobs"}


def _config(**overrides):
    return {"services": {"svc": SERVICE}, "compute": COMPUTE, "driver": DRIVER, "job": JOB, **overrides}


def test_implicit_placement_single_compute():
    config = SubmitConfig.model_validate(_config())
    assert config.services["svc"].placement == "cluster"


def test_explicit_valid_placement():
    config = SubmitConfig.model_validate(_config(services={"svc": {**SERVICE, "placement": "cluster"}}))
    assert config.services["svc"].placement == "cluster"


def test_multiple_compute_raises():
    with pytest.raises(ValidationError, match="Multiple compute resources are not supported yet"):
        SubmitConfig.model_validate(_config(compute=COMPUTE_TWO))


def test_invalid_placement_raises():
    with pytest.raises(ValidationError, match="does not match any compute resource"):
        SubmitConfig.model_validate(_config(services={"svc": {**SERVICE, "placement": "nonexistent"}}))


def test_valid_policy_model():
    config = SubmitConfig.model_validate(_config(driver={**DRIVER, "policy_model": "svc"}))
    assert config.driver.policy_model == "svc"


def test_invalid_policy_model_raises():
    with pytest.raises(ValidationError, match="does not match any service"):
        SubmitConfig.model_validate(_config(driver={**DRIVER, "policy_model": "nonexistent"}))


def test_no_policy_model():
    config = SubmitConfig.model_validate(_config())
    assert config.driver.policy_model is None


def test_policy_model_injects_run_args():
    config = SubmitConfig.model_validate(_config(driver={**DRIVER, "policy_model": "svc"}))
    benchmark = config.driver.benchmarks["gsm8k"]
    assert benchmark.run["policy_base_url"] == "http://localhost:8000/v1"
    assert benchmark.run["policy_model_name"] == "org/model"
    assert benchmark.run["policy_api_key"] == "dummy"  # pragma: allowlist secret


def test_policy_model_conflict_raises():
    driver = {**DRIVER, "policy_model": "svc", "benchmarks": {"gsm8k": {"run": {"policy_base_url": "http://other"}}}}
    with pytest.raises(ValidationError, match="already sets"):
        SubmitConfig.model_validate(_config(driver=driver))


def test_service_env_accepted():
    config = SubmitConfig.model_validate(_config(services={"svc": {**SERVICE, "env": {"FOO": "bar"}}}))
    assert config.services["svc"].env == {"FOO": "bar"}


def test_driver_env_accepted():
    config = SubmitConfig.model_validate(_config(driver={**DRIVER, "env": {"KEY": "val"}}))
    assert config.driver.env == {"KEY": "val"}


def test_service_unknown_field_raises():
    with pytest.raises(ValidationError):
        SubmitConfig.model_validate(_config(services={"svc": {**SERVICE, "unknown_field": "x"}}))


# ---------------------------------------------------------------------------
# number_of_instances
# ---------------------------------------------------------------------------

_MULTI_SERVICE = {**SERVICE, "number_of_instances": 4}


def test_number_of_instances_accepted():
    config = SubmitConfig.model_validate(_config(services={"svc": _MULTI_SERVICE}))
    assert config.services["svc"].number_of_instances == 4


def test_number_of_instances_defaults_to_1():
    config = SubmitConfig.model_validate(_config())
    assert config.services["svc"].number_of_instances == 1


def test_number_of_instances_zero_raises():
    with pytest.raises(ValidationError):
        SubmitConfig.model_validate(_config(services={"svc": {**SERVICE, "number_of_instances": 0}}))


def test_distributed_backend_is_not_a_settable_field():
    # distributed_backend (ray vs single-node mp) is an internal implementation detail, fully
    # derived from node count / number_of_instances - not part of the public schema.
    with pytest.raises(ValidationError):
        SubmitConfig.model_validate(_config(services={"svc": {**SERVICE, "distributed_backend": {"type": "ray"}}}))


# ---------------------------------------------------------------------------
# multi-node (ray-backed) deployment
# ---------------------------------------------------------------------------

COMPUTE_MULTI_NODE = {
    "cluster": {
        "type": "slurm",
        "account": "my-account",
        "hostname": "foo",
        "node_pools": {"compute": {"partition": "batch", "nodes": 2, "gpus_per_node": 4}},
    }
}


def test_multi_node_single_instance_accepted():
    # Node count alone is enough to span a single instance's TP/PP footprint across nodes.
    config = SubmitConfig.model_validate(_config(compute=COMPUTE_MULTI_NODE))
    assert config.services["svc"].number_of_instances == 1


def test_multi_node_dp_evenly_divisible_accepted():
    # COMPUTE_MULTI_NODE has 2 nodes; 4 instances split evenly (2 per node).
    config = SubmitConfig.model_validate(
        _config(services={"svc": {**SERVICE, "number_of_instances": 4}}, compute=COMPUTE_MULTI_NODE)
    )
    assert config.services["svc"].number_of_instances == 4


def test_multi_node_dp_uneven_split_raises():
    with pytest.raises(ValidationError, match="evenly divisible"):
        SubmitConfig.model_validate(
            _config(services={"svc": {**SERVICE, "number_of_instances": 3}}, compute=COMPUTE_MULTI_NODE)
        )


# ---------------------------------------------------------------------------
# GPU footprint vs node pool capacity - multi-node
# ---------------------------------------------------------------------------


def test_multi_node_single_instance_footprint_exact_fit_accepted():
    # COMPUTE_MULTI_NODE has 2 nodes x 4 GPUs = 8 total; TP8 spans the whole allocation via ray.
    config = SubmitConfig.model_validate(
        _config(services={"svc": {**SERVICE, "tensor_parallel_size": 8}}, compute=COMPUTE_MULTI_NODE)
    )
    assert config.services["svc"].tensor_parallel_size == 8


def test_multi_node_single_instance_footprint_exceeds_total_raises():
    with pytest.raises(ValidationError, match="exceeds the total GPUs across all nodes"):
        SubmitConfig.model_validate(
            _config(services={"svc": {**SERVICE, "tensor_parallel_size": 9}}, compute=COMPUTE_MULTI_NODE)
        )


def test_multi_node_dp_per_node_footprint_exact_fit_accepted():
    # 4 instances / 2 nodes = 2 local replicas/node; TP2 x 2 local replicas = 4 GPUs, matches gpus_per_node.
    config = SubmitConfig.model_validate(
        _config(
            services={"svc": {**SERVICE, "tensor_parallel_size": 2, "number_of_instances": 4}},
            compute=COMPUTE_MULTI_NODE,
        )
    )
    assert config.services["svc"].number_of_instances == 4


def test_multi_node_dp_per_node_footprint_exceeds_raises():
    # 4 instances / 2 nodes = 2 local replicas/node; TP3 x 2 local replicas = 6 GPUs > gpus_per_node (4).
    with pytest.raises(ValidationError, match="exceeds a single node's gpus_per_node"):
        SubmitConfig.model_validate(
            _config(
                services={"svc": {**SERVICE, "tensor_parallel_size": 3, "number_of_instances": 4}},
                compute=COMPUTE_MULTI_NODE,
            )
        )


def test_multi_instance_per_instance_multi_node_tp_uses_ray_serve_gateway():
    # Each instance's own TP/PP footprint (TP5) exceeds a single node's GPU count (4): the Ray
    # Serve gateway path is forced on automatically (no use_ray_serve needed) since only Ray's own
    # placement-group scheduler - not vLLM's multi-node DP - can place such an instance. Aggregate
    # footprint (5 x 2 = 10) exceeds COMPUTE_MULTI_NODE's total (2 nodes x 4 = 8), so this still
    # raises, just against total cluster capacity rather than "not supported".
    with pytest.raises(ValidationError, match="exceeds the total GPUs across all nodes"):
        SubmitConfig.model_validate(
            _config(
                services={"svc": {**SERVICE, "tensor_parallel_size": 5, "number_of_instances": 2}},
                compute=COMPUTE_MULTI_NODE,
            )
        )


COMPUTE_4_NODES_8_GPUS = {
    "cluster": {
        "type": "slurm",
        "account": "my-account",
        "hostname": "foo",
        "node_pools": {"compute": {"partition": "batch", "nodes": 4, "gpus_per_node": 8}},
    }
}


def test_multi_instance_per_instance_multi_node_tp_accepted_when_footprint_fits():
    # TP8 x PP2 = 16 GPUs/instance > 8 gpus_per_node, so an instance must itself span nodes - the
    # Ray Serve gateway path is forced on. 2 instances x 16 = 32 == 4 nodes x 8 gpus_per_node: fits
    # exactly. This is the previously-forbidden topology that Ray Serve now supports.
    config = SubmitConfig.model_validate(
        _config(
            services={
                "svc": {**SERVICE, "tensor_parallel_size": 8, "pipeline_parallel_size": 2, "number_of_instances": 2}
            },
            compute=COMPUTE_4_NODES_8_GPUS,
            driver=DRIVER_WITH_GYM_INSTALL,
        )
    )
    assert config.services["svc"].number_of_instances == 2


def test_multi_instance_per_instance_multi_node_tp_requires_gym_install():
    # Same topology as above but no driver.gym_install set - the Ray Serve gateway script has
    # nowhere to be fetched from, so this must fail fast at config-validation time rather than
    # only when the sbatch script is built.
    with pytest.raises(ValidationError, match="gym_install"):
        SubmitConfig.model_validate(
            _config(
                services={
                    "svc": {
                        **SERVICE,
                        "tensor_parallel_size": 8,
                        "pipeline_parallel_size": 2,
                        "number_of_instances": 2,
                    }
                },
                compute=COMPUTE_4_NODES_8_GPUS,
            )
        )


COMPUTE_4_NODES_6_GPUS = {
    "cluster": {
        "type": "slurm",
        "account": "my-account",
        "hostname": "foo",
        "node_pools": {"compute": {"partition": "batch", "nodes": 4, "gpus_per_node": 6}},
    }
}


def test_multi_instance_per_instance_multi_node_tp_number_of_instances_need_not_divide_nodes():
    # 3 instances don't evenly divide 4 nodes, which would be rejected for vLLM's own multi-node DP
    # - but Ray's placement-group scheduler packs flexibly, so the Ray Serve gateway path doesn't
    # require this. tp_pp=8 (fits in one node) forces effective_ray_serve via use_ray_serve here so
    # the mismatched node count is exercised without also tripping the footprint check (8 x 3 = 24
    # exactly matches 4 nodes x 6 gpus_per_node, so no idle-GPU warning either).
    config = SubmitConfig.model_validate(
        _config(
            services={"svc": {**SERVICE, "tensor_parallel_size": 8, "number_of_instances": 3, "use_ray_serve": True}},
            compute=COMPUTE_4_NODES_6_GPUS,
            driver=DRIVER_WITH_GYM_INSTALL,
        )
    )
    assert config.services["svc"].number_of_instances == 3


# ---------------------------------------------------------------------------
# use_ray_serve opt-in
# ---------------------------------------------------------------------------


def test_use_ray_serve_defaults_to_false():
    config = SubmitConfig.model_validate(_config())
    assert config.services["svc"].use_ray_serve is False


def test_use_ray_serve_opt_in_single_node_multi_instance_accepted():
    # Single-node, multi-instance: vLLM's own --data-parallel-size would normally handle this, but
    # a user can still opt into the Ray Serve gateway for it.
    service = {**SERVICE, "number_of_instances": 4, "use_ray_serve": True}
    config = SubmitConfig.model_validate(
        _config(services={"svc": service}, compute=COMPUTE_8_GPUS_PER_NODE, driver=DRIVER_WITH_GYM_INSTALL)
    )
    assert config.services["svc"].use_ray_serve is True


def test_use_ray_serve_opt_in_does_not_require_gpus_per_node():
    # No node_pools/gpus_per_node info at all - opting in still validates fine (nothing to check
    # against), matching how the default path also skips footprint validation without that info.
    service = {**SERVICE, "use_ray_serve": True}
    config = SubmitConfig.model_validate(_config(services={"svc": service}, driver=DRIVER_WITH_GYM_INSTALL))
    assert config.services["svc"].use_ray_serve is True


def test_use_ray_serve_opt_in_without_gym_install_raises():
    service = {**SERVICE, "use_ray_serve": True}
    with pytest.raises(ValidationError, match="gym_install"):
        SubmitConfig.model_validate(_config(services={"svc": service}))


# ---------------------------------------------------------------------------
# GPU footprint vs node pool capacity
# ---------------------------------------------------------------------------

COMPUTE_8_GPUS_PER_NODE = {
    "cluster": {
        "type": "slurm",
        "account": "my-account",
        "hostname": "foo",
        "node_pools": {"compute": {"partition": "batch", "gpus_per_node": 8}},
    }
}


def test_gpu_footprint_exact_fit_accepted():
    service = {**SERVICE, "tensor_parallel_size": 2, "number_of_instances": 4}
    config = SubmitConfig.model_validate(_config(services={"svc": service}, compute=COMPUTE_8_GPUS_PER_NODE))
    assert config.services["svc"].number_of_instances == 4


def test_gpu_footprint_exceeds_node_raises():
    service = {**SERVICE, "tensor_parallel_size": 2, "number_of_instances": 8}
    with pytest.raises(ValidationError, match="exceeds the node pool's gpus_per_node"):
        SubmitConfig.model_validate(_config(services={"svc": service}, compute=COMPUTE_8_GPUS_PER_NODE))


def test_gpu_footprint_underutilized_warns():
    service = {**SERVICE, "tensor_parallel_size": 2, "number_of_instances": 2}
    with pytest.warns(UserWarning, match="leaving 4 GPU"):
        config = SubmitConfig.model_validate(_config(services={"svc": service}, compute=COMPUTE_8_GPUS_PER_NODE))
    assert config.services["svc"].number_of_instances == 2


def test_gpu_footprint_no_node_pools_skips_validation():
    # Default COMPUTE fixture has no node_pools, so nothing to validate against.
    config = SubmitConfig.model_validate(_config(services={"svc": _MULTI_SERVICE}))
    assert config.services["svc"].number_of_instances == 4
