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
# number_of_instances / distributed_backend
# ---------------------------------------------------------------------------

_MULTI_SERVICE = {**SERVICE, "number_of_instances": 4, "distributed_backend": {"type": "mp"}}


def test_number_of_instances_with_backend_accepted():
    config = SubmitConfig.model_validate(_config(services={"svc": _MULTI_SERVICE}))
    svc = config.services["svc"]
    assert svc.number_of_instances == 4
    assert svc.distributed_backend is not None
    assert svc.distributed_backend.type == "mp"


def test_number_of_instances_defaults_to_1():
    config = SubmitConfig.model_validate(_config())
    assert config.services["svc"].number_of_instances == 1
    assert config.services["svc"].distributed_backend is None


def test_multi_instance_without_backend_defaults_to_mp():
    config = SubmitConfig.model_validate(_config(services={"svc": {**SERVICE, "number_of_instances": 4}}))
    svc = config.services["svc"]
    assert svc.distributed_backend is not None
    assert svc.distributed_backend.type == "mp"


def test_single_instance_with_backend_raises():
    with pytest.raises(ValidationError, match="should not be set"):
        SubmitConfig.model_validate(_config(services={"svc": {**SERVICE, "distributed_backend": {"type": "mp"}}}))


def test_number_of_instances_zero_raises():
    with pytest.raises(ValidationError):
        SubmitConfig.model_validate(_config(services={"svc": {**SERVICE, "number_of_instances": 0}}))


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
    service = {
        **SERVICE,
        "tensor_parallel_size": 2,
        "number_of_instances": 4,
        "distributed_backend": {"type": "mp"},
    }
    config = SubmitConfig.model_validate(_config(services={"svc": service}, compute=COMPUTE_8_GPUS_PER_NODE))
    assert config.services["svc"].number_of_instances == 4


def test_gpu_footprint_exceeds_node_raises():
    service = {
        **SERVICE,
        "tensor_parallel_size": 2,
        "number_of_instances": 8,
        "distributed_backend": {"type": "mp"},
    }
    with pytest.raises(ValidationError, match="exceeds the largest available"):
        SubmitConfig.model_validate(_config(services={"svc": service}, compute=COMPUTE_8_GPUS_PER_NODE))


def test_gpu_footprint_underutilized_warns():
    service = {
        **SERVICE,
        "tensor_parallel_size": 2,
        "number_of_instances": 2,
        "distributed_backend": {"type": "mp"},
    }
    with pytest.warns(UserWarning, match="leaving 4 GPU"):
        config = SubmitConfig.model_validate(_config(services={"svc": service}, compute=COMPUTE_8_GPUS_PER_NODE))
    assert config.services["svc"].number_of_instances == 2


def test_gpu_footprint_no_node_pools_skips_validation():
    # Default COMPUTE fixture has no node_pools, so nothing to validate against.
    config = SubmitConfig.model_validate(_config(services={"svc": _MULTI_SERVICE}))
    assert config.services["svc"].number_of_instances == 4
