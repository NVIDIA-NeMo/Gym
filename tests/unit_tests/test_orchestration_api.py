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
    assert benchmark.run["policy_api_key"] == "dummy"


def test_policy_model_conflict_raises():
    driver = {**DRIVER, "policy_model": "svc", "benchmarks": {"gsm8k": {"run": {"policy_base_url": "http://other"}}}}
    with pytest.raises(ValidationError, match="already sets"):
        SubmitConfig.model_validate(_config(driver=driver))
