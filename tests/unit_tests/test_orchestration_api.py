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

from nemo_gym.orchestration.api import SAVED_CONFIG, SubmitConfig


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


def _policy_driver(*, run=None, command=None):
    bench = {"command": command} if command else {"run": run or {}}
    return {**DRIVER, "benchmarks": {"gsm8k": bench}}


def _gsm8k(config):
    return config.driver.benchmarks["gsm8k"]


@pytest.mark.parametrize(
    "field, value, match",
    [
        ("policy_model", "svc", "driver.policy_model was removed.*service:<name>"),
        ("policy_host", "localhost", "driver.policy_host was removed.*HEAD_NODE_IP"),
        ("policy_model_type", "vllm_model", "driver.policy_model_type was removed.*--model-type"),
    ],
)
def test_removed_policy_fields_get_a_tailored_error(field, value, match):
    with pytest.raises(ValidationError, match=match):
        SubmitConfig.model_validate(_config(driver={**DRIVER, field: value}))


def test_the_sole_model_service_is_bound_as_the_policy_implicitly():
    benchmark = _gsm8k(SubmitConfig.model_validate(_config()))
    assert benchmark.policy_service == "svc"
    assert benchmark.policy_model_type == "openai_model"
    assert benchmark.run["policy_model"] == {
        "responses_api_models": {
            "openai_model": {
                "openai_base_url": "http://localhost:8000/v1",
                "openai_model": "org/model",
                "openai_api_key": "dummy",  # pragma: allowlist secret
            }
        }
    }
    assert benchmark.run["policy_base_url"] == "http://localhost:8000/v1"
    assert benchmark.run["policy_model_name"] == "org/model"
    assert benchmark.run["policy_api_key"] == "dummy"  # pragma: allowlist secret


def test_the_implicit_policy_uses_the_served_model_name():
    config = SubmitConfig.model_validate(_config(services={"svc": {**SERVICE, "served_model_name": "my-model"}}))
    assert _gsm8k(config).run["policy_model_name"] == "my-model"


def test_declarative_and_ray_services_are_not_implicit_policy_candidates():
    ray = {"type": "ray", "container": "c", "node_pools": ["main"]}
    config = SubmitConfig.model_validate(_config(compute=POOLS, services={"svc": SERVICE, "judge": JUDGE, "r": ray}))
    assert _gsm8k(config).policy_service == "svc"


def test_no_model_service_means_no_policy():
    benchmark = _gsm8k(SubmitConfig.model_validate(_config(services={"judge": JUDGE})))
    assert benchmark.policy_service is None
    assert "policy_model" not in benchmark.run
    assert "policy_base_url" not in benchmark.run


def test_several_model_services_need_an_explicit_policy():
    with pytest.raises(ValidationError, match=r"Benchmark 'gsm8k' has no policy_model.*\(svc, other\)"):
        SubmitConfig.model_validate(_config(services={"svc": SERVICE, "other": {**SERVICE, "port": 8001}}))


def test_an_explicit_policy_reference_picks_among_several_and_keeps_its_type():
    run = {"policy_model": {"responses_api_models": {"vllm_model": "service:other"}}}
    config = SubmitConfig.model_validate(
        _config(
            services={"svc": SERVICE, "other": {**SERVICE, "port": 8001, "model": "org/other"}},
            driver=_policy_driver(run=run),
        )
    )
    benchmark = _gsm8k(config)
    assert (benchmark.policy_service, benchmark.policy_model_type) == ("other", "vllm_model")
    assert benchmark.run["policy_model"]["responses_api_models"]["vllm_model"]["base_url"] == (
        "http://localhost:8001/v1"
    )
    assert benchmark.run["policy_model_name"] == "org/other"


def test_an_empty_policy_model_opts_out():
    config = SubmitConfig.model_validate(
        _config(
            services={"svc": SERVICE, "other": {**SERVICE, "port": 8001}},
            driver=_policy_driver(run={"policy_model": {}}),
        )
    )
    benchmark = _gsm8k(config)
    assert benchmark.policy_service is None
    assert benchmark.run == {}


def test_a_policy_node_without_a_reference_adds_nothing():
    node = {"responses_api_models": {"openai_model": {"openai_base_url": "http://elsewhere/v1"}}}
    benchmark = _gsm8k(SubmitConfig.model_validate(_config(driver=_policy_driver(run={"policy_model": node}))))
    assert benchmark.run == {"policy_model": node}
    assert benchmark.policy_model_type is None


def test_a_policy_model_that_is_not_a_mapping_is_rejected():
    with pytest.raises(ValidationError, match="policy_model must be a mapping"):
        SubmitConfig.model_validate(_config(driver=_policy_driver(run={"policy_model": "service:svc"})))


def test_a_declarative_service_can_be_the_policy_explicitly():
    run = {"policy_model": {"responses_api_models": {"openai_model": "service:judge"}}}
    config = SubmitConfig.model_validate(_config(services={"judge": JUDGE}, driver=_policy_driver(run=run)))
    benchmark = _gsm8k(config)
    assert (benchmark.policy_service, benchmark.policy_model_type) == ("judge", "openai_model")
    assert benchmark.run["policy_model"]["responses_api_models"]["openai_model"] == {
        "openai_base_url": JUDGE["base_url"],
        "openai_model": JUDGE["model"],
        "openai_api_key": "dummy",  # pragma: allowlist secret
    }
    assert (benchmark.run["policy_base_url"], benchmark.run["policy_model_name"]) == (
        JUDGE["base_url"],
        JUDGE["model"],
    )
    assert config.policy_service == "judge"


def test_an_explicit_policy_naming_an_unknown_service_names_the_benchmark():
    run = {"policy_model": {"responses_api_models": {"openai_model": "service:nope"}}}
    with pytest.raises(ValidationError, match="Benchmark 'gsm8k' run config: .*does not match any service"):
        SubmitConfig.model_validate(_config(driver=_policy_driver(run=run)))


def test_policy_variables_set_by_the_benchmark_are_kept():
    config = SubmitConfig.model_validate(_config(driver=_policy_driver(run={"policy_base_url": "http://other"})))
    assert _gsm8k(config).run["policy_base_url"] == "http://other"


def test_a_command_benchmark_gets_the_policy_variables_but_no_node():
    benchmark = _gsm8k(SubmitConfig.model_validate(_config(driver=_policy_driver(command="./run.sh"))))
    assert benchmark.policy_service == "svc"
    assert benchmark.policy_model_type is None
    assert benchmark.run["policy_base_url"] == "http://localhost:8000/v1"
    assert "policy_model" not in benchmark.run


def test_a_command_benchmark_with_several_model_services_gets_no_policy():
    config = SubmitConfig.model_validate(
        _config(
            services={"svc": SERVICE, "other": {**SERVICE, "port": 8001}}, driver=_policy_driver(command="./run.sh")
        )
    )
    assert _gsm8k(config).policy_service is None
    assert _gsm8k(config).run == {}


def test_the_policy_binding_is_derived_and_cannot_be_written():
    driver = _policy_driver(run={})
    driver["benchmarks"]["gsm8k"]["policy_service"] = "svc"
    with pytest.raises(ValidationError, match="policy_service is set by submit"):
        SubmitConfig.model_validate(_config(driver=driver))


@pytest.mark.parametrize(
    "run, services",
    [
        ({}, {"svc": SERVICE}),
        ({"policy_model": {}}, {"svc": SERVICE}),
        ({"policy_model": {"responses_api_models": {"vllm_model": "service:svc"}}}, {"svc": SERVICE}),
        (
            {"policy_model": {"responses_api_models": {"openai_model": "service:judge"}}},
            {"judge": {"type": "openai_compatible", "base_url": "https://j.example.com/v1", "model": "m"}},
        ),
    ],
    ids=["implicit", "opt_out", "explicit", "declarative"],
)
def test_a_saved_config_reloads_to_the_same_policy_binding(run, services):
    config = SubmitConfig.model_validate(_config(services=services, driver=_policy_driver(run=run)))
    saved = config.model_dump(mode="json")
    reloaded = SubmitConfig.model_validate(saved, context={SAVED_CONFIG: True})
    before, after = _gsm8k(config), _gsm8k(reloaded)
    assert (after.policy_service, after.policy_model_type, after.run) == (
        before.policy_service,
        before.policy_model_type,
        before.run,
    )


def test_a_saved_command_benchmark_keeps_its_policy_variables():
    config = SubmitConfig.model_validate(_config(driver=_policy_driver(command="./run.sh")))
    reloaded = SubmitConfig.model_validate(config.model_dump(mode="json"), context={SAVED_CONFIG: True})
    assert _gsm8k(reloaded).run == _gsm8k(config).run
    assert _gsm8k(reloaded).policy_service == "svc"


def test_a_server_per_node_service_is_the_implicit_policy_through_vllm_model():
    pools = {
        "cluster": {**POOLS["cluster"], "node_pools": {"main": {"partition": "p", "nodes": 2, "gpus_per_node": 8}}}
    }
    service = {**SERVICE, "node_pool": "main", "server_per_node": True}
    benchmark = _gsm8k(SubmitConfig.model_validate(_config(compute=pools, services={"svc": service})))
    urls = [f"http://${{oc.env:GYM_POOL_MAIN_NODE_{i}}}:8000/v1" for i in range(2)]
    assert benchmark.policy_model_type == "vllm_model"
    assert benchmark.run["policy_model"]["responses_api_models"]["vllm_model"]["base_url"] == urls
    assert benchmark.run["policy_base_url"] == urls


def test_service_env_accepted():
    config = SubmitConfig.model_validate(_config(services={"svc": {**SERVICE, "env": {"FOO": "lit:bar"}}}))
    assert config.services["svc"].env == {"FOO": "bar"}


def test_driver_env_accepted():
    config = SubmitConfig.model_validate(_config(driver={**DRIVER, "env": {"KEY": "lit:val"}}))
    assert config.driver.env == {"KEY": "val"}


def test_env_lit_prefix_strips_to_literal():
    config = SubmitConfig.model_validate(_config(services={"svc": {**SERVICE, "env": {"FOO": "lit:600"}}}))
    assert config.services["svc"].env == {"FOO": "600"}


def test_env_no_prefix_raises():
    with pytest.raises(ValidationError, match="must start with one of the prefixes"):
        SubmitConfig.model_validate(_config(services={"svc": {**SERVICE, "env": {"FOO": "bar"}}}))


def test_env_host_prefix_reads_submitting_env(monkeypatch):
    monkeypatch.setenv("MY_HOST_VAR", "host-value")
    config = SubmitConfig.model_validate(_config(services={"svc": {**SERVICE, "env": {"FOO": "host:MY_HOST_VAR"}}}))
    assert config.services["svc"].env == {"FOO": "host-value"}


def test_env_host_prefix_missing_var_raises(monkeypatch):
    monkeypatch.delenv("MY_MISSING_HOST_VAR", raising=False)
    with pytest.raises(ValidationError, match="MY_MISSING_HOST_VAR"):
        SubmitConfig.model_validate(_config(services={"svc": {**SERVICE, "env": {"FOO": "host:MY_MISSING_HOST_VAR"}}}))


def test_env_runtime_prefix_left_unresolved():
    config = SubmitConfig.model_validate(
        _config(services={"svc": {**SERVICE, "env": {"FOO": "runtime:NEL_INVOCATION_ID"}}})
    )
    assert config.services["svc"].env == {"FOO": "runtime:NEL_INVOCATION_ID"}


def test_env_host_prefix_invalid_var_name_raises():
    with pytest.raises(ValidationError, match="not a valid environment variable name"):
        SubmitConfig.model_validate(_config(services={"svc": {**SERVICE, "env": {"FOO": "host:1BAD"}}}))


def test_env_runtime_prefix_invalid_var_name_raises():
    with pytest.raises(ValidationError, match="not a valid environment variable name"):
        SubmitConfig.model_validate(_config(services={"svc": {**SERVICE, "env": {"FOO": "runtime:1BAD"}}}))


def test_driver_env_host_prefix_reads_submitting_env(monkeypatch):
    monkeypatch.setenv("MY_DRIVER_HOST_VAR", "driver-host-value")
    config = SubmitConfig.model_validate(_config(driver={**DRIVER, "env": {"KEY": "host:MY_DRIVER_HOST_VAR"}}))
    assert config.driver.env == {"KEY": "driver-host-value"}


def test_driver_env_runtime_prefix_left_unresolved():
    config = SubmitConfig.model_validate(_config(driver={**DRIVER, "env": {"KEY": "runtime:NEL_INVOCATION_ID"}}))
    assert config.driver.env == {"KEY": "runtime:NEL_INVOCATION_ID"}


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
    # TP5 exceeds a single node's GPU count (4), forcing the Ray Serve gateway path automatically.
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
    # TP8 x PP2 = 16 GPUs/instance spans nodes; 2 instances x 16 fits exactly in 4 nodes x 8 GPUs.
    config = SubmitConfig.model_validate(
        _config(
            services={
                "svc": {**SERVICE, "tensor_parallel_size": 8, "pipeline_parallel_size": 2, "number_of_instances": 2}
            },
            compute=COMPUTE_4_NODES_8_GPUS,
        )
    )
    assert config.services["svc"].number_of_instances == 2


COMPUTE_4_NODES_6_GPUS = {
    "cluster": {
        "type": "slurm",
        "account": "my-account",
        "hostname": "foo",
        "node_pools": {"compute": {"partition": "batch", "nodes": 4, "gpus_per_node": 6}},
    }
}


def test_multi_instance_per_instance_multi_node_tp_number_of_instances_need_not_divide_nodes():
    # 3 instances don't evenly divide 4 nodes - fine for the Ray Serve gateway path (opted in here).
    config = SubmitConfig.model_validate(
        _config(
            services={"svc": {**SERVICE, "tensor_parallel_size": 8, "number_of_instances": 3, "use_ray_serve": True}},
            compute=COMPUTE_4_NODES_6_GPUS,
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
    service = {**SERVICE, "number_of_instances": 4, "use_ray_serve": True}
    config = SubmitConfig.model_validate(_config(services={"svc": service}, compute=COMPUTE_8_GPUS_PER_NODE))
    assert config.services["svc"].use_ray_serve is True


def test_use_ray_serve_opt_in_does_not_require_gpus_per_node():
    service = {**SERVICE, "use_ray_serve": True}
    config = SubmitConfig.model_validate(_config(services={"svc": service}))
    assert config.services["svc"].use_ray_serve is True


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


# ---------------------------------------------------------------------------
# KubernetesComputeConfig
# ---------------------------------------------------------------------------

COMPUTE_K8S = {"cluster": {"type": "kubernetes", "namespace": "eng-test", "gpus_per_node": 8, "pvc_name": "workspace"}}


def test_kubernetes_compute_accepted():
    config = SubmitConfig.model_validate(_config(compute=COMPUTE_K8S))
    assert config.services["svc"].placement == "cluster"


def test_kubernetes_gpu_footprint_exact_fit_accepted():
    service = {**SERVICE, "tensor_parallel_size": 2, "number_of_instances": 4}
    config = SubmitConfig.model_validate(_config(services={"svc": service}, compute=COMPUTE_K8S))
    assert config.services["svc"].number_of_instances == 4


def test_kubernetes_gpu_footprint_exceeds_node_raises():
    service = {**SERVICE, "tensor_parallel_size": 2, "number_of_instances": 8}
    with pytest.raises(ValidationError, match="exceeds the node pool's gpus_per_node"):
        SubmitConfig.model_validate(_config(services={"svc": service}, compute=COMPUTE_K8S))


def test_kubernetes_node_pool_on_service_raises():
    # Kubernetes v1 has no named pools; a service pinned to one has nothing to match.
    service = {**SERVICE, "node_pool": "compute"}
    with pytest.raises(ValidationError, match="does not match any node pool"):
        SubmitConfig.model_validate(_config(services={"svc": service}, compute=COMPUTE_K8S))


def test_kubernetes_compute_discriminated_from_slurm():
    k8s_config = SubmitConfig.model_validate(_config(compute=COMPUTE_K8S))
    slurm_config = SubmitConfig.model_validate(_config())
    assert type(next(iter(k8s_config.compute.values()))).__name__ == "KubernetesComputeConfig"
    assert type(next(iter(slurm_config.compute.values()))).__name__ == "SlurmComputeConfig"


def test_kubernetes_unknown_type_rejected():
    with pytest.raises(ValidationError):
        SubmitConfig.model_validate(_config(compute={"cluster": {"type": "not-a-real-backend"}}))


def test_kubernetes_pvc_name_is_required():
    # The driver container's own volumeMounts always references the output volume
    # (kubernetes_script.py's build_job_manifest); without a pvc_name that volume is never
    # declared, and the pod is rejected outright by the API server. Fail at config-validation
    # time with a clear message instead.
    with pytest.raises(ValidationError, match="pvc_name"):
        SubmitConfig.model_validate(_config(compute={"cluster": {"type": "kubernetes", "namespace": "eng-test"}}))


# ---------------------------------------------------------------------------
# resumable / ResumeConfig
# ---------------------------------------------------------------------------


def test_resumable_defaults_to_false():
    config = SubmitConfig.model_validate(_config())
    benchmark = config.driver.benchmarks["gsm8k"]
    assert benchmark.resumable is False
    assert benchmark.resume_config is None


def test_resumable_true_uses_resume_config_defaults():
    driver = {**DRIVER, "benchmarks": {"gsm8k": {"resumable": True}}}
    config = SubmitConfig.model_validate(_config(driver=driver))
    resume = config.driver.benchmarks["gsm8k"].resume_config
    assert resume.max_retries == 3
    assert resume.max_walltime is None


def test_resumable_object_overrides_defaults():
    driver = {**DRIVER, "benchmarks": {"gsm8k": {"resumable": {"max_retries": 7, "max_walltime": "48:00:00"}}}}
    config = SubmitConfig.model_validate(_config(driver=driver))
    resume = config.driver.benchmarks["gsm8k"].resume_config
    assert resume.max_retries == 7
    assert resume.max_walltime == "48:00:00"


@pytest.mark.parametrize("resumable", [True, {"max_retries": 5}])
def test_resumable_is_rejected_on_command_benchmarks(resumable):
    driver = {**DRIVER, "benchmarks": {"gsm8k": {"command": "bash run.sh", "resumable": resumable}}}
    with pytest.raises(ValidationError, match="both `command` and `resumable`"):
        SubmitConfig.model_validate(_config(driver=driver))


def test_command_benchmark_without_resumable_is_accepted():
    driver = {**DRIVER, "benchmarks": {"gsm8k": {"command": "bash run.sh", "resumable": False}}}
    config = SubmitConfig.model_validate(_config(driver=driver))
    assert config.driver.benchmarks["gsm8k"].resume_config is None


JUDGE = {"type": "openai_compatible", "base_url": "https://judge.example.com:8443/v1", "model": "gpt-4o"}
POOLS = {
    "cluster": {
        "type": "slurm",
        "account": "a",
        "hostname": "h",
        "node_pools": {"main": {"partition": "p", "nodes": 1}, "aux": {"partition": "p", "nodes": 1}},
    }
}


def _run(**run):
    return _config(driver={**DRIVER, "benchmarks": {"gsm8k": {"run": run}}})


def test_declarative_service_is_not_deployed():
    config = SubmitConfig.model_validate(_config(services={"svc": SERVICE, "judge": JUDGE}))
    assert list(config.deployed_services) == ["svc"]


def test_declarative_service_rejects_deployment_fields():
    with pytest.raises(ValidationError, match="container"):
        SubmitConfig.model_validate(_config(services={"judge": {**JUDGE, "container": "x"}}))


def test_whole_server_reference_expands_by_server_type():
    cfg = _run(judge={"responses_api_models": {"openai_model": "service:judge"}})
    config = SubmitConfig.model_validate({**cfg, "services": {"svc": SERVICE, "judge": JUDGE}})
    server = config.driver.benchmarks["gsm8k"].run["judge"]["responses_api_models"]["openai_model"]
    assert server == {
        "openai_base_url": "https://judge.example.com:8443/v1",
        "openai_model": "gpt-4o",
        "openai_api_key": "dummy",  # pragma: allowlist secret
    }


def test_whole_server_reference_to_vllm_service_uses_vllm_fields():
    cfg = _run(m={"responses_api_models": {"vllm_model": "service:svc"}})
    run = SubmitConfig.model_validate(cfg).driver.benchmarks["gsm8k"].run
    assert run["m"]["responses_api_models"]["vllm_model"] == {
        "base_url": "http://localhost:8000/v1",
        "model": "org/model",
        "api_key": "dummy",  # pragma: allowlist secret
    }


def test_field_reference_fills_one_value():
    cfg = _run(openai_base_url="service:judge.url", port="service:judge.port", host="service:judge.host")
    config = SubmitConfig.model_validate({**cfg, "services": {"judge": JUDGE}})
    run = config.driver.benchmarks["gsm8k"].run
    assert (run["openai_base_url"], run["port"], run["host"]) == (JUDGE["base_url"], "8443", "judge.example.com")


def test_served_model_name_is_the_model_field():
    cfg = _run(name="service:svc.model")
    config = SubmitConfig.model_validate({**cfg, "services": {"svc": {**SERVICE, "served_model_name": "alias"}}})
    assert config.driver.benchmarks["gsm8k"].run["name"] == "alias"
    assert config.deferred_host_services == []


def test_pinned_service_host_is_deferred_to_a_driver_env_var():
    cfg = _run(url="service:svc.url")
    config = SubmitConfig.model_validate(
        {**cfg, "compute": POOLS, "services": {"svc": {**SERVICE, "node_pool": "aux"}}}
    )
    assert config.driver.benchmarks["gsm8k"].run["url"] == "http://${oc.env:GYM_SERVICE_SVC_HOST}:8000/v1"
    assert config.deferred_host_services == ["svc"]


def test_a_saved_config_still_defers_the_hosts_its_resolved_references_use():
    cfg = _run(url="service:svc.url")
    config = SubmitConfig.model_validate(
        {**cfg, "compute": POOLS, "services": {"svc": {**SERVICE, "node_pool": "aux"}}}
    )
    resumed = SubmitConfig.model_validate(config.model_dump(mode="json"), context={SAVED_CONFIG: True})
    assert resumed.deferred_host_services == ["svc"]


def test_a_server_per_node_service_references_every_nodes_url():
    pools = {
        "cluster": {
            **POOLS["cluster"],
            "node_pools": {"main": {"partition": "p", "nodes": 3, "gpus_per_node": 8}},
        }
    }
    service = {**SERVICE, "node_pool": "main", "server_per_node": True}
    cfg = _run(urls="service:svc.url", host="service:svc.host")
    config = SubmitConfig.model_validate({**cfg, "compute": pools, "services": {"svc": service}})
    run = config.driver.benchmarks["gsm8k"].run
    assert run["urls"] == [f"http://${{oc.env:GYM_POOL_MAIN_NODE_{i}}}:8000/v1" for i in range(3)]
    assert run["host"] == "${oc.env:GYM_SERVICE_SVC_HOST}"


def test_unreferenced_pinned_service_defers_nothing():
    config = SubmitConfig.model_validate(
        _config(
            compute=POOLS,
            services={"svc": {**SERVICE, "node_pool": "aux"}},
            driver=_policy_driver(run={"policy_model": {}}),
        )
    )
    assert config.deferred_host_services == []


@pytest.mark.parametrize(
    "run, match",
    [
        ({"u": "service:nope.url"}, "does not match any service"),
        ({"u": "service:svc.bogus"}, "unknown field 'bogus'"),
        ({"u": "service:svc"}, "not a server type"),
    ],
)
def test_bad_service_reference_names_the_benchmark(run, match):
    with pytest.raises(ValidationError, match=f"Benchmark 'gsm8k' run config: .*{match}"):
        SubmitConfig.model_validate(_run(**run))


def test_reference_to_a_ray_service_is_rejected():
    ray = {"type": "ray", "container": "c", "node_pools": ["main"]}
    cfg = _run(u="service:r.url")
    with pytest.raises(ValidationError, match="does not serve a model"):
        SubmitConfig.model_validate({**cfg, "compute": POOLS, "services": {"r": ray}})


def test_resolving_service_references_leaves_the_input_config_untouched():
    raw = _run(outer={"inner": {"url": "service:svc.url"}, "server": {"vllm_model": "service:svc"}})
    config = SubmitConfig.model_validate(raw)
    assert raw["driver"]["benchmarks"]["gsm8k"]["run"]["outer"] == {
        "inner": {"url": "service:svc.url"},
        "server": {"vllm_model": "service:svc"},
    }
    assert config.driver.benchmarks["gsm8k"].run["outer"]["inner"]["url"] == "http://localhost:8000/v1"


@pytest.mark.parametrize(
    "base_url", ["judge.example.com/v1", "ftp://judge.example.com", "https:///v1", "http://h:99999/v1"]
)
def test_declarative_service_base_url_must_be_an_http_url_with_a_host(base_url):
    with pytest.raises(ValidationError, match="base_url"):
        SubmitConfig.model_validate(_config(services={"judge": {**JUDGE, "base_url": base_url}}))


def test_declarative_service_port_defaults_from_the_scheme():
    cfg = _run(a="service:j.port", b="service:k.port")
    services = {
        "j": {**JUDGE, "base_url": "https://judge.example.com/v1"},
        "k": {**JUDGE, "base_url": "http://judge.example.com/v1"},
    }
    run = SubmitConfig.model_validate({**cfg, "services": services}).driver.benchmarks["gsm8k"].run
    assert (run["a"], run["b"]) == ("443", "80")


@pytest.mark.parametrize(
    "field, value",
    [
        ("base_url", "https://h/v1?a=1,2"),
        ("base_url", "https://h/v1#frag"),
        ("base_url", "https://h/${oc.env:HOME}"),
        ("base_url", "http://[::1]:8000/v1"),
        ("model", "my model"),
        ("model", "${oc.env:SECRET}"),
    ],
)
def test_declarative_service_rejects_values_hydra_cannot_carry(field, value):
    with pytest.raises(ValidationError, match="Hydra override"):
        SubmitConfig.model_validate(_config(services={"judge": {**JUDGE, field: value}}))


def test_two_referenced_pinned_services_sharing_a_host_variable_are_rejected():
    pinned = {**SERVICE, "node_pool": "aux"}
    services = {"a-b": pinned, "a_b": {**pinned, "port": 8001}}
    cfg = _run(p="service:a-b.url", q="service:a_b.url", policy_model={})
    with pytest.raises(ValidationError, match="GYM_SERVICE_A_B_HOST"):
        SubmitConfig.model_validate({**cfg, "compute": POOLS, "services": services})


def test_service_references_resolve_alongside_policy_injection():
    cfg = _config(driver=_policy_driver(run={"u": "service:judge.url"}))
    config = SubmitConfig.model_validate({**cfg, "services": {"svc": SERVICE, "judge": JUDGE}})
    run = config.driver.benchmarks["gsm8k"].run
    assert run["u"] == JUDGE["base_url"]
    assert run["policy_base_url"] == "http://localhost:8000/v1"


def test_a_deployed_model_service_describes_its_own_fields():
    config = SubmitConfig.model_validate(_config(services={"svc": {**SERVICE, "served_model_name": "alias"}}))
    accessors = config.services["svc"].accessors("svc")
    assert (accessors.url, accessors.host, accessors.port, accessors.model) == (
        "http://localhost:8000/v1",
        "localhost",
        "8000",
        "alias",
    )
    assert not accessors.host_deferred


def test_a_pinned_service_defers_its_host_to_the_name_derived_variable():
    config = SubmitConfig.model_validate(
        _config(compute=POOLS, services={"my-judge": {**SERVICE, "node_pool": "aux"}})
    )
    accessors = config.services["my-judge"].accessors("my-judge")
    assert accessors.host == "${oc.env:GYM_SERVICE_MY_JUDGE_HOST}"
    assert accessors.url == "http://${oc.env:GYM_SERVICE_MY_JUDGE_HOST}:8000/v1"
    assert accessors.host_deferred


def test_a_declarative_service_describes_its_own_fields():
    config = SubmitConfig.model_validate(_config(services={"judge": JUDGE}))
    accessors = config.services["judge"].accessors("judge")
    assert (accessors.url, accessors.host, accessors.port, accessors.model) == (
        "https://judge.example.com:8443/v1",
        "judge.example.com",
        "8443",
        "gpt-4o",
    )
    assert not accessors.host_deferred
