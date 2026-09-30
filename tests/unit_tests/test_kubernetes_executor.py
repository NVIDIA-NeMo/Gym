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

import json
import subprocess

import pytest

from nemo_gym.orchestration.api import SubmitConfig
from nemo_gym.orchestration.executors import kubernetes as kubernetes_module
from nemo_gym.orchestration.executors.kubernetes import KubernetesExecutor
from nemo_gym.orchestration.executors.kubernetes_script import _dns_label, _scale_quantity, job_name
from nemo_gym.orchestration.jobs import SubmissionRecord


def _submit_config(tmp_path, benchmarks, services=None, compute_overrides=None):
    return SubmitConfig.model_validate(
        {
            "services": services or {},
            "compute": {
                "cluster": {
                    "type": "kubernetes",
                    "namespace": "eng-test",
                    "pvc_name": "workspace",
                    **(compute_overrides or {}),
                }
            },
            "driver": {"container": "gym:latest", "benchmarks": {name: {} for name in benchmarks}},
            "job": {"output_path": str(tmp_path / "jobs")},
            "otel": {"enabled": False},
        }
    )


class _FakeKubectl:
    """Answers `_kubectl(...)` calls in the order they are made, without touching a cluster.

    Patches `kubernetes_module._kubectl` directly rather than `subprocess.run`: the latter is the
    same module object every caller shares (including `jobs.installed_gym_commit`'s own `git`
    subprocess calls), so replacing it globally breaks unrelated code the executor also calls.
    """

    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = []

    def __call__(self, compute, *args, input=None):
        self.calls.append((args, input))
        returncode, stdout, stderr = self._replies.pop(0)
        return subprocess.CompletedProcess(args, returncode, stdout=stdout, stderr=stderr)


def _install(monkeypatch, fake_kubectl):
    monkeypatch.setattr(kubernetes_module, "_kubectl", fake_kubectl)
    monkeypatch.setattr(kubernetes_module.shutil, "which", lambda name: "/usr/bin/kubectl")


def test_dns_label_sanitizes_gym_names_for_kubernetes():
    assert _dns_label("vllm_model") == "vllm-model"
    assert _dns_label("tau2.airline") == "tau2-airline"
    assert _dns_label("GPQA") == "gpqa"


def test_job_name_is_a_valid_dns_label():
    name = job_name("gym-job-20260101T000000Z-abcdef", "gpqa_diamond")
    assert name == "gym-gym-job-20260101t000000z-abcdef-gpqa-diamond"


def test_scale_quantity_multiplies_the_numeric_part():
    assert _scale_quantity("32Gi", 2) == "64Gi"
    assert _scale_quantity("512Mi", 1) == "512Mi"


def test_scale_quantity_rejects_an_unparseable_value():
    with pytest.raises(ValueError, match="not a supported memory quantity"):
        _scale_quantity("lots", 2)


def test_gpu_sidecar_gets_a_memory_request_scaled_by_gpu_count(tmp_path):
    from nemo_gym.orchestration.executors.kubernetes_script import build_job_manifest

    services = {
        "vllm_model": {
            "type": "vllm",
            "container": "vllm/vllm-openai:latest",
            "model": "org/model",
            "tensor_parallel_size": 2,
        }
    }
    config = _submit_config(tmp_path, ["bench_a"], services=services)
    compute = next(iter(config.compute.values()))
    benchmark = config.driver.benchmarks["bench_a"]

    job = build_job_manifest(
        config,
        "bench_a",
        benchmark,
        compute,
        tmp_path / "run",
        name="gym-test-bench-a",
        gym_job_id="gym-job-test",
        resolved_config="",
        manifest="",
    )

    sidecar = job["spec"]["template"]["spec"]["initContainers"][0]
    assert sidecar["resources"]["requests"]["memory"] == "64Gi"
    driver = job["spec"]["template"]["spec"]["containers"][0]
    assert driver["resources"]["requests"]["memory"]


def test_gpu_sidecar_has_a_startup_probe_that_gates_the_driver(tmp_path):
    # readinessProbe alone does not delay when kubelet starts the next (driver) container for a
    # native sidecar -- only startupProbe does. Missing this meant the driver ran before vLLM was
    # actually serving, observed for real against the cluster.
    from nemo_gym.orchestration.executors.kubernetes_script import build_job_manifest

    services = {
        "vllm_model": {
            "type": "vllm",
            "container": "vllm/vllm-openai:latest",
            "model": "org/model",
        }
    }
    config = _submit_config(tmp_path, ["bench_a"], services=services)
    compute = next(iter(config.compute.values()))
    benchmark = config.driver.benchmarks["bench_a"]

    job = build_job_manifest(
        config,
        "bench_a",
        benchmark,
        compute,
        tmp_path / "run",
        name="gym-test-bench-a",
        gym_job_id="gym-job-test",
        resolved_config="",
        manifest="",
    )

    sidecar = job["spec"]["template"]["spec"]["initContainers"][0]
    assert "startupProbe" in sidecar
    assert sidecar["startupProbe"]["httpGet"]["path"] == "/health"


def test_run_returns_a_record_naming_every_benchmark(tmp_path, monkeypatch):
    fake = _FakeKubectl([(0, "job.batch/x created", ""), (0, "job.batch/y created", "")])
    _install(monkeypatch, fake)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    record = KubernetesExecutor().run(_submit_config(tmp_path, ["bench_a", "bench_b"]))

    assert record is not None
    assert [b.benchmark for b in record.benchmarks] == ["bench_a", "bench_b"]
    assert all(b.job_id is not None for b in record.benchmarks)
    assert record.cluster == "cluster"
    assert record.executor == "kubernetes"
    assert record.hostname is None
    assert record.executor_metadata["namespace"] == "eng-test"
    assert record.run_dir.endswith(record.gym_job_id)


def test_run_records_a_failed_benchmark_without_disturbing_others(tmp_path, monkeypatch):
    fake = _FakeKubectl(
        [
            (0, "job.batch/x created", ""),
            (1, "", "Error from server (Forbidden): jobs.batch is forbidden"),
        ]
    )
    _install(monkeypatch, fake)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    record = KubernetesExecutor().run(_submit_config(tmp_path, ["bench_a", "bench_b"]))

    by_name = {b.benchmark: b for b in record.benchmarks}
    assert by_name["bench_a"].job_id is not None
    assert by_name["bench_a"].error is None
    assert by_name["bench_b"].job_id is None
    assert "Forbidden" in by_name["bench_b"].error


def test_run_writes_the_local_index(tmp_path, monkeypatch):
    fake = _FakeKubectl([(0, "job.batch/x created", "")])
    _install(monkeypatch, fake)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    record = KubernetesExecutor().run(_submit_config(tmp_path, ["bench_a"]))

    index = tmp_path / "cache" / "nemo-gym" / "jobs" / f"{record.gym_job_id}.json"
    assert SubmissionRecord.load(json.loads(index.read_text())) == record


def test_dry_run_returns_none_and_never_calls_kubectl(tmp_path, monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("kubectl should not be invoked on a dry run")

    monkeypatch.setattr(kubernetes_module, "_kubectl", _boom)

    result = KubernetesExecutor().run(_submit_config(tmp_path, ["bench_a"]), dry_run=True)

    assert result is None


def test_otel_enabled_by_default_is_rejected(tmp_path):
    services = {
        "vllm_model": {
            "type": "vllm",
            "container": "vllm/vllm-openai:latest",
            "model": "Qwen/Qwen2.5-1.5B-Instruct",
        }
    }
    config = SubmitConfig.model_validate(
        {
            "services": services,
            "compute": {"cluster": {"type": "kubernetes", "namespace": "eng-test"}},
            "driver": {"container": "gym:latest", "benchmarks": {"bench_a": {}}},
            "job": {"output_path": str(tmp_path / "jobs")},
        }
    )

    with pytest.raises(ValueError, match="otel"):
        KubernetesExecutor().run(config, dry_run=True)


def test_ray_serve_service_is_rejected(tmp_path):
    services = {
        "vllm_model": {
            "type": "vllm",
            "container": "vllm/vllm-openai:latest",
            "model": "Qwen/Qwen2.5-1.5B-Instruct",
            "use_ray_serve": True,
        }
    }
    config = _submit_config(tmp_path, ["bench_a"], services=services)

    with pytest.raises(ValueError, match="use_ray_serve"):
        KubernetesExecutor().run(config, dry_run=True)


# ---------------------------------------------------------------------------
# multi-node data-parallel deployment
# ---------------------------------------------------------------------------

_MULTI_NODE_SERVICES = {
    "vllm_model": {
        "type": "vllm",
        "container": "vllm/vllm-openai:latest",
        "model": "org/model",
        "tensor_parallel_size": 2,
        "number_of_instances": 4,
        "extra_args": "--api-server-count 4",
    }
}


def _multi_node_job(tmp_path):
    from nemo_gym.orchestration.executors.kubernetes_script import build_job_manifest

    config = _submit_config(
        tmp_path, ["bench_a"], services=_MULTI_NODE_SERVICES, compute_overrides={"nodes": 2, "gpus_per_node": 4}
    )
    compute = next(iter(config.compute.values()))
    benchmark = config.driver.benchmarks["bench_a"]
    return build_job_manifest(
        config,
        "bench_a",
        benchmark,
        compute,
        tmp_path / "run",
        name="gym-test-bench-a",
        gym_job_id="gym-job-test",
        resolved_config="",
        manifest="",
    )


def test_number_of_instances_greater_than_one_rejected_on_single_node_job(tmp_path):
    from nemo_gym.orchestration.executors.kubernetes_script import build_job_manifest

    services = {
        "vllm_model": {
            "type": "vllm",
            "container": "vllm/vllm-openai:latest",
            "model": "org/model",
            "number_of_instances": 2,
        }
    }
    config = _submit_config(tmp_path, ["bench_a"], services=services)
    compute = next(iter(config.compute.values()))
    benchmark = config.driver.benchmarks["bench_a"]

    with pytest.raises(ValueError, match="compute.nodes > 1"):
        build_job_manifest(
            config,
            "bench_a",
            benchmark,
            compute,
            tmp_path / "run",
            name="gym-test-bench-a",
            gym_job_id="gym-job-test",
            resolved_config="",
            manifest="",
        )


def test_multi_node_job_uses_indexed_completion_mode(tmp_path):
    job = _multi_node_job(tmp_path)
    assert job["spec"]["completionMode"] == "Indexed"
    assert job["spec"]["parallelism"] == 2
    assert job["spec"]["completions"] == 2


def test_single_node_job_has_no_indexed_completion_fields(tmp_path):
    from nemo_gym.orchestration.executors.kubernetes_script import build_job_manifest

    services = {"vllm_model": {"type": "vllm", "container": "vllm/vllm-openai:latest", "model": "org/model"}}
    config = _submit_config(tmp_path, ["bench_a"], services=services)
    compute = next(iter(config.compute.values()))
    benchmark = config.driver.benchmarks["bench_a"]

    job = build_job_manifest(
        config,
        "bench_a",
        benchmark,
        compute,
        tmp_path / "run",
        name="gym-test-bench-a",
        gym_job_id="gym-job-test",
        resolved_config="",
        manifest="",
    )

    assert "completionMode" not in job["spec"]
    assert "parallelism" not in job["spec"]
    assert "completions" not in job["spec"]


def test_multi_node_vllm_sidecar_branches_on_job_completion_index(tmp_path):
    job = _multi_node_job(tmp_path)
    sidecar = job["spec"]["template"]["spec"]["initContainers"][0]
    assert sidecar["command"][:2] == ["bash", "-lc"]
    script = sidecar["command"][2]
    assert "JOB_COMPLETION_INDEX" in script
    assert "--data-parallel-size 4" in script
    assert "--data-parallel-size-local 2" in script
    assert "--headless" in script
    assert "--data-parallel-start-rank $(( JOB_COMPLETION_INDEX * 2 ))" in script
    assert "gym-test-bench-a-head.eng-test.svc.cluster.local" in script


def test_multi_node_worker_strips_api_server_count(tmp_path):
    job = _multi_node_job(tmp_path)
    script = job["spec"]["template"]["spec"]["initContainers"][0]["command"][2]
    head_branch, _, worker_branch = script.partition("else")
    assert "--api-server-count" in head_branch
    assert "--api-server-count" not in worker_branch


def test_multi_node_sidecar_uses_exec_probe_not_httpget(tmp_path):
    job = _multi_node_job(tmp_path)
    sidecar = job["spec"]["template"]["spec"]["initContainers"][0]
    assert "httpGet" not in sidecar["startupProbe"]
    assert "exec" in sidecar["startupProbe"]
    assert "JOB_COMPLETION_INDEX" in sidecar["startupProbe"]["exec"]["command"][2]
    assert "exec" in sidecar["readinessProbe"]


def test_multi_node_driver_placeholders_non_zero_index(tmp_path):
    job = _multi_node_job(tmp_path)
    driver_script = job["spec"]["template"]["spec"]["containers"][0]["command"][2]
    assert "sleep infinity" in driver_script


def test_multi_node_driver_command_includes_self_delete_trap(tmp_path):
    job = _multi_node_job(tmp_path)
    driver_script = job["spec"]["template"]["spec"]["containers"][0]["command"][2]
    assert "trap _gym_k8s_cleanup EXIT" in driver_script
    assert "gym-test-bench-a" in driver_script
    assert "gym-test-bench-a-head" in driver_script


def test_single_node_driver_command_has_no_self_delete_trap(tmp_path):
    from nemo_gym.orchestration.executors.kubernetes_script import build_job_manifest

    services = {"vllm_model": {"type": "vllm", "container": "vllm/vllm-openai:latest", "model": "org/model"}}
    config = _submit_config(tmp_path, ["bench_a"], services=services)
    compute = next(iter(config.compute.values()))
    benchmark = config.driver.benchmarks["bench_a"]

    job = build_job_manifest(
        config,
        "bench_a",
        benchmark,
        compute,
        tmp_path / "run",
        name="gym-test-bench-a",
        gym_job_id="gym-job-test",
        resolved_config="",
        manifest="",
    )

    driver_script = job["spec"]["template"]["spec"]["containers"][0]["command"][2]
    assert "_gym_k8s_cleanup" not in driver_script


def test_head_service_manifest_selects_completion_index_zero():
    from nemo_gym.orchestration.executors.kubernetes_script import build_head_service_manifest

    compute = SubmitConfig.model_validate(
        {
            "services": _MULTI_NODE_SERVICES,
            "compute": {"cluster": {"type": "kubernetes", "namespace": "eng-test", "nodes": 2, "gpus_per_node": 4}},
            "driver": {"container": "gym:latest", "benchmarks": {"bench_a": {}}},
            "job": {"output_path": "/tmp/jobs"},
            "otel": {"enabled": False},
        }
    ).compute["cluster"]
    labels = {"gym-job-id": "gym-job-test", "gym-benchmark": "bench-a"}

    service = build_head_service_manifest(compute, "gym-test-bench-a", labels, vllm_port=8000)

    assert service["kind"] == "Service"
    assert service["metadata"]["name"] == "gym-test-bench-a-head"
    assert service["spec"]["selector"]["batch.kubernetes.io/job-completion-index"] == "0"
    ports = {p["port"] for p in service["spec"]["ports"]}
    assert ports == {8000, 13345}


def test_head_service_manifest_is_headless_with_not_ready_addresses_published():
    # A normal ClusterIP is a virtual address no pod's network interface actually owns, so vLLM's
    # head process (which binds a ZMQ socket directly to this address, not just advertises it to
    # workers) can't bind to it. Must be headless so DNS resolves straight to the pod's real,
    # bindable IP, and must publish not-ready addresses since the head needs to resolve/bind its
    # own address before it can ever become Ready.
    from nemo_gym.orchestration.executors.kubernetes_script import build_head_service_manifest

    compute = SubmitConfig.model_validate(
        {
            "services": _MULTI_NODE_SERVICES,
            "compute": {"cluster": {"type": "kubernetes", "namespace": "eng-test", "nodes": 2, "gpus_per_node": 4}},
            "driver": {"container": "gym:latest", "benchmarks": {"bench_a": {}}},
            "job": {"output_path": "/tmp/jobs"},
            "otel": {"enabled": False},
        }
    ).compute["cluster"]
    labels = {"gym-job-id": "gym-job-test", "gym-benchmark": "bench-a"}

    service = build_head_service_manifest(compute, "gym-test-bench-a", labels, vllm_port=8000)

    assert service["spec"]["clusterIP"] == "None"
    assert service["spec"]["publishNotReadyAddresses"] is True


def test_build_manifests_includes_service_doc_only_when_multi_node(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    from datetime import datetime, timezone

    config = _submit_config(tmp_path, ["bench_a"])
    compute = next(iter(config.compute.values()))
    docs = KubernetesExecutor()._build_manifests(
        config, compute, "gym-job-test", datetime.now(timezone.utc), tmp_path / "jobs" / "gym-job-test", ["bench_a"]
    )
    assert len(docs[0][1]) == 1

    multi_config = _submit_config(
        tmp_path, ["bench_a"], services=_MULTI_NODE_SERVICES, compute_overrides={"nodes": 2, "gpus_per_node": 4}
    )
    multi_compute = next(iter(multi_config.compute.values()))
    multi_docs = KubernetesExecutor()._build_manifests(
        multi_config,
        multi_compute,
        "gym-job-test",
        datetime.now(timezone.utc),
        tmp_path / "jobs" / "gym-job-test",
        ["bench_a"],
    )
    assert len(multi_docs[0][1]) == 2
    assert multi_docs[0][1][1]["kind"] == "Service"


def test_run_applies_combined_multi_document_manifest_for_multi_node_job(tmp_path, monkeypatch):
    fake = _FakeKubectl([(0, "job.batch/x created", "")])
    _install(monkeypatch, fake)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    config = _submit_config(
        tmp_path, ["bench_a"], services=_MULTI_NODE_SERVICES, compute_overrides={"nodes": 2, "gpus_per_node": 4}
    )

    record = KubernetesExecutor().run(config)

    assert record is not None
    _, rendered_input = fake.calls[0]
    assert rendered_input.count("kind: Job") == 1
    assert "kind: Service" in rendered_input
