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
    # native sidecar -- only startupProbe does. Missing this means the driver runs before vLLM is
    # actually serving.
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
    assert "readinessProbe" in sidecar


def test_gpu_sidecar_mounts_a_larger_dev_shm(tmp_path):
    # Kubernetes' default 64Mi /dev/shm is too small for vLLM's multiprocess engine at TP>1 or
    # number_of_instances>1 ("Insufficient space in /dev/shm ..."), observed for real against a
    # cluster.
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
    assert sidecar["volumeMounts"] == [{"name": "dshm", "mountPath": "/dev/shm"}]
    volumes = {v["name"]: v for v in job["spec"]["template"]["spec"]["volumes"]}
    assert volumes["dshm"]["emptyDir"]["sizeLimit"] == "4Gi"


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
            "compute": {"cluster": {"type": "kubernetes", "namespace": "eng-test", "pvc_name": "workspace"}},
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


def test_multi_instance_uses_plain_data_parallel_flag(tmp_path):
    # All replicas run as local ranks in this one pod -- vLLM's plain single-node data-parallel
    # mode, no head/worker split, no second pod.
    from nemo_gym.orchestration.executors.kubernetes_script import build_job_manifest

    services = {
        "vllm_model": {
            "type": "vllm",
            "container": "vllm/vllm-openai:latest",
            "model": "org/model",
            "tensor_parallel_size": 2,
            "number_of_instances": 4,
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
    assert sidecar["command"][-2:] == ["--data-parallel-size", "4"]


def test_multi_instance_sidecar_requests_gpus_for_all_replicas(tmp_path):
    # tensor_parallel_size=2 x number_of_instances=4 = 8 GPUs, not just tensor_parallel_size=2 --
    # all replicas share this one pod.
    from nemo_gym.orchestration.executors.kubernetes_script import build_job_manifest

    services = {
        "vllm_model": {
            "type": "vllm",
            "container": "vllm/vllm-openai:latest",
            "model": "org/model",
            "tensor_parallel_size": 2,
            "number_of_instances": 4,
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
    assert sidecar["resources"]["limits"]["nvidia.com/gpu"] == 8


def test_job_has_ttl_seconds_after_finished_by_default(tmp_path):
    from nemo_gym.orchestration.executors.kubernetes_script import build_job_manifest

    config = _submit_config(tmp_path, ["bench_a"])
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

    assert job["spec"]["ttlSecondsAfterFinished"] == 60 * 60 * 24
    assert "activeDeadlineSeconds" not in job["spec"]


def test_job_uses_configured_active_deadline_seconds(tmp_path):
    from nemo_gym.orchestration.executors.kubernetes_script import build_job_manifest

    config = _submit_config(
        tmp_path, ["bench_a"], compute_overrides={"active_deadline_seconds": 3600, "ttl_seconds_after_finished": 60}
    )
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

    assert job["spec"]["activeDeadlineSeconds"] == 3600
    assert job["spec"]["ttlSecondsAfterFinished"] == 60


class _TimingOutKubectl:
    def __call__(self, compute, *args, input=None):
        raise subprocess.TimeoutExpired(cmd=["kubectl", *args], timeout=kubernetes_module._KUBECTL_TIMEOUT_SECONDS)


def test_kubectl_timeout_is_recorded_as_that_benchmarks_error(tmp_path, monkeypatch):
    _install(monkeypatch, _TimingOutKubectl())
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    record = KubernetesExecutor().run(_submit_config(tmp_path, ["bench_a"]))

    assert record is not None
    benchmark = record.benchmarks[0]
    assert benchmark.job_id is None
    assert "timed out" in benchmark.error


def test_benchmark_command_is_forwarded_to_the_driver(tmp_path):
    # A `command` benchmark replaces `gym eval run` with its own harness -- unlike the Slurm
    # executor, this was silently dropped, so the benchmark always ran `gym eval run` instead.
    from nemo_gym.orchestration.executors.kubernetes_script import build_job_manifest

    config = SubmitConfig.model_validate(
        {
            "services": {},
            "compute": {"cluster": {"type": "kubernetes", "namespace": "eng-test", "pvc_name": "workspace"}},
            "driver": {
                "container": "gym:latest",
                "benchmarks": {"bench_a": {"command": "./run_my_harness.sh"}},
            },
            "job": {"output_path": str(tmp_path / "jobs")},
            "otel": {"enabled": False},
        }
    )
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
    assert "./run_my_harness.sh" in driver_script
    assert "GYM_CMD" not in driver_script


def test_benchmark_command_gets_policy_and_bench_dir_env_vars(tmp_path):
    # driver.policy_model injects policy_base_url/model_name/api_key into benchmark.run *after*
    # construction (see SubmitConfig._resolve_and_validate_placements), so this works even for a
    # `command` benchmark -- it just reaches the policy via env vars instead of CLI args.
    from nemo_gym.orchestration.executors.kubernetes_script import build_job_manifest

    config = SubmitConfig.model_validate(
        {
            "services": {"vllm_model": {"type": "vllm", "container": "vllm:latest", "model": "org/model"}},
            "compute": {"cluster": {"type": "kubernetes", "namespace": "eng-test", "pvc_name": "workspace"}},
            "driver": {
                "container": "gym:latest",
                "policy_model": "vllm_model",
                "benchmarks": {"bench_a": {"command": "./run_my_harness.sh"}},
            },
            "job": {"output_path": str(tmp_path / "jobs")},
            "otel": {"enabled": False},
        }
    )
    compute = next(iter(config.compute.values()))
    benchmark = config.driver.benchmarks["bench_a"]
    run_dir = tmp_path / "run"

    job = build_job_manifest(
        config,
        "bench_a",
        benchmark,
        compute,
        run_dir,
        name="gym-test-bench-a",
        gym_job_id="gym-job-test",
        resolved_config="",
        manifest="",
    )

    env = {e["name"]: e["value"] for e in job["spec"]["template"]["spec"]["containers"][0]["env"]}
    assert env["NEMO_GYM_BENCH_DIR"] == str(run_dir)
    assert env["NEMO_GYM_POLICY_BASE_URL"] == "http://localhost:8000/v1"
    assert env["NEMO_GYM_POLICY_MODEL_NAME"] == "org/model"
    assert env["NEMO_GYM_POLICY_API_KEY"] == "dummy"
