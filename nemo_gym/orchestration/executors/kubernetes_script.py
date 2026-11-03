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

"""Renders the Kubernetes Job manifest for one benchmark.

v1 scope: single node, one Job per benchmark, one pod. Model/ray services become native sidecar
containers (`initContainers` with `restartPolicy: Always`, k8s >= 1.29) so kubelet enforces
start-before-driver ordering and readiness via probes, instead of the hand-rolled bash health-check
loop `slurm_script.py` needs on Slurm. The driver is the pod's single regular container.
"""

import re
import shlex
from pathlib import Path
from typing import Any

from nemo_gym.orchestration.api import (
    BenchmarkRunConfig,
    KubernetesComputeConfig,
    RayServiceConfig,
    SubmitConfig,
    VllmServiceConfig,
)
from nemo_gym.orchestration.executors.script_templates import (
    render_driver_entrypoint,
    render_gym_cmd,
    render_write_file_from_base64,
)
from nemo_gym.orchestration.executors.utils import flatten_run_args


GPU_RESOURCE_KEY = "nvidia.com/gpu"
LOGS_DIRNAME = "logs"
ARTIFACTS_DIRNAME = "artifacts"
OUTPUT_VOLUME_NAME = "gym-output"
# Flat memory request for the driver container, which does no GPU work of its own: enough that it
# isn't QoS class BestEffort either (see `_gpu_resources`), without needing its own config knob.
DRIVER_MEMORY_REQUEST = "2Gi"
# K8s defaults /dev/shm to 64Mi; vLLM's multiprocess engine needs more at TP>1 or instances>1
# ("Insufficient space in /dev/shm ... Increase /dev/shm"). 4Gi comfortably covers single-node use.
SHM_VOLUME_NAME = "dshm"
SHM_SIZE = "4Gi"

_QUANTITY_RE = re.compile(r"^(\d+)([A-Za-z]*)$")


def _dns_label(value: str) -> str:
    """Kubernetes object/container names are RFC 1123 labels: lowercase alnum and '-' only, <=63
    chars, must start/end alphanumeric. Gym service and benchmark names allow '.', '_' (e.g.
    `vllm_model`, `tau2.airline`), which are valid there but rejected by the k8s API server, so
    every name that becomes a k8s name goes through this first."""
    return re.sub(r"[^a-z0-9-]", "-", value.lower()).strip("-")[:63]


def job_name(gym_job_id: str, benchmark_name: str) -> str:
    return f"gym-{_dns_label(gym_job_id)}-{_dns_label(benchmark_name)}"[:63].rstrip("-")


def _env_list(env: dict[str, str]) -> list[dict[str, str]]:
    # `env` is already resolved (lit:/host:/runtime:) by `resolve_env_dict` at validation time;
    # `runtime:VAR` values are left as `runtime:VAR` there for an executor to turn into a live
    # reference. Kubernetes has no shell to expand `runtime:` against, so it isn't supported here.
    resolved = []
    for key, value in env.items():
        if value.startswith("runtime:"):
            raise ValueError(
                f"env[{key!r}] uses {value!r}, which the kubernetes executor does not support "
                "(no shell to resolve it against). Use 'lit:' or 'host:' instead."
            )
        resolved.append({"name": key, "value": value})
    return resolved


def _reject_unsupported_mounts(mounts: list[str]) -> None:
    # Pyxis-style "src", "src:dst", "src:dst:flags" mounts don't have a Kubernetes analog without
    # a matching PVC/hostPath per entry, which v1 doesn't model. The one mount every container
    # that needs it gets -- the job's own output directory -- is added by the caller directly.
    if mounts:
        raise ValueError(
            f"service/driver `mounts` ({mounts}) are not supported by the kubernetes executor yet; "
            "only the PVC at `compute.pvc_name`, mounted at `job.output_path`, is available."
        )


def _output_volume_mount(output_path: str) -> dict[str, Any]:
    # Mounted at the SAME fixed path (`job.output_path`) for every job sharing this PVC, not at
    # the job's own `run_dir`: a volume mount re-roots the PVC at whatever `mountPath` is given,
    # so mounting it per-job-run-dir would make every job's `run_dir/logs` etc. land at the same
    # PVC-relative `logs/` and collide. `run_dir` is a subdirectory *under* this fixed mount.
    return {"name": OUTPUT_VOLUME_NAME, "mountPath": output_path}


def _scale_quantity(quantity: str, factor: int) -> str:
    """Scale a Kubernetes memory quantity ("32Gi") by an integer factor ("64Gi" for factor=2)."""
    match = _QUANTITY_RE.match(quantity)
    if not match:
        raise ValueError(f"{quantity!r} is not a supported memory quantity (expected e.g. '32Gi', '512Mi').")
    number, unit = match.groups()
    return f"{int(number) * factor}{unit}"


def _gpu_resources(gpu_count: int, memory_per_gpu: str) -> dict[str, Any]:
    if gpu_count <= 0:
        return {}
    memory = _scale_quantity(memory_per_gpu, gpu_count)
    return {
        "limits": {GPU_RESOURCE_KEY: gpu_count},
        # A memory *request* (not just a GPU limit) so the pod isn't QoS class BestEffort, which
        # the kubelet kills first under node memory pressure.
        "requests": {"memory": memory},
    }


def _vllm_command(service: VllmServiceConfig, port: int) -> list[str]:
    cmd = ["vllm", "serve", service.model, "--port", str(port)]
    if service.served_model_name:
        cmd += ["--served-model-name", service.served_model_name]
    if service.tensor_parallel_size > 1:
        cmd += ["--tensor-parallel-size", str(service.tensor_parallel_size)]
    if service.pipeline_parallel_size > 1:
        cmd += ["--pipeline-parallel-size", str(service.pipeline_parallel_size)]
    if service.number_of_instances > 1:
        # vLLM's plain single-node DP mode: replicas run as local ranks in this one pod/port.
        # Only viable when replica*TP*PP fits this node's GPUs (api.py's gpus_per_node check).
        cmd += ["--data-parallel-size", str(service.number_of_instances)]
    if service.trust_remote_code:
        cmd.append("--trust-remote-code")
    if service.extra_args:
        cmd += shlex.split(service.extra_args)
    return cmd


def _probe(path: str, port: int, *, period: int, failure_threshold: int) -> dict[str, Any]:
    return {
        "httpGet": {"path": path, "port": port},
        "periodSeconds": period,
        "failureThreshold": failure_threshold,
    }


# Ongoing readinessProbe cadence, once startup has already succeeded once: quick to notice a real
# problem without needing `health_check.timeout_seconds`' full startup allowance every time.
_READINESS_PERIOD_SECONDS = 5
_READINESS_FAILURE_THRESHOLD = 3


def _sidecar_containers(config: SubmitConfig, compute: KubernetesComputeConfig) -> list[dict[str, Any]]:
    containers = []
    for name, service in config.services.items():
        if isinstance(service, RayServiceConfig):
            raise ValueError(f"Service '{name}': ray services are not supported by the kubernetes executor yet.")
        if service.use_ray_serve:
            raise ValueError(f"Service '{name}': use_ray_serve is not supported by the kubernetes executor yet.")
        _reject_unsupported_mounts(service.mounts)
        # number_of_instances replicas all run in this one pod (vLLM's single-node data-parallel
        # mode -- see _vllm_command), so the pod needs GPUs for all of them.
        gpu_count = service.tensor_parallel_size * service.pipeline_parallel_size * service.number_of_instances
        container: dict[str, Any] = {
            "name": _dns_label(name),
            "image": service.container,
            "restartPolicy": "Always",  # Native sidecar (k8s >= 1.29): torn down after the
            # driver finishes. Only startupProbe (below) delays the driver starting; readinessProbe
            # alone doesn't gate it.
            "command": _vllm_command(service, service.port),
            "ports": [{"containerPort": service.port}],
            "resources": _gpu_resources(gpu_count, compute.memory_per_gpu),
        }
        if gpu_count > 0:
            container["volumeMounts"] = [{"name": SHM_VOLUME_NAME, "mountPath": "/dev/shm"}]
        if service.env:
            container["env"] = _env_list(service.env)
        if service.health_check:
            port = service.health_check.port or service.port
            period = 5
            failure_threshold = max(1, service.health_check.timeout_seconds // period)
            container["startupProbe"] = _probe(
                service.health_check.path, port, period=period, failure_threshold=failure_threshold
            )
            container["readinessProbe"] = _probe(
                service.health_check.path,
                port,
                period=_READINESS_PERIOD_SECONDS,
                failure_threshold=_READINESS_FAILURE_THRESHOLD,
            )
        containers.append(container)
    return containers


def _command_env(benchmark: BenchmarkRunConfig, run_dir: str) -> dict[str, str]:
    """Env a `command` benchmark gets in place of `gym eval run` arguments -- mirrors
    slurm_script.py's _command_env, kept local to avoid coupling the two executor modules."""
    env = {"NEMO_GYM_BENCH_DIR": run_dir}
    for key, name in (
        ("policy_base_url", "NEMO_GYM_POLICY_BASE_URL"),
        ("policy_model_name", "NEMO_GYM_POLICY_MODEL_NAME"),
        ("policy_api_key", "NEMO_GYM_POLICY_API_KEY"),
    ):
        value = benchmark.run.get(key)
        if value is not None:
            env[name] = str(value)
    return env


def _driver_command(
    config: SubmitConfig,
    benchmark_name: str,
    benchmark: BenchmarkRunConfig,
    run_dir: str,
    manifest_writes: list[str],
) -> list[str]:
    gi = config.driver.gym_install
    prepare_cmd = None
    if benchmark.prepare:
        prepare_cmd = "gym eval prepare " + " ".join(flatten_run_args(benchmark.prepare))

    if benchmark.command is None:
        output_path = f"+output_jsonl_fpath={run_dir}/{ARTIFACTS_DIRNAME}/rollouts.jsonl"
        policy_type = config.driver.policy_model_type
        extra_flags = (
            [f"--model-type {shlex.quote(policy_type)}"] if config.driver.policy_model and policy_type else []
        )
        gym_cmd = render_gym_cmd("eval run", "GYM_CMD", [output_path] + extra_flags + flatten_run_args(benchmark.run))
    else:
        # A command replaces `gym eval run`, so it reaches the policy/output path via env vars
        # instead (see _command_env) -- build_job_manifest merges those into the driver's env.
        gym_cmd = ""
    entrypoint = render_driver_entrypoint(
        repo=gi.repo if gi else None, ref=gi.ref if gi else None, prepare_cmd=prepare_cmd, command=benchmark.command
    )

    script_lines = [
        "set -euo pipefail",
        f"mkdir -p {shlex.quote(run_dir)}/{LOGS_DIRNAME} {shlex.quote(run_dir)}/{ARTIFACTS_DIRNAME}",
        *manifest_writes,
        gym_cmd,
        entrypoint,
    ]
    return ["bash", "-c", "\n".join(line for line in script_lines if line)]


def _manifest_write_commands(resolved_config: str, manifest: str, run_dir: str) -> list[str]:
    """Preamble lines that write the resolved config / job manifest onto the mounted PVC.

    Stands in for `Connection.write_text` (Slurm) / a ConfigMap (rejected -- see kubernetes.py):
    the k8s Job name is chosen before submission, so both files' content is fully known up front
    and can be embedded directly in the driver container's own startup command.
    """
    return [
        render_write_file_from_base64(resolved_config, f"{run_dir}/resolved-config.yaml"),
        render_write_file_from_base64(manifest, f"{run_dir}/gym-job.json"),
    ]


def build_job_manifest(
    config: SubmitConfig,
    benchmark_name: str,
    benchmark: BenchmarkRunConfig,
    compute: KubernetesComputeConfig,
    run_dir: Path,
    *,
    name: str,
    gym_job_id: str,
    resolved_config: str,
    manifest: str,
) -> dict[str, Any]:
    run_dir_str = str(run_dir)
    manifest_writes = _manifest_write_commands(resolved_config, manifest, run_dir_str)
    _reject_unsupported_mounts(config.driver.mounts)

    volumes = [
        {"name": SHM_VOLUME_NAME, "emptyDir": {"medium": "Memory", "sizeLimit": SHM_SIZE}},
        {"name": OUTPUT_VOLUME_NAME, "persistentVolumeClaim": {"claimName": compute.pvc_name}},
    ]

    driver_env = dict(config.driver.env)
    if benchmark.command is not None:
        driver_env |= _command_env(benchmark, run_dir_str)

    driver_container: dict[str, Any] = {
        "name": "driver",
        "image": config.driver.container,
        "command": _driver_command(config, benchmark_name, benchmark, run_dir_str, manifest_writes),
        "volumeMounts": [_output_volume_mount(config.job.output_path)],
        "resources": {"requests": {"memory": DRIVER_MEMORY_REQUEST}},
    }
    if driver_env:
        driver_container["env"] = _env_list(driver_env)

    pod_spec: dict[str, Any] = {
        "restartPolicy": "Never",
        "initContainers": _sidecar_containers(config, compute),
        "containers": [driver_container],
        "volumes": volumes,
    }
    if compute.node_selector:
        pod_spec["nodeSelector"] = compute.node_selector
    if compute.service_account:
        pod_spec["serviceAccountName"] = compute.service_account

    # Set on both the Job and its pod template: the Job's own labels are what `kubectl get jobs
    # -l ...` filters on, but the *pod* -- what `kubectl get pods -l ...`/`logs -l ...` see -- only
    # gets labels declared here, not the Job's; Kubernetes does not copy them across.
    labels = {
        "app.kubernetes.io/managed-by": "nemo-gym",
        "gym-job-id": _dns_label(gym_job_id),
        "gym-benchmark": _dns_label(benchmark_name),
    }

    job_spec: dict[str, Any] = {
        "backoffLimit": 0,
        "ttlSecondsAfterFinished": compute.ttl_seconds_after_finished,
        "template": {"metadata": {"labels": labels}, "spec": pod_spec},
    }
    if compute.active_deadline_seconds is not None:
        job_spec["activeDeadlineSeconds"] = compute.active_deadline_seconds

    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": name, "namespace": compute.namespace, "labels": labels},
        "spec": job_spec,
    }
