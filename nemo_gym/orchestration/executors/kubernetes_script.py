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
# Kubernetes defaults a container's /dev/shm to 64Mi, which is enough for most workloads but not
# vLLM's multiprocess engine: with tensor_parallel_size > 1 it talks to its GPU worker processes
# over a /dev/shm-backed ring buffer, and starting one on the default allocation fails outright
# ("Insufficient space in /dev/shm ... Increase /dev/shm (e.g. --shm-size or --ipc=host)") --
# observed in practice at TP=2. A well-known vLLM-on-Kubernetes gotcha; 4Gi comfortably covers
# ordinary single-node tensor-parallel sizes without eating meaningfully into `memory_per_gpu`.
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
        # A memory *request* (not just a GPU limit) so the pod isn't QoS class BestEffort --
        # BestEffort pods are the kubelet's first choice to kill under node memory pressure, which
        # otherwise silently SIGKILLs a GPU sidecar mid-startup on a busy shared node.
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
    if service.trust_remote_code:
        cmd.append("--trust-remote-code")
    if service.extra_args:
        cmd += shlex.split(service.extra_args)
    return cmd


# vLLM refuses `--api-server-count` in headless mode ("no API servers are started in headless
# mode") and exits before loading anything. Mirrors slurm_script.py's identical concern; kept
# local here (not shared) to avoid coupling the two executor modules over one regex.
_HEADLESS_INCOMPATIBLE_FLAG = re.compile(r"\s--api-server-count(?:[= ]\S+)?")

_DP_RPC_PORT = 13345


def _strip_headless_incompatible_flags(cmd: str) -> str:
    return _HEADLESS_INCOMPATIBLE_FLAG.sub("", cmd)


def _head_service_name(job_name_str: str) -> str:
    # <=63 chars (Service names are RFC 1035 labels); truncate the job name so "-head" always fits.
    return f"{job_name_str[:57].rstrip('-')}-head"


def _head_service_fqdn(service_name: str, namespace: str) -> str:
    return f"{service_name}.{namespace}.svc.cluster.local"


def _vllm_multi_node_command(
    service: VllmServiceConfig, port: int, *, total_nodes: int, head_service_fqdn: str
) -> list[str]:
    """Head/worker vLLM multi-node data-parallel command, branching on Kubernetes' own
    `JOB_COMPLETION_INDEX` env var (auto-injected per pod by an Indexed Job) -- the k8s equivalent
    of Slurm's $SLURM_NODEID. A k8s container `command` is an argv array, not a shell string a
    caller re-parses (unlike Slurm's `srun ... &` line), so no extra escaping layer is needed here
    the way slurm_script.py needs escape_for_single_quoted_block.

    number_of_instances is guaranteed evenly divisible by total_nodes here -- api.py's
    SubmitConfig validation enforces this before build_job_manifest is ever called.
    """
    dp_size_local = service.number_of_instances // total_nodes
    base = shlex.join(_vllm_command(service, port))
    dp_flags = (
        f" --data-parallel-size {service.number_of_instances}"
        f" --data-parallel-size-local {dp_size_local}"
        f' --data-parallel-address "{head_service_fqdn}"'
        f" --data-parallel-rpc-port {_DP_RPC_PORT}"
    )
    head_cmd = base + dp_flags
    worker_cmd = (
        _strip_headless_incompatible_flags(base)
        + dp_flags
        + " --headless"
        + f" --data-parallel-start-rank $(( JOB_COMPLETION_INDEX * {dp_size_local} ))"
    )
    script = f'if [ "$JOB_COMPLETION_INDEX" = "0" ]; then\n    {head_cmd}\nelse\n    {worker_cmd}\nfi'
    return ["bash", "-lc", script]


def _probe(path: str, port: int, *, period: int, failure_threshold: int) -> dict[str, Any]:
    return {
        "httpGet": {"path": path, "port": port},
        "periodSeconds": period,
        "failureThreshold": failure_threshold,
    }


def _multi_node_probe(path: str, port: int, *, period: int, failure_threshold: int) -> dict[str, Any]:
    # A headless worker (index != 0) starts no HTTP server at all, and an Indexed Job shares one
    # pod template across every index -- an httpGet probe here would never succeed on worker pods,
    # permanently blocking kubelet from starting the driver-slot container there. Only check
    # health on index 0; trivially succeed elsewhere.
    check = f'[ "$JOB_COMPLETION_INDEX" != "0" ] || curl -sf http://localhost:{port}{path} > /dev/null'
    return {
        "exec": {"command": ["sh", "-c", check]},
        "periodSeconds": period,
        "failureThreshold": failure_threshold,
    }


# Ongoing readinessProbe cadence, once startup has already succeeded once: quick to notice a real
# problem without needing `health_check.timeout_seconds`' full startup allowance every time.
_READINESS_PERIOD_SECONDS = 5
_READINESS_FAILURE_THRESHOLD = 3


def _sidecar_containers(
    config: SubmitConfig,
    compute: KubernetesComputeConfig,
    *,
    total_nodes: int,
    head_service_fqdn: str | None,
) -> list[dict[str, Any]]:
    containers = []
    for name, service in config.services.items():
        if isinstance(service, RayServiceConfig):
            raise ValueError(f"Service '{name}': ray services are not supported by the kubernetes executor yet.")
        if service.use_ray_serve:
            raise ValueError(f"Service '{name}': use_ray_serve is not supported by the kubernetes executor yet.")
        is_multi_node = total_nodes > 1 and service.number_of_instances > 1
        if service.number_of_instances > 1 and total_nodes == 1:
            raise ValueError(
                f"Service '{name}': number_of_instances={service.number_of_instances} needs multiple nodes on the "
                "kubernetes executor -- set compute.nodes > 1 for a multi-node data-parallel deployment (v1 does "
                "not support multiple engine replicas sharing a single pod)."
            )
        _reject_unsupported_mounts(service.mounts)
        gpu_count = service.tensor_parallel_size * service.pipeline_parallel_size
        if is_multi_node:
            assert head_service_fqdn is not None
            command = _vllm_multi_node_command(
                service, service.port, total_nodes=total_nodes, head_service_fqdn=head_service_fqdn
            )
        else:
            command = _vllm_command(service, service.port)
        container: dict[str, Any] = {
            "name": _dns_label(name),
            "image": service.container,
            "restartPolicy": "Always",  # Native sidecar (k8s >= 1.29): torn down after the pod's
            # regular (driver) container finishes. A `startupProbe` (added below when a
            # health_check is set) is what actually makes the driver WAIT for this one to be
            # healthy first -- without it, kubelet starts the driver the instant this container's
            # process launches, readinessProbe or not; readinessProbe alone never gates that.
            "command": command,
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
            probe_fn = _multi_node_probe if is_multi_node else _probe
            container["startupProbe"] = probe_fn(
                service.health_check.path, port, period=period, failure_threshold=failure_threshold
            )
            container["readinessProbe"] = probe_fn(
                service.health_check.path,
                port,
                period=_READINESS_PERIOD_SECONDS,
                failure_threshold=_READINESS_FAILURE_THRESHOLD,
            )
        containers.append(container)
    return containers


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

    output_path = f"+output_jsonl_fpath={run_dir}/{ARTIFACTS_DIRNAME}/rollouts.jsonl"
    policy_type = config.driver.policy_model_type
    extra_flags = [f"--model-type {shlex.quote(policy_type)}"] if config.driver.policy_model and policy_type else []
    gym_cmd = render_gym_cmd("eval run", "GYM_CMD", [output_path] + extra_flags + flatten_run_args(benchmark.run))
    entrypoint = render_driver_entrypoint(
        repo=gi.repo if gi else None, ref=gi.ref if gi else None, prepare_cmd=prepare_cmd
    )

    script_lines = [
        "set -euo pipefail",
        f"mkdir -p {shlex.quote(run_dir)}/{LOGS_DIRNAME} {shlex.quote(run_dir)}/{ARTIFACTS_DIRNAME}",
        *manifest_writes,
        gym_cmd,
        entrypoint,
    ]
    return ["bash", "-c", "\n".join(script_lines)]


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


def _render_self_delete_trap(job_name: str, namespace: str, head_service_name: str) -> str:
    """Deletes this Job (cascades to all its pods, including this one) and its head Service when
    the driver script on pod 0 exits, success or failure -- the k8s equivalent of a Slurm
    allocation being released when the sbatch script exits. Worker pods (index != 0) never exit on
    their own (see _wrap_driver_command_for_multi_node), so nothing else tears them down.

    Requires compute.service_account to be bound to a Role granting `delete` on batch/jobs and
    core/services in this namespace -- documented in the multi-node example YAML. Prefers curl;
    falls back to python3/python's stdlib urllib so no new binary is strictly required beyond
    what render_driver_entrypoint's own gym_install may already need.
    """
    return f"""\
_gym_k8s_cleanup() {{
    ns="$(cat /var/run/secrets/kubernetes.io/serviceaccount/namespace 2>/dev/null || echo {shlex.quote(namespace)})"
    tok="/var/run/secrets/kubernetes.io/serviceaccount/token"
    ca="/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
    api="https://kubernetes.default.svc"
    for res in "apis/batch/v1/namespaces/${{ns}}/jobs/{job_name}" "api/v1/namespaces/${{ns}}/services/{head_service_name}"; do
        if command -v curl >/dev/null 2>&1; then
            curl -sS -X DELETE --cacert "$ca" -H "Authorization: Bearer $(cat "$tok")" \\
                "${{api}}/${{res}}?propagationPolicy=Background" >/dev/null 2>&1 || true
        elif command -v python3 >/dev/null 2>&1 || command -v python >/dev/null 2>&1; then
            "$(command -v python3 || command -v python)" - "$ns" "$tok" "$ca" "$api" "$res" <<'PYEOF' || true
import ssl, sys, urllib.request
ns, tok, ca, api, res = sys.argv[1:]
req = urllib.request.Request(f"{{api}}/{{res}}?propagationPolicy=Background", method="DELETE",
                              headers={{"Authorization": f"Bearer {{open(tok).read().strip()}}"}})
ctx = ssl.create_default_context(cafile=ca)
urllib.request.urlopen(req, context=ctx, timeout=10)
PYEOF
        else
            echo "WARNING: neither curl nor python available; cannot self-delete ${{res}}." >&2
        fi
    done
}}
trap _gym_k8s_cleanup EXIT"""


def _wrap_driver_command_for_multi_node(driver_cmd: list[str], *, job_name: str, namespace: str) -> list[str]:
    """The Indexed Job's pod template is shared across every index, so the driver-slot container's
    command must branch on $JOB_COMPLETION_INDEX too: index 0 is the real driver (plus a cleanup
    trap that tears the whole Job down on exit); other indices just idle, since nothing else would
    stop their vllm-worker sidecar otherwise.
    """
    assert driver_cmd[:2] == ["bash", "-c"]
    inner_script = driver_cmd[2]
    delete_trap = _render_self_delete_trap(job_name, namespace, _head_service_name(job_name))
    wrapped = (
        f'if [ "$JOB_COMPLETION_INDEX" = "0" ]; then\n{delete_trap}\n{inner_script}\nelse\n    sleep infinity\nfi'
    )
    return ["bash", "-c", wrapped]


def build_head_service_manifest(
    compute: KubernetesComputeConfig, job_name: str, labels: dict[str, str], *, vllm_port: int
) -> dict[str, Any]:
    """A ClusterIP Service selecting only pod index 0 of the given Indexed Job, so worker pods have
    a stable DNS name for --data-parallel-address -- the k8s equivalent of Slurm's $HEAD_NODE_IP,
    resolved here at manifest-build time (as a plain string) since both the Service name and
    compute.namespace are known before `kubectl apply`, unlike Slurm's runtime scontrol lookup.

    Selects on `batch.kubernetes.io/job-completion-index`, a pod label Kubernetes' Job controller
    sets automatically for every pod of an Indexed Job -- consistent with this file's existing
    k8s >= 1.29 native-sidecar assumption elsewhere.
    """
    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": _head_service_name(job_name), "namespace": compute.namespace, "labels": labels},
        "spec": {
            "selector": {**labels, "batch.kubernetes.io/job-completion-index": "0"},
            "ports": [
                {"name": "api", "port": vllm_port, "targetPort": vllm_port},
                {"name": "dp-rpc", "port": _DP_RPC_PORT, "targetPort": _DP_RPC_PORT},
            ],
        },
    }


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
    total_nodes = compute.nodes
    head_service_fqdn = _head_service_fqdn(_head_service_name(name), compute.namespace) if total_nodes > 1 else None

    volumes = [{"name": SHM_VOLUME_NAME, "emptyDir": {"medium": "Memory", "sizeLimit": SHM_SIZE}}]
    if compute.pvc_name:
        volumes.append({"name": OUTPUT_VOLUME_NAME, "persistentVolumeClaim": {"claimName": compute.pvc_name}})

    driver_cmd = _driver_command(config, benchmark_name, benchmark, run_dir_str, manifest_writes)
    if total_nodes > 1:
        driver_cmd = _wrap_driver_command_for_multi_node(driver_cmd, job_name=name, namespace=compute.namespace)

    driver_container: dict[str, Any] = {
        "name": "driver",
        "image": config.driver.container,
        "command": driver_cmd,
        "volumeMounts": [_output_volume_mount(config.job.output_path)],
        "resources": {"requests": {"memory": DRIVER_MEMORY_REQUEST}},
    }
    if config.driver.env:
        driver_container["env"] = _env_list(config.driver.env)

    pod_spec: dict[str, Any] = {
        "restartPolicy": "Never",
        "initContainers": _sidecar_containers(
            config, compute, total_nodes=total_nodes, head_service_fqdn=head_service_fqdn
        ),
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
        "template": {"metadata": {"labels": labels}, "spec": pod_spec},
    }
    if total_nodes > 1:
        job_spec["completionMode"] = "Indexed"
        job_spec["parallelism"] = total_nodes
        job_spec["completions"] = total_nodes

    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": name, "namespace": compute.namespace, "labels": labels},
        "spec": job_spec,
    }
