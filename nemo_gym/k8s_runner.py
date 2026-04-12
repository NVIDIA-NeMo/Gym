# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import time
from typing import Optional


class K8sJobRunner:
    """Runs a command as a Kubernetes Job and returns its output."""

    def __init__(self, namespace: str = "default"):
        self._namespace = namespace
        self._batch_v1 = None
        self._core_v1 = None

    def _ensure_clients(self) -> None:
        if self._batch_v1 is not None:
            return
        from kubernetes import client, config as k8s_config

        try:
            k8s_config.load_incluster_config()
        except Exception:
            k8s_config.load_kube_config()

        self._batch_v1 = client.BatchV1Api()
        self._core_v1 = client.CoreV1Api()

    def _build_job(
        self,
        job_name: str,
        image: str,
        command: list[str],
        timeout: int,
        env: Optional[dict[str, str]],
        volume_mounts: Optional[list[dict]],
        volumes: Optional[list[dict]],
        resource_limits: Optional[dict[str, str]] = None,
    ):
        from kubernetes import client

        env_vars = [client.V1EnvVar(name=k, value=v) for k, v in (env or {}).items()]

        k8s_volume_mounts = None
        if volume_mounts:
            k8s_volume_mounts = [
                client.V1VolumeMount(
                    name=vm["name"],
                    mount_path=vm["mountPath"],
                    sub_path=vm.get("subPath"),
                    read_only=vm.get("readOnly", False),
                )
                for vm in volume_mounts
            ]

        k8s_volumes = None
        if volumes:
            k8s_volumes = []
            for v in volumes:
                if "persistentVolumeClaim" in v:
                    k8s_volumes.append(
                        client.V1Volume(
                            name=v["name"],
                            persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                                claim_name=v["persistentVolumeClaim"]["claimName"]
                            ),
                        )
                    )
                elif "emptyDir" in v:
                    k8s_volumes.append(
                        client.V1Volume(name=v["name"], empty_dir=client.V1EmptyDirVolumeSource())
                    )
                elif "configMap" in v:
                    k8s_volumes.append(
                        client.V1Volume(
                            name=v["name"],
                            config_map=client.V1ConfigMapVolumeSource(name=v["configMap"]["name"]),
                        )
                    )
                elif "hostPath" in v:
                    k8s_volumes.append(
                        client.V1Volume(
                            name=v["name"],
                            host_path=client.V1HostPathVolumeSource(
                                path=v["hostPath"]["path"],
                                type=v["hostPath"].get("type", ""),
                            ),
                        )
                    )

        resources = None
        if resource_limits:
            resources = client.V1ResourceRequirements(limits=resource_limits)

        container = client.V1Container(
            name="runner",
            image=image,
            command=command,
            env=env_vars if env_vars else None,
            volume_mounts=k8s_volume_mounts,
            resources=resources,
        )

        return client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=client.V1ObjectMeta(
                name=job_name,
                namespace=self._namespace,
                labels={"app": "nemogym", "nemogym-job": "true"},
            ),
            spec=client.V1JobSpec(
                template=client.V1PodTemplateSpec(
                    metadata=client.V1ObjectMeta(labels={"job-name": job_name}),
                    spec=client.V1PodSpec(
                        restart_policy="Never",
                        containers=[container],
                        volumes=k8s_volumes,
                    ),
                ),
                backoff_limit=0,
                ttl_seconds_after_finished=300,
                active_deadline_seconds=timeout,
            ),
        )

    def _sync_create(self, job) -> None:
        self._ensure_clients()
        self._batch_v1.create_namespaced_job(namespace=self._namespace, body=job)

    def _sync_status(self, job_name: str) -> str:
        """Returns 'running' | 'succeeded' | 'failed'."""
        self._ensure_clients()
        job = self._batch_v1.read_namespaced_job(name=job_name, namespace=self._namespace)
        if job.status.succeeded and job.status.succeeded > 0:
            return "succeeded"
        if job.status.failed and job.status.failed > 0:
            return "failed"
        return "running"

    def _sync_get_output(self, job_name: str) -> tuple[int, str]:
        """Returns (exit_code, logs)."""
        self._ensure_clients()
        pods = self._core_v1.list_namespaced_pod(
            namespace=self._namespace,
            label_selector=f"job-name={job_name}",
        )
        if not pods.items:
            return -1, "(no pod found)"

        pod = pods.items[0]
        exit_code = -1
        for cs in pod.status.container_statuses or []:
            if cs.name == "runner" and cs.state and cs.state.terminated:
                exit_code = cs.state.terminated.exit_code
                break

        try:
            logs = self._core_v1.read_namespaced_pod_log(
                name=pod.metadata.name,
                namespace=self._namespace,
                container="runner",
            )
        except Exception as exc:
            logs = f"(could not fetch logs: {exc})"

        return exit_code, logs

    def _sync_delete(self, job_name: str) -> None:
        from kubernetes import client

        self._ensure_clients()
        try:
            self._batch_v1.delete_namespaced_job(
                name=job_name,
                namespace=self._namespace,
                body=client.V1DeleteOptions(propagation_policy="Background"),
            )
        except Exception:
            pass

    async def run_job(
        self,
        job_name: str,
        image: str,
        command: list[str],
        timeout: int,
        env: Optional[dict[str, str]] = None,
        volume_mounts: Optional[list[dict]] = None,
        volumes: Optional[list[dict]] = None,
        resource_limits: Optional[dict[str, str]] = None,
        cleanup: bool = True,
        poll_interval: float = 1.0,
    ) -> tuple[int, str, str]:
        """Create a K8s Job, wait for completion, return (exit_code, stdout, stderr).

        stdout and stderr are merged in the returned stdout field; stderr is empty
        unless there is a runner-level error such as a timeout.
        """
        loop = asyncio.get_running_loop()
        job_obj = self._build_job(job_name, image, command, timeout, env, volume_mounts, volumes, resource_limits)

        try:
            await loop.run_in_executor(None, self._sync_create, job_obj)

            deadline = time.monotonic() + timeout + 15
            status = "running"
            while time.monotonic() < deadline:
                status = await loop.run_in_executor(None, self._sync_status, job_name)
                if status != "running":
                    break
                await asyncio.sleep(poll_interval)

            if status == "running":
                return -1, "", f"timeout: job {job_name!r} did not complete within {timeout}s"

            exit_code, logs = await loop.run_in_executor(None, self._sync_get_output, job_name)
            return exit_code, logs, ""

        finally:
            if cleanup:
                await loop.run_in_executor(None, self._sync_delete, job_name)

    async def delete_job(self, job_name: str) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._sync_delete, job_name)
