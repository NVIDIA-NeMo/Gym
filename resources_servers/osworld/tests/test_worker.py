# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess

import pytest
from fastapi import HTTPException

from resources_servers.osworld.worker import ContainerCreateRequest, WorkerRuntime


def _completed(command: list[str], stdout: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")


def test_worker_owns_idempotent_container_lifecycle(monkeypatch, tmp_path) -> None:
    vm_path = tmp_path / "Ubuntu.qcow2"
    vm_path.write_bytes(b"qcow2")
    monkeypatch.setenv("OSWORLD_WORKER_ID", "worker-a")
    monkeypatch.setenv("NEMO_GYM_HEAD_URL", "http://head.example")
    monkeypatch.setenv("OSWORLD_WORKER_DATA_HOST", "worker-routed.example")
    monkeypatch.setenv("OSWORLD_WORKER_VM_PATH", str(vm_path))
    monkeypatch.setenv("OSWORLD_DEPLOYMENT_ID", "deployment-a")
    monkeypatch.setenv("OSWORLD_WORKER_RAM_SIZE", "12G")
    monkeypatch.setenv("OSWORLD_WORKER_CPU_CORES", "6")
    monkeypatch.setenv("OSWORLD_WORKER_DISK_SIZE", "48G")
    monkeypatch.setenv("OSWORLD_WORKER_IMAGE", "registry.example/osworld:test")
    monkeypatch.setenv("OSWORLD_WORKER_CAPACITY", "2")
    runtime = WorkerRuntime()
    commands: list[list[str]] = []

    port_payload = {
        f"{port}/tcp": [{"HostIp": "0.0.0.0", "HostPort": str(20000 + index)}]
        for index, port in enumerate((5000, 9222, 8006, 8080, 5900))
    }

    def fake_run(command, **_kwargs):
        commands.append(list(command))
        if command[:3] == ["docker", "inspect", "--format"]:
            return _completed(command, json.dumps(port_payload))
        return _completed(command)

    monkeypatch.setattr(runtime, "_run", fake_run)
    request = ContainerCreateRequest(
        session_id="session-a",
        deployment_id="deployment-a",
    )

    with pytest.raises(HTTPException, match="worker belongs to deployment"):
        runtime.create_container(
            ContainerCreateRequest(
                session_id="foreign-session",
                deployment_id="foreign-deployment",
            )
        )

    first = runtime.create_container(request)
    second = runtime.create_container(request)

    assert first == second
    assert first["data_host"] == "worker-routed.example"
    docker_runs = [command for command in commands if command[:3] == ["docker", "run", "-d"]]
    assert len(docker_runs) == 1
    assert "--device" in docker_runs[0]
    assert "/dev/kvm" in docker_runs[0]
    assert f"{vm_path}:/System.qcow2:ro" in docker_runs[0]
    assert docker_runs[0][-1] == "registry.example/osworld:test"

    assert runtime.delete_container("session-a") is True
    assert runtime.delete_container("session-a") is True


def test_registration_metadata_contains_runtime_contract_not_credentials(monkeypatch, tmp_path) -> None:
    vm_path = tmp_path / "Ubuntu.qcow2"
    vm_path.write_bytes(b"qcow2")
    monkeypatch.setenv("OSWORLD_WORKER_ID", "worker-a")
    monkeypatch.setenv("NEMO_GYM_HEAD_URL", "http://head.example")
    monkeypatch.setenv("NEMO_GYM_REGISTRATION_TOKEN", "registration-secret")
    monkeypatch.setenv("OSWORLD_WORKER_DATA_HOST", "worker-routed.example")
    monkeypatch.setenv("OSWORLD_WORKER_VM_PATH", str(vm_path))
    monkeypatch.setenv("OSWORLD_DEPLOYMENT_ID", "deployment-a")
    monkeypatch.setenv("OSWORLD_WORKER_RAM_SIZE", "12G")
    monkeypatch.setenv("OSWORLD_WORKER_CPU_CORES", "6")
    monkeypatch.setenv("OSWORLD_WORKER_DISK_SIZE", "48G")
    runtime = WorkerRuntime()
    runtime.control_url = "http://worker-routed.example:28080"

    registration = runtime._registration_body()

    assert registration["service_type"] == "osworld_worker"
    assert registration["url"] == runtime.control_url
    assert registration["metadata"]["deployment_id"] == "deployment-a"
    assert registration["metadata"]["vm_filename"] == "Ubuntu.qcow2"
    assert registration["metadata"]["ram_size"] == "12G"
    assert registration["metadata"]["cpu_cores"] == "6"
    assert registration["metadata"]["disk_size"] == "48G"
    assert "registration-secret" not in repr(registration)
