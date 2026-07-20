# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pytest

from nemo_gym.sandbox import SandboxEndpoint, SandboxStatus
from responses_api_agents.osworld_agent import sandbox_provider as osworld_sandbox


class FakeSandbox:
    instances: list["FakeSandbox"] = []

    def __init__(self, provider: dict[str, Any]) -> None:
        self.provider = provider
        self.spec = None
        self.stopped = 0
        FakeSandbox.instances.append(self)

    def start(self, spec: Any) -> "FakeSandbox":
        self.spec = spec
        return self

    def endpoint(self, port: int) -> SandboxEndpoint:
        offsets = {5000: 50, 9222: 51, 8006: 52, 8080: 53}
        return SandboxEndpoint(endpoint=f"http://127.0.0.1:{30000 + offsets[port]}")

    def status(self) -> SandboxStatus:
        return SandboxStatus.RUNNING

    def stop(self) -> None:
        self.stopped += 1


def _patch_kvm(monkeypatch: pytest.MonkeyPatch) -> None:
    real_exists = osworld_sandbox.os.path.exists
    real_access = osworld_sandbox.os.access
    monkeypatch.setattr(
        osworld_sandbox.os.path,
        "exists",
        lambda path: True if path == "/dev/kvm" else real_exists(path),
    )
    monkeypatch.setattr(
        osworld_sandbox.os,
        "access",
        lambda path, mode: True if path == "/dev/kvm" else real_access(path, mode),
    )


def test_build_spec_mounts_read_only_snapshot_and_requests_runtime(tmp_path, monkeypatch) -> None:
    _patch_kvm(monkeypatch)
    vm_path = tmp_path / "Ubuntu.qcow2"
    vm_path.write_bytes(b"qcow2")
    provider = osworld_sandbox.GymSandboxDesktopProvider(
        {"docker": {}},
        {
            "image": "docker://osworld@sha256:abc",
            "ports": None,
            "resources": {"cpu": 4, "memory_mib": 16384},
            "provider_options": {"run_args": ["--security-opt", "label=disable"]},
        },
    )

    spec = provider._build_spec(str(vm_path), headless=True, os_type="Ubuntu")

    assert spec.image == "docker://osworld@sha256:abc"
    assert spec.ports == osworld_sandbox.OSWORLD_SERVICE_PORTS
    assert spec.entrypoint == list(osworld_sandbox.OSWORLD_IMAGE_ENTRYPOINT)
    assert spec.env["HEADLESS"] == "Y"
    assert spec.env["KVM"] == "Y"
    assert spec.resources.cpu == 4
    assert f"{vm_path.resolve()}:/System.qcow2:ro" in spec.provider_options["volumes"]
    assert osworld_sandbox._has_option(spec.provider_options["run_args"], "--cap-add", "NET_ADMIN")
    assert osworld_sandbox._has_option(spec.provider_options["run_args"], "--device", "/dev/kvm")


def test_build_spec_docker_tcg_mode_does_not_map_kvm(tmp_path) -> None:
    vm_path = tmp_path / "Ubuntu.qcow2"
    vm_path.write_bytes(b"qcow2")
    provider = osworld_sandbox.GymSandboxDesktopProvider(
        {"docker": {}},
        {"image": "osworld:fixed"},
        require_kvm=False,
    )

    spec = provider._build_spec(str(vm_path), headless=True, os_type="Ubuntu")

    assert spec.env["KVM"] == "N"
    assert not osworld_sandbox._has_option(spec.provider_options["run_args"], "--device", "/dev/kvm")
    assert osworld_sandbox._has_option(spec.provider_options["run_args"], "--cap-add", "NET_ADMIN")


def test_provider_rejects_non_docker_config() -> None:
    with pytest.raises(ValueError, match="requires Gym's Docker provider"):
        osworld_sandbox.GymSandboxDesktopProvider(
            {"apptainer": {}},
            {"image": "osworld:fixed"},
        )


def test_build_spec_rejects_non_string_docker_options(tmp_path, monkeypatch) -> None:
    _patch_kvm(monkeypatch)
    vm_path = tmp_path / "Ubuntu.qcow2"
    vm_path.write_bytes(b"qcow2")
    provider = osworld_sandbox.GymSandboxDesktopProvider(
        {"docker": {}},
        {"image": "osworld:fixed", "provider_options": {"volumes": [123]}},
    )

    with pytest.raises(TypeError, match="volumes must be a string or list of strings"):
        provider._build_spec(str(vm_path), headless=True, os_type="Ubuntu")


def test_endpoint_contract_rejects_proxy_headers_and_paths() -> None:
    assert osworld_sandbox._parse_plain_http_endpoint(
        SandboxEndpoint("http://127.0.0.1:5000"),
        5000,
    ) == ("127.0.0.1", 5000)
    with pytest.raises(ValueError, match="requires headers"):
        osworld_sandbox._parse_plain_http_endpoint(
            SandboxEndpoint("http://127.0.0.1:5000", {"authorization": "secret"}),
            5000,
        )
    with pytest.raises(ValueError, match="plain origin"):
        osworld_sandbox._parse_plain_http_endpoint(
            SandboxEndpoint("http://127.0.0.1:5000/proxy/path"),
            5000,
        )


def test_lifecycle_recreates_from_snapshot_and_close_is_idempotent(tmp_path, monkeypatch) -> None:
    FakeSandbox.instances.clear()
    _patch_kvm(monkeypatch)
    monkeypatch.setattr(osworld_sandbox, "Sandbox", FakeSandbox)
    vm_path = tmp_path / "Ubuntu.qcow2"
    vm_path.write_bytes(b"qcow2")
    provider = osworld_sandbox.GymSandboxDesktopProvider(
        {"docker": {}},
        {"image": "osworld:fixed"},
    )
    monkeypatch.setattr(provider, "_wait_for_vm_ready", lambda *_args: None)

    provider.start_emulator(str(vm_path), headless=True, os_type="Ubuntu")
    assert provider.get_ip_address(str(vm_path)) == "127.0.0.1:30050:30051:30052:30053"
    first = FakeSandbox.instances[0]
    provider.revert_to_snapshot(str(vm_path), "init_state")
    provider.start_emulator(str(vm_path), headless=True, os_type="Ubuntu")
    second = FakeSandbox.instances[1]
    provider.stop_emulator(str(vm_path))
    provider.stop_emulator(str(vm_path))

    assert first.stopped == 1
    assert second.stopped == 1
    assert first.spec.provider_options["volumes"] == second.spec.provider_options["volumes"]


def test_start_failure_cleans_up_sandbox(tmp_path, monkeypatch) -> None:
    class BadEndpointSandbox(FakeSandbox):
        def endpoint(self, port: int) -> SandboxEndpoint:
            return SandboxEndpoint(
                endpoint=f"https://proxy.example/{port}",
                headers={"authorization": "secret"},
            )

    _patch_kvm(monkeypatch)
    monkeypatch.setattr(osworld_sandbox, "Sandbox", BadEndpointSandbox)
    vm_path = tmp_path / "Ubuntu.qcow2"
    vm_path.write_bytes(b"qcow2")
    provider = osworld_sandbox.GymSandboxDesktopProvider(
        {"docker": {}},
        {"image": "osworld:fixed"},
    )

    with pytest.raises(ValueError, match="requires headers"):
        provider.start_emulator(str(vm_path), headless=True, os_type="Ubuntu")
    assert BadEndpointSandbox.instances[-1].stopped == 1
