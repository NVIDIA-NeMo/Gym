# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import http.server
import threading
from typing import Any

import pytest
import requests

from nemo_gym.sandbox import SandboxEndpoint, SandboxStatus
from responses_api_agents.osworld_agent import sandbox_provider as osworld_sandbox
from responses_api_agents.osworld_agent.local_forwarder import start_forwarder


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


def test_build_spec_mounts_read_only_snapshot_and_requests_runtime(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OSWORLD_RUN_ID", "smoke-run")
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
    assert osworld_sandbox._has_option(
        spec.provider_options["run_args"],
        "--label",
        osworld_sandbox.OSWORLD_WORKLOAD_LABEL,
    )
    assert osworld_sandbox._has_option(
        spec.provider_options["run_args"],
        "--label",
        "nemo-gym.run-id=smoke-run",
    )


def test_build_spec_leaves_kvm_validation_to_docker_host(tmp_path, monkeypatch) -> None:
    vm_path = tmp_path / "Ubuntu.qcow2"
    vm_path.write_bytes(b"qcow2")
    real_exists = osworld_sandbox.os.path.exists
    real_access = osworld_sandbox.os.access
    monkeypatch.setattr(
        osworld_sandbox.os.path,
        "exists",
        lambda path: False if path == "/dev/kvm" else real_exists(path),
    )
    monkeypatch.setattr(
        osworld_sandbox.os,
        "access",
        lambda path, mode: False if path == "/dev/kvm" else real_access(path, mode),
    )
    provider = osworld_sandbox.GymSandboxDesktopProvider(
        {"docker": {}},
        {"image": "osworld:fixed"},
    )

    spec = provider._build_spec(str(vm_path), headless=True, os_type="Ubuntu")

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


def test_build_spec_uses_sdk_compatibility_image_for_opensandbox_pool(monkeypatch) -> None:
    monkeypatch.setenv("OSWORLD_RUN_ID", "opensandbox-run")
    provider = osworld_sandbox.GymSandboxDesktopProvider(
        {
            "opensandbox": {
                "connection": {
                    "domain": "http://sandbox.example",
                    "use_server_proxy": False,
                }
            }
        },
        {
            "ttl_s": 1800,
            "image": "busybox:1.36",
            "entrypoint": ["/run/entry.sh"],
            "env": {"KVM": "Y"},
            "resources": {"cpu": 4, "memory_mib": 16384},
            "provider_options": {
                "skip_health_check": True,
                "extensions": {"poolRef": "osworld-kvm"},
            },
        },
    )

    spec = provider._build_spec(
        "/opensandbox/Ubuntu.qcow2",
        headless=True,
        os_type="Ubuntu",
    )

    assert spec.image == "busybox:1.36"
    assert spec.ttl_s == 1800
    assert spec.ports == osworld_sandbox.OSWORLD_SERVICE_PORTS
    assert spec.provider_options == {
        "skip_health_check": True,
        "extensions": {"poolRef": "osworld-kvm"},
    }
    assert spec.metadata["osworld-provider"] == "gym-opensandbox-sandbox"
    assert spec.metadata["run-id"] == "opensandbox-run"
    assert spec.entrypoint is None
    assert spec.env == {}
    assert spec.resources.cpu is None


def test_build_spec_rejects_invalid_opensandbox_pool_spec() -> None:
    provider = osworld_sandbox.GymSandboxDesktopProvider(
        {"opensandbox": {}},
        {"provider_options": {"extensions": {}}},
    )
    with pytest.raises(ValueError, match="requires sandbox_spec.image"):
        provider._build_spec(
            "/opensandbox/Ubuntu.qcow2",
            headless=True,
            os_type="Ubuntu",
        )

    provider = osworld_sandbox.GymSandboxDesktopProvider(
        {"opensandbox": {}},
        {
            "image": "busybox:1.36",
            "provider_options": {"extensions": {}},
        },
    )
    with pytest.raises(ValueError, match="poolRef"):
        provider._build_spec(
            "/opensandbox/Ubuntu.qcow2",
            headless=True,
            os_type="Ubuntu",
        )


def test_provider_rejects_non_docker_config() -> None:
    with pytest.raises(ValueError, match="Docker or OpenSandbox provider"):
        osworld_sandbox.GymSandboxDesktopProvider(
            {"apptainer": {}},
            {"image": "osworld:fixed"},
        )


def test_build_spec_rejects_non_string_docker_options(tmp_path) -> None:
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


def test_local_forwarder_maps_proxy_path_headers_and_cdp_url(monkeypatch) -> None:
    seen: dict[str, str] = {}

    class Upstream(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args: object) -> None:
            del args

        def do_GET(self) -> None:
            seen["path"] = self.path
            seen["route"] = self.headers.get("X-Route", "")
            content = b'{"webSocketDebuggerUrl":"ws://100.100.1.2:9222/devtools/browser/test"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

    upstream = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    forwarder, port = start_forwarder(
        f"http://127.0.0.1:{upstream.server_address[1]}/proxy/9222",
        {"X-Route": "gateway"},
    )
    try:
        with requests.Session() as session:
            session.trust_env = False
            response = session.get(
                f"http://127.0.0.1:{port}/json/version",
                timeout=10,
            )
        assert response.status_code == 200
        assert seen == {
            "path": "/proxy/9222/json/version",
            "route": "gateway",
        }
        assert response.json()["webSocketDebuggerUrl"] == (f"ws://127.0.0.1:{port}/devtools/browser/test")
    finally:
        forwarder.shutdown()
        forwarder.server_close()
        upstream.shutdown()
        upstream.server_close()


def test_lifecycle_recreates_from_snapshot_and_close_is_idempotent(tmp_path, monkeypatch) -> None:
    FakeSandbox.instances.clear()
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

    monkeypatch.setattr(osworld_sandbox, "Sandbox", BadEndpointSandbox)
    vm_path = tmp_path / "Ubuntu.qcow2"
    vm_path.write_bytes(b"qcow2")
    provider = osworld_sandbox.GymSandboxDesktopProvider(
        {"docker": {}},
        {"image": "osworld:fixed"},
    )
    monkeypatch.setattr(
        osworld_sandbox,
        "start_forwarder",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("forwarder failed")),
    )

    with pytest.raises(RuntimeError, match="forwarder failed"):
        provider.start_emulator(str(vm_path), headless=True, os_type="Ubuntu")
    assert BadEndpointSandbox.instances[-1].stopped == 1
