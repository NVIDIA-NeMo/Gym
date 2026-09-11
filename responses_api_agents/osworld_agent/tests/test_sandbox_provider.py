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


def test_opensandbox_pool_skips_local_docker_vm_resolution() -> None:
    assert (
        osworld_sandbox._resolve_pool_vm_path(
            {"opensandbox": {"connection": {}}},
            None,
        )
        == osworld_sandbox.OPENSANDBOX_POOL_VM_PATH
    )


def test_explicit_vm_path_is_preserved_for_opensandbox() -> None:
    assert (
        osworld_sandbox._resolve_pool_vm_path(
            {"opensandbox": {"connection": {}}},
            "caller-supplied-path",
        )
        == "caller-supplied-path"
    )


def test_docker_sandbox_does_not_receive_pool_sentinel() -> None:
    assert osworld_sandbox._resolve_pool_vm_path({"docker": {}}, None) is None


def test_build_spec_mounts_read_only_snapshot_and_requests_runtime(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OSWORLD_RUN_ID", "smoke-run")
    vm_path = tmp_path / "Ubuntu.qcow2"
    vm_path.write_bytes(b"qcow2")
    provider = osworld_sandbox.GymSandboxDesktopProvider(
        {"docker": {}},
        {
            "image": "docker://osworld@sha256:abc",
            "metadata": {osworld_sandbox.EXECUTION_ID_SANDBOX_METADATA_KEY: "execution-test-001"},
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
    assert osworld_sandbox._has_option(
        spec.provider_options["run_args"],
        "--label",
        f"{osworld_sandbox.EXECUTION_ID_SANDBOX_METADATA_KEY}=execution-test-001",
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


def test_build_spec_skips_local_vm_and_kvm_checks_on_remote_docker_host(monkeypatch) -> None:
    monkeypatch.setenv("DOCKER_HOST", "ssh://remote-docker")
    provider = osworld_sandbox.GymSandboxDesktopProvider(
        {"docker": {}},
        {"image": "osworld:fixed"},
    )

    spec = provider._build_spec(
        "/srv/osworld-assets/Ubuntu.qcow2",
        headless=True,
        os_type="Ubuntu",
    )

    assert "/srv/osworld-assets/Ubuntu.qcow2:/System.qcow2:ro" in spec.provider_options["volumes"]
    assert osworld_sandbox._has_option(spec.provider_options["run_args"], "--device", "/dev/kvm")


def test_build_spec_honors_explicit_remote_daemon_behind_unix_proxy(monkeypatch) -> None:
    monkeypatch.setenv("DOCKER_HOST", "unix:///run/osworld-docker.sock")
    monkeypatch.setenv("OSWORLD_DOCKER_REMOTE", "true")
    provider = osworld_sandbox.GymSandboxDesktopProvider(
        {"docker": {}},
        {"image": "osworld:fixed"},
    )

    spec = provider._build_spec(
        "/srv/osworld-assets/Ubuntu.qcow2",
        headless=True,
        os_type="Ubuntu",
    )

    assert "/srv/osworld-assets/Ubuntu.qcow2:/System.qcow2:ro" in spec.provider_options["volumes"]
    assert osworld_sandbox._has_option(spec.provider_options["run_args"], "--device", "/dev/kvm")


def test_build_spec_rejects_invalid_remote_daemon_override(monkeypatch) -> None:
    monkeypatch.setenv("OSWORLD_DOCKER_REMOTE", "remote")
    provider = osworld_sandbox.GymSandboxDesktopProvider(
        {"docker": {}},
        {"image": "osworld:fixed"},
    )

    with pytest.raises(ValueError, match="OSWORLD_DOCKER_REMOTE"):
        provider._build_spec(
            "/srv/osworld-assets/Ubuntu.qcow2",
            headless=True,
            os_type="Ubuntu",
        )


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
            "metadata": {osworld_sandbox.EXECUTION_ID_SANDBOX_METADATA_KEY: "execution-test-001"},
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
    assert spec.metadata[osworld_sandbox.EXECUTION_ID_SANDBOX_METADATA_KEY] == ("execution-test-001")
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
    with pytest.raises(ValueError, match="Docker, OpenSandbox or E2B provider"):
        osworld_sandbox.GymSandboxDesktopProvider(
            {"apptainer": {}},
            {"image": "osworld:fixed"},
        )


def test_build_spec_uses_template_and_slim_vnc_port_for_agentenv(monkeypatch) -> None:
    """AgentENV boots a template snapshot: no qcow2, no entrypoint, noVNC on 6901."""
    monkeypatch.setenv("OSWORLD_RUN_ID", "agentenv-run")
    provider = osworld_sandbox.GymSandboxDesktopProvider(
        {"e2b": {"connection": {"api_url": "http://agentenv.example"}}},
        {
            "image": "osworld-slim",
            "ttl_s": 7200,
            # Docker-profile leftovers that do not apply to a prebuilt template.
            "entrypoint": ["/usr/bin/tini", "-s", "/run/entry.sh"],
            "env": {"KVM": "Y", "RAM_SIZE": "4G"},
            "resources": {"cpu": 4, "memory_mib": 16384},
        },
    )

    # No local qcow2 exists; AgentENV restores a snapshot instead of booting one.
    spec = provider._build_spec("", headless=True, os_type="Ubuntu")

    assert spec.image == "osworld-slim"
    assert spec.ttl_s == 7200
    # 6901, not the Docker image's 8006.
    assert spec.ports == (5000, 9222, 6901, 8080)
    assert osworld_sandbox.OSWORLD_DOCKER_VNC_PORT not in spec.ports
    assert spec.metadata["osworld-provider"] == "gym-e2b-sandbox"
    assert spec.metadata["run-id"] == "agentenv-run"
    # E2B rejects entrypoint outright and the template fixes the machine shape.
    assert spec.entrypoint is None
    assert spec.env == {}
    assert spec.resources.cpu is None


def test_agentenv_vnc_guest_port_is_overridable() -> None:
    provider = osworld_sandbox.GymSandboxDesktopProvider(
        {"e2b": {}},
        {"image": "osworld-slim"},
        vnc_guest_port=5901,
    )
    assert provider._service_ports == (5000, 9222, 5901, 8080)


def test_vnc_guest_port_must_not_collide_with_another_service() -> None:
    with pytest.raises(ValueError, match="collides with another OSWorld service port"):
        osworld_sandbox.GymSandboxDesktopProvider(
            {"e2b": {}},
            {"image": "osworld-slim"},
            vnc_guest_port=5000,
        )


def test_agentenv_drops_provider_options_e2b_does_not_accept() -> None:
    """OmegaConf merges, so the base config's OpenSandbox options survive.

    Writing `provider_options: {}` into the generated env.yaml does not clear
    `skip_health_check` and `extensions.poolRef`; they arrive anyway and E2B
    rejects every option it does not know, failing all 361 rollouts.
    """
    provider = osworld_sandbox.GymSandboxDesktopProvider(
        {"e2b": {}},
        {
            "image": "osworld-slim",
            "provider_options": {
                "skip_health_check": True,
                "extensions": {"poolRef": "osworld-kvm"},
            },
        },
    )
    spec = provider._build_spec("", headless=True, os_type="Ubuntu")
    assert spec.provider_options == {}


def test_agentenv_keeps_an_explicit_template_option() -> None:
    provider = osworld_sandbox.GymSandboxDesktopProvider(
        {"e2b": {}},
        {"provider_options": {"template": "tpl-123", "skip_health_check": True}},
    )
    spec = provider._build_spec("", headless=True, os_type="Ubuntu")
    assert spec.provider_options == {"template": "tpl-123"}


def test_agentenv_gets_a_sentinel_vm_path_not_an_empty_one() -> None:
    """An empty path sends DesktopEnv to DockerVMManager, which downloads.

    The template already holds the guest, so resolving a local qcow2 is pure
    waste -- ~11 GB per task, 361 times over. The sentinel exists only to stop
    DesktopEnv reaching for the manager.
    """
    assert osworld_sandbox._resolve_pool_vm_path({"e2b": {}}, "") == osworld_sandbox.AGENTENV_TEMPLATE_VM_PATH
    assert osworld_sandbox._resolve_pool_vm_path({"e2b": {}}, None) == osworld_sandbox.AGENTENV_TEMPLATE_VM_PATH
    # An explicit path still wins, and the other backends are unchanged.
    assert osworld_sandbox._resolve_pool_vm_path({"e2b": {}}, "/tmp/x.qcow2") == "/tmp/x.qcow2"
    assert osworld_sandbox._resolve_pool_vm_path({"opensandbox": {}}, "") == osworld_sandbox.OPENSANDBOX_POOL_VM_PATH
    assert osworld_sandbox._resolve_pool_vm_path({"docker": {}}, "") == ""


def test_generated_env_yaml_never_gives_agentenv_an_empty_vm_path(tmp_path) -> None:
    from benchmarks.osworld import prepare

    env_path = tmp_path / "env.yaml"
    prepare.write_env(
        env_path=env_path,
        config_paths=prepare.PROFILE_CONFIGS["nano_omni"] + (prepare.AGENTENV_CONFIG,),
        input_jsonl=tmp_path / "in.jsonl",
        output_jsonl=tmp_path / "out.jsonl",
        policy_base_url="http://127.0.0.1:8000/v1",
        policy_api_key="EMPTY",  # pragma: allowlist secret
        policy_model_name="m",
        setup_cache_dir=tmp_path,
        agent_name="osworld_nano_omni_agent",
        execution_backend="gym_agentenv",
        max_steps=200,
        force=True,
    )
    text = env_path.read_text()
    assert f'vm_path: "{osworld_sandbox.AGENTENV_TEMPLATE_VM_PATH}"' in text
    assert 'vm_path: ""' not in text


def test_vnc_guest_port_reaches_every_hop() -> None:
    """The port is configured in yaml and consumed five modules away.

    It travelled prepare.py -> app.py -> sandbox_desktop_env.py -> the adapter
    but not into run_osworld_task, so every rollout died with
    `unexpected keyword argument 'sandbox_vnc_guest_port'` -- after the servers
    were up, which reads as a backend fault rather than a plumbing gap.

    Checked statically so the assertion holds without OSWorld's runtime, which
    only the managed agent environment installs.
    """
    import ast
    from pathlib import Path

    agent_dir = Path(__file__).resolve().parents[1]
    hops = {
        "client.py": "run_osworld_task",
        "sandbox_desktop_env.py": "__init__",
    }
    for filename, function in hops.items():
        tree = ast.parse((agent_dir / filename).read_text())
        found = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function
        ]
        assert found, f"{filename} has no {function}"
        assert any("sandbox_vnc_guest_port" in [a.arg for a in n.args.args + n.args.kwonlyargs] for n in found), (
            f"{filename}:{function} does not accept sandbox_vnc_guest_port"
        )

    # And the value has to be forwarded, not merely accepted.
    assert '"sandbox_vnc_guest_port": sandbox_vnc_guest_port' in (agent_dir / "client.py").read_text()


def test_docker_keeps_its_own_vnc_port() -> None:
    """The AgentENV default must not move the Docker image's noVNC port."""
    provider = osworld_sandbox.GymSandboxDesktopProvider(
        {"docker": {}},
        {"image": "osworld:fixed"},
    )
    assert provider._service_ports == (5000, 9222, 8006, 8080)


def test_build_spec_rejects_agentenv_spec_without_a_template() -> None:
    provider = osworld_sandbox.GymSandboxDesktopProvider({"e2b": {}}, {})
    with pytest.raises(ValueError, match="prebuilt template"):
        provider._build_spec("", headless=True, os_type="Ubuntu")


def test_build_spec_accepts_agentenv_template_via_provider_options() -> None:
    """A tagged name or template ID is ambiguous in `image`, so it goes here."""
    provider = osworld_sandbox.GymSandboxDesktopProvider(
        {"e2b": {}},
        {"provider_options": {"template": "01a04eb6-dcca-73c3-ac7b-f1988d17be60"}},
    )
    spec = provider._build_spec("", headless=True, os_type="Ubuntu")
    assert spec.provider_options == {"template": "01a04eb6-dcca-73c3-ac7b-f1988d17be60"}
    assert spec.image is None


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
