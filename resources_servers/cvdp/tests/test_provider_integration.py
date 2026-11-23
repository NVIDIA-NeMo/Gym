# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in integration test for real sandbox providers."""

import os
import shlex
from contextlib import AsyncExitStack
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from nemo_gym.sandbox import SandboxSpec, create_provider
from resources_servers.cvdp.testbench_runner import TestbenchRunner as CVDPTestbenchRunner


@pytest.mark.asyncio
async def test_apptainer_tasks_cannot_read_host_or_sibling_workspaces(tmp_path: Path) -> None:
    image = os.environ.get("CVDP_ISOLATION_TEST_IMAGE")
    if not image:
        pytest.skip("set CVDP_ISOLATION_TEST_IMAGE to run real Apptainer isolation checks")
    root = Path(__file__).resolve().parents[3]
    config = yaml.safe_load(
        (root / "responses_api_agents/cvdp_agent/configs/cvdp_agent_generic_claude.yaml").read_text()
    )["cvdp_agent_generic_claude"]["responses_api_agents"]["cvdp_agent"]["sandbox_provider"]
    provider = create_provider(config)
    host_marker = tmp_path / "host-only-marker"
    host_marker.write_text("host-only")
    handles = []
    async with AsyncExitStack() as cleanup:
        for label in ("first", "second"):
            handle = await provider.create(SandboxSpec(image=image))
            cleanup.push_async_callback(provider.close, handle)
            handles.append(handle)
            result = await provider.exec(
                handle,
                f"echo {label} > /code/isolation-marker; "
                f"echo {label} > /tmp/isolation-marker; echo {label} > /var/tmp/isolation-marker",
            )
            assert result.return_code == 0 and result.error_type is None, result
        for index, label in enumerate(("first", "second")):
            handle = handles[index]
            sibling_path = handles[1 - index].raw.staging_dir / "isolation-marker"
            forbidden = [
                host_marker,
                sibling_path,
                root / "pyproject.toml",
                Path.home() / ".local/apptainer/bin/apptainer",
            ]
            checks = [f"test ! -e {shlex.quote(str(path))}" for path in forbidden]
            checks += [
                f'test "$(cat {path})" = {label}'
                for path in ("/code/isolation-marker", "/tmp/isolation-marker", "/var/tmp/isolation-marker")
            ]
            result = await provider.exec(handle, " && ".join(checks))
            assert result.return_code == 0 and result.error_type is None, result
            # Heavy repositories compile in /tmp. Containment must retain disk-backed
            # temporary storage rather than Apptainer's small session tmpfs.
            result = await provider.exec(
                handle,
                "python3 -c \"from pathlib import Path; p=Path('/tmp/storage-check'); "
                'p.write_bytes(bytes(128*1024*1024)); p.unlink()"',
            )
            assert result.return_code == 0 and result.error_type is None, result


def _provider_config(name: str) -> dict:
    if name == "apptainer":
        return {"apptainer": {"probe": {"command": None}}}
    if name == "opensandbox":
        domain = os.environ.get("OPENSANDBOX_DOMAIN")
        api_key = os.environ.get("OPENSANDBOX_API_KEY")
        if not domain or not api_key:
            pytest.skip("OpenSandbox integration credentials are not configured")
        return {
            "opensandbox": {
                "connection": {
                    "domain": domain,
                    "api_key": api_key,
                    "protocol": os.environ.get("OPENSANDBOX_PROTOCOL", "http"),
                    "use_server_proxy": True,
                    "request_timeout_s": 300,
                },
                "create": {"timeout_s": 600, "request_timeout_s": 600, "skip_health_check": True},
                "probe": {"command": None},
            }
        }
    raise ValueError(f"unsupported integration provider: {name}")


@pytest.mark.asyncio
async def test_real_provider_workspace_round_trip() -> None:
    provider_name = os.environ.get("CVDP_SANDBOX_PROVIDER")
    if not provider_name:
        pytest.skip("set CVDP_SANDBOX_PROVIDER to run sandbox integration")

    config = SimpleNamespace(
        oss_sim_image="ghcr.io/hdl/sim/osvb",
        oss_pnr_image="ghcr.io/hdl/impl/pnr",
        eda_sim_image="",
        container_timeout=600,
        num_processes=1,
        harness_workspace_dir="",
        container_workspace="/code",
        container_transfer_dir="/sandbox",
        container_tmp_bind_path="",
        sandbox_provider=_provider_config(provider_name),
        sandbox_spec={},
        prepared_images={},
        prepared_image_manifest="",
        image_overrides={},
    )
    runner = CVDPTestbenchRunner(config)
    compose = """
services:
  build:
    image: ghcr.io/hdl/sim/osvb
    volumes:
      - ./src:/src:ro
    working_dir: /code/rundir
    command: /bin/sh -c "test -f /src/check.sh && test -f /code/rtl/design.sv && /bin/sh /src/check.sh"
  check:
    image: ghcr.io/hdl/sim/osvb
    working_dir: /code/rundir
    command: /bin/sh -c "test -s /code/rundir/design.out && echo second-service-sees-the-first-ones-file"
"""
    exit_code, output, services = await runner.run(
        rtl_files={"rtl/design.sv": "module design; endmodule\n"},
        harness_files={
            "docker-compose.yml": compose,
            "src/check.sh": (
                "#!/bin/sh\n"
                "iverilog -g2012 -s design -o /code/rundir/design.out /code/rtl/design.sv\n"
                "test -x /code/rundir/design.out\n"
                "echo provider-round-trip-ok\n"
            ),
        },
        task_id=f"integration-{provider_name}",
    )

    assert exit_code == 0, output
    # The second service starts from the archive the first one handed on, so it sees the file the first one built.
    assert [(s["service"], s["exit_code"]) for s in services] == [("build", 0), ("check", 0)]
    assert "provider-round-trip-ok" in output
    assert "second-service-sees-the-first-ones-file" in output
