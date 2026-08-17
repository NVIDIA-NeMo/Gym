# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in integration test for real sandbox providers."""

import os
from types import SimpleNamespace

import pytest

from resources_servers.cvdp.testbench_runner import TestbenchRunner as CVDPTestbenchRunner


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
    )
    runner = CVDPTestbenchRunner(config)
    compose = """
services:
  direct:
    image: ghcr.io/hdl/sim/osvb
    volumes:
      - ./src:/src:ro
    working_dir: /code/rundir
    command: /bin/sh -c "test -f /src/check.sh && test -f /code/rtl/design.sv && /bin/sh /src/check.sh"
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
    assert services[0]["exit_code"] == 0
    assert "provider-round-trip-ok" in output
