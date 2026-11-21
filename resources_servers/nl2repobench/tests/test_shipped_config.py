# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import yaml


OPENCODE_CONFIG = Path(__file__).resolve().parents[3] / "benchmarks/nl2repobench/opencode.yaml"


def test_the_shipped_preset_references_no_private_runtime() -> None:
    text = OPENCODE_CONFIG.read_text()
    agent = yaml.safe_load(text)["nl2repobench_opencode_sandboxed_agent"]["responses_api_agents"][
        "opencode_sandboxed_agent"
    ]

    assert "/mnt/s3-data" not in text
    # with the fields unset the agent downloads the installer, which is what open-egress modes need
    assert not agent.get("remote_opencode_install_script_path")
    assert not agent.get("remote_opencode_binary_path")
    assert not agent.get("preinstalled_opencode")
    assert not agent.get("require_opencode_runtime")
