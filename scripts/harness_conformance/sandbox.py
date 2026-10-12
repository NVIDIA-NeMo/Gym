# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned local Docker runtime for the dedicated sandboxed OpenCode adapter."""

import json
import subprocess
from pathlib import Path

from responses_api_agents.opencode_agent.runtime import OPENCODE_VERSION


IMAGE = f"nemo-gym-conformance-opencode:{OPENCODE_VERSION}"


def build_image() -> None:
    """Build the credential-free probe image; agent startup installs the shared plugin."""
    subprocess.run(
        [
            "docker",
            "build",
            "--build-arg",
            f"OPENCODE_VERSION={OPENCODE_VERSION}",
            "-t",
            IMAGE,
            "-f",
            str(Path(__file__).with_name("opencode.Dockerfile")),
            str(Path(__file__).parent),
        ],
        check=True,
    )


def check_image() -> dict:
    """Require the prepared image and retain its immutable identity in runtime provenance."""
    result = subprocess.run(
        ["docker", "image", "inspect", IMAGE], capture_output=True, text=True, check=True, timeout=10
    )
    image = json.loads(result.stdout)[0]
    return {"image": IMAGE, "image_id": image["Id"], "version": OPENCODE_VERSION}
