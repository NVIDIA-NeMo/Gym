# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures of the Compose tests.

The main and sidecar image references, their recorded configurations, the overlay document and
the fixture loader that test_compose_config.py, test_compose_fields.py and test_compose_volumes.py
all read. Not a pytest conftest, so each test module imports the names it uses.
"""

from pathlib import Path

import yaml


MAIN = "org/task-env@sha256:" + "a" * 64
SIDECAR = "org/task-sidecar@sha256:" + "b" * 64
IMAGE_CONFIGS = {
    MAIN: {
        "os": "linux",
        "architecture": "amd64",
        "image": "registry/task-env@sha256:" + "a" * 64,
        "config": {"WorkingDir": "/app"},
    },
    SIDECAR: {
        "os": "linux",
        "architecture": "amd64",
        "image": "registry/task-sidecar@sha256:" + "b" * 64,
        "config": {
            "Entrypoint": ["/entry.sh"],
            "Cmd": ["serve"],
            "User": "pwuser",
            "ExposedPorts": {"3080/tcp": {}},
            "Healthcheck": {"Test": ["CMD-SHELL", "curl -sf localhost:3080"], "Interval": 5_000_000_000, "Retries": 4},
        },
    },
}
OVERLAY = {
    "services": {
        "main": {"image": MAIN, "depends_on": ["browser"]},
        "browser": {
            "image": SIDECAR,
            "shm_size": "1gb",
            "environment": ["MCP_PORT=3080", "BROWSER_URL=http://workspace:18073"],
            "expose": ["9223"],
        },
        "workspace": {"image": MAIN, "expose": ["18073"]},
    }
}

FIXTURES = Path(__file__).parent / "fixtures" / "harbor_compose_fields"


def _fixture(name):
    return yaml.safe_load((FIXTURES / name / "environment" / "docker-compose.yaml").read_text())
