# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SANITIZER = REPO_ROOT / "scripts" / "ci" / "sanitize_env.sh"

BEHAVIOR_CHANGING_ENV = {
    "SKIP": "ruff",
    "NEMO_GYM_EXTRA_ROOTS": "/tmp/external-gym",
    "NEMO_GYM_CONFIG_DICT": '{"search_dir": "/tmp/external-gym"}',
    "NEMO_GYM_ALLOW_PRERELEASE": "true",
    "PYTHONPATH": "/tmp/python",
    "PYTEST_ADDOPTS": "-m injected-selection",
}


def _environment_after_sanitizing(stage: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(BEHAVIOR_CHANGING_ENV)
    env["GYM_CI_PRESERVED_SENTINEL"] = "preserved"
    command = f'source "{SANITIZER}"; gym_ci_sanitize_environment "{stage}"; env -0'
    result = subprocess.run(
        ["bash", "-c", command],
        check=True,
        capture_output=True,
        env=env,
    )
    return {
        key.decode(): value.decode()
        for item in result.stdout.split(b"\0")
        if item
        for key, value in [item.split(b"=", 1)]
    }


@pytest.mark.parametrize(
    ("stage", "removed"),
    [
        ("lint", {"SKIP"}),
        ("core", {"NEMO_GYM_EXTRA_ROOTS", "NEMO_GYM_CONFIG_DICT", "PYTHONPATH"}),
        (
            "server",
            {
                "NEMO_GYM_EXTRA_ROOTS",
                "NEMO_GYM_CONFIG_DICT",
                "NEMO_GYM_ALLOW_PRERELEASE",
                "PYTHONPATH",
                "PYTEST_ADDOPTS",
            },
        ),
    ],
)
def test_ci_stage_removes_only_its_behavior_changing_environment(stage: str, removed: set[str]) -> None:
    sanitized = _environment_after_sanitizing(stage)

    assert removed.isdisjoint(sanitized)
    assert sanitized["GYM_CI_PRESERVED_SENTINEL"] == "preserved"
    for name, value in BEHAVIOR_CHANGING_ENV.items():
        if name not in removed:
            assert sanitized[name] == value


def test_ci_environment_sanitizer_rejects_unknown_stage() -> None:
    result = subprocess.run(
        ["bash", "-c", f'source "{SANITIZER}"; gym_ci_sanitize_environment unknown'],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "unknown Gym CI stage: unknown" in result.stderr
