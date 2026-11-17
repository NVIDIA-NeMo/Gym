# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the actual request-transform contract with the runtime's JavaScript engine."""

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest


def test_latest_browser_screenshot_request_transform():
    node = shutil.which("node")
    if node is None:
        spec = importlib.util.find_spec("playwright")
        if spec and spec.origin:
            candidate = Path(spec.origin).parent / "driver/node"
            if candidate.is_file():
                node = str(candidate)
    if node is None:
        pytest.skip("Screenshot transform requires Node or the provisioned Playwright runtime")
    result = subprocess.run(
        [node, "--test", str(Path(__file__).with_suffix(".mjs"))],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
