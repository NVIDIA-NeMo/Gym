# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Include the plugin behavior suites in the normal pytest/conformance test path."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("bun") is None, reason="Bun is required for OpenCode plugin tests")
@pytest.mark.parametrize("suite", ["test_final_answer.mjs", "test_eval_correlation.mjs"])
def test_bun_plugin_suite(suite):
    directory = Path(__file__).resolve().parent
    result = subprocess.run(
        [shutil.which("bun"), "test", str(directory / suite)],
        cwd=directory,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
