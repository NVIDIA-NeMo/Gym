# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is required to execute the gzip hooks")
def test_javascript_gzip_compressor_and_harness_hooks():
    script = Path(__file__).with_name("fixtures") / "request_gzip.mjs"
    completed = subprocess.run(
        [shutil.which("node"), "--test", str(script)],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
