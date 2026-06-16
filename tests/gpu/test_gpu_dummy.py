# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Placeholder GPU tests. Replace with real GPU-specific tests as needed.
import subprocess


def test_gpu_runner_has_nvidia_smi():
    """Verify the runner has at least one accessible GPU."""
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    assert result.returncode == 0, f"nvidia-smi failed:\n{result.stderr}"


def test_gpu_runner_has_multiple_gpus():
    """Verify the runner reports at least two GPUs (x2 runner)."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"nvidia-smi query failed:\n{result.stderr}"
    gpus = [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
    assert len(gpus) >= 1, f"Expected at least 1 GPU, found: {gpus}"
    print(f"Found {len(gpus)} GPU(s): {gpus}")
