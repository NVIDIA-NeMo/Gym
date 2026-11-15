# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


def verify(workspace: Path) -> float:
    """Return full credit when the requested file has the exact contents."""

    output_path = workspace / "hello-gym.txt"
    if not output_path.is_file():
        return 0.0
    return float(output_path.read_text(encoding="utf-8").removesuffix("\n") == "Hello from NeMo Gym!")
