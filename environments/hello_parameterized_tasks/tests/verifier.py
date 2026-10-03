# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from pathlib import Path


def verify(workspace: Path, task_data: Mapping[str, str]) -> float:
    """Return full credit when the submitted answer is correct."""

    output_path = workspace / "answer.txt"
    if not output_path.is_file():
        return 0.0
    return float(output_path.read_text(encoding="utf-8").strip() == task_data["expected_answer"])
