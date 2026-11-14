# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent implementation of Harvey LAB's bounded finish gate.

Contract: https://github.com/harveyai/harvey-labs/blob/
9e9fa7fe126103de1fceb2227b7c70bf59b67f73/lab_core/harness/tools.py (MIT).
"""

from dataclasses import dataclass


@dataclass
class CompletionState:
    """Accept completion after checking self-reported files, with at most two rejections."""

    finished: bool = False
    summary: str | None = None
    rejections: int = 0

    def finish(self, *, summary: str, missing: list[str]) -> str:
        if missing and self.rejections < 2:
            self.rejections += 1
            return "Not finished: these files do not exist in the output directory: " + ", ".join(missing)
        self.finished = True
        self.summary = summary or None
        return "Finished."
