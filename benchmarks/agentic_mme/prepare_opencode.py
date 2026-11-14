# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the Agentic-MME track for ``opencode.yaml``: OpenCode with the task images as files in a lent sandbox.

``prepare.py`` converts the release once and writes every track. This returns the file that ``opencode.yaml`` reads, as
``gym eval prepare`` requires.
"""

from pathlib import Path

from benchmarks.agentic_mme.prepare import OPENCODE_OUTPUT_FPATH
from benchmarks.agentic_mme.prepare import prepare as prepare_all_tracks


def prepare() -> Path:
    prepare_all_tracks()
    return OPENCODE_OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
