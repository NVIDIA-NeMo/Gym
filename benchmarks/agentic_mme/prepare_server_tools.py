# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the Agentic-MME track for ``server_tools.yaml``: the 14 image tools served by the resources server.

``prepare.py`` converts the release once and writes every track. This returns the file that ``server_tools.yaml`` reads, as
``gym eval prepare`` requires.
"""

from pathlib import Path

from benchmarks.agentic_mme.prepare import SERVER_TOOLS_OUTPUT_FPATH
from benchmarks.agentic_mme.prepare import prepare as prepare_all_tracks


def prepare() -> Path:
    prepare_all_tracks()
    return SERVER_TOOLS_OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
