# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the Agentic-MME track for ``opencode_server_tools.yaml``: OpenCode with the 14 image tools over MCP.

``prepare.py`` converts the release once and writes every track. This returns the file that ``opencode_server_tools.yaml`` reads, as
``gym eval prepare`` requires.
"""

from pathlib import Path

from benchmarks.agentic_mme.prepare import OPENCODE_SERVER_TOOLS_OUTPUT_FPATH
from benchmarks.agentic_mme.prepare import prepare as prepare_all_tracks


def prepare() -> Path:
    prepare_all_tracks()
    return OPENCODE_SERVER_TOOLS_OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
