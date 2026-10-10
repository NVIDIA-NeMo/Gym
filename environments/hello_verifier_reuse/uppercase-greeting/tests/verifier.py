# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from nemo_gym.verifiers.files import read_text


def verify(workspace: Path) -> float:
    """Return full credit for the expected uppercase greeting."""

    content = read_text(workspace, "shout.txt")
    return float(content == "HELLO FROM NEMO GYM!" and content.isupper())
