# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the desktop runtime's imports without creating a VM or downloading models."""

import os
import subprocess
import sys


def test_desktop_runtime_imports_with_cpu_torch() -> None:
    """Catch excluded direct and transitive dependencies before session creation."""
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from desktop_env.desktop_env import DesktopEnv; import cryptography, flask, torch; "
            "assert DesktopEnv.__name__ == 'DesktopEnv'; assert torch.version.cuda is None",
        ],
        env=os.environ | {"PYNPUT_BACKEND": "dummy"},
        check=True,
    )
