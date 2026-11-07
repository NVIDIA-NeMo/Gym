# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trusted entrypoint inside the renderer; never run submitted code on the host."""

import os
import runpy


if __name__ == "__main__":
    os.makedirs("/workspace/.tmp", exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    os.chdir("/workspace")
    runpy.run_path(
        "/workspace/code.py",
        run_name="__main__",
        init_globals={
            "OUTPUT_PATH": os.environ["OUTPUT_PATH"],
            "TARGET_WIDTH": int(os.environ["TARGET_WIDTH"]),
            "TARGET_HEIGHT": int(os.environ["TARGET_HEIGHT"]),
        },
    )
