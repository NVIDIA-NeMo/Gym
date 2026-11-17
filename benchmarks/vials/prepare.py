# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare pinned VIALS Harbor tasks for Gym's Harbor environment server."""

import os
from pathlib import Path

from benchmarks.vials.harbor.adapter import DEFAULT_OUTPUT_DIR
from benchmarks.vials.harbor.adapter import prepare as export


def prepare() -> Path:
    """Download the pinned source and return the native task-folder target."""
    return export(
        docker_image=os.environ.get("VIALS_HARBOR_IMAGE", "vials-inputs:local"),
        output_dir=DEFAULT_OUTPUT_DIR,
        overwrite=True,
    )


if __name__ == "__main__":
    prepare()
