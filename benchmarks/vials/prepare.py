# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare pinned VIALS Harbor tasks for Gym's shared Harbor bridge."""

import os
import shutil
from pathlib import Path

from benchmarks.vials.harbor.adapter import DEFAULT_OUTPUT_DIR
from benchmarks.vials.harbor.adapter import prepare as export


def prepare() -> Path:
    index = export(
        docker_image=os.environ.get("VIALS_HARBOR_IMAGE", "vials-inputs:local"),
        output_dir=DEFAULT_OUTPUT_DIR,
        overwrite=True,
    )
    shutil.copy2(index.parent / "example.jsonl", DEFAULT_OUTPUT_DIR.parent / "harbor_example.jsonl")
    return index


if __name__ == "__main__":
    prepare()
