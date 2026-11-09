# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the upstream downloader with VisIT-Bench's pinned Parquet export."""

import runpy
import sys
from pathlib import Path


VISITBENCH_REVISION = "41983853920321bb56a53c99db70f74db17060ba"  # pragma: allowlist secret
VISITBENCH_PARQUET = (
    "https://huggingface.co/datasets/mlfoundations/VisIT-Bench/resolve/"
    f"{VISITBENCH_REVISION}/default/test/0000.parquet"
)


def run_downloader(script: str | Path) -> None:
    import datasets

    original_load = datasets.load_dataset

    def load_dataset(path: str, *args, **kwargs):
        # Avoid the image-folder metadata incompatibility; preserve the export's row order.
        if path == "mlfoundations/VisIT-Bench":
            return original_load("parquet", data_files={"test": VISITBENCH_PARQUET}, *args, **kwargs)
        return original_load(path, *args, **kwargs)

    datasets.load_dataset = load_dataset
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        datasets.load_dataset = original_load


if __name__ == "__main__":
    run_downloader(sys.argv[1])
