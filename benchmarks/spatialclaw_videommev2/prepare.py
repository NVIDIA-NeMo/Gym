# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

from resources_servers.spatialclaw.prepare_data import prepare_spatialclaw_benchmark


OUTPUT_FPATH = Path(__file__).parent / "data" / "spatialclaw_videommev2.jsonl"


def prepare() -> Path:
    return prepare_spatialclaw_benchmark(
        dataset_config="videommev2.json",
        output_path=OUTPUT_FPATH,
    )


if __name__ == "__main__":
    prepare()
