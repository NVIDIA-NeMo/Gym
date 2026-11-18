# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download preprocessed HS3 evaluation tasks."""

from pathlib import Path

from resources_servers.genrm_eval.preferences import validate_metadata
from resources_servers.genrm_eval.prepare import download_prepared_dataset


OUTPUT = Path(__file__).parent / "data/hs3_validation.jsonl"


def prepare(output_fpath: Path | str = OUTPUT, version: str = "0.1.0") -> Path:
    return download_prepared_dataset(
        "genrm_hs3", version, OUTPUT.name, output_fpath, validate_metadata=validate_metadata
    )


if __name__ == "__main__":
    print(prepare())
