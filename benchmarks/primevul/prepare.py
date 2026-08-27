# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the pinned PrimeVul paired test split as prompt-agnostic Gym rows."""

from pathlib import Path

import orjson

from nemo_gym.global_config import HF_TOKEN_KEY_NAME, get_global_config_dict
from resources_servers.primevul.primevul_data import load_pairs, raw_row


DATA_DIR = Path(__file__).parent / "data"
OUTPUT_FPATH = DATA_DIR / "primevul_benchmark.jsonl"


def prepare(output_path: Path = OUTPUT_FPATH, max_pairs: int | None = None, seed: int = 0) -> Path:
    """Write all 435 test pairs, or a reproducible whole-pair subset."""
    output_path = Path(output_path)
    hf_token = get_global_config_dict().get(HF_TOKEN_KEY_NAME)
    records = load_pairs("benchmark", max_pairs=max_pairs, seed=seed, hf_token=hf_token)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".jsonl.tmp")
    with temporary_path.open("wb") as f:
        for record in records:
            f.write(orjson.dumps(raw_row(record)) + b"\n")
    temporary_path.replace(output_path)

    print(f"Wrote {len(records)} rows ({len(records) // 2} pairs) to {output_path}")
    return output_path


if __name__ == "__main__":
    prepare()
