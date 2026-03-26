# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Prepare Ruler benchmark data.

Adapted from https://github.com/NVIDIA-NeMo/Skills/blob/54d2e113c2f64bf74bda72e15f23f01b524850da/nemo_skills/dataset/ruler/prepare.py#L79"""

import json
from os import environ
from pathlib import Path
from subprocess import run

from nemo_gym.global_config import get_hf_token


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "ruler_benchmark.jsonl"


def prepare(model: str, length: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    skills_dir = BENCHMARK_DIR / "Skills"
    if skills_dir.exists():
        print("Skipping git clone as the repository is already cloned!")
    else:
        run(
            """git clone https://github.com/NVIDIA-NeMo/Skills \
&& cd Skills \
&& git checkout 54d2e113c2f64bf74bda72e15f23f01b524850da \
&& uv venv --python 3.12 --seed \
&& source .venv/bin/activate \
&& uv pip install '-e .' wonderwords html2text tenacity nltk""",
            check=True,
            shell=True,
            cwd=BENCHMARK_DIR,
        )

    maybe_hf_token = get_hf_token()
    env_vars = dict()
    if maybe_hf_token:
        env_vars["HF_TOKEN"] = maybe_hf_token

    run(
        """source .venv/bin/activate \
LENGTH=262144
MODEL=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
python nemo_skills/dataset/ruler/prepare.py \
    --data_format=chat \
    --setup=$MODEL-$LENGTH \
    --max_seq_length=$LENGTH \
    --tokenizer_path=$MODEL \
    --max_seq_length=$LENGTH \
    --tmp_data_dir=ruler
""",
        check=True,
        shell=True,
        cwd=skills_dir,
        env=environ | env_vars,
    )

    samples = []
    for row in samples:
        sample = {
            "responses_create_params": {"input": [{"role": "user", "content": ""}]},
        }
        samples.append(sample)

    with OUTPUT_FPATH.open("w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    print(f"Wrote {len(samples)} samples to {OUTPUT_FPATH}")

    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
