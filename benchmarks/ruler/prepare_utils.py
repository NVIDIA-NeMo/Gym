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
from shutil import rmtree
from subprocess import run

import requests

from nemo_gym.global_config import get_hf_token


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"


def prepare_helper(output_name: str, model: str, length: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_fpath = DATA_DIR / output_name

    response = requests.get(
        "https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/54d2e113c2f64bf74bda72e15f23f01b524850da/nemo_skills/dataset/ruler/prepare.py"
    )
    assert response.ok

    prepare_script_fpath = BENCHMARK_DIR / "ruler_prepare_script.py"
    with prepare_script_fpath.open("wb") as f:
        f.write(response.content)

    venv_dir = BENCHMARK_DIR / ".venv"
    if not venv_dir.exists():
        run(
            """uv venv --python 3.12 --seed .venv \
&& source .venv/bin/activate \
&& uv pip install bs4 scipy wonderwords html2text tenacity nltk""",
            check=True,
            shell=True,
            cwd=BENCHMARK_DIR,
            executable="/bin/bash",
        )

    maybe_hf_token = get_hf_token()
    env_vars = dict()
    if maybe_hf_token:
        env_vars["HF_TOKEN"] = maybe_hf_token

    tmp_data_dir = BENCHMARK_DIR / "temp_ruler_data_dir" / model / str(length)
    ruler_dir = tmp_data_dir / "RULER"
    rmtree(ruler_dir, ignore_errors=True)
    run(
        f"""source .venv/bin/activate \
&& python ruler_prepare_script.py \
    --data_format=chat \
    --setup={model}-{length} \
    --max_seq_length={length} \
    --tokenizer_path={model} \
    --max_seq_length={length} \
    --tmp_data_dir={tmp_data_dir.absolute()}
""",
        check=True,
        shell=True,
        cwd=BENCHMARK_DIR,
        env=environ | env_vars,
        executable="/bin/bash",
    )

    samples = []
    for subset_dir in (tmp_data_dir / "ruler_data").iterdir():
        subset_file = subset_dir / "test.jsonl"
        with subset_file.open() as f:
            subset_samples = list(map(json.loads, f))

        for sample in subset_samples:
            sample = {
                "responses_create_params": {"input": [{"role": "user", "content": sample["input"]}]},
                "outputs": sample["outputs"],
                "length": sample["length"],
                "subset": subset_dir.name,
            }
            samples.append(sample)

    with output_fpath.open("w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    print(f"Wrote {len(samples)} samples to {output_fpath}")

    return output_fpath
