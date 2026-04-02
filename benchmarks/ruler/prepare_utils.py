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


def prepare_helper(output_name: str, model: str, length: str, add_answer_prefix: bool = True) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_fpath = DATA_DIR / output_name

    skills_dir = BENCHMARK_DIR / "Skills"
    if skills_dir.exists():
        print("Skipping git clone as the repository is already cloned!")
    else:
        run(
            """git clone https://github.com/NVIDIA-NeMo/Skills \
&& cd Skills \
&& git lfs install \
&& git checkout 54d2e113c2f64bf74bda72e15f23f01b524850da \
&& uv venv --python 3.12 --seed .venv \
&& source .venv/bin/activate \
&& uv pip install '-e .' scipy wonderwords html2text tenacity nltk""",
            check=True,
            shell=True,
            cwd=BENCHMARK_DIR,
        )

    maybe_hf_token = get_hf_token()
    env_vars = dict()
    if maybe_hf_token:
        env_vars["HF_TOKEN"] = maybe_hf_token

    tmp_data_dir = skills_dir / "ruler" / model / str(length)
    run(
        f"""source .venv/bin/activate \
&& python nemo_skills/dataset/ruler/prepare.py \
    --data_format=chat \
    --setup={model}-{length} \
    --max_seq_length={length} \
    --tokenizer_path={model} \
    --max_seq_length={length} \
    --tmp_data_dir={tmp_data_dir.absolute()}
""",
        check=True,
        shell=True,
        cwd=skills_dir,
        env=environ | env_vars,
    )

    samples = []
    for subset_dir in (tmp_data_dir / "ruler_data").iterdir():
        subset_file = subset_dir / "test.jsonl"
        with subset_file.open() as f:
            subset_samples = list(map(json.loads, f))

        for sample in subset_samples:
            answer_prefix = sample["answer_prefix"].strip()
            sample = {
                "responses_create_params": {"input": [{"role": "user", "content": sample["input"]}]},
                "outputs": sample["outputs"],
                "length": sample["length"],
                "subset": subset_dir.name,
            }
            if add_answer_prefix:
                # status is needed in response mode but optional in chat completion mode.
                sample["responses_create_params"]["input"].append({"role": "assistant", "content": answer_prefix, "status": "in_progress"})
            samples.append(sample)

    with output_fpath.open("w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    print(f"Wrote {len(samples)} samples to {output_fpath}")

    return output_fpath
