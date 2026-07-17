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
TEXT_COMPLETION_OUTPUT_TOKENS = {"niah": 128, "vt": 30, "cwe": 120, "fwe": 50, "qa": 32}
SUPPORTED_DATA_FORMATS = {"chat", "default", "base"}


def prepare(model: str, length: int, data_format: str = "chat") -> Path:
    output_name = "ruler.jsonl" if data_format == "chat" else "ruler_pretrain.jsonl"
    return prepare_helper(output_name=output_name, model=model, length=length, data_format=data_format)


def _to_gym_sample(sample: dict, subset: str, data_format: str) -> dict:
    content = sample["input"]
    create_params = {"input": [{"role": "user", "content": content}]}
    if data_format != "chat":
        separator = "" if data_format == "default" else "\n"
        create_params["input"][0]["content"] += separator + sample["answer_prefix"].strip()
        task_family = subset.split("_", maxsplit=1)[0]
        create_params["max_output_tokens"] = TEXT_COMPLETION_OUTPUT_TOKENS[task_family]

    return {
        "responses_create_params": create_params,
        "outputs": sample["outputs"],
        "length": sample["length"],
        "subset": subset,
    }


def prepare_helper(output_name: str, model: str, length: int, data_format: str = "chat") -> Path:
    if data_format not in SUPPORTED_DATA_FORMATS:
        raise ValueError(f"Unsupported RULER data format: {data_format}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_fpath = DATA_DIR / output_name

    run(
        """uv venv --python 3.12 --allow-existing --seed .venv \
&& source .venv/bin/activate \
&& uv pip install pyyaml bs4 scipy wonderwords html2text tenacity nltk transformers""",
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

    run(
        f"""source .venv/bin/activate \
&& python ruler_prepare_script.py \
    --data_format={data_format} \
    --setup={model}-{length} \
    --max_seq_length={length} \
    --tokenizer_path={model} \
    --max_seq_length={length} \
    --ruler_parent_dir={BENCHMARK_DIR} \
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

        samples.extend(_to_gym_sample(sample, subset_dir.name, data_format) for sample in subset_samples)

    with output_fpath.open("w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    print(f"Wrote {len(samples)} samples to {output_fpath}")

    return output_fpath
