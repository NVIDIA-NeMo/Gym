# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
The prepare_* functions in this file are written to exactly match the input observed in the VLMEvalKit OpenAI API call.
"""

from pathlib import Path
from subprocess import run

import orjson
from vlmeval.dataset.image_mcq import ImageMCQDataset
from vlmeval.dataset.image_vqa import OCRBench


def setup_VLMEvalKit():
    this_dir = Path(__file__).parent.absolute()
    # We freeze the commit SHA for now.
    # We pip install with no-deps since we have the deps in the pyproject.toml already.
    setup_command = f"""cd {this_dir} \
&& source .venv/bin/activate \
&& if [ ! -d VLMEvalKit ]; then git clone https://github.com/open-compass/VLMEvalKit/; fi \
&& cd VLMEvalKit \
&& git checkout 00804217f868058f871f5ff252a7b9623c3475d9 \
&& uv pip install '-e .' --no-deps \
&& sed -i '' 's/import clip/# import clip/' vlmeval/dataset/utils/SArena/FID.py
"""
    print(f"Running VLMEvalKit setup command: {setup_command}")
    run(setup_command, shell=True, check=True)


def prepare_OCRBench():
    dataset_name = "OCRBench"

    data = OCRBench(dataset=dataset_name).load_data(dataset_name)

    print(f"Columns: {data.columns}")
    print(data.head())

    assert list(data.columns) == ["index", "image", "question", "answer", "category"]

    f = open(f"data/{dataset_name}_validation.jsonl", "wb")
    for _, vlmevalkit_row in data.iterrows():
        gym_row = {
            "responses_create_params": {
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{vlmevalkit_row['image']}",
                                "detail": "high",
                            },
                            {
                                "type": "input_text",
                                "text": vlmevalkit_row["question"],
                            },
                        ],
                    }
                ],
            },
            "answer": eval(vlmevalkit_row["answer"]),
            "category": vlmevalkit_row["category"],
            "eval_fn": f"_score_{dataset_name}",
        }
        f.write(orjson.dumps(gym_row) + b"\n")


def prepare_MMBench_DEV_EN_V11():
    dataset_name = "MMBench_DEV_EN_V11"

    dataset = ImageMCQDataset(dataset=dataset_name)
    data = dataset.load_data(dataset_name)

    print(f"""Columns: {data.columns}
Data:
{data}
Data head:
{data.head()}""")

    assert list(data.columns) == [
        "index",
        "question",
        "hint",
        "A",
        "B",
        "C",
        "D",
        "answer",
        "category",
        "image",
        "l2-category",
        "split",
    ]

    f = open(f"data/{dataset_name}_validation.jsonl", "wb")
    for _, vlmevalkit_row in data.iterrows():
        messages = dataset.build_prompt(vlmevalkit_row)

        gym_row = {
            "responses_create_params": {
                "input": [
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{vlmevalkit_row['image']}",
                        "detail": "high",
                    },
                    {
                        "type": "input_text",
                        "text": messages[-1]["value"],
                    },
                ]
            },
            "answer": vlmevalkit_row["answer"],
            "category": vlmevalkit_row["category"],
            "eval_fn": f"_score_{dataset_name}",
        }
        f.write(orjson.dumps(gym_row) + b"\n")


if __name__ == "__main__":
    setup_VLMEvalKit()

    # prepare_OCRBench()
    prepare_MMBench_DEV_EN_V11()
