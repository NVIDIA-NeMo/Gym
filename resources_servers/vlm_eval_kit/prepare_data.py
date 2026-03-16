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

from collections import Counter

import orjson
from app import VlmEvalKitResourcesServer
from pandas import DataFrame
from vlmeval.dataset.image_mcq import ImageMCQDataset
from vlmeval.dataset.image_vqa import OCRBench
from vlmeval.dataset.utils.multiple_choice import build_choices


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
            "benchmark_name": dataset_name,
        }
        f.write(orjson.dumps(gym_row) + b"\n")


def prepare_MMBench_DEV_EN_V11():
    dataset_name = "MMBench_DEV_EN_V11"

    dataset = ImageMCQDataset(dataset=dataset_name)
    data: DataFrame = dataset.load_data(dataset_name)

    # Uncomment these lines to test a single sample
    # samples = [3757, 1003757, 2003757, 3003757]
    # data = data[data["index"].isin(samples)]

    print(f"""Columns: {data.columns}
Data:
{data}
Data head:
{data.head()}""")

    # From https://github.com/open-compass/VLMEvalKit/blob/00804217f868058f871f5ff252a7b9623c3475d9/vlmeval/dataset/utils/multiple_choice.py#L513
    get_group = lambda i: int(i % 1e6)
    group_counts = Counter(map(get_group, data["index"]))

    # We sort this dataset so that samples in a group are adjacent to each other rather than spread apart
    # At runtime, this data will be read in order and this results in much more efficient processing
    # This key is the same as get_group, just for a pd.Series
    data = data.sort_values("index", key=lambda i: i.astype(int) % 1e6)

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

        group = get_group(vlmevalkit_row["index"])

        has_image = group == int(vlmevalkit_row["index"])
        if has_image:
            image = vlmevalkit_row["image"]
        if not has_image:  # Is not valid image, rather is an image reference
            image = data[data["index"] == int(vlmevalkit_row["image"])].iloc[0]["image"]

        gym_row = {
            "responses_create_params": {
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{image}",
                                "detail": "high",
                            },
                            {
                                "type": "input_text",
                                "text": messages[-1]["value"],
                            },
                        ],
                    },
                ]
            },
            "answer": vlmevalkit_row["answer"],
            "category": vlmevalkit_row["category"],
            "benchmark_name": dataset_name,
            "group": group,
            "group_size": group_counts[group],
            # Choices is built here https://github.com/open-compass/VLMEvalKit/blob/00804217f868058f871f5ff252a7b9623c3475d9/vlmeval/dataset/utils/multiple_choice.py#L337
            "choices": build_choices(vlmevalkit_row),
        }
        f.write(orjson.dumps(gym_row) + b"\n")


if __name__ == "__main__":
    VlmEvalKitResourcesServer.setup_VLMEvalKit(None)

    # prepare_OCRBench()
    prepare_MMBench_DEV_EN_V11()
