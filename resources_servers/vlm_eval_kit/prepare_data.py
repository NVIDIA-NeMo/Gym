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
import orjson
from vlmeval.dataset.image_vqa import OCRBench


def load_and_dump(dataset_cls, dataset_name: str):
    data = dataset_cls(dataset=dataset_name).load_data(dataset_name)
    print(data.head())

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
            "eval_fn": "_score_OCRBench",
        }
        f.write(orjson.dumps(gym_row) + b"\n")


if __name__ == "__main__":
    load_and_dump(OCRBench, "OCRBench")
