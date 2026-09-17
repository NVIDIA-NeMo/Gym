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

from pytest import mark

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.prompt import PromptConfig
from nemo_gym.vision_input import build_image_input


@mark.parametrize("system", [None, "Answer carefully."])
@mark.parametrize("image", ["", "aGVsbG8=", "data:image/jpeg;base64,aGVsbG8="])
def test_materialized_image_input(system, image):
    prompt = PromptConfig(system=system, user="Question: {question}")
    messages = build_image_input(prompt, "What is shown?", image)
    text = "Question: What is shown?"
    expected_content = text
    if image:
        expected_content = [
            {"type": "input_text", "text": text},
            {
                "type": "input_image",
                "image_url": image if image.startswith("data:") else "data:image/png;base64,aGVsbG8=",
                "detail": "high",
            },
        ]
    expected = [{"role": "user", "content": expected_content}]
    if system is not None:
        expected.insert(0, {"role": "system", "content": system})
    assert messages == expected
    NeMoGymResponseCreateParamsNonStreaming(input=messages)
