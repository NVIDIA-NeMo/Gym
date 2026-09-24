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

"""Shared prompt and image input construction for HLE datasets."""

from nemo_gym.prompt import PromptConfig, fill_prompt


def _to_data_uri(image: str) -> str:
    """Normalize an HLE image field into a data URI usable as an ``image_url``.

    The ``cais/hle`` image column stores a base64 data URI string (e.g.
    ``data:image/png;base64,...``) for image questions. If a raw base64 payload
    is encountered instead, wrap it with a PNG data-URI prefix.
    """
    if image.startswith("data:"):
        return image
    return f"data:image/png;base64,{image}"


def build_image_input(prompt_config: PromptConfig, question: str, image: str) -> list[dict]:
    """Build a materialized ``responses_create_params.input`` for one row.

    Applies the shared prompt template, then (for image questions) rewrites the
    user turn into a multimodal content list with an ``input_image`` block.
    """
    messages = fill_prompt(prompt_config, {"question": question})
    if not image:
        return messages

    # Rewrite the (string) user turn into a multimodal content list so the image
    # travels alongside the question text.
    for msg in messages:
        if msg["role"] == "user":
            msg["content"] = [
                {"type": "input_text", "text": msg["content"]},
                {"type": "input_image", "image_url": _to_data_uri(image), "detail": "high"},
            ]
            break
    return messages
