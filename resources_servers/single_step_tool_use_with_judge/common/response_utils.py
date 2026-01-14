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
from typing import Optional, Union

from nemo_gym.openai_utils import (
    NeMoGymContent,
    NeMoGymResponse,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)


def extract_tool_call_or_output_message_from_response(
    response: NeMoGymResponse,
) -> Optional[Union[NeMoGymResponseFunctionToolCall, NeMoGymResponseOutputMessage]]:
    result = None
    for output_item in response.output:
        if output_item.type == "function_call":
            return output_item

        elif output_item.type == "message" and output_item.role == "assistant" and result is None:
            if extract_output_text_from_content(output_item.content) is not None:
                result = output_item

    return result


def extract_output_text_from_response(response: NeMoGymResponse) -> Optional[NeMoGymResponseOutputText]:
    for output_item in response.output:
        if output_item.type == "message" and output_item.role == "assistant":
            output_text = extract_output_text_from_content(output_item.content)
            if output_text is not None:
                return output_text

    return None


def extract_output_text_from_content(content: list[NeMoGymContent]) -> Optional[NeMoGymResponseOutputText]:
    for content_item in content:
        if content_item.type == "output_text":
            return content_item

    return None


def extract_canonical_text(text: str) -> str:
    return text.strip().removeprefix("```json").removesuffix("```")
