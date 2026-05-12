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

from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseFunctionToolCall, NeMoGymResponseOutputText
from resources_servers.single_step_tool_use_with_argument_comparison.common.verification_utils import (
    ExpectedFunctionCall,
    ExpectedFunctionCallBatch,
)


def extract_tool_call_or_text(
    response: NeMoGymResponse,
) -> Optional[Union[NeMoGymResponseFunctionToolCall, ExpectedFunctionCallBatch, NeMoGymResponseOutputText]]:
    result = None
    tool_calls: list[NeMoGymResponseFunctionToolCall] = []
    for output_item in response.output:
        if output_item.type == "function_call":
            tool_calls.append(output_item)

        elif output_item.type == "message" and output_item.role == "assistant" and result is None:
            for content_item in output_item.content:
                if content_item.type == "output_text":
                    result = content_item
                    break

    if len(tool_calls) == 1:
        return tool_calls[0]

    if len(tool_calls) > 1:
        return ExpectedFunctionCallBatch(
            type="function_call_batch",
            calls=[
                ExpectedFunctionCall(
                    type="function_call",
                    name=tool_call.name,
                    arguments=tool_call.arguments,
                )
                for tool_call in tool_calls
            ],
        )

    return result
