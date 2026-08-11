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
from typing import Optional

from nemo_gym.openai_utils import NeMoGymResponse
from resources_servers.single_step_tool_use_with_argument_comparison.common.verification_utils import (
    ExpectedAction,
    FunctionCallAction,
    FunctionCallBatchAction,
    MessageAction,
)


def extract_action(response: NeMoGymResponse) -> Optional[ExpectedAction]:
    """Normalize a model response into the canonical action shape that dataset rows also use.

    Tool calls take precedence over assistant text, so a response that both narrates and calls tools is
    judged on the calls. Several tool calls in one response become a batch, which the comparator then
    matches without regard to the order they were emitted in.
    """
    tool_calls: list[FunctionCallAction] = []
    assistant_text: Optional[str] = None

    for output_item in response.output:
        if output_item.type == "function_call":
            tool_calls.append(
                FunctionCallAction(
                    type="function_call",
                    name=output_item.name,
                    arguments=output_item.arguments,
                )
            )

        elif output_item.type == "message" and output_item.role == "assistant" and assistant_text is None:
            for content_item in output_item.content:
                if content_item.type == "output_text":
                    assistant_text = content_item.text
                    break

    if len(tool_calls) == 1:
        return tool_calls[0]

    if tool_calls:
        return FunctionCallBatchAction(type="function_call_batch", calls=tool_calls)

    if assistant_text is not None:
        return MessageAction(type="message", content=assistant_text)

    return None
