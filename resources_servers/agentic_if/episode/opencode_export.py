# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenCode export projection for agentic IF evidence.

The projection of ``OpenCodeSandboxedAgent._opencode_export_to_output_items``
(``responses_api_agents/opencode_sandboxed_agent/app.py``), except that tool calls carry the arguments and results the
model saw.
"""

import json

from openai.types.responses import ResponseInputTextParam

from nemo_gym.core import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputItem,
)
from nemo_gym.responses_converter import ResponsesConverter
from resources_servers.agentic_if.episode.opencode_profile import model_visible_tool_input, model_visible_tool_output


def export_output(opencode_export: dict) -> list[NeMoGymResponseOutputItem]:
    """Convert an OpenCode session export to Responses API output items, in order."""
    messages = []
    for message in opencode_export["messages"]:
        if message["info"]["role"] == "user":
            message_parts = []
            for part in message["parts"]:
                if part["type"] != "text":
                    continue

                message_parts.append(ResponseInputTextParam(text=part["text"], type="input_text"))

            messages.append(NeMoGymEasyInputMessage(content=message_parts, role="user"))
        elif message["info"]["role"] == "assistant":
            converter = ResponsesConverter(return_token_id_information=True)
            for part in message["parts"]:
                if part["type"] == "text":
                    output_items = converter.postprocess_assistant_message_dict(
                        message_dict={
                            "content": part["text"],
                            "role": "assistant",
                        }
                    )
                    messages.extend(output_items)
                elif part["type"] == "reasoning":
                    output_items = converter.postprocess_assistant_message_dict(
                        message_dict={
                            "content": converter._wrap_reasoning_in_think_tags([part["text"]]),
                            "role": "assistant",
                        }
                    )
                    messages.extend(output_items)
                elif part["type"] == "tool":
                    state = part["state"]
                    messages.append(
                        NeMoGymResponseFunctionToolCall(
                            arguments=json.dumps(model_visible_tool_input(state)),
                            call_id=part["callID"],
                            name=part["tool"],
                        )
                    )
                    messages.append(
                        NeMoGymFunctionCallOutput(
                            call_id=part["callID"],
                            # A failed call (status "error", e.g. invalid arguments) has no output; the model was
                            # shown its error text, so record that.
                            output=str(model_visible_tool_output(state) or ""),
                        )
                    )
                elif part["type"] in ("step-finish", "step-start", "patch"):
                    pass
                else:
                    # Fail on part types this projection does not know rather than drop them.
                    raise NotImplementedError(part)
        else:
            raise NotImplementedError(message)

    return messages
