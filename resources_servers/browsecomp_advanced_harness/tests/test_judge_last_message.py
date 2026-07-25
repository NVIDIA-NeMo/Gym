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
"""Part A: the judge grades only the LAST assistant message item, not the concatenation
of every assistant message (the old Response.output_text behavior, wrong ~49% of the time)."""

from nemo_gym.openai_utils import (
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from resources_servers.browsecomp_advanced_harness.app import _last_assistant_text


def _msg(text: str) -> NeMoGymResponseOutputMessage:
    return NeMoGymResponseOutputMessage(
        id="msg_id",
        content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
        role="assistant",
        status="completed",
        type="message",
    )


def _response(output: list) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="resp",
        created_at=0.0,
        model="m",
        object="response",
        output=output,
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )


def test_last_assistant_text_returns_final_message_not_concat():
    resp = _response(
        [
            _msg("Exact Answer: WRONG"),
            NeMoGymResponseFunctionToolCall(
                id="fc_001", call_id="c1", name="search", arguments="{}", type="function_call"
            ),
            NeMoGymFunctionCallOutput(type="function_call_output", call_id="c1", output="some tool result"),
            _msg("Exact Answer: RIGHT"),
        ]
    )
    # the fix: grade ONLY the last assistant message
    assert _last_assistant_text(resp) == "Exact Answer: RIGHT"
    # document the bug: stock output_text concatenates BOTH assistant messages (judge saw two answers)
    assert "WRONG" in resp.output_text and "RIGHT" in resp.output_text


def test_last_assistant_text_empty_when_no_assistant_message():
    resp = _response(
        [NeMoGymFunctionCallOutput(type="function_call_output", call_id="c1", output="only a tool result")]
    )
    assert _last_assistant_text(resp) == ""
