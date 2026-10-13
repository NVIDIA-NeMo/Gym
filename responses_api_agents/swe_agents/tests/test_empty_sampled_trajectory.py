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
from copy import deepcopy

import pytest

from responses_api_agents.swe_agents.app import SWEBenchWrapper
from responses_api_models.vllm_model.app import VLLMConverter, split_responses_input_output_items


@pytest.mark.parametrize("content", [None, ""])
@pytest.mark.parametrize("generation_ids", [[11], [2], [12, 13, 11]])
@pytest.mark.parametrize("provider_fields", [False, True])
def test_empty_sampled_final_turn_preserves_training_metadata(content, generation_ids, provider_fields):
    prior = {
        "role": "assistant",
        "content": "prior answer",
        "prompt_token_ids": [10],
        "generation_token_ids": [20],
        "generation_log_probs": [-0.3],
    }
    history = [{"role": "user", "content": "task"}, prior, {"role": "user", "content": "continue"}]
    token_fields = {
        "prompt_token_ids": [10, 20, 30],
        "generation_token_ids": generation_ids,
        "generation_log_probs": [-0.2] * len(generation_ids),
    }
    final = {"role": "assistant", "content": content}
    data = {"messages": deepcopy(history), "response": {"choices": [{"message": final}]}}
    if provider_fields:
        data["provider_specific_fields"] = token_fields
    else:
        final.update(token_fields)

    messages, tools = SWEBenchWrapper._materialize_trajectory(data)

    assert tools == []
    assert messages[:-1] == history
    assert messages[-1]["content"] == content
    assert {key: messages[-1][key] for key in token_fields} == token_fields
    # Exercise the downstream converter used by the agent, not only the append.
    converter = VLLMConverter(return_token_id_information=True)
    _, output = split_responses_input_output_items(converter.chat_completions_messages_to_responses_items(messages))
    generated = [item.model_dump() for item in output if getattr(item, "generation_token_ids", None)]
    assert [item["generation_token_ids"] for item in generated] == [[20], generation_ids]
    assert generated[-1]["generation_log_probs"] == token_fields["generation_log_probs"]
    assert generated[-1]["prompt_token_ids"] == token_fields["prompt_token_ids"]
    # The next prompt remains an exact extension of the preceding sampled turn.
    assert (
        generated[1]["prompt_token_ids"][:2] == generated[0]["prompt_token_ids"] + generated[0]["generation_token_ids"]
    )


@pytest.mark.parametrize("content", [None, ""])
@pytest.mark.parametrize("generation_ids", [None, []])
def test_empty_unsampled_response_does_not_create_a_turn(content, generation_ids):
    history = [{"role": "user", "content": "task"}]
    final = {"role": "assistant", "content": content}
    if generation_ids is not None:
        # Empty token bundles also represent no sampled generation.
        final.update(prompt_token_ids=[], generation_token_ids=generation_ids, generation_log_probs=[])
    messages, tools = SWEBenchWrapper._materialize_trajectory(
        {"messages": history, "response": {"choices": [{"message": final}]}}
    )
    assert messages == history
    assert tools == []


@pytest.mark.parametrize("content", [None, ""])
def test_empty_provider_token_bundle_does_not_create_a_turn(content):
    history = [{"role": "user", "content": "task"}]
    messages, _ = SWEBenchWrapper._materialize_trajectory(
        {
            "messages": history,
            "response": {"choices": [{"message": {"role": "assistant", "content": content}}]},
            "provider_specific_fields": {
                "prompt_token_ids": [10],
                "generation_token_ids": [],
                "generation_log_probs": [],
            },
        }
    )
    assert messages == history


@pytest.mark.parametrize(
    "final",
    [
        {"role": "assistant", "content": "answer"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "finish", "type": "function", "function": {"name": "finish", "arguments": "{}"}}],
        },
    ],
)
def test_nonempty_content_and_tool_calls_remain_in_trajectory(final):
    messages, _ = SWEBenchWrapper._materialize_trajectory({"response": {"choices": [{"message": final}]}})
    assert messages == [final]


def test_missing_response_keeps_existing_history():
    history = [{"role": "user", "content": "task"}]
    assert SWEBenchWrapper._materialize_trajectory({"messages": history}) == (history, [])


@pytest.mark.parametrize("inline_ids,provider_ids", [([99], [11]), ([99], []), ([], [11])])
def test_provider_generation_metadata_controls_final_turn_retention(inline_ids, provider_ids):
    final = {
        "role": "assistant",
        "content": None,
        "prompt_token_ids": [1],
        "generation_token_ids": inline_ids,
        "generation_log_probs": [-1.0] * len(inline_ids),
    }
    provider_fields = {
        "prompt_token_ids": [10],
        "generation_token_ids": provider_ids,
        "generation_log_probs": [-0.2] * len(provider_ids),
    }
    messages, _ = SWEBenchWrapper._materialize_trajectory(
        {"response": {"choices": [{"message": final}]}, "provider_specific_fields": provider_fields}
    )
    if provider_ids:
        assert len(messages) == 1
        assert {key: messages[0][key] for key in provider_fields} == provider_fields
    else:
        assert messages == []
