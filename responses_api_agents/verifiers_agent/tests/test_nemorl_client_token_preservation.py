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

from verifiers.clients.nemorl_chat_completions_client import _attach_trajectory_tokens_to_prompt
from verifiers.types import AssistantMessage, UserMessage


def test_verifiers_pr_1231_attaches_trajectory_tokens_to_prior_assistant_messages() -> None:
    """Guard PrimeIntellect-ai/verifiers#1231 behavior used by multi-turn NeMo RL training."""
    assistant = AssistantMessage(content="turn0")
    prompt = [UserMessage(content="u0"), assistant]
    state = {
        "trajectory": [
            {
                "tokens": {
                    "prompt_ids": [1, 2],
                    "completion_ids": [3, 4],
                    "completion_logprobs": [-0.1, -0.2],
                }
            }
        ]
    }

    _attach_trajectory_tokens_to_prompt(prompt, state)

    assert assistant.prompt_token_ids == [1, 2]
    assert assistant.generation_token_ids == [3, 4]
    assert assistant.generation_log_probs == [-0.1, -0.2]


def test_verifiers_pr_1231_pairs_tokens_with_last_assistant_messages() -> None:
    """PR #1231 ignores leading few-shot assistants when fewer trajectory steps exist."""
    few_shot = AssistantMessage(content="few-shot")
    assistant = AssistantMessage(content="turn0")
    prompt = [UserMessage(content="demo"), few_shot, UserMessage(content="u0"), assistant]
    state = {
        "trajectory": [
            {
                "tokens": {
                    "prompt_ids": [7],
                    "completion_ids": [8],
                    "completion_logprobs": [-0.3],
                }
            }
        ]
    }

    _attach_trajectory_tokens_to_prompt(prompt, state)

    assert not hasattr(few_shot, "prompt_token_ids") or few_shot.prompt_token_ids is None
    assert assistant.prompt_token_ids == [7]
    assert assistant.generation_token_ids == [8]
    assert assistant.generation_log_probs == [-0.3]


def test_verifiers_pr_1231_helper_exists_with_clear_message() -> None:
    """Fail explicitly if the installed verifiers package predates PR #1231."""
    assert callable(_attach_trajectory_tokens_to_prompt), (
        "Installed verifiers package is missing PrimeIntellect-ai/verifiers#1231 "
        "NeMoRL trajectory token preservation support."
    )
