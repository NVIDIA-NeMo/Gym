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
import time
from dataclasses import dataclass
from typing import Any, Dict, List
from uuid import uuid4

from openai.types.responses.response_input_text_param import ResponseInputTextParam

from nemo_gym.openai_utils import NeMoGymMessage, NeMoGymResponseOutputMessageForTraining, NeMoGymResponseOutputText


@dataclass
class MiniSWEAgentUtils:
    @staticmethod
    def chat_cmp_to_responses(messages: List[Dict[str, Any]], responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        nemo_gym_responses = []
        responses_idx = 0
        for message in messages:
            status = "completed"
            msg_type = "message"

            role = message["role"]
            content = message["content"]

            if role in ["user", "system"]:
                wrapped_message = NeMoGymMessage(
                    content=[
                        ResponseInputTextParam(
                            type="input_text",
                            text=content,
                        )
                    ],
                    role=role,
                    status=status,
                    type=msg_type,
                )
            elif role == "assistant":
                assistant_response = responses[responses_idx]
                provider_specific_fields = assistant_response.get("provider_specific_fields", {})
                prompt_token_ids = provider_specific_fields.get("prompt_token_ids", [])
                generation_token_ids = provider_specific_fields.get("generation_token_ids", [])
                generation_log_probs = provider_specific_fields.get("generation_log_probs", [])

                wrapped_message = NeMoGymResponseOutputMessageForTraining(
                    id=f"cht_{str(uuid4())}",
                    content=[
                        NeMoGymResponseOutputText(
                            annotations=[],
                            text=content,
                            type="output_text",
                            logprobs=None,
                        ),
                    ],
                    role=role,
                    status=status,
                    type=msg_type,
                    prompt_token_ids=prompt_token_ids,
                    generation_token_ids=generation_token_ids,
                    generation_log_probs=generation_log_probs,
                )
                responses_idx += 1

            nemo_gym_responses.append(wrapped_message.model_dump())

        return nemo_gym_responses

    @staticmethod
    def get_default_response_object() -> Dict[str, Any]:
        return {
            "id": f"resp_{str(uuid4())}",
            "created_at": int(time.time()),
            "error": None,
            "incomplete_details": None,
            "instructions": None,
            "metadata": {},
            "object": "response",
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "background": False,
            "max_output_tokens": None,
            "max_tool_calls": None,
            "previous_response_id": None,
            "prompt": None,
            "reasoning": {
                "effort": None,
                "generate_summary": None,
                "summary": None,
            },
            "service_tier": "default",
            "status": "completed",
            "text": {"format": {"type": "text"}, "verbosity": "medium"},
            "top_logprobs": 0,
            "truncation": "disabled",
            "usage": {
                "input_tokens": 0,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 0,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 0,
            },
            "user": None,
            "prompt_cache_key": None,
            "safety_identifier": None,
            "store": True,
        }

    @staticmethod
    def is_resolved(instance_id: str, eval_report: dict[str, Any]) -> float:
        try:
            if not eval_report:
                return False
            eval_report = eval_report["eval_report"][instance_id]
            resolved = eval_report["resolved"]
            if not eval_report.get("tests_status"):
                return False

            tests_status = eval_report["tests_status"]
            f2f = tests_status.get("FAIL_TO_PASS", {})
            p2p = tests_status.get("PASS_TO_PASS", {})
            f2f_success = len(f2f.get("success", []))
            f2f_failure = len(f2f.get("failure", []))
            p2p_success = len(p2p.get("success", []))
            p2p_failure = len(p2p.get("failure", []))

            if f2f_success == 0 and f2f_failure == 0 and p2p_success == 0 and p2p_failure == 0:
                return False
            return resolved
        except Exception as e:
            print(f"Error in is_resolved: {e}")
            return False
