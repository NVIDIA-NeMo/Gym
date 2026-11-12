# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hand-authored evidence contracts, independent of harnesses and collector code.

Keep the representations separate so mutations can expose conflicting copies.
These records specify checker inputs; they make no claims about any producer.
"""

from copy import deepcopy


def evidence_record() -> dict:
    """Return two synthetic policy steps with calls, executed tools and zero reward."""
    model_ref = {"type": "responses_api_models", "name": "synthetic-model"}
    conversation = [{"role": "user", "content": "Read two values."}]
    captures, calls, turns, tools = [], [], [], []
    for number in (1, 2):
        reference = {"model_call_id": f"attempt-{number}", "response_id": f"response-{number}", "model_ref": model_ref}
        request = {"input": deepcopy(conversation)}
        tool_request = {
            "type": "function_call",
            "call_id": f"tool-{number}",
            "name": "read_value",
            "arguments": '{"key": "example"}',
        }
        tool_result = {"type": "function_call_output", "call_id": f"tool-{number}", "output": "value"}
        response = {
            "id": reference["response_id"],
            "status": "completed",
            "output": [tool_request],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "input_tokens_details": {"cached_tokens": 2},
                "output_tokens_details": {"reasoning_tokens": 3},
            },
        }
        metadata = {
            "model_ref": model_ref,
            "model": "synthetic-model",
            "response_id": reference["response_id"],
            "dialect": "responses",
            "status_code": 200,
            "response_status": "completed",
            "finish_reason": None,
            "error_category": None,
            "client_session_id": "invocation",
        }
        captures.append(
            {
                **reference,
                **metadata,
                "call_index": number - 1,
                "tokens_in": 10,
                "tokens_out": 5,
                "tokens_total": 15,
                "tokens_reasoning": 3,
                "cached_tokens": 2,
            }
        )
        calls.append(
            {
                "model_call_id": reference["model_call_id"],
                "started_at": float(number),
                "completed_at": float(number) + 0.5,
                "response_metadata": {key: value for key, value in metadata.items() if key != "client_session_id"},
                "request": request,
                "response": response,
                "token_stats": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                    "reasoning_tokens": 3,
                    "cached_tokens": 2,
                },
            }
        )
        turns.append(
            {
                "invocation_id": "invocation",
                "task_id": "0",
                "rollout_id": "0-0",
                "turn_no": number,
                "step_count": number,
                "timestamp": float(number),
                "question": deepcopy(conversation),
                "answer": [tool_request],
                "resolved": None,
                "model_calls": [reference],
            }
        )
        tools.append(
            {
                "kind": "tool_call",
                "invocation_id": "invocation",
                "tool_call_id": f"tool-{number}",
                "tool_name": "read_value",
                "status": "completed",
                "output": "value",
            }
        )
        conversation.extend([tool_request, tool_result])
    invocation = {
        "kind": "agent_invocation",
        "invocation_id": "invocation",
        "parent_invocation_id": None,
        "model_calls": [deepcopy(turn["model_calls"][0]) for turn in turns],
        "conversation": conversation,
    }
    return {
        "_ng_task_index": 0,
        "_ng_rollout_index": 0,
        "reward": 0.0,
        "evaluation_completed": True,
        "mask_sample": False,
        "response": {"output": deepcopy(conversation[1:])},
        "ng_model_call_capture": {"rollout_id": "0-0", "calls": deepcopy(captures)},
        "ng_agent_observations": {
            "records": deepcopy(
                [invocation, *[{key: value for key, value in tool.items() if key != "output"} for tool in tools]]
            ),
            "gaps": [],
        },
        "ng_trajectory": {
            "schema_version": "1.0",
            "task_id": "0",
            "rollout_id": "0-0",
            "gaps": [],
            "invocations": [deepcopy(invocation)],
            "model_calls": deepcopy(calls),
            "turns": deepcopy(turns),
            "tool_calls": deepcopy(tools),
        },
    }


def evidence_and_witness() -> tuple[dict, dict]:
    """Return independent saved evidence and matching controlled observations."""
    record = evidence_record()
    witness = {
        "seeded": 1,
        "finished": True,
        "violations": [],
        "verifications": [{"reward": 0.0, "answer_seen": True}],
        "tool_calls": [
            {
                "id": f"tool-{number}",
                "name": "read_value",
                "arguments": {"key": "example"},
                "exit_code": 0,
                "outputs": ["value"],
                "executed": True,
                "result_seen": True,
            }
            for number in (1, 2)
        ],
        "attempts": [
            {"request": c["request"], "response": c["response"], "status_code": c["response_metadata"]["status_code"]}
            for c in record["ng_trajectory"]["model_calls"]
        ],
    }
    return deepcopy(record), witness
