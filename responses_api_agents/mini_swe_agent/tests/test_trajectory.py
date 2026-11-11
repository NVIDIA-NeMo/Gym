# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import base64
import json

from nemo_gym.config import ModelServerRef
from nemo_gym.trajectory import AgentInvocation
from responses_api_agents.mini_swe_agent.trajectory import observation_records, original_response_id, project


MODEL = ModelServerRef(type="responses_api_models", name="policy_model")


def model_response(response_id: str, *, call_id: str | None = None, command: str = "ls", usage: bool = True) -> dict:
    """A Responses API object as mini-SWE 2.4.6's litellm_response model saves it."""
    output = [
        {
            "type": "message",
            "id": f"msg_{response_id}",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": f"thinking {response_id}", "annotations": []}],
        }
    ]
    if call_id is not None:
        output.append(
            {
                "type": "function_call",
                "id": f"fc_{call_id}",
                "call_id": call_id,
                "name": "bash",
                "arguments": json.dumps({"command": command}),
                "status": "completed",
            }
        )
    return {
        "id": response_id,
        "object": "response",
        "created_at": 1,
        "model": "served-model",
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": 10,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 5,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 15,
        }
        if usage
        else None,
    }


def step(response: dict, *actions: str) -> dict:
    return {**response, "extra": {"actions": [{"command": "ls", "tool_call_id": a} for a in actions], "timestamp": 7}}


def observation(call_id: str, output: str, *, returncode: int = 0, timed: bool = True, **extra) -> dict:
    timing = {"started_at": 100.0, "completed_at": 101.5, "duration_ms": 1500.0} if timed else {}
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": output,
        "extra": {"returncode": returncode, **timing, **extra},
    }


def native_trajectory() -> dict:
    """Two steps and a submission, with one response rejected for its format in between."""
    rejected = model_response("resp_rejected")
    return {
        "info": {"exit_status": "Submitted", "mini_version": "2.4.6"},
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Please solve this issue: fix it"},
            step(model_response("resp_1", call_id="call_1"), "call_1"),
            observation("call_1", "<returncode>0</returncode>", returncode=0),
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "format error"}],
                "extra": {"interrupt_type": "FormatError", "response": rejected},
            },
            step(
                model_response("resp_2", call_id="call_2", command="echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"),
                "call_2",
            ),
            {
                "role": "exit",
                "content": "",
                "extra": {
                    "exit_status": "Submitted",
                    "submission": "",
                    "tool_call_id": "call_2",
                    "output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n",
                    "started_at": 200.0,
                    "completed_at": 200.1,
                    "duration_ms": 100.0,
                },
            },
        ],
    }


def test_the_projection_keeps_every_model_call_and_replays_only_accepted_responses() -> None:
    projection = project(native_trajectory(), [])

    # A rejected response is a model call mini-SWE paid for, so it counts, but it is not replayed.
    assert [response.id for response in projection.responses] == ["resp_1", "resp_rejected", "resp_2"]
    third_request = projection.history[2]["input"]
    assert [item.get("id") or item.get("call_id") or item.get("role") for item in third_request] == [
        "system",
        "user",
        "msg_resp_1",
        "fc_call_1",
        "call_1",
        "user",
    ]
    assert projection.usage.total_tokens == 45
    types = [(item.type, getattr(item, "call_id", None)) for item in projection.output()]
    assert types == [
        ("message", None),
        ("function_call", "call_1"),
        ("function_call_output", "call_1"),
        ("message", None),
        ("message", None),
        ("function_call", "call_2"),
        # The runner keeps the submission command's output, which mini-SWE itself does not save.
        ("function_call_output", "call_2"),
    ]


def test_a_call_in_a_response_mini_swe_rejected_never_reaches_the_response_output() -> None:
    trajectory = native_trajectory()
    # mini-SWE runs only bash calls: a direct call to another tool fails the format check and never runs.
    rejected = model_response("resp_rejected", call_id="call_weather")
    rejected["output"][1]["name"] = "get_weather"
    trajectory["messages"][4]["extra"]["response"] = rejected

    projection = project(trajectory, [])

    items = projection.output()
    assert "call_weather" not in [getattr(item, "call_id", None) for item in items]
    assert "msg_resp_rejected" in [item.id for item in items if item.type == "message"]
    # The call was made and billed, so it keeps its model-call reference and its usage.
    assert projection.usage.total_tokens == 45
    invocation = AgentInvocation(invocation_id="r")
    observation_records(projection, invocation=invocation, model_server=MODEL, sandbox_id=None, stopped=None)
    assert [ref.response_id for ref in invocation.model_calls] == ["resp_1", "resp_rejected", "resp_2"]


def test_tool_status_and_timing_come_from_the_saved_observations() -> None:
    trajectory = native_trajectory()
    trajectory["messages"][3] = observation("call_1", "boom", returncode=2, timed=False, timestamp=150.0)
    projection = project(trajectory, [])

    first, submitted = projection.tools
    assert (first.status, first.duration_ms, first.completed_at) == ("failed", None, 150.0)
    # mini-SWE saves no observation for the submission; the exit message carries its output and timing.
    assert (submitted.status, submitted.duration_ms) == ("completed", 100.0)
    assert submitted.output == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n"

    invocation = AgentInvocation(invocation_id="rollout-1")
    records, gaps = observation_records(
        projection, invocation=invocation, model_server=MODEL, sandbox_id="box", stopped=None
    )
    assert [gap.code for gap in gaps] == ["tool_timing_unavailable"]
    assert [ref.response_id for ref in invocation.model_calls] == ["resp_1", "resp_rejected", "resp_2"]
    assert [(record.tool_call_id, record.status, record.timing_source) for record in records] == [
        ("call_1", "failed", "artifact"),
        ("call_2", "completed", "artifact"),
    ]


def test_a_timed_out_command_and_a_stopped_run_mark_their_tool_calls() -> None:
    trajectory = native_trajectory()
    trajectory["messages"][3] = observation("call_1", "", returncode=-1, exception_type="TimeoutExpired")
    # The run was stopped while the submission command was running: mini-SWE saved no exit message.
    del trajectory["messages"][-1]
    projection = project(trajectory, [])

    records, _ = observation_records(
        projection,
        invocation=AgentInvocation(invocation_id="r"),
        model_server=MODEL,
        sandbox_id=None,
        stopped="timeout",
    )
    assert [(record.status, record.error_type) for record in records] == [
        ("timeout", "TimeoutExpired"),
        ("timeout", "invocation_timeout"),
    ]


def test_usage_stays_unknown_when_one_model_call_reported_none() -> None:
    trajectory = native_trajectory()
    trajectory["messages"][4]["extra"]["response"] = model_response("resp_rejected", usage=False)

    assert project(trajectory, []).usage is None


def test_usage_stays_unknown_when_a_model_call_saved_no_response() -> None:
    # mini-SWE counts a model call before making it; a stop or a failure during the call leaves no response.
    trajectory = native_trajectory()
    trajectory["info"]["model_stats"] = {"api_calls": 3}
    assert project(trajectory, []).usage.total_tokens == 45

    trajectory["info"]["model_stats"] = {"api_calls": 4}
    assert project(trajectory, []).usage is None


def test_a_malformed_saved_response_stops_the_projection() -> None:
    trajectory = native_trajectory()
    trajectory["messages"][5] = step({**model_response("resp_2"), "output": "not a list"}, "call_2")

    projection = project(trajectory, [])

    assert projection.invalid_response
    assert [response.id for response in projection.responses] == ["resp_1", "resp_rejected"]


def test_litellm_response_id_envelopes_are_undone_to_join_captured_calls() -> None:
    wrapped = "resp_" + base64.b64encode(b"litellm:custom_llm_provider:openai;response_id:resp_gym_42").decode()

    assert original_response_id(wrapped) == "resp_gym_42"
    assert original_response_id("resp_gym_42") == "resp_gym_42"
    assert original_response_id("resp_not-base64!") == "resp_not-base64!"
    trajectory = native_trajectory()
    trajectory["messages"][2] = step(model_response(wrapped, call_id="call_1"), "call_1")
    assert project(trajectory, []).responses[0].id == "resp_gym_42"


def test_mcp_calls_follow_the_bash_call_that_made_them() -> None:
    calls = [
        {
            "tool_call_id": "call_1",
            "server": "task_tools",
            "tool": "get_weather",
            "arguments": {"city": "SF"},
            "started_at": 100.2,
            "completed_at": 100.4,
            "duration_ms": 200.0,
            "is_error": False,
            "output": '{"content": [{"type": "text", "text": "sunny"}]}',
        },
        # A call from a process the model left running in the background, outside any bash call.
        {
            "tool_call_id": None,
            "server": "task_tools",
            "tool": "ping",
            "arguments": {},
            "is_error": True,
            "output": "x",
        },
    ]
    projection = project(native_trajectory(), calls)

    items = projection.output()
    names = [(item.type, getattr(item, "name", None), getattr(item, "call_id", None)) for item in items]
    assert names[1:5] == [
        ("function_call", "bash", "call_1"),
        ("function_call", "mcp__task_tools__get_weather", "call_1-mcp-0"),
        ("function_call_output", None, "call_1-mcp-0"),
        ("function_call_output", None, "call_1"),
    ]
    assert json.loads(items[2].arguments) == {"city": "SF"}
    assert names[-2:] == [
        ("function_call", "mcp__task_tools__ping", "mcp-mcp-1"),
        ("function_call_output", None, "mcp-mcp-1"),
    ]
    invocation = AgentInvocation(invocation_id="r")
    records, _ = observation_records(
        projection, invocation=invocation, model_server=MODEL, sandbox_id=None, stopped=None
    )
    # The model requested the bash calls; an MCP call is part of one, not a tool call record of its own.
    assert [r.tool_name for r in records] == ["bash", "bash"]
    assert not any("mcp__" in str(item) for item in invocation.conversation)


def test_no_trajectory_projects_to_nothing() -> None:
    projection = project(None, [])

    assert projection.responses == [] and projection.output() == [] and projection.usage is None
