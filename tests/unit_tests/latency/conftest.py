# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixtures for the offline latency analysis: two synthetic rollout rows in the shape the collector writes.

``benchmarks/nemotron_3.5_super/analysis`` is not an importable package name (the directory has a dot), so the
analysis directory is put on ``sys.path`` here, the way ``timeline.py`` does for the command line.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import pytest


ANALYSIS_DIR = Path(__file__).resolve().parents[3] / "benchmarks" / "nemotron_3.5_super" / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

NODE = "node:nvl-1"
SANDBOX = "sandbox:sb-1"
T0 = 1_700_000_000.0


def _call(**kwargs: Any) -> dict[str, Any]:
    call = {
        "model_call_id": None,
        "client_call_id": None,
        "invocation_id": "root",
        "clock_domain": NODE,
        "started_at": None,
        "completed_at": None,
        "duration_ms": None,
        "processed_at": None,
        "server_started_at": None,
        "server_completed_at": None,
        "server_duration_ms": None,
        "status": "ok",
        "model_call_purpose": "agent_step",
        "model_response_kind": "text",
        "response_metadata": {"response_id": None, "finish_reason": "stop", "status_code": 200, "engine": None},
        "token_stats": {"prompt_tokens": 1000, "completion_tokens": 60, "reasoning_tokens": 20, "cached_tokens": 200},
        "attempts": [],
        "parsed_tool_calls": None,
        "task_complete": None,
        "context_overflow": None,
        "rejected_request_tokens": None,
    }
    call.update(kwargs)
    return call


def _engine(queue_ms: float, ttft_ms: float, generation_ms: float) -> dict[str, Any]:
    return {
        "queue_time_ms": queue_ms,
        "time_to_first_token_ms": ttft_ms,
        "generation_time_ms": generation_ms,
        "mean_itl_ms": 20.0,
        "tokens_per_second": 50.0,
        "speculative_decoding": {"num_drafts": 10, "num_accepted_tokens": 31},
    }


def terminus_row() -> dict[str, Any]:
    """One Terminus rollout: setup, two agent steps (the second retried once), two command batches, one proactive
    compaction, verify. Harness records on the node clock; server records on the same node clock."""
    t = T0
    calls = [
        _call(  # step 1
            model_call_id="cap-1",
            client_call_id="t2-1",
            started_at=t + 20.0,
            completed_at=t + 30.0,
            duration_ms=10_000.0,
            processed_at=t + 30.2,
            server_started_at=t + 20.01,
            server_completed_at=t + 29.99,
            server_duration_ms=9_980.0,
            parsed_tool_calls=2,
            task_complete=False,
            response_metadata={
                "response_id": "resp_1",
                "finish_reason": "stop",
                "status_code": 200,
                "engine": _engine(50.0, 1_050.0, 8_800.0),
            },
            attempts=[
                {
                    "attempt_index": 1,
                    "started_at": t + 20.0,
                    "completed_at": t + 30.0,
                    "duration_ms": 10_000.0,
                    "status": "completed",
                    "status_code": 200,
                    "clock_domain": NODE,
                }
            ],
        ),
        _call(  # compaction summary call inside the compaction
            model_call_id="cap-2",
            client_call_id="t2-2",
            started_at=t + 40.0,
            completed_at=t + 45.0,
            duration_ms=5_000.0,
            server_started_at=t + 40.01,
            server_completed_at=t + 44.99,
            server_duration_ms=4_980.0,
            model_call_purpose="compaction_summary",
            response_metadata={
                "response_id": "resp_2",
                "finish_reason": "stop",
                "status_code": 200,
                "engine": _engine(10.0, 510.0, 4_400.0),
            },
            token_stats={"prompt_tokens": 9_000, "completion_tokens": 400, "reasoning_tokens": 0, "cached_tokens": 0},
        ),
        _call(  # step 2: a timed-out attempt then a successful one; declares the task complete
            model_call_id="cap-3",
            client_call_id="t2-3",
            started_at=t + 46.0,
            completed_at=t + 70.0,
            duration_ms=24_000.0,
            processed_at=t + 70.1,
            server_started_at=t + 60.01,
            server_completed_at=t + 69.99,
            server_duration_ms=9_980.0,
            parsed_tool_calls=1,
            task_complete=True,
            response_metadata={
                "response_id": "resp_3",
                "finish_reason": "stop",
                "status_code": 200,
                "engine": _engine(20.0, 1_020.0, 8_900.0),
            },
            token_stats={
                "prompt_tokens": 2_000,
                "completion_tokens": 80,
                "reasoning_tokens": 30,
                "cached_tokens": 1_900,
            },
            attempts=[
                {
                    "attempt_index": 1,
                    "started_at": t + 46.0,
                    "completed_at": t + 60.0,
                    "duration_ms": 14_000.0,
                    "status": "timeout",
                    "error_type": "TimeoutError",
                    "clock_domain": NODE,
                },
                {
                    "attempt_index": 2,
                    "started_at": t + 60.0,
                    "completed_at": t + 70.0,
                    "duration_ms": 10_000.0,
                    "status": "completed",
                    "status_code": 200,
                    "clock_domain": NODE,
                },
            ],
        ),
    ]
    tools = [
        {
            "kind": "tool_call",
            "invocation_id": "root",
            "tool_call_id": "cmd_a",
            "tool_name": "terminal",
            "operation": "ls -la",
            "requested_at": t + 30.2,
            "started_at": t + 30.3,
            "completed_at": t + 31.3,
            "response_received_at": t + 32.5,
            "duration_ms": 1_000.0,
            "timing_source": "harness",
            "status": "completed",
            "clock_domain": NODE,
            "batch_id": "batch-1",
            "batch_index": 1,
            "requested_wait_s": 1.0,
            "output_trajectory_index": 3,
            "output_chars": 240,
        },
        {
            "kind": "tool_call",
            "invocation_id": "root",
            "tool_call_id": "cmd_b",
            "tool_name": "terminal",
            "operation": "pytest -q",
            "requested_at": t + 30.2,
            "started_at": t + 31.3,
            "completed_at": t + 32.4,
            "response_received_at": t + 32.5,
            "duration_ms": 1_100.0,
            "timing_source": "harness",
            "status": "completed",
            "clock_domain": NODE,
            "batch_id": "batch-1",
            "batch_index": 2,
            "requested_wait_s": 1.0,
            "output_trajectory_index": 3,
            "output_chars": 240,
        },
        {
            "kind": "tool_call",
            "invocation_id": "root",
            "tool_call_id": "cmd_c",
            "tool_name": "terminal",
            "operation": "",
            "requested_at": t + 70.1,
            "started_at": t + 70.2,
            "completed_at": t + 71.2,
            "response_received_at": t + 71.3,
            "duration_ms": 1_000.0,
            "timing_source": "harness",
            "status": "completed",
            "clock_domain": NODE,
            "batch_id": "batch-2",
            "batch_index": 1,
            "requested_wait_s": 1.0,
            "output_trajectory_index": 5,
            "output_chars": 10,
        },
    ]
    phases = [
        {
            "kind": "agent_phase",
            "name": "seed_session",
            "started_at": t + 1.0,
            "completed_at": t + 3.0,
            "duration_ms": 2_000.0,
            "status": "completed",
            "clock_domain": NODE,
        },
        {
            "kind": "agent_phase",
            "name": "connect_sandbox",
            "started_at": t + 3.0,
            "completed_at": t + 4.0,
            "duration_ms": 1_000.0,
            "status": "completed",
            "clock_domain": NODE,
        },
        {
            "kind": "agent_phase",
            "name": "agent_setup",
            "started_at": t + 4.0,
            "completed_at": t + 19.0,
            "duration_ms": 15_000.0,
            "status": "completed",
            "clock_domain": NODE,
        },
        {
            "kind": "agent_phase",
            "name": "agent_execute",
            "started_at": t + 4.0,
            "completed_at": t + 73.0,
            "duration_ms": 69_000.0,
            "status": "completed",
            "clock_domain": NODE,
        },
        {
            "kind": "agent_phase",
            "name": "verify",
            "started_at": t + 73.0,
            "completed_at": t + 93.0,
            "duration_ms": 20_000.0,
            "status": "completed",
            "clock_domain": NODE,
        },
        {
            "kind": "agent_phase",
            "name": "sandbox_stop",
            "started_at": t + 93.0,
            "completed_at": t + 94.0,
            "duration_ms": 1_000.0,
            "status": "completed",
            "clock_domain": NODE,
        },
    ]
    return {
        "_ng_task_index": 4,
        "_ng_rollout_index": 1,
        "task_id": "task-4",
        "reward": 1.0,
        "terminus2_completed": True,
        "model_call_times": [10.0, 5.0, 24.0],
        "command_exec_times": [2.1, 1.0],
        "terminus2_time_taken": 69.0,
        "usages": [
            {"input_tokens": 1000, "output_tokens": 60, "output_tokens_details": {"reasoning_tokens": 20}},
            {"input_tokens": 9000, "output_tokens": 400, "output_tokens_details": {"reasoning_tokens": 0}},
            {"input_tokens": 2000, "output_tokens": 80, "output_tokens_details": {"reasoning_tokens": 30}},
        ],
        "num_compactions": 0,
        "num_proactive_compactions": 1,
        "model_calls_gt_10min": 1,
        "response": {
            "output": [
                {"type": "message", "role": "user", "content": "task"},
                {"type": "message", "role": "assistant", "content": '{"commands": [], "task_complete": false}'},
                {"type": "message", "role": "user", "content": "summary request"},
                {"type": "message", "role": "user", "content": "x" * 240},
                {"type": "message", "role": "assistant", "content": '{"commands": [], "task_complete": true}'},
                {"type": "message", "role": "user", "content": "ten chars!"},
            ],
        },
        "ng_perf": {
            "total_latency_ms": 95_000.0,
            "rollout_started_at": t,
            "rollout_completed_at": t + 95.0,
            "rollout_duration_ms": 94_999.5,
            "num_model_calls": 3,
        },
        "ng_trajectory": {
            "task_id": "task-4",
            "rollout_id": "4-1",
            "invocations": [
                {
                    "invocation_id": "root",
                    "status": "completed",
                    "started_at": t + 19.0,
                    "completed_at": t + 72.0,
                    "duration_ms": 53_000.0,
                    "clock_domain": NODE,
                }
            ],
            "model_calls": calls,
            "tool_calls": tools,
            "compactions": [
                {
                    "invocation_id": "root",
                    "observed_at": t + 39.0,
                    "completed_at": t + 45.5,
                    "duration_ms": 6_500.0,
                    "trigger": "proactive",
                    "tokens_before": 1080,
                    "tokens_after": 2000,
                    "outcome": "completed",
                    "clock_domain": NODE,
                    "model_calls": [{"model_call_id": "cap-2"}],
                }
            ],
            "phases": phases,
            "gaps": [],
        },
        "ng_agent_observations": {
            "source": "terminus2",
            "records": [{"kind": "agent_invocation", "invocation_id": "root", "status": "completed"}],
            "gaps": [],
        },
    }


def opencode_row() -> dict[str, Any]:
    """One OpenCode rollout: two steps in the root session, a sub-agent session with one step, a bash tool and
    the task tool that spawned the sub-agent; artifact times on the sandbox clock, server times on the node."""
    t = T0
    offset = 0.3  # sandbox clock runs 0.3 s ahead of the node
    s = t + offset
    calls = [
        _call(  # root step 1: paired with capture cap-1; the send time is the plugin's
            model_call_id="cap-1",
            client_call_id="oc-1",
            invocation_id="root",
            clock_domain=SANDBOX,
            started_at=s + 25.0,
            completed_at=s + 33.0,
            processed_at=s + 33.6,
            server_started_at=t + 25.2,
            server_completed_at=t + 32.9,
            server_duration_ms=7_700.0,
            model_response_kind="tool_call",
            response_metadata={
                "response_id": "cc-1",
                "finish_reason": "tool_calls",
                "status_code": 200,
                "engine": _engine(5.0, 1_005.0, 6_500.0),
            },
        ),
        _call(  # sub-agent step
            model_call_id="cap-2",
            client_call_id="oc-2",
            invocation_id="child",
            clock_domain=SANDBOX,
            started_at=s + 35.0,
            completed_at=s + 40.0,
            processed_at=s + 40.2,
            server_started_at=t + 35.2,
            server_completed_at=t + 39.9,
            server_duration_ms=4_700.0,
            model_call_purpose="subagent_step",
            response_metadata={
                "response_id": "cc-2",
                "finish_reason": "stop",
                "status_code": 200,
                "engine": _engine(5.0, 505.0, 4_000.0),
            },
            token_stats={"prompt_tokens": 500, "completion_tokens": 30, "reasoning_tokens": 10, "cached_tokens": 0},
        ),
        _call(  # root step 2: final text
            model_call_id="cap-3",
            client_call_id="oc-3",
            invocation_id="root",
            clock_domain=SANDBOX,
            started_at=s + 61.0,
            completed_at=s + 66.0,
            processed_at=s + 66.4,
            server_started_at=t + 61.2,
            server_completed_at=t + 65.9,
            server_duration_ms=4_700.0,
            response_metadata={
                "response_id": "cc-3",
                "finish_reason": "stop",
                "status_code": 200,
                "engine": _engine(5.0, 1_005.0, 3_500.0),
            },
            token_stats={
                "prompt_tokens": 1_800,
                "completion_tokens": 40,
                "reasoning_tokens": 10,
                "cached_tokens": 1_700,
            },
        ),
    ]
    tools = [
        {
            "kind": "tool_call",
            "invocation_id": "root",
            "tool_call_id": "call-bash",
            "tool_name": "bash",
            "operation": "pytest -q",
            "requested_at": s + 33.1,
            "started_at": s + 33.1,
            "completed_at": s + 45.0,
            "response_received_at": s + 61.0,
            "duration_ms": 11_900.0,
            "timing_source": "artifact",
            "status": "completed",
            "clock_domain": SANDBOX,
        },
        {
            "kind": "tool_call",
            "invocation_id": "root",
            "tool_call_id": "call-task",
            "tool_name": "task",
            "operation": None,
            "requested_at": s + 33.2,
            "started_at": s + 33.2,
            "completed_at": s + 60.5,
            "response_received_at": s + 61.0,
            "duration_ms": 27_300.0,
            "timing_source": "artifact",
            "status": "completed",
            "clock_domain": SANDBOX,
        },
    ]
    phases = [
        {
            "kind": "agent_phase",
            "name": "seed_session",
            "started_at": t + 1.0,
            "completed_at": t + 3.0,
            "duration_ms": 2_000.0,
            "status": "completed",
            "clock_domain": NODE,
        },
        {
            "kind": "agent_phase",
            "name": "start_sandbox",
            "started_at": t + 3.0,
            "completed_at": t + 5.0,
            "duration_ms": 2_000.0,
            "status": "completed",
            "clock_domain": NODE,
        },
        {
            "kind": "agent_phase",
            "name": "agent_sandbox_exec",
            "started_at": t + 5.0,
            "completed_at": t + 80.0,
            "duration_ms": 75_000.0,
            "status": "completed",
            "clock_domain": NODE,
        },
        {
            "kind": "agent_phase",
            "name": "agent_export",
            "started_at": t + 80.0,
            "completed_at": t + 82.0,
            "duration_ms": 2_000.0,
            "status": "completed",
            "clock_domain": NODE,
        },
        {
            "kind": "agent_phase",
            "name": "verify",
            "started_at": t + 83.0,
            "completed_at": t + 110.0,
            "duration_ms": 27_000.0,
            "status": "completed",
            "clock_domain": NODE,
        },
        {
            "kind": "agent_phase",
            "name": "sandbox_stop",
            "started_at": t + 110.0,
            "completed_at": t + 111.0,
            "duration_ms": 1_000.0,
            "status": "completed",
            "clock_domain": NODE,
        },
    ]
    return {
        "_ng_task_index": 9,
        "_ng_rollout_index": 0,
        "instance_id": "repo__pkg-1",
        "reward": 0.0,
        "resolved": False,
        "opencode_finished": True,
        "response": {
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call-bash",
                    "name": "bash",
                    "arguments": '{"command": "pytest -q"}',
                },
                {"type": "function_call_output", "call_id": "call-bash", "output": "." * 300},
                {"type": "function_call", "call_id": "call-task", "name": "task", "arguments": '{"description": "x"}'},
                {"type": "function_call_output", "call_id": "call-task", "output": "found"},
            ],
            "usage": {"input_tokens": 900, "output_tokens": 100, "input_tokens_details": {"cached_tokens": 1_900}},
        },
        "ng_perf": {
            "total_latency_ms": 112_000.0,
            "rollout_started_at": t,
            "rollout_completed_at": t + 112.0,
            "rollout_duration_ms": 111_999.0,
            "num_model_calls": 3,
            "num_tool_calls": 2,
        },
        "ng_trajectory": {
            "task_id": "repo__pkg-1",
            "rollout_id": "9-0",
            "invocations": [
                {
                    "invocation_id": "root",
                    "status": "completed",
                    "started_at": s + 20.0,
                    "completed_at": s + 66.0,
                    "process_started_at": s + 8.0,
                    "process_exited_at": s + 70.0,
                    "clock_domain": SANDBOX,
                },
                {
                    "invocation_id": "child",
                    "parent_invocation_id": "root",
                    "spawned_by_tool_call_id": "call-task",
                    "status": "completed",
                    "started_at": s + 34.0,
                    "completed_at": s + 40.0,
                    "requested_at": s + 33.2,
                    "response_observed_at": s + 60.5,
                    "clock_domain": SANDBOX,
                },
            ],
            "model_calls": calls,
            "tool_calls": tools,
            "compactions": [],
            "phases": phases,
            "gaps": [{"code": "tool_response_boundary_approximate", "invocation_id": "root"}],
        },
        "ng_agent_observations": {
            "source": "opencode",
            "records": [
                {"kind": "agent_invocation", "invocation_id": "root", "status": "completed"},
                {
                    "kind": "sandbox",
                    "role": "agent",
                    "provider": "opensandbox",
                    "sandbox_id": "sb-1",
                    "outcome": "completed",
                    "clock_offset_s": offset,
                    "clock_offset_uncertainty_s": 0.04,
                },
            ],
            "gaps": [],
        },
    }


@pytest.fixture
def terminus_record() -> dict[str, Any]:
    return copy.deepcopy(terminus_row())


@pytest.fixture
def opencode_record() -> dict[str, Any]:
    return copy.deepcopy(opencode_row())
