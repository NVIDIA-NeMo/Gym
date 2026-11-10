# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Legacy consumers keep native identities before model-derived turns are joined."""

import json
from pathlib import Path

import pytest

from nemo_gym.rollouts import enrich_run_result
from nemo_gym.rollouts.harness_capabilities.behavior import tool_checks
from nemo_gym.trajectory import AgentInvocation, ToolCallObservation
from responses_api_agents.opencode_agent.artifacts import parse_opencode_observations
from responses_api_agents.opencode_agent.tests.test_app import _session_db


@pytest.mark.parametrize("scenario", ["tool_success", "tool_failure", "retry_429"])
@pytest.mark.parametrize("include_payloads", [True, False])
async def test_legacy_parser_joins_native_model_turns_without_message_headers(tmp_path, scenario, include_payloads):
    # Reduced provider/witness evidence from public conformance run 41811044198, Kilo 7.4.15.
    # The uploaded artifact has no SQLite file: reconstruct parts from explicit witness IDs,
    # arguments and results; do not infer a native/normalized alias in the implementation.
    fixtures = json.loads(Path(__file__).with_name("fixtures").joinpath("kilo_tool_witness.json").read_text())
    fixture = fixtures[scenario]
    captures, witness = fixture["captures"], fixture["witness"]
    assert all(call.get("client_assistant_message_id") is None for call in captures)
    session = next(call["client_session_id"] for call in captures if call.get("client_session_id"))
    parts = [
        {
            "type": "tool",
            "callID": item["id"],
            "tool": item["name"],
            "state": {
                "status": "completed",
                "input": item["arguments"],
                "output": item["outputs"][0],
                "metadata": {"exit": item["exit_code"]},
                "time": {"start": 1000, "end": 2000},
            },
        }
        for item in witness
    ]
    db = _session_db(tmp_path, [(session, "assistant", parts)], sessions=[(session, None)])
    observations = parse_opencode_observations(db, "0-0", require_terminal_finish=True)
    assert observations.schema_version == "1.0"
    capture_dir = tmp_path / "capture"
    capture_dir.mkdir()
    (capture_dir / "0-0.capture.jsonl").write_text("".join(json.dumps(call) + "\n" for call in captures))
    row = {"_ng_task_index": 0, "_ng_rollout_index": 0}
    result = {"ng_agent_observations": observations.model_dump(mode="json")}
    # Keep the actual failure shape: native model-derived turns already exist when
    # subsequent export omits captured bodies. Omission must not change their IDs.
    await enrich_run_result(row, result, capture_dirs=[capture_dir], observability_enabled=True)
    first = None
    for _ in range(3):
        await enrich_run_result(
            row, result, capture_dirs=[capture_dir], observability_enabled=True, include_payloads=include_payloads
        )
        checks = tool_checks(result, witness)
        assert all(check["status"] == "pass" for check in checks), checks
        trajectory = result["ng_trajectory"]
        tools = trajectory["tool_calls"]
        assert all(tool["timing_source"] == "artifact" for tool in tools)
        assert not any(gap["code"] == "tool_native_identity_unrepresentable" for gap in trajectory["gaps"])
        assert trajectory["turns"]
        assert all(turn["model_calls"] for turn in trajectory["turns"])
        if first is None:
            first = tools
        assert tools == first
        if not include_payloads:
            assert all(
                call.get("response") is None and call.get("request") is None for call in trajectory["model_calls"]
            )


@pytest.mark.parametrize("schema", ["1.0", "2.0"])
@pytest.mark.parametrize("native_id", [None, "native"])
def test_version_selects_partial_timing_and_missing_identity_before_references(tmp_path, schema, native_id):
    part = {"type": "tool", "tool": "bash", "state": {"status": "running", "time": {"start": 1000}}}
    if native_id:
        part["callID"] = native_id
    db = _session_db(tmp_path, [("assistant", [part])])
    bundle = parse_opencode_observations(db, "capture", observation_schema_version=schema)
    tools = [record for record in bundle.records if isinstance(record, ToolCallObservation)]
    invocation = next(record for record in bundle.records if isinstance(record, AgentInvocation))
    (request,) = [item for item in invocation.conversation if item.type == "function_call"]
    if schema == "1.0":
        assert request.call_id == native_id if native_id else request.call_id.startswith("call-")
        assert len(tools) == bool(native_id)
        assert all(tool.started_at is None and tool.duration_ms is None for tool in tools)
        assert "tool_timing_unavailable" in {gap.code for gap in bundle.gaps}
    else:
        assert request.call_id == tools[0].tool_call_id == "call:p0"
        assert tools[0].native_tool_call_id == native_id
        assert tools[0].timings[0].started_at == 1
        assert tools[0].timings[0].duration_ms is None
        assert "tool_completion_unavailable" in {gap.code for gap in bundle.gaps}


@pytest.mark.parametrize(
    "configured,selected,expected",
    [(None, None, "1.0"), ("2.0", None, "2.0"), ("2.0", "1.0", "1.0"), ("1.0", "2.0", "2.0")],
)
@pytest.mark.parametrize("envelope", ["direct", "episode_result", "episode_failure"])
async def test_raw_examples_honor_output_schema_without_enrichment(tmp_path, configured, selected, expected, envelope):
    from copy import deepcopy
    from types import SimpleNamespace
    from unittest.mock import patch

    from nemo_gym.rollouts import RolloutCollectionHelper
    from nemo_gym.trajectory import TrajectoryRecord
    from responses_api_agents.opencode_agent.artifacts import parse_opencode_session

    db = _session_db(
        tmp_path,
        [
            (
                "assistant",
                [
                    {
                        "type": "tool",
                        "callID": "native",
                        "tool": "bash",
                        "state": {
                            "status": "completed",
                            "input": {"command": "echo done"},
                            "output": "done",
                            "time": {"start": 1000, "end": 2000},
                        },
                    }
                ],
            )
        ],
    )
    trajectory = TrajectoryRecord(task_id="0", rollout_id="0-0")
    bundle = parse_opencode_observations(db, "0-0", trajectory, observation_schema_version="2.0")
    native_response, _ = parse_opencode_session(db)
    evidence = {
        "ng_agent_observations": bundle.model_dump(mode="json"),
        "ng_trajectory": trajectory.model_dump(mode="json"),
        "response": {"output": [item.model_dump(mode="json") for item in native_response]},
        "reward": 1,
    }
    if envelope == "direct":
        raw = evidence
    elif envelope == "episode_result":
        raw = {
            "episode_id": {"rollout_id": "0-0", "attempt": 0},
            "task_id": {"taskset": "test", "task_id": "0"},
            "result": evidence,
        }
    else:
        raw = {
            "episode_id": {"rollout_id": "0-0", "attempt": 0},
            "task_id": {"taskset": "test", "task_id": "0"},
            "failure": {
                "agent_observations": evidence["ng_agent_observations"],
                "trajectory": evidence["ng_trajectory"],
                "partial_response": evidence["response"],
            },
        }
    before = deepcopy(raw)
    config = {"model_call_capture": {"trajectory_schema_version": configured}} if configured else {}

    class Helper(RolloutCollectionHelper):
        def setup_server_client(self, *args):
            return SimpleNamespace(global_config_dict=config)

        def _run_examples_with_metadata(self, examples, *args, **kwargs):
            async def complete():
                return SimpleNamespace(row=examples[0], result=deepcopy(raw))

            yield complete()

    with patch(
        "nemo_gym.rollouts.rollout_collection.enrich_run_result", side_effect=AssertionError("must not enrich")
    ):
        _, result = await next(Helper().run_examples([{}], enrich_results=False, trajectory_schema_version=selected))
    assert raw == before
    if envelope == "episode_failure":
        actual = result["failure"]["agent_observations"]
        assert result["failure"]["partial_response"] == evidence["response"]
    else:
        returned = result if envelope == "direct" else result["result"]
        actual = returned["ng_agent_observations"]
        assert returned["response"] == evidence["response"] and returned["reward"] == 1
    tool = next(record for record in actual["records"] if record["kind"] == "tool_call")
    if expected == "2.0":
        assert result == before
        assert actual["schema_version"] == "2.0" and tool["native_tool_call_id"] == "native"
        assert tool["span_id"] and tool["timings"]
    else:
        assert "schema_version" not in actual
        assert tool["tool_call_id"] == "native"
        assert "span_id" not in tool and "timings" not in tool
        assert tool["duration_ms"] == 1000
