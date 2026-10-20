# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""/run body in compaction mode: ``responses`` attached, or the row masked (imports app.py: slow)."""

import json

from responses_api_agents.gdpval_sandboxed_agent import app as agent_app
from responses_api_agents.gdpval_sandboxed_agent.tests import test_compaction as fixtures


def _body():
    return {
        "task_id": "task-7",
        "responses_create_params": {"input": []},
        "response": {"id": "gdpval-x", "output": [{"type": "message"}]},
        "policy_failure_reason": None,
        "agent_finished": True,
        "incomplete_reason": None,
        "agent_error": None,
    }


def test_segments_are_attached_to_the_run_body(tmp_path):
    fixtures._write(tmp_path, fixtures.compacting_rollout())
    body = _body()
    agent_app.GDPValSandboxedAgent._attach_segment_responses(body, tmp_path, model="gdpval-policy", session_id="task-7")
    assert body["agent_finished"] is True and body["response"]["id"] == "gdpval-x"
    assert [r["metadata"]["segment_boundary_reason"] for r in body["responses"]] == ["", "compaction", "post_compaction"]
    assert all(r["metadata"]["session_id"] == "task-7" for r in body["responses"])
    assert json.loads(body["responses"][1]["metadata"]["request_ids"]) == ["chatcmpl-2"]


def test_unreconstructable_segments_mask_the_row(tmp_path):
    payloads = fixtures.compacting_rollout()
    payloads[1]["messages"] = [fixtures.SYSTEM, fixtures.TASK, fixtures._tool_message(0)]  # assistant turn missing
    fixtures._write(tmp_path, payloads)
    body = _body()
    agent_app.GDPValSandboxedAgent._attach_segment_responses(body, tmp_path, model="gdpval-policy", session_id=None)
    assert "responses" not in body
    assert body["agent_finished"] is False and body["incomplete_reason"] == "agent_exec_incomplete"
    assert body["agent_error"].startswith("segment export: segment 0: 1 assistant messages")
    assert body["response"]["output"] == [] and body["response"]["status"] == "incomplete"
    assert body["response"]["id"].startswith("gdpval-segments-")


def test_segments_read_the_downloaded_trace_layout(tmp_path):
    """The exact files ``_download_trace`` leaves behind: ``<local>/trace/completions/*.json``."""
    import zipfile

    archive = tmp_path / "trace.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for n, payload in enumerate(fixtures.compacting_rollout()):
            bundle.writestr(f"trace/completions/gdpval-policy-{payload['timestamp']}.{n}.json", json.dumps(payload))
        bundle.writestr("trace/agent-status.json", json.dumps({"agent_state": "finished", "finished": True}))
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            destination = tmp_path.joinpath(*member.filename.split("/"))
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(bundle.read(member))
    # Same root ``export_trajectory`` reads: ``local`` with the trace below it.
    assert list((tmp_path / "trace/completions").glob("*.json"))
    body = _body()
    agent_app.GDPValSandboxedAgent._attach_segment_responses(body, tmp_path, model="gdpval-policy", session_id="task-7")
    assert body["agent_finished"] is True and len(body["responses"]) == 3


def test_config_flag_defaults_off():
    fields = agent_app.GDPValSandboxedAgentConfig.model_fields
    assert fields["context_summarization_cutoff"].default is None
    assert fields["context_summarization_max_event_chars"].default == 10_000
