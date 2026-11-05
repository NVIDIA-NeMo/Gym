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
"""The fault classes are one shared default in the failure vocabulary; a policy timeout's class is a run setting."""

import pytest

from nemo_gym.core import failure_kinds
from nemo_gym.orchestration import recovery
from nemo_gym.orchestration.recovery import (
    GateCriteria,
    fault_class,
    fault_report,
    gate_summary_from_collected,
    trial_verdict,
)
from tests.unit_tests.orchestration.test_eval_resubmit_decision import (  # noqa: F401
    complete_job,
    fake_executor,
    run_cli,
)
from tests.unit_tests.orchestration.test_evaluation_health import _gate, _run


def test_the_gate_uses_the_shared_class_lists():
    assert recovery.MODEL_FAULT_KINDS is failure_kinds.MODEL_FAULT_KINDS
    assert recovery.SERVER_FAULT_KINDS is failure_kinds.SERVER_FAULT_KINDS
    # The shared default covers what a policy-failure list counts as the policy's own.
    assert {
        "agent_timeout",
        "agent_command_not_allowed",
        "agent_limit",
        "agent_invalid_output",
        "model_context_limit",
    } <= failure_kinds.MODEL_FAULT_KINDS


def test_a_command_the_harness_refused_is_the_models_fault():
    from collections import Counter

    assert fault_class("agent_command_not_allowed") == "model"
    report = fault_report(Counter({"agent_command_not_allowed": 1}), 100)
    assert report["model"]["kinds"] == {"agent_command_not_allowed": 1}
    assert report["evaluation"]["count"] == 0


@pytest.mark.parametrize(
    "kind",
    [
        "truncated_answer",
        "empty_answer",
        "malformed_answer",
        "model_server_error",
        "non_policy_model_rejection",
        "invalid_output_budget",
        "invalid_request_parameter",
        "agent_sandbox_oom",
    ],
)
def test_the_gate_derived_classes_are_registered_but_no_producer_may_emit_them(kind):
    assert kind in failure_kinds.GATE_DERIVED_KINDS and failure_kinds.is_gate_derived(kind)
    assert kind not in failure_kinds.FAILURE_KINDS
    assert not failure_kinds.is_registered(kind)
    with pytest.raises(ValueError, match="derived by the acceptance gate"):
        failure_kinds.require_failure_kind(kind)


def _timed_out_run():
    """One trial's policy request timed out on every attempt; 99 others are graded."""
    return _run(99, settled={"model_policy_response_timeout": 1})


def test_a_policy_timeout_is_the_models_fault_by_default():
    collected = _timed_out_run()
    _, (passed, reason, report), task = _gate(collected)
    assert report["model"]["kinds"] == {"model_policy_response_timeout": 1}, reason
    assert report["evaluation"]["server"] == 0
    assert task["criteria"]["evaluation"]["policy_timeout_counts_as_model"] is True


def test_a_run_that_owns_its_serving_can_count_a_policy_timeout_as_a_server_fault():
    from nemo_gym.orchestration.recovery import decide_from_collected

    collected = _timed_out_run()
    decision = decide_from_collected(collected, max_infra_retries=1, max_walltime_s=None, live_job=False)
    passed, reason, report = trial_verdict(decision, collected, policy_timeout_counts_as_model=False)
    assert report["evaluation"]["kinds"] == {"model_policy_response_timeout": 1}, reason
    assert report["evaluation"]["server"] == 1 and report["model"]["count"] == 0
    assert "(server 1: model_policy_response_timeout 1)" in reason
    task = gate_summary_from_collected(
        decision,
        collected,
        max_unhealthy_fraction=None,
        criteria=GateCriteria(policy_timeout_counts_as_model=False),
    )
    assert task["evaluation_health"]["evaluation"]["server"] == 1
    assert task["criteria"]["evaluation"]["policy_timeout_counts_as_model"] is False


def test_the_setting_is_a_strict_boolean():
    with pytest.raises(ValueError):
        GateCriteria(policy_timeout_counts_as_model="no")


@pytest.mark.parametrize("value,expected", [("false", False), ("true", True)])
def test_the_recover_command_sets_it_on_the_gate(tmp_path, fake_executor, capsys, value, expected):
    fake_executor.job_dir = complete_job(tmp_path)
    _, out = run_cli(
        ["eval", "resubmit", "gym-job-k", "bench", "--recover", "--policy-timeout-counts-as-model", value], capsys
    )
    assert out["gate_task"]["criteria"]["evaluation"]["policy_timeout_counts_as_model"] is expected


def test_ungraded_harness_failure_is_evaluation_side_not_model_invalid_output():
    from collections import Counter

    report = fault_report(Counter({"agent_route_failed": 1, "agent_invalid_output": 1}), 2)
    assert report["evaluation"]["kinds"] == {"agent_route_failed": 1}
    assert report["model"]["kinds"] == {"agent_invalid_output": 1}


def _agent_oom_zero():
    """A saved explicit-zero policy result, with its admitted provider event and observation."""
    from nemo_gym.core import NgFailureEvent
    from nemo_gym.core.failure_records import SandboxFailureDetails
    from nemo_gym.core.failure_transport import FailureReportRow
    from nemo_gym.core.typed_exceptions import failure_report_for
    from nemo_gym.trajectory import NG_AGENT_SANDBOX_OBSERVATION_KEY, SandboxObservation
    from tests.unit_tests.test_evaluation_health import _empty

    collected = _run(1)
    row = collected.results[0]
    _empty(row)
    event = NgFailureEvent(
        boundary="sandbox",
        failure_kind="provider_oom_killed",
        failure_reason="Confirmed sandbox lifecycle OOMKilled",
        operation="pty",
        details=SandboxFailureDetails(provider="opensandbox", sandbox_id="agent-sandbox"),
    )
    report = failure_report_for(event).model_copy(update={"failure_id": None})
    observation = SandboxObservation(role="agent", phase="agent_execution", outcome="oom", failure_id=event.event_id)
    row.update(
        FailureReportRow.fields(report)
        | {
            "failure_kind": "provider_oom_killed",
            "agent_failure_action": "count_as_zero",
            "evaluation_completed": False,
            "mask_sample": False,
            "sandbox_handle": "agent-sandbox",
        }
    )
    row["response"]["output"] = []
    row["response"]["metadata"] = {NG_AGENT_SANDBOX_OBSERVATION_KEY: observation.model_dump_json()}
    return collected


def test_policy_counted_agent_oom_is_a_model_fault_with_a_sandbox_cause():
    from nemo_gym.orchestration.recovery import trial_failure_cause

    collected = _agent_oom_zero()
    decision, (passed, reason, report), task = _gate(collected)
    assert passed is True, reason
    assert report["model"]["kinds"] == {"agent_sandbox_oom": 1}
    assert report["evaluation"]["count"] == 0 and report["evaluation"]["server"] == 0
    assert decision.counts["scored"] == 1
    assert task["failures"]["metric_coverage"]["trials"] == 1
    assert task["failures"]["trial_failures"]["by_cause"]["sandbox"] == {"count": 1, "rate": 1.0}
    assert trial_failure_cause("agent_sandbox_oom") == "sandbox"
    # The canonical observation stays provider-neutral, and provider kinds alone remain evaluation-side.
    assert collected.results[0]["failure_kind"] == "provider_oom_killed"
    assert fault_class("provider_oom_killed") == "server"


@pytest.mark.parametrize(
    "change",
    [
        "missing_action",
        "fail",
        "verify",
        "retry",
        "nonzero_reward",
        "boolean_reward",
        "masked",
        "graded",
        "missing_completed",
        "verifier",
        "other_phase",
        "not_oom",
        "missing_event_id",
        "wrong_event_id",
        "selected_terminal_event",
        "omitted_report",
        "wrong_event_kind",
        "wrong_boundary",
        "missing_details",
        "missing_provider",
        "wrong_sandbox",
        "missing_sandbox",
        "conflicting_provider",
        "conflicting_sandbox",
        "nonempty_response",
        "invalid_observation",
    ],
)
def test_only_explicit_policy_zero_with_matching_agent_oom_evidence_is_model_side(change):
    import json

    from nemo_gym.trajectory import NG_AGENT_SANDBOX_OBSERVATION_KEY

    collected = _agent_oom_zero()
    row = collected.results[0]
    [event] = row["_ng_failure_events"].values()
    metadata = row["response"]["metadata"]
    observed = json.loads(metadata[NG_AGENT_SANDBOX_OBSERVATION_KEY])
    if change == "missing_action":
        row.pop("agent_failure_action")
    elif change in {"fail", "verify", "retry"}:
        row["agent_failure_action"] = change
    elif change == "nonzero_reward":
        row["reward"] = 1.0
    elif change == "boolean_reward":
        row["reward"] = False
    elif change == "masked":
        row["mask_sample"] = True
    elif change == "graded":
        row["evaluation_completed"] = True
    elif change == "missing_completed":
        row.pop("evaluation_completed")
    elif change == "verifier":
        observed["role"] = "verifier"
    elif change == "other_phase":
        observed["phase"] = "verification"
    elif change == "not_oom":
        observed["outcome"] = "sandbox_error"
    elif change == "missing_event_id":
        observed.pop("failure_id")
    elif change == "wrong_event_id":
        observed["failure_id"] = "missing-event"
    elif change == "selected_terminal_event":
        row["_ng_failure_id"] = event["event_id"]
    elif change == "omitted_report":
        row["_ng_failure_events_omitted"] = True
    elif change == "wrong_event_kind":
        event["failure_kind"] = "provider_unavailable"
    elif change == "wrong_boundary":
        event["boundary"] = "agent"
    elif change == "missing_details":
        event["details"] = None
    elif change == "missing_provider":
        event["details"]["provider"] = None
    elif change == "wrong_sandbox":
        row["sandbox_handle"] = "unrelated-sandbox"
    elif change == "missing_sandbox":
        row.pop("sandbox_handle")
    elif change == "conflicting_provider":
        observed["provider"] = "another-provider"
    elif change == "conflicting_sandbox":
        observed["sandbox_id"] = "another-sandbox"
    elif change == "nonempty_response":
        row["response"]["output"] = [{"type": "message", "role": "assistant", "content": []}]
    metadata[NG_AGENT_SANDBOX_OBSERVATION_KEY] = "invalid" if change == "invalid_observation" else json.dumps(observed)
    assert recovery._scored_kind(row) == "provider_oom_killed", change
    _, (_, _, report), _ = _gate(collected)
    if report is not None:
        assert report["model"]["count"] == 0, change
        assert report["evaluation"]["kinds"] == {"provider_oom_killed": 1}


def test_collector_round_trips_policy_zero_attribution_without_diagnostic_metadata(tmp_path):
    import json

    from nemo_gym.trajectory import NG_AGENT_SANDBOX_OBSERVATION_KEY
    from tests.unit_tests.test_eval_resubmit_decision import collect

    source = _agent_oom_zero()
    row = source.results[0]
    metadata = row["response"]["metadata"]
    observation = json.loads(metadata[NG_AGENT_SANDBOX_OBSERVATION_KEY])
    observation["error_message"] = "private-command-output-marker"
    metadata[NG_AGENT_SANDBOX_OBSERVATION_KEY] = json.dumps(observation)
    metadata["unrelated_metadata"] = "private-metadata-marker"
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    for name, rows in (
        ("rollouts_materialized_inputs.jsonl", source.materialized),
        ("rollouts.jsonl", source.results),
        ("rollouts_failures.jsonl", []),
    ):
        (artifacts / name).write_text("".join(json.dumps(item) + "\n" for item in rows))
    collected = collect(tmp_path)
    projected = collected.results[0]
    assert projected["agent_failure_action"] == "count_as_zero"
    assert projected["sandbox_handle"] == "agent-sandbox"
    projected_text = json.dumps(projected)
    assert "private-command-output-marker" not in projected_text
    assert "private-metadata-marker" not in projected_text
    decision, (passed, reason, report), task = _gate(collected)
    assert decision.counts == {"scored": 1}
    # This minimal directory has no quality/provenance report, so it cannot certify the whole run.
    assert passed is None and "UNVERIFIED" in reason
    assert report["model"]["kinds"] == {"agent_sandbox_oom": 1}
    assert report["evaluation"]["count"] == 0
    assert task["failures"]["metric_coverage"]["trials"] == 1
