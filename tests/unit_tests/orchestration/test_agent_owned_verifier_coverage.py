# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""An agent that grades its own rollouts and names no resources server (verifiers_agent) owns its verifier.

Its trials are observed only through the grading record the agent saved, which names the agent. A run of such an agent
saved without that record keeps the verdict it had: the trial gate reads it as before.
"""

import pytest

from tests.unit_tests.orchestration.test_tool_trial_coverage import fixture
from tests.unit_tests.orchestration.test_verifier_recorded_trial_coverage import _gate


AGENT = "simple"


def _agent_owned(case="repeats", *, record="agent"):
    """The fixture as a self-grading agent saves it: no resources server, and per row either the agent's grading
    record (``record="agent"``), a record naming someone else, or no verifier invocation at all (an earlier run)."""
    collected = fixture(case)
    servers = collected.quality["provenance"]["servers"]
    servers[AGENT] |= {"implementation": "verifiers_agent", "resources_server": None}
    del servers["mcqa"]
    for row in collected.results:
        observed = row["ng_trial_observations"]
        if record is None:
            observed["verifier_invocations"] = []
            continue
        [verifier] = observed["verifier_invocations"]
        verifier["grading_operations"] = None
        verifier["grading_record"] = {
            "verifier": AGENT if record == "agent" else record,
            "score": row["reward"],
            "verdict": "scored",
            "result_binding_sha256": row["_ng_collected_grading_binding"],
        }
    return collected


def test_a_self_grading_agents_own_record_observes_its_trials():
    passed, reason, counts, covered = _gate(_agent_owned())
    assert covered is True and passed is True, reason
    assert counts == {"observed_healthy": 2, "unhealthy": 0, "unobserved": 0, "verifier_recorded": 2}


@pytest.mark.parametrize("record", ["mcqa", "another_agent"])
def test_a_record_that_does_not_name_the_self_grading_agent_stays_unobserved(record):
    passed, reason, counts, covered = _gate(_agent_owned(record=record))
    assert covered is False and passed is None, reason
    assert counts["verifier_recorded"] == 0 and counts["unobserved"] == 2


def _environment_scored(collected):
    """Each row with the environment's own reward record of the episode, as an environment that scores it saves it."""
    for row in collected.results:
        row["_ng_collected_grading_facts"] = {"environment_reward": row["reward"]}
    return collected


def test_a_self_grading_agents_trial_without_a_verifier_is_observed_through_the_environments_reward():
    # No verifier ran: the environment scored the episode and saved its reward record (a tau2-style agent).
    passed, reason, counts, covered = _gate(_environment_scored(_agent_owned(record=None)))
    assert covered is True and passed is True, reason
    assert counts["unobserved"] == 0


def test_an_environment_reward_beside_a_verifier_that_ran_does_not_observe_a_self_grading_agents_trial():
    # A verifier ran and saved no record naming the agent: the environment's reward does not stand in for it.
    passed, reason, counts, covered = _gate(_environment_scored(_agent_owned(record="mcqa")))
    assert covered is False and passed is None, reason


def test_a_self_grading_agents_trial_without_a_verifier_is_observed_through_its_rubrics_saved_score():
    # No verifier ran: the agent's own rubric saved its guardrail-gated score and the counts it follows from.
    collected = _agent_owned(record=None)
    for row in collected.results:
        row["_ng_collected_grading_facts"] = {
            "guarded_objectives": {
                "aa_headline": row["reward"],
                "objectives_passed": row["reward"],
                "objectives_total": 1.0,
                "guardrails_violated": 0.0,
                "guardrails_total": 1.0,
            }
        }
    passed, reason, counts, covered = _gate(collected)
    assert covered is True and passed is True, reason
    assert counts["unobserved"] == 0


# The verdict an earlier self-grading agent run had before agents could own their verifier, read from that code.
UNOBSERVED_REASON = (
    "no completed grading operation or grading record was recorded by the verifier (a resources server the run "
    "provenance does not name)"
)


@pytest.mark.parametrize(("case", "trials"), [("repeats", 2), ("healthy", 1), ("wrong", 1)])
def test_an_earlier_run_of_a_self_grading_agent_keeps_exactly_its_verdict(case, trials):
    """No verifier invocation was saved, as before this change: verdict, reason and counts are what they were."""
    passed, reason, counts, covered = _gate(_agent_owned(case, record=None))
    assert (passed, covered) == (None, False)
    assert reason == (
        f"trial: UNOBSERVED {trials} of {trials} expected trials: {UNOBSERVED_REASON}, so the run is unverified, not "
        f"healthy; observed healthy 0, unhealthy 0; known so far: evaluation-side faults: 0 of {trials} (0.00%) is "
        f"below 2%; model faults: 0 of {trials} (0.00%); unverified: this withholds a pass and is no health failure"
    )
    assert counts == {
        "observed_healthy": 0,
        "unhealthy": 0,
        "unobserved": trials,
        "verifier_recorded": 0,
        "unobserved_reason": UNOBSERVED_REASON,
    }


def test_an_agent_with_a_resources_server_is_not_its_own_verifier():
    """A record naming the agent does not stand in for the agent's resources server."""
    collected = fixture("repeats")
    for row in collected.results:
        [verifier] = row["ng_trial_observations"]["verifier_invocations"]
        verifier["grading_operations"] = None
        verifier["grading_record"] = {
            "verifier": AGENT,
            "score": row["reward"],
            "verdict": "scored",
            "result_binding_sha256": row["_ng_collected_grading_binding"],
        }
    passed, reason, counts, covered = _gate(collected)
    assert covered is False and passed is None, reason
    assert counts["verifier_recorded"] == 0
