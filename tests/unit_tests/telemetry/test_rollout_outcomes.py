# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``gym.rollout.completed_total``: the driver-side attempt counter, against a real in-memory reader."""

import pytest

from nemo_gym import rollout_collection
from nemo_gym.telemetry import gym_metrics
from nemo_gym.telemetry.span_groups import GymSpanGroup
from tests.unit_tests.telemetry.test_sandbox_active import collected_metrics  # noqa: F401 - fixture


pytest.importorskip("opentelemetry.sdk.metrics")

COMPLETED = gym_metrics.ROLLOUT_COMPLETED_INSTRUMENT
OUTCOME = gym_metrics.ROLLOUT_OUTCOME_ATTRIBUTE
DISPATCH = gym_metrics.DISPATCH_NAME_ATTRIBUTE
CLASS = gym_metrics.FAILURE_CLASS_ATTRIBUTE
KIND = gym_metrics.FAILURE_KIND_ATTRIBUTE
TYPE = gym_metrics.FAILURE_TYPE_ATTRIBUTE


def _by_attrs(collected):
    return {tuple(sorted(p.attributes.items())): p.value for p in collected().get(COMPLETED, [])}


def _key(**attrs):
    return tuple(sorted(attrs.items()))


def _rollout_group_only(monkeypatch):
    monkeypatch.setattr(rollout_collection, "is_span_group_enabled", lambda group: group == GymSpanGroup.ROLLOUT)


def test_outcomes_are_counted_with_bounded_failure_attributes(collected_metrics):  # noqa: F811
    gym_metrics.record_rollout_completed("scored", dispatch_name="env")
    gym_metrics.record_rollout_completed("scored", dispatch_name="env")
    gym_metrics.record_rollout_completed(
        "failed",
        dispatch_name="env",
        failure_class="agent_run_error",
        failure_kind="agent_run_error",
        failure_type="ClientResponseError",
    )
    gym_metrics.record_rollout_completed("omitted", dispatch_name="env", failure_class="kill_shaped")
    assert _by_attrs(collected_metrics) == {
        _key(**{OUTCOME: "scored", DISPATCH: "env"}): 2,
        _key(
            **{
                OUTCOME: "failed",
                DISPATCH: "env",
                CLASS: "agent_run_error",
                KIND: "agent_run_error",
                TYPE: "ClientResponseError",
            }
        ): 1,
        _key(**{OUTCOME: "omitted", DISPATCH: "env", CLASS: "kill_shaped"}): 1,
    }


def test_driver_helper_reads_the_record_and_respects_the_span_group(collected_metrics, monkeypatch):  # noqa: F811
    _rollout_group_only(monkeypatch)
    rollout_collection._record_rollout_outcome({"reward": 1.0}, "scored", "env")
    rollout_collection._record_rollout_outcome(
        {
            rollout_collection.NG_FAILURE_CLASS_KEY: "agent_request_failed",
            "failure_kind": "transport_timeout",
            "_ng_failure_type": "ServerTimeoutError",
            "_ng_failure_message": "free text that must not become an attribute",
        },
        "failed",
        "env",
    )
    monkeypatch.setattr(rollout_collection, "is_span_group_enabled", lambda group: False)
    rollout_collection._record_rollout_outcome({"reward": 1.0}, "scored", "env")
    assert _by_attrs(collected_metrics) == {
        _key(**{OUTCOME: "scored", DISPATCH: "env"}): 1,
        _key(
            **{
                OUTCOME: "failed",
                DISPATCH: "env",
                CLASS: "agent_request_failed",
                KIND: "transport_timeout",
                TYPE: "ServerTimeoutError",
            }
        ): 1,
    }


def test_an_unregistered_kind_is_bucketed(collected_metrics, monkeypatch):  # noqa: F811
    _rollout_group_only(monkeypatch)
    for kind in ("connect to 10.0.0.7:8000 refused after 3 retries", "session_lost", "tb4:sandbox_gone"):
        rollout_collection._record_rollout_outcome(
            {rollout_collection.NG_FAILURE_CLASS_KEY: "infrastructure_error", "failure_kind": kind}, "failed", "env"
        )
    assert {dict(key)[KIND] for key in _by_attrs(collected_metrics)} == {
        gym_metrics.UNREGISTERED_FAILURE_KIND,
        "session_lost",
        "tb4:sandbox_gone",
    }


def test_failure_fields_on_a_scored_row_are_not_labels(collected_metrics, monkeypatch):  # noqa: F811
    _rollout_group_only(monkeypatch)
    rollout_collection._record_rollout_outcome(
        {"reward": 0.0, "failure_kind": "verifier_error", "_ng_failure_type": "StaleEcho"}, "scored", "env"
    )
    assert _by_attrs(collected_metrics) == {_key(**{OUTCOME: "scored", DISPATCH: "env"}): 1}
