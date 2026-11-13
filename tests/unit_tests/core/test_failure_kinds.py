# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The shared failure vocabulary has to stay small, stable and label-safe."""

import logging
import re

from nemo_gym.core import failure_kinds
from nemo_gym.core.failure_kinds import (
    FAILURE_KINDS,
    is_namespaced,
    is_registered,
    validate_failure_kind,
)


class TestNamingConvention:
    def test_every_name_matches_one_convention(self) -> None:
        assert all(re.fullmatch(r"[a-z][a-z0-9_]*", name) for name in FAILURE_KINDS)

    def test_no_name_differs_from_another_only_by_case_or_separator(self) -> None:
        """Two spellings of one failure defeat the grouping this vocabulary exists for."""
        normalized = [name.lower().replace("_", "") for name in FAILURE_KINDS]

        assert len(set(normalized)) == len(normalized)

    def test_every_name_states_the_layer_that_observed_the_failure(self) -> None:
        """`transport_timeout` and `agent_timeout` are different failures, not one."""
        domains = (
            "model_",
            "transport_",
            "agent_",
            "environment_",
            "judge_",
            "verifier_",
            "provider_",
            "session_",
            "cohort_",
            "persistence_",
            "environment_",
            "harness_",
            "collection_",
            # Gym's server supervisor, which starts every server of a run.
            "server_",
            # The driver step of an orchestrated job, before collection runs anything.
            "driver_",
            # A rollout's own lifecycle across allocations.
            "rollout_",
        )
        lifecycle = {"cancelled", "kill_shaped", "shutdown"}
        # Registered for saved rows of shipped producers; the rule binds every name added after them.
        producer_labels = {"harbor_failed", "infrastructure_error"}

        undomained = {n for n in FAILURE_KINDS if not n.startswith(domains)} - lifecycle - producer_labels

        assert undomained == set()


class TestRegistryLookup:
    def test_a_registered_name_is_recognized(self) -> None:
        assert is_registered("judge_failed")

    def test_a_provider_safety_refusal_has_its_own_model_kind(self) -> None:
        assert is_registered(failure_kinds.MODEL_SAFETY_REFUSAL)
        assert failure_kinds.MODEL_SAFETY_REFUSAL == "model_safety_refusal"

    def test_an_unregistered_name_is_not(self) -> None:
        assert not is_registered("something_invented")

    def test_an_environment_can_namespace_its_own(self) -> None:
        assert is_namespaced("lexmount_browser:quota_exhausted")

    def test_a_bare_name_is_not_namespaced(self) -> None:
        assert not is_namespaced("judge_failed")

    def test_a_trailing_newline_is_not_a_legal_name(self) -> None:
        """`$` also matches before a final newline, so `match` would admit a second,
        silently different label for the same failure."""
        assert not is_namespaced("my_server:odd_case\n")
        assert not is_registered("judge_failed\n")


class TestValidation:
    def test_a_registered_name_passes_silently(self, caplog) -> None:
        with caplog.at_level(logging.WARNING):
            assert validate_failure_kind("judge_failed") == "judge_failed"

        assert caplog.records == []

    def test_a_namespaced_name_passes_silently(self, caplog) -> None:
        with caplog.at_level(logging.WARNING):
            assert validate_failure_kind("my_server:odd_case") == "my_server:odd_case"

        assert caplog.records == []

    def test_an_unknown_name_warns_but_survives(self, caplog) -> None:
        """Rejecting it would trade a visible wrong name for an invisible dropped failure."""
        failure_kinds._WARNED_UNKNOWN.discard("legacy_label")
        with caplog.at_level(logging.WARNING):
            result = validate_failure_kind("legacy_label")

        assert result == "legacy_label"
        assert any("legacy_label" in record.getMessage() for record in caplog.records)

    def test_a_malformed_namespaced_value_takes_the_unknown_path(self, caplog) -> None:
        """It stays observable instead of passing silently as a namespaced name."""
        with caplog.at_level(logging.WARNING):
            result = validate_failure_kind("my_server:odd_case\n")

        assert result == "my_server:odd_case\n"
        assert len(caplog.records) == 1

    def test_the_same_unknown_name_is_reported_once(self, caplog) -> None:
        """A failing component emits the same label on every rollout; warning per call
        buries the one line that matters."""
        failure_kinds._WARNED_UNKNOWN.discard("repeated_legacy")
        with caplog.at_level(logging.WARNING):
            for _ in range(1000):
                assert validate_failure_kind("repeated_legacy") == "repeated_legacy"

        assert len(caplog.records) == 1

    def test_each_distinct_unknown_name_is_still_reported(self, caplog) -> None:
        for name in ("legacy_one", "legacy_two"):
            failure_kinds._WARNED_UNKNOWN.discard(name)
        with caplog.at_level(logging.WARNING):
            validate_failure_kind("legacy_one")
            validate_failure_kind("legacy_two")

        assert len(caplog.records) == 2

    def test_absent_is_not_a_failure(self, caplog) -> None:
        with caplog.at_level(logging.WARNING):
            assert validate_failure_kind(None) is None

        assert caplog.records == []


class TestLabelSafety:
    def test_the_vocabulary_stays_small_enough_to_be_a_metric_label(self) -> None:
        """A name is a metric dimension. Free text belongs in `failure_reason` instead."""
        assert len(FAILURE_KINDS) < 64

    def test_no_name_carries_occurrence_detail(self) -> None:
        assert all(len(name) < 40 and " " not in name for name in FAILURE_KINDS)


class TestNamesTheRepoAlreadyProduces:
    """Labels main emits today must be in the vocabulary, or it describes nothing."""

    def test_rollout_collection_row_routing_classes(self) -> None:
        from nemo_gym.rollouts.rollout_collection import (
            AGENT_REQUEST_FAILED_FAILURE_CLASS,
            AGENT_RUN_ERROR_FAILURE_CLASS,
            ENVIRONMENT_SERVER_FAILURE_CLASS,
        )

        assert is_registered(AGENT_RUN_ERROR_FAILURE_CLASS)
        assert is_registered(AGENT_REQUEST_FAILED_FAILURE_CLASS)
        assert is_registered(ENVIRONMENT_SERVER_FAILURE_CLASS)

    def test_the_judge_failsafe_class(self) -> None:
        from nemo_gym.eval.rollout_reverification import JUDGE_FAILED_FAILURE_CLASS

        assert is_registered(JUDGE_FAILED_FAILURE_CLASS)


class TestDependencyWeight:
    def test_the_module_imports_nothing_from_the_server_stack(self) -> None:
        """Data tooling reads the vocabulary without installing any server's requirements."""
        source = open(failure_kinds.__file__).read()

        assert "import fastapi" not in source
        assert "from nemo_gym" not in source


def test_the_shipped_producer_labels_are_registered() -> None:
    from nemo_gym.core.failure_kinds import is_registered

    assert is_registered("harbor_failed") and is_registered("infrastructure_error")


class TestSharedClassification:
    def test_registered_and_unknown_defaults_are_distinct(self):
        from nemo_gym.core.failure_kinds import FailureClass, classify_failure

        assert classify_failure(failure_kind="judge_failed") == FailureClass.EVALUATION
        assert classify_failure(failure_kind="provider_io_timeout", stage="seed") == FailureClass.EVALUATION
        assert classify_failure(failure_kind="provider_oom_killed") == FailureClass.UNKNOWN
        assert classify_failure(failure_kind="provider_new_unregistered") == FailureClass.UNKNOWN
        assert classify_failure(failure_kind=None) == FailureClass.UNKNOWN
        assert classify_failure(failure_kind="grading_without_verdict") == FailureClass.EVALUATION

    def test_http_evidence_is_shared_configurable_and_strict(self):
        from nemo_gym.core.failure_kinds import FailureClassificationConfig, classify_failure, reporting_failure_kind

        assert classify_failure(failure_kind="model_request_error", http_status=503) == "evaluation"
        assert classify_failure(failure_kind="model_request_error", http_status=400) == "model"
        assert classify_failure(failure_kind="model_request_error", http_status=400, policy=False) == "evaluation"
        assert classify_failure(failure_kind="model_request_error", http_status="503") == "model"
        assert classify_failure(failure_kind="model_request_error", http_status=True) == "model"
        config = FailureClassificationConfig(reporting_kind_by_status_code={503: None})
        assert classify_failure(failure_kind="model_request_error", http_status=503, config=config) == "model"
        assert reporting_failure_kind("model_request_error", 503, config=config) == "model_request_error"
        assert reporting_failure_kind("model_request_error", 500, config=config) == "model_server_error"

    def test_stage_classification_precedes_generic_http_evidence(self):
        from nemo_gym.core.failure_kinds import FailureClassificationConfig, classify_failure

        assert classify_failure(failure_kind="provider_oom_killed", stage="agent", http_status=502) == "model"
        config = FailureClassificationConfig(by_status_code={502: "unknown"})
        assert (
            classify_failure(failure_kind="provider_oom_killed", stage="agent", http_status=502, config=config)
            == "unknown"
        )

    def test_partial_overrides_round_trip_and_do_not_share_state(self):
        from nemo_gym.core.failure_kinds import FailureClassificationConfig, classify_failure

        config = FailureClassificationConfig.model_validate_json(
            '{"by_stage":{"agent":{"by_failure_kind":{"provider_io_timeout":"unknown"}}},"by_failure_kind":{"model_policy_response_timeout":"evaluation"}}'
        )
        assert classify_failure(failure_kind="provider_oom_killed", stage="agent", config=config) == "model"
        assert classify_failure(failure_kind="provider_io_timeout", stage="agent", config=config) == "unknown"
        assert classify_failure(failure_kind="model_policy_response_timeout", config=config) == "evaluation"
        assert FailureClassificationConfig.model_validate_json(config.model_dump_json()) == config
        config.by_stage["agent"].by_failure_kind["provider_oom_killed"] = "unknown"
        assert classify_failure(failure_kind="provider_oom_killed", stage="agent") == "model"

    def test_server_is_an_evaluation_subcategory(self):
        from nemo_gym.core.failure_kinds import FailureClassificationConfig, classify_failure, is_server_failure

        assert is_server_failure("provider_io_timeout")
        assert not is_server_failure("judge_failed")
        assert classify_failure(failure_kind="judge_failed") == "evaluation"
        assert is_server_failure("provider_oom_killed")
        assert not is_server_failure("provider_typo")
        config = FailureClassificationConfig(by_failure_kind={"model_policy_response_timeout": "evaluation"})
        assert is_server_failure("model_policy_response_timeout", config=config)

    def test_invalid_overrides_fail_at_configuration_load(self):
        import pytest
        from pydantic import ValidationError

        from nemo_gym.core.failure_kinds import FailureClassificationConfig

        for value in (
            {"by_failure_kind": {"provider_typo": "evaluation"}},
            {"by_failure_kind": {"provider_io_timeout": "infra"}},
            {"by_stage": {"agent_execution": {}}},
            {"by_stage": {"agent": {"by_failure_kind": {"typo": "model"}}}},
            {"by_status_code": {600: "evaluation"}},
            {"reporting_kind_by_status_code": {503: "typo"}},
        ):
            with pytest.raises(ValidationError):
                FailureClassificationConfig.model_validate(value)


def test_classification_accepts_namespaced_producer_kinds_without_expanding_action_registry():
    import pytest
    from pydantic import ValidationError

    from nemo_gym.core.failure_kinds import FailureClassificationConfig, classify_failure, require_failure_kind
    from nemo_gym.core.outcome_policy import OutcomePolicyConfig

    kind = "lexmount_browser:quota_exhausted"
    assert require_failure_kind(kind) == kind
    assert classify_failure(failure_kind=kind) == "unknown"
    config = FailureClassificationConfig.model_validate(
        {
            "by_failure_kind": {kind: "evaluation"},
            "by_stage": {"agent": {"by_failure_kind": {kind: "model"}}},
        }
    )
    assert classify_failure(failure_kind=kind, config=config) == "evaluation"
    assert classify_failure(failure_kind=kind, stage="agent", config=config) == "model"
    assert FailureClassificationConfig.model_validate_json(config.model_dump_json()) == config
    with pytest.raises(ValidationError, match="Unregistered failure kinds"):
        FailureClassificationConfig(by_failure_kind={"lexmount_browser:quota_exhausted\n": "evaluation"})
    # Action overrides retain their existing registered-kind restriction. A boundary
    # handles custom names through class rules instead of silently widening its API.
    with pytest.raises(ValidationError, match="Unregistered failure kinds"):
        OutcomePolicyConfig(default_action="fail", by_failure_kind={kind: "retry"})


def test_repeated_default_classification_avoids_rebuilding_config_and_custom_rules_do_not_leak(monkeypatch):
    from nemo_gym.core.failure_kinds import FailureClassificationConfig, classify_failure, reporting_failure_kind

    custom = FailureClassificationConfig(by_failure_kind={"model_request_error": "evaluation"})
    # Warm the read-only default before prohibiting additional allocations.
    assert classify_failure(failure_kind="model_request_error") == "model"

    def unexpected_config_build(*args, **kwargs):
        raise AssertionError("Per-row classification rebuilt the default configuration")

    monkeypatch.setattr(failure_kinds, "FailureClassificationConfig", unexpected_config_build)
    for _ in range(100):
        assert classify_failure(failure_kind="model_request_error", http_status=503) == "evaluation"
        assert reporting_failure_kind("model_request_error", 503) == "model_server_error"
        assert classify_failure(failure_kind="model_request_error", config=custom) == "evaluation"
        assert classify_failure(failure_kind="model_request_error") == "model"
