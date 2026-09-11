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

from nemo_gym import failure_kinds
from nemo_gym.failure_kinds import (
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
        domains = ("transport_", "agent_", "judge_", "verifier_", "provider_", "session_", "cohort_", "persistence_")
        lifecycle = {"cancelled", "kill_shaped", "shutdown"}

        undomained = {n for n in FAILURE_KINDS if not n.startswith(domains)} - lifecycle

        assert undomained == set()


class TestRegistryLookup:
    def test_a_registered_name_is_recognized(self) -> None:
        assert is_registered("judge_failed")

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
        from nemo_gym.rollout_collection import AGENT_REQUEST_FAILED_FAILURE_CLASS, AGENT_RUN_ERROR_FAILURE_CLASS

        assert is_registered(AGENT_RUN_ERROR_FAILURE_CLASS)
        assert is_registered(AGENT_REQUEST_FAILED_FAILURE_CLASS)

    def test_the_judge_failsafe_class(self) -> None:
        from nemo_gym.rollout_reverification import JUDGE_FAILED_FAILURE_CLASS

        assert is_registered(JUDGE_FAILED_FAILURE_CLASS)


class TestDependencyWeight:
    def test_the_module_imports_nothing_from_the_server_stack(self) -> None:
        """Data tooling reads the vocabulary without installing any server's requirements."""
        source = open(failure_kinds.__file__).read()

        assert "import fastapi" not in source
        assert "from nemo_gym" not in source
