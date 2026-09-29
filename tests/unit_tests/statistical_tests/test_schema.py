# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
from pydantic import ValidationError

from nemo_gym.statistical_tests.schema import DEFAULT_STAT_TEST, StatTestConfig


BASE = {"baseline_rollouts_jsonl_fpath": "a.jsonl", "candidate_rollouts_jsonl_fpaths": ["b.jsonl"]}


class TestStatTestConfigValidation:
    """The rules every test inherits, asserted on the base directly rather than through a subclass."""

    def test_valid_minimal_config(self):
        config = StatTestConfig.model_validate(BASE)
        assert config.test == DEFAULT_STAT_TEST
        assert config.alpha == 0.05
        assert config.report_format == "both"
        assert config.output_dirpath is None

    def test_more_than_one_candidate_is_rejected(self):
        with pytest.raises(ValidationError, match="more than 1 candidate"):
            StatTestConfig.model_validate({**BASE, "candidate_rollouts_jsonl_fpaths": ["b.jsonl", "c.jsonl"]})

    def test_mismatched_candidate_agent_names_length_is_rejected(self):
        with pytest.raises(ValidationError, match="candidate_agent_names has 2 entries"):
            StatTestConfig.model_validate({**BASE, "candidate_agent_names": ["a", "b"]})

    def test_mismatched_candidate_agg_metrics_length_is_rejected(self):
        with pytest.raises(ValidationError, match="candidate_aggregate_metrics_fpaths has 2 entries"):
            StatTestConfig.model_validate({**BASE, "candidate_aggregate_metrics_fpaths": ["x.json", "y.json"]})

    @pytest.mark.parametrize("alpha", [0, 1, -0.1, 1.5])
    def test_alpha_out_of_range_is_rejected(self, alpha):
        with pytest.raises(ValidationError, match="--alpha must be between 0 and 1"):
            StatTestConfig.model_validate({**BASE, "alpha": alpha})

    def test_base_filename_parts_is_empty_so_the_stem_is_just_the_test_and_alpha(self):
        # A test that adds no filename_parts() still gets a distinct, non-colliding stem.
        assert StatTestConfig.model_validate(BASE).filename_parts() == []
