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
import pytest
from pydantic import ValidationError

from nemo_gym.statistical_tests.schema import PairedTestConfig


BASE = {"baseline_rollouts_jsonl_fpath": "a.jsonl", "candidate_rollouts_jsonl_fpaths": ["b.jsonl"]}


class TestPairedTestConfigValidation:
    def test_valid_minimal_config(self):
        config = PairedTestConfig.model_validate(BASE)
        assert config.metric is None
        assert config.margin is None
        assert config.alpha == 0.05
        assert config.report_format == "both"

    def test_more_than_one_candidate_is_rejected(self):
        with pytest.raises(ValidationError, match="more than 1 candidate"):
            PairedTestConfig.model_validate({**BASE, "candidate_rollouts_jsonl_fpaths": ["b.jsonl", "c.jsonl"]})

    def test_mismatched_candidate_agent_names_length_is_rejected(self):
        with pytest.raises(ValidationError, match="candidate_agent_names has 2 entries"):
            PairedTestConfig.model_validate({**BASE, "candidate_agent_names": ["a", "b"]})

    def test_mismatched_candidate_agg_metrics_length_is_rejected(self):
        with pytest.raises(ValidationError, match="candidate_aggregate_metrics_fpaths has 2 entries"):
            PairedTestConfig.model_validate({**BASE, "candidate_aggregate_metrics_fpaths": ["x.json", "y.json"]})

    @pytest.mark.parametrize("margin", [0, -0.01, -5])
    def test_non_positive_margin_is_rejected(self, margin):
        with pytest.raises(ValidationError, match="--margin must be a positive number"):
            PairedTestConfig.model_validate({**BASE, "margin": margin})

    @pytest.mark.parametrize("alpha", [0, 1, -0.1, 1.5])
    def test_alpha_out_of_range_is_rejected(self, alpha):
        with pytest.raises(ValidationError, match="--alpha must be between 0 and 1"):
            PairedTestConfig.model_validate({**BASE, "alpha": alpha})

    def test_positive_margin_and_in_range_alpha_are_accepted(self):
        config = PairedTestConfig.model_validate({**BASE, "margin": 0.01, "alpha": 0.1})
        assert config.margin == 0.01
        assert config.alpha == 0.1
