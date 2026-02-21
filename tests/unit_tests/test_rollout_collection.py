# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import json
from pathlib import Path

from nemo_gym.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper


class TestRolloutCollection:
    def test_preprocess_rows_from_config(self, tmp_path: Path) -> None:
        fpath = tmp_path / "input.jsonl"
        samples = [json.dumps({"responses_create_params": {"input": []}, "x": i}) for i in range(10)]
        fpath.write_text("\n".join(samples) + "\n")

        config = RolloutCollectionConfig(
            agent_name="my_agent",
            input_jsonl_fpath=str(fpath),
            output_jsonl_fpath="abcd",
            limit=3,
            num_repeats=2,
            num_repeats_add_seed=True,
            num_samples_in_parallel=None,
            responses_create_params=dict(temperature=0.1),
        )

        rows = RolloutCollectionHelper._preprocess_rows_from_config(None, config)
        assert rows == [
            {"_task_index": 0, "responses_create_params": {"input": [], "seed": 0, "temperature": 0.1}, "x": 0},
            {"_task_index": 0, "responses_create_params": {"input": [], "seed": 1, "temperature": 0.1}, "x": 0},
            {"_task_index": 1, "responses_create_params": {"input": [], "seed": 0, "temperature": 0.1}, "x": 1},
            {"_task_index": 1, "responses_create_params": {"input": [], "seed": 1, "temperature": 0.1}, "x": 1},
            {"_task_index": 2, "responses_create_params": {"input": [], "seed": 0, "temperature": 0.1}, "x": 2},
            {"_task_index": 2, "responses_create_params": {"input": [], "seed": 1, "temperature": 0.1}, "x": 2},
        ]
