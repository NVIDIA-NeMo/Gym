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
"""Failure-classed verify results must never enter or leave the verify cache."""

import json
from pathlib import Path


class TestVerifyCacheScreening:
    def test_failure_classed_results_are_neither_written_nor_reused(self, tmp_path: Path) -> None:
        from responses_api_agents.stirrup_agent.app import (
            NG_FAILURE_CLASS_KEY,
            StirrupAgentWrapper,
            _verify_cache_path,
        )

        deliverables = tmp_path / "repeat_0"
        deliverables.mkdir()
        failure_result = {"reward": 0.0, NG_FAILURE_CLASS_KEY: "reference_missing"}

        StirrupAgentWrapper._write_cached_verify(None, str(deliverables), failure_result)
        cache_path = _verify_cache_path(str(deliverables))
        assert cache_path is not None and not cache_path.exists()

        # A failure-classed entry already on disk must read as a miss.
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(failure_result))
        assert StirrupAgentWrapper._read_cached_verify(None, str(deliverables)) is None

        success_result = {"reward": 1.0, "judge_response": {}}
        cache_path.write_text(json.dumps(success_result))
        assert StirrupAgentWrapper._read_cached_verify(None, str(deliverables)) == success_result
