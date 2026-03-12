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
from typing import Dict, Optional, Union

from nemo_gym.metrics.base import BaseMetrics


class MathMetrics(BaseMetrics):
    """Metrics for math benchmarks with symbolic and judge-based verification.

    Reports symbolic_accuracy, judge_accuracy, and overall accuracy.
    Enables majority@k via extracted_answer.
    """

    def get_score_dict(self, result: dict) -> Dict[str, Union[float, bool]]:
        scores: Dict[str, Union[float, bool]] = {}

        if "library_reward" in result:
            scores["symbolic_accuracy"] = result["library_reward"]

        if "judge_evaluations" in result and result["judge_evaluations"] is not None:
            scores["judge_accuracy"] = result["reward"]

        scores["accuracy"] = result["reward"]
        return scores

    def get_answer(self, result: dict) -> Optional[str]:
        return result.get("extracted_answer")
