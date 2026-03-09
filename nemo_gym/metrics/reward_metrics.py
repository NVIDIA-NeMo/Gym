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
from typing import Dict, Union

from nemo_gym.metrics.base import BaseMetrics


class RewardMetrics(BaseMetrics):
    """Default metrics: uses the reward field from any verify response."""

    def get_score_dict(self, result: dict) -> Dict[str, Union[float, bool]]:
        if "reward" not in result:
            raise ValueError(f"Verify result missing required 'reward' field. Got keys: {sorted(result.keys())}")
        return {"reward": result["reward"]}
