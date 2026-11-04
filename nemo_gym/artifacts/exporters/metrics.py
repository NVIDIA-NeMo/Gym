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
from typing import Any, Iterable


_PRIMITIVE_TYPES = (bool, int, float, str, type(None))


def flatten_aggregate_metrics(agent_entries: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """The scalar metrics a tracking backend gets from per-agent aggregate-metrics entries.

    Live runs and post-run export both build their metrics here, so a run looks the same in the backend
    either way: `<agent>/<metric>` and `key_metrics/<agent>/<metric>`, primitive values only.
    """
    # Imported here because `global_config` imports this package to set up exporters.
    from nemo_gym.config.global_config import AGENT_REF_KEY_NAME

    metrics: dict[str, Any] = {}
    for entry in agent_entries:
        agent_name = entry[AGENT_REF_KEY_NAME]["name"]
        metrics.update(
            {
                f"{agent_name}/{k}": v
                for k, v in entry.get("agent_metrics", {}).items()
                if isinstance(v, _PRIMITIVE_TYPES)
            }
        )
        metrics.update(
            {
                f"key_metrics/{agent_name}/{k}": v
                for k, v in entry.get("key_metrics", {}).items()
                if isinstance(v, _PRIMITIVE_TYPES)
            }
        )
    return metrics
