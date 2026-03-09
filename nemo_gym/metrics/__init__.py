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
import importlib
from typing import Optional

from nemo_gym.metrics.base import NOT_FOUND, BaseMetrics, MetricsOutput, compute_statistics
from nemo_gym.metrics.reward_metrics import RewardMetrics


def get_metrics(metrics_type: Optional[str] = None) -> BaseMetrics:
    """Resolve a metrics class from a class path string.

    The default RewardMetrics simply reads the "reward" field from verify results.
    Custom metrics override get_score_dict() for multi-score reporting (e.g. symbolic_accuracy
    + judge_accuracy + accuracy), get_answer() for majority@k voting, and optionally
    _add_derived_metrics() for cross-sample computations like precision/recall/F1.

    Args:
        metrics_type: Either None/"reward" for default RewardMetrics,
            or a class path like "resources_servers.code_gen.metrics::CodeGenMetrics".

    Returns:
        An instance of the resolved BaseMetrics subclass.
    """
    if metrics_type is None or metrics_type == "reward":
        return RewardMetrics()

    if "::" not in metrics_type:
        raise ValueError(
            f"Invalid metrics_type '{metrics_type}'. Expected format: 'module.path::ClassName' "
            f"(e.g. 'resources_servers.code_gen.metrics::CodeGenMetrics') or 'reward' for default."
        )

    module_path, class_name = metrics_type.rsplit("::", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    if not isinstance(cls, type) or not issubclass(cls, BaseMetrics):
        raise TypeError(f"{metrics_type} does not resolve to a BaseMetrics subclass")

    return cls()


__all__ = ["NOT_FOUND", "BaseMetrics", "MetricsOutput", "RewardMetrics", "compute_statistics", "get_metrics"]
