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
from pathlib import Path


def failures_path_for(output_fpath: Path) -> Path:
    return output_fpath.with_name(output_fpath.stem + "_failures.jsonl")


def aggregate_metrics_path_for(output_fpath: Path) -> Path:
    """`results/rollouts.jsonl` -> `results/rollouts_aggregate_metrics.json`.

    Mirrors how rollout collection and reverification name the file they write, so consumers
    (e.g. `gym compare`) derive the same path the writers produced.
    """
    return output_fpath.with_stem(output_fpath.stem + "_aggregate_metrics").with_suffix(".json")
