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
    (e.g. `gym eval compare`) derive the same path the writers produced.
    """
    return output_fpath.with_stem(output_fpath.stem + "_aggregate_metrics").with_suffix(".json")


def resolve_run_output_dir(
    rollouts_jsonl_fpath: str | Path, output_dirpath: str | None = None, subdir: str | None = None
) -> Path:
    """Directory to write output files *about* the run identified by given `rollouts_jsonl_fpath`.

    A relative path is anchored at the user's cwd, never resolved against the install root: this is a
    *write* target, and on a wheel install that root is site-packages.
    """
    if output_dirpath:
        p = Path(output_dirpath)
    else:
        p = Path(rollouts_jsonl_fpath).parent
    if not p.is_absolute():
        p = Path.cwd() / p
    if subdir:
        p = p / subdir
    return p
