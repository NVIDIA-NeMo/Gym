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


def resubmit_offset_path_for(output_fpath: Path) -> Path:
    """Where `gym eval resubmit` records the failures sidecar's line count; failures before it no longer count."""
    return output_fpath.with_name(output_fpath.stem + "_resubmit_offset")


def materialized_path_for(output_fpath: Path) -> Path:
    """`results/rollouts.jsonl` -> `results/rollouts_materialized_inputs.jsonl`."""
    return output_fpath.with_stem(output_fpath.stem + "_materialized_inputs").with_suffix(".jsonl")


def aggregate_metrics_path_for(output_fpath: Path) -> Path:
    """`results/rollouts.jsonl` -> `results/rollouts_aggregate_metrics.json`.

    Mirrors how rollout collection and reverification name the file they write, so consumers
    (e.g. `gym eval compare`) derive the same path the writers produced.
    """
    return output_fpath.with_stem(output_fpath.stem + "_aggregate_metrics").with_suffix(".json")


def aggregation_sources_path_for(output_fpath: Path) -> Path:
    """`results/rollouts.jsonl` -> `results/rollouts_sources.json`: the shards `gym eval aggregate` merged into it."""
    return output_fpath.with_stem(output_fpath.stem + "_sources").with_suffix(".json")


def run_summary_path_for(output_fpath: Path) -> Path:
    """`results/rollouts.jsonl` -> `results/rollouts_run_summary.json`.

    The per-run summary a Harbor run writes beside its rollouts (`nemo_gym.data.tasks.harbor.run_summary`),
    named like the other sidecars so a renamed `--output` keeps them together.
    """
    return output_fpath.with_stem(output_fpath.stem + "_run_summary").with_suffix(".json")


def resolved_config_path_for(output_fpath: Path) -> Path:
    """`results/rollouts.jsonl` -> `results/rollouts_resolved_config.yaml`."""
    return output_fpath.with_stem(output_fpath.stem + "_resolved_config").with_suffix(".yaml")


def exporter_run_id_path_for(output_fpath: Path, exporter: str) -> Path:
    """`results/rollouts.jsonl`, `mlflow` -> `results/rollouts_mlflow_run_id`: the run a live exporter logs to."""
    return output_fpath.with_name(f"{output_fpath.stem}_{exporter}_run_id")


def export_done_path_for(output_fpath: Path, exporter: str) -> Path:
    """`results/rollouts.jsonl`, `mlflow` -> `results/rollouts_mlflow_export_done`: set once the run was exported."""
    return output_fpath.with_name(f"{output_fpath.stem}_{exporter}_export_done")
