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
from dataclasses import dataclass
from pathlib import Path

from nemo_gym.utils.path_utils import (
    aggregate_metrics_path_for,
    export_done_path_for,
    exporter_run_id_path_for,
    failures_path_for,
    materialized_path_for,
    resolved_config_path_for,
)


@dataclass(frozen=True)
class RunArtifacts:
    """The files `gym eval run` leaves for one run, all derived from the rollouts file's path.

    Paths are derived, not checked: any of the sidecars may be missing (for example a run from before
    `resolved_config` was saved), and it is up to each exporter to decide which ones it requires.
    """

    rollouts_jsonl_fpath: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "rollouts_jsonl_fpath", Path(self.rollouts_jsonl_fpath))

    @property
    def failures_jsonl_fpath(self) -> Path:
        return failures_path_for(self.rollouts_jsonl_fpath)

    @property
    def materialized_inputs_jsonl_fpath(self) -> Path:
        return materialized_path_for(self.rollouts_jsonl_fpath)

    @property
    def aggregate_metrics_json_fpath(self) -> Path:
        return aggregate_metrics_path_for(self.rollouts_jsonl_fpath)

    @property
    def resolved_config_yaml_fpath(self) -> Path:
        return resolved_config_path_for(self.rollouts_jsonl_fpath)

    def exporter_run_id_fpath(self, exporter: str) -> Path:
        """The run a live exporter logged this run to. It exists once the run opened, whether or not it finished."""
        return exporter_run_id_path_for(self.rollouts_jsonl_fpath, exporter)

    def export_done_fpath(self, exporter: str) -> Path:
        """Marker a post-run export writes, holding the backend run id, once the whole run has been sent."""
        return export_done_path_for(self.rollouts_jsonl_fpath, exporter)
