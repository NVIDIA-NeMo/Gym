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
from nemo_gym.artifacts.exporters.base import Exporter, ExportResult
from nemo_gym.artifacts.run_artifacts import RunArtifacts
from nemo_gym.trajectory.atif_export import AtifExportError, ExportAtifConfig, export_rollouts_to_atif


class AtifExporter(Exporter):
    """Convert a run's trajectories to ATIF files plus a manifest in a new directory."""

    name = "atif"
    config_model = ExportAtifConfig

    config: ExportAtifConfig

    def export(self, artifacts: RunArtifacts) -> ExportResult:
        config = self.config.model_copy(update={"rollouts_jsonl_fpath": artifacts.rollouts_jsonl_fpath})
        try:
            result = export_rollouts_to_atif(config)
        except (AtifExportError, OSError) as exc:
            return ExportResult.failed(str(exc))
        return ExportResult.succeeded(
            output_dirpath=str(result.output_dirpath),
            trajectory_count=result.trajectory_count,
            manifest_fpath=str(result.manifest_fpath),
        )
