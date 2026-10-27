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
import hashlib
import json
import os
from pathlib import Path
from typing import ClassVar

from pydantic import Field

from nemo_gym.config_types import RunExportConfig
from nemo_gym.exporters.base import Exporter, ExportResult, unexportable_reason
from nemo_gym.run_artifacts import RunArtifacts


class SummaryExportConfig(RunExportConfig):
    """Settings of `gym eval export summary`."""

    destination_dirpath: Path = Field(description="Directory the summary is written to.")


class SummaryExporter(Exporter):
    """Write a small JSON summary of a finished run."""

    name: ClassVar[str] = "summary"
    config_model: ClassVar[type[RunExportConfig]] = SummaryExportConfig

    config: SummaryExportConfig

    def export(self, artifacts: RunArtifacts) -> ExportResult:
        try:
            problem = unexportable_reason(artifacts)
            if problem is not None:
                return ExportResult.failed(problem)

            # Runs often share a rollouts filename, so name the summary after the whole path.
            rollouts = artifacts.rollouts_jsonl_fpath
            run_id = hashlib.sha256(str(rollouts.resolve()).encode()).hexdigest()[:8]
            destination = self.config.destination_dirpath / f"{rollouts.stem}_{run_id}_summary.json"
            if destination.exists():
                return ExportResult.skipped(reason="already exported", destination=str(destination))

            with rollouts.open("rb") as lines:
                rollout_count = sum(1 for line in lines if line.strip())
            summary = {
                "rollouts": rollout_count,
                "aggregate_metrics": json.loads(artifacts.aggregate_metrics_json_fpath.read_text()),
            }
            # Publish only a complete file, because the existence check above treats any file as done.
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
            try:
                temporary.write_text(json.dumps(summary, indent=2))
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
        except OSError as exc:
            return ExportResult.failed(str(exc))
        return ExportResult.succeeded(destination=str(destination), rollouts=rollout_count)
