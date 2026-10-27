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
import shutil
from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field

from nemo_gym.config_types import RunExportConfig
from nemo_gym.exporters.base import unexportable_reason
from nemo_gym.exporters.isolated import ExportPreparationError, IsolatedExporter
from nemo_gym.run_artifacts import RunArtifacts


class AcmeExportConfig(RunExportConfig):
    """Settings of `gym eval export acme`."""

    project: str = Field(description="Acme project to upload to.")
    requirement: str = Field(default="acme-sdk", description="pip requirement for the Acme SDK.")


class AcmeExporter(IsolatedExporter):
    """Upload a finished run's aggregate metrics to Acme, with the SDK in its own virtual environment."""

    name: ClassVar[str] = "acme"
    config_model: ClassVar[type[RunExportConfig]] = AcmeExportConfig
    worker_fpath: ClassVar[Path] = Path(__file__).with_name("acme_worker.py")

    config: AcmeExportConfig

    def required_packages(self) -> tuple[str, ...]:
        return (self.config.requirement,)

    def build_payload(self, artifacts: RunArtifacts, workdir: Path) -> dict[str, Any]:
        problem = unexportable_reason(artifacts, rollouts_required=False)
        if problem is not None:
            raise ExportPreparationError(problem)

        staged_dir = workdir / "run"
        staged_dir.mkdir()
        try:
            shutil.copy2(artifacts.aggregate_metrics_json_fpath, staged_dir)
        except OSError as exc:
            raise ExportPreparationError(f"Could not stage the aggregate metrics: {exc}") from exc
        return {"project": self.config.project, "staged_dir": str(staged_dir)}
