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
from typing import Any, ClassVar, Optional

from omegaconf import DictConfig, ListConfig, OmegaConf
from pydantic import Field, field_validator

from nemo_gym.config_types import RunExportConfig
from nemo_gym.exporters.base import unexportable_reason
from nemo_gym.exporters.isolated import ExportPreparationError, IsolatedExporter
from nemo_gym.run_artifacts import RunArtifacts


class EvalRDLExportConfig(RunExportConfig):
    """Configuration for `gym eval export evalrdl`.

    Credentials are not settings here: the EvalRDL SDK finds its token in the environment or its login cache.
    """

    label: Optional[str] = Field(default=None, description="Label for the uploaded run.")
    tags: list[str] = Field(default_factory=list, description="Tags for the uploaded run.")
    meta: dict[str, Any] = Field(default_factory=dict, description="Extra metadata for the uploaded run.")
    parallel: int = Field(default=4, ge=1, description="Number of parallel upload parts.")
    wait: bool = Field(default=True, description="Block until EvalRDL has a verdict on the upload.")
    timeout: float = Field(default=1800, gt=0, description="Seconds to wait for the verdict.")
    url: Optional[str] = Field(
        default=None, description="EvalRDL service: `prod`, `stg` or a URL. Defaults to the SDK's own choice."
    )
    requirement: str = Field(
        default="evalrdl", description="pip requirement for the EvalRDL SDK, for example a pinned version."
    )

    @field_validator("tags", "meta", mode="before")
    @classmethod
    def plain_containers(cls, value: Any) -> Any:
        return OmegaConf.to_container(value, resolve=True) if isinstance(value, (DictConfig, ListConfig)) else value


class EvalRDLExporter(IsolatedExporter):
    """Upload a finished run to EvalRDL.

    The `evalrdl` SDK lives in the exporter's own environment. Install it from the index your organization
    provides by configuring uv's package index settings for the export command; this exporter does not know
    one. The upload is deduplicated by EvalRDL, so exporting the same run again reports `skipped`.
    """

    name: ClassVar[str] = "evalrdl"
    config_model: ClassVar[type[RunExportConfig]] = EvalRDLExportConfig
    worker_fpath: ClassVar[Path] = Path(__file__).with_name("_evalrdl_worker.py")

    config: EvalRDLExportConfig

    def required_packages(self) -> tuple[str, ...]:
        return (self.config.requirement,)

    def build_payload(self, artifacts: RunArtifacts, workdir: Path) -> dict[str, Any]:
        """Stage the run as one folder, and describe the upload for the worker.

        The files are copied rather than linked so the folder can be packed. The saved config and the aggregate
        metrics are required; the failures and materialized-inputs sidecars go along when the run has them.
        """
        unexportable = unexportable_reason(artifacts)
        if unexportable is not None:
            raise ExportPreparationError(unexportable)
        staged_dir = workdir / "run"
        staged_dir.mkdir()
        for source in (
            artifacts.rollouts_jsonl_fpath,
            artifacts.aggregate_metrics_json_fpath,
            artifacts.resolved_config_yaml_fpath,
            artifacts.failures_jsonl_fpath,
            artifacts.materialized_inputs_jsonl_fpath,
        ):
            if source.is_file():
                shutil.copy2(source, staged_dir / source.name)
        config = self.config
        return {
            "staged_dir": str(staged_dir),
            "label": config.label,
            "tags": list(config.tags),
            "meta": config.meta,
            "parallel": config.parallel,
            "wait": config.wait,
            "timeout": config.timeout,
            "url": config.url,
        }
