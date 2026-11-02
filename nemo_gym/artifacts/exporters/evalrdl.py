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
from typing import Any, ClassVar, Optional

from omegaconf import DictConfig, ListConfig, OmegaConf
from pydantic import Field, field_validator

from nemo_gym.artifacts.exporters.base import unexportable_reason
from nemo_gym.artifacts.exporters.isolated import ExportPreparationError, IsolatedExporter
from nemo_gym.artifacts.run_artifacts import RunArtifacts
from nemo_gym.config.config_types import RunExportConfig


# Time the upload itself may take, before the wait for a verdict; past it the worker is killed.
UPLOAD_ALLOWANCE_S = 3600


class EvalRDLExportConfig(RunExportConfig):
    """Configuration for `gym eval export evalrdl`.

    Credentials are not settings here: the EvalRDL SDK reads its token from `$EVALRDL_TOKEN`, `$GITLAB_TOKEN` or
    `$GITLAB_PRIVATE_TOKEN`, then from `glab`.
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

    def worker_timeout_s(self, payload: dict[str, Any]) -> float:
        """The wait for the verdict (when waiting) plus an allowance for the upload itself, which `timeout` excludes."""
        return UPLOAD_ALLOWANCE_S + (self.config.timeout if self.config.wait else 0)

    def build_payload(self, artifacts: RunArtifacts, workdir: Path) -> dict[str, Any]:
        """Describe the upload for the worker: the directory the rollouts are in, uploaded as it is.

        Everything in that directory is part of the run (the rollouts and their sidecars, health reports,
        model-call captures, benchmark-specific files), so nothing is picked out, staged or copied. The saved
        config and the aggregate metrics are still required first, as Gym's own rule for a complete run.
        """
        unexportable = unexportable_reason(artifacts)
        if unexportable is not None:
            raise ExportPreparationError(unexportable)
        # `absolute`, not `resolve`: a symlinked rollouts file must not move the upload to its target's directory.
        rollouts = artifacts.rollouts_jsonl_fpath.absolute()
        others = sorted(
            path.name
            for path in rollouts.parent.glob("*rollouts.jsonl")
            if path.name != rollouts.name and (path.name == "rollouts.jsonl" or path.name.endswith("_rollouts.jsonl"))
        )
        if others:
            raise ExportPreparationError(
                f"{rollouts.parent} also holds {', '.join(others)}. EvalRDL reads a directory as one run, so these "
                "would be mixed into it; keep each run in a directory of its own."
            )
        config = self.config
        return {
            "run_dir": str(rollouts.parent),
            "label": config.label,
            "tags": list(config.tags),
            "meta": config.meta,
            "parallel": config.parallel,
            "wait": config.wait,
            "timeout": config.timeout,
            "url": config.url,
        }
