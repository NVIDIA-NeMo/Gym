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
import json
import os
from pathlib import Path
from typing import Any, Optional

from omegaconf import DictConfig

from nemo_gym.config_types import RunExportConfig
from nemo_gym.exporters import load_run_exporter_class
from nemo_gym.exporters.base import ExportResult, ExportState
from nemo_gym.run_artifacts import RunArtifacts


RUNNING_STATE = "running"
_FINAL_STATES = (ExportState.SUCCEEDED.value, ExportState.SKIPPED.value)


def _write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, default=str))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_state(path: Path) -> Optional[str]:
    try:
        state = json.loads(path.read_text()).get("state")
    except (OSError, ValueError, AttributeError):
        return None
    return state if isinstance(state, str) else None


def _result_payload(result: ExportResult) -> dict[str, Any]:
    return {"state": result.state.value, "exit_code": result.exit_code, "error": result.error, "detail": result.detail}


def run_export(name: str, global_config_dict: DictConfig) -> ExportResult:
    """Run the post-run exporter `name` on the run `rollouts_jsonl_fpath` points at.

    With `result_json_fpath` set, the outcome is also kept in that file, which is how a job that runs several
    exports can tell them apart afterwards: no file means never attempted, `running` means interrupted (for
    example at the walltime), and anything else is a final result. An export that already succeeded or was
    skipped is not repeated; a failed or interrupted one is retried. A configuration error is recorded as a
    failed result too, and still raised.
    """
    settings = RunExportConfig.model_validate(global_config_dict)
    result_fpath = settings.result_json_fpath
    if result_fpath is not None:
        previous = _read_state(result_fpath)
        if previous in _FINAL_STATES:
            return ExportResult.skipped(reason="a previous attempt already exported this run", previous_state=previous)
        _write_json_atomically(result_fpath, {"state": RUNNING_STATE})
    try:
        exporter = load_run_exporter_class(name).from_global_config(global_config_dict)
        result = exporter.export(RunArtifacts(settings.rollouts_jsonl_fpath))
    except Exception as exc:
        if result_fpath is not None:
            _write_json_atomically(
                result_fpath,
                _result_payload(ExportResult.failed(f"{type(exc).__name__}: {exc}")),
            )
        raise
    if result_fpath is not None:
        _write_json_atomically(result_fpath, _result_payload(result))
    return result
