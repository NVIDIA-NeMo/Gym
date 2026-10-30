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
"""One post-run export of a submitted benchmark, with its outcome kept in a result file.

The driver step of a job runs this after `gym eval run`, once per configured export:
`python3 -m nemo_gym.orchestration.export_step <exporter> +key=value ...`. The exporters themselves know nothing
about it; what a job needs on top of them (telling several exports apart afterwards, not repeating one that
finished, retrying one that was interrupted) lives here.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

from omegaconf import DictConfig
from pydantic import TypeAdapter

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

    `result_json_fpath` is a setting of this step, not of the exporters. When it is set, the outcome is also kept
    in that file, which is how a job that runs several exports can tell them apart afterwards: no file means never attempted, `running` means interrupted (for
    example at the walltime), and anything else is a final result. An export that already succeeded or was
    skipped is not repeated; a failed or interrupted one is retried. A configuration error is recorded as a
    failed result too, and still raised.
    """
    # Only the result file is read here; the rest of the shared settings are validated inside the recording
    # boundary below, so an error in them is written to the result file too.
    result_fpath = TypeAdapter(Optional[Path]).validate_python(global_config_dict.get("result_json_fpath"))
    if result_fpath is not None:
        previous = _read_state(result_fpath)
        if previous in _FINAL_STATES:
            return ExportResult.skipped(reason="a previous attempt already exported this run", previous_state=previous)
        _write_json_atomically(result_fpath, {"state": RUNNING_STATE})
    try:
        settings = RunExportConfig.model_validate(global_config_dict)
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


def main() -> None:
    """`python3 -m nemo_gym.orchestration.export_step <exporter> +key=value ...`: the settings are Hydra overrides."""
    from nemo_gym.cli.utils import exit_cleanly_on_config_error
    from nemo_gym.global_config import GlobalConfigDictParserConfig, get_global_config_dict

    @exit_cleanly_on_config_error
    def run() -> None:
        name = sys.argv[1]
        # The exporter's name is not an override.
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        # The step opens its own backend session, so no live exporter may be started beside it.
        global_config_dict = get_global_config_dict(
            global_config_dict_parser_config=GlobalConfigDictParserConfig(skip_exporters=True)
        )
        result = run_export(name, global_config_dict)

        print(f"Export {name}: {result.state.value}")
        for key, value in result.detail.items():
            print(f"  {key}: {value}")
        if result.state is ExportState.FAILED:
            print(f"Error: {result.error}", file=sys.stderr)
            raise SystemExit(result.exit_code)

    if len(sys.argv) < 2:
        raise SystemExit("usage: python3 -m nemo_gym.orchestration.export_step <exporter> [+key=value ...]")
    run()


if __name__ == "__main__":  # pragma: no cover
    main()
