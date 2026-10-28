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
import shlex
import subprocess
import sys
import tempfile
from abc import abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Optional

from omegaconf import DictConfig

from nemo_gym.config.config_types import RunExportConfig
from nemo_gym.config.global_config import UV_VENV_DIR_KEY_NAME, WORKING_DIR
from nemo_gym.exporters.base import Exporter, ExportResult, ExportState
from nemo_gym.run_artifacts import RunArtifacts
from nemo_gym.utils.venv_setup import setup_environment


WORKER_HARNESS_FPATH = Path(__file__).with_name("_isolated_worker.py")


class ExporterEnvironmentError(RuntimeError):
    """The virtual environment an isolated exporter needs could not be built."""


class ExportPreparationError(RuntimeError):
    """The run cannot be handed to an isolated exporter's worker; the message says why."""


def exporter_venv_path(venv_root: Path, name: str, requirements: tuple[str, ...]) -> Path:
    """`<venv_root>/exporters/<name>/<digest of requirements>/.venv`.

    Keying on the requirements gives a changed requirement list a fresh environment, while a given list keeps
    its environment, which an SDK is free to update itself within.
    """
    digest = hashlib.sha256("\n".join(sorted(requirements)).encode()).hexdigest()[:12]
    return Path(venv_root) / "exporters" / name / digest / ".venv"


def ensure_exporter_environment(venv_root: Path, name: str, requirements: tuple[str, ...]) -> Path:
    """Build the exporter's virtual environment if it is not ready, and return its Python.

    Concurrent exports share one build: `setup_environment` serializes installers with a file lock and skips a
    ready environment.
    """
    venv = exporter_venv_path(venv_root, name, requirements)
    command = f"uv venv --allow-existing --python {shlex.quote(sys.executable)} {shlex.quote(str(venv))}"
    if requirements:
        command += f" && uv pip install --python {shlex.quote(str(venv / 'bin/python'))} " + " ".join(
            shlex.quote(requirement) for requirement in requirements
        )
    try:
        return_code = setup_environment(venv, command, skip_if_ready=True)
    except (OSError, RuntimeError) as exc:
        raise ExporterEnvironmentError(f"Could not build the {name} exporter environment at {venv}: {exc}") from exc
    if return_code:
        raise ExporterEnvironmentError(
            f"Could not build the {name} exporter environment at {venv} (setup exited with {return_code})."
        )
    return venv / "bin/python"


class IsolatedExporter(Exporter):
    """An exporter whose backend work runs in its own virtual environment.

    The SDK and its dependencies never enter Gym's environment, and an SDK that updates itself cannot disturb
    it. The main process validates the config and prepares what the backend needs (`build_payload`); the worker,
    a standalone file run with the environment's Python, does the backend call.

    The payload goes to the worker as JSON on stdin and may carry credentials, which therefore stay out of the
    process list. The worker file defines `run(payload) -> dict` and must not import `nemo_gym`.
    """

    # pip requirement specifiers installed into the exporter's environment; override `required_packages` to
    # take them from the config instead.
    requirements: ClassVar[tuple[str, ...]] = ()
    # The worker file; see `_isolated_worker.py` for its contract.
    worker_fpath: ClassVar[Path]

    def __init__(self, config: RunExportConfig, *, venv_root: Path) -> None:
        super().__init__(config)
        self.venv_root = Path(venv_root).resolve()

    @classmethod
    def from_global_config(cls, global_config_dict: DictConfig) -> "IsolatedExporter":
        config = cls.config_model.model_validate(global_config_dict)
        return cls(config, venv_root=Path(global_config_dict.get(UV_VENV_DIR_KEY_NAME) or WORKING_DIR))

    def required_packages(self) -> tuple[str, ...]:
        return self.requirements

    def worker_timeout_s(self, payload: dict[str, Any]) -> Optional[float]:
        """Seconds the worker may run before it is killed and the export fails; None for no limit.

        A backend SDK can hang (a stalled upload, a retry loop), and an export that runs in a job script would
        hold the job's allocation until then.
        """
        return None

    @abstractmethod
    def build_payload(self, artifacts: RunArtifacts, workdir: Path) -> dict[str, Any]:
        """The JSON-serializable input for the worker. Files it stages go under `workdir`, removed afterwards.

        Raise `ExportPreparationError` when the run cannot be exported.
        """

    def export(self, artifacts: RunArtifacts) -> ExportResult:
        with tempfile.TemporaryDirectory(prefix=f"gym-export-{self.name}-") as workdir_name:
            workdir = Path(workdir_name)
            try:
                payload = self.build_payload(artifacts, workdir)
                python = ensure_exporter_environment(self.venv_root, self.name, self.required_packages())
            except (ExportPreparationError, ExporterEnvironmentError) as exc:
                return ExportResult.failed(str(exc))
            return self._run_worker(python, payload, workdir)

    def _run_worker(self, python: Path, payload: dict[str, Any], workdir: Path) -> ExportResult:
        result_fpath = workdir / "worker_result.json"
        python = Path(os.path.abspath(python))
        try:
            completed = subprocess.run(
                [str(python), str(WORKER_HARNESS_FPATH), str(result_fpath), str(Path(self.worker_fpath).resolve())],
                input=json.dumps(payload).encode(),
                cwd=workdir,
                env=_worker_environment(python),
                timeout=self.worker_timeout_s(payload),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return ExportResult.failed(f"The {self.name} exporter worker was killed after {exc.timeout:g}s.")
        except OSError as exc:
            return ExportResult.failed(f"Could not launch the {self.name} exporter worker: {exc}")
        if not result_fpath.is_file():
            return ExportResult.failed(
                f"The {self.name} exporter worker exited with code {completed.returncode} without a result."
            )
        try:
            outcome = json.loads(result_fpath.read_text())
            state = ExportState(outcome["state"])
        except (ValueError, KeyError) as exc:
            return ExportResult.failed(f"The {self.name} exporter worker wrote an unreadable result: {exc}")
        detail = outcome.get("detail") or {}
        if completed.returncode and state is not ExportState.FAILED:
            return ExportResult.failed(
                f"The {self.name} exporter worker reported {state.value} but then exited with code "
                f"{completed.returncode}."
            )
        if state is ExportState.FAILED:
            return ExportResult(
                state,
                exit_code=outcome.get("exit_code") or 1,
                error=outcome.get("error") or "unknown error",
                detail=detail,
            )
        return ExportResult(state, detail=detail)


def _worker_environment(python: Path) -> dict[str, str]:
    """This process's environment minus what would point the worker's Python back at Gym's packages.

    The exporter environment's `bin` leads `PATH`, so a bare `python` or an SDK-installed command resolves there.
    """
    environment = {
        key: value for key, value in os.environ.items() if key not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")
    }
    environment["PATH"] = os.pathsep.join(filter(None, [str(python.parent), environment.get("PATH")]))
    return environment
