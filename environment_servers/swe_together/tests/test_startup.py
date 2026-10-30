# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import runpy
import subprocess
import sys
from pathlib import Path

from nemo_gym.server_utils import SimpleServer


def test_app_import_from_component_working_directory():
    """Gym puts the component directory first, where stdlib names must remain safe."""
    component = Path(__file__).resolve().parents[1]
    repository = component.parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import app; from types import MappingProxyType; "
            "assert app.SWETEnvironmentServer.request_model.__name__ == 'SWETRequest'; "
            "assert MappingProxyType({'ready': True})['ready']",
        ],
        cwd=component,
        env=os.environ | {"PYTHONPATH": os.pathsep.join((str(component), str(repository)))},
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_module_exports_app_for_multi_worker_import(monkeypatch):
    """With several workers, uvicorn imports the entrypoint in each worker and serves its module-level app."""
    worker_app = object()
    monkeypatch.setattr("nemo_gym.utils.server_process.is_nemo_gym_fastapi_entrypoint", lambda _: True)
    monkeypatch.setattr(SimpleServer, "run_webserver", classmethod(lambda _: worker_app))

    namespace = runpy.run_path(
        str(Path(__file__).parents[1] / "app.py"),
        run_name="environment_servers.swe_together.worker_test",
    )

    assert namespace["app"] is worker_app
