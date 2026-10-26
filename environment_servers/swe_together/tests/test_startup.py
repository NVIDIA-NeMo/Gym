# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
from pathlib import Path


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
