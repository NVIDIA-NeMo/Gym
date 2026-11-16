# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import runpy
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI

from nemo_gym.server_runtime.base_resources_server import SimpleResourcesServer


@pytest.mark.parametrize(
    ("worker", "module_name", "serves_app"),
    [
        (False, "test_pdf_analyst_entrypoint", False),
        (True, "test_pdf_analyst_entrypoint", True),
        (True, "__mp_main__", False),
    ],
)
def test_only_worker_import_creates_the_fastapi_app(monkeypatch, worker, module_name, serves_app):
    entrypoint = Path(__file__).resolve().parents[1] / "app.py"
    monkeypatch.setenv("IS_NEMO_GYM_FASTAPI_WORKER", "1" if worker else "0")
    monkeypatch.setenv("IS_NEMO_GYM_FASTAPI_ENTRYPOINT", "resources_servers/pdf_analyst_qa/app.py")
    application = FastAPI()
    launch = MagicMock(return_value=application)
    monkeypatch.setattr(SimpleResourcesServer, "run_webserver", launch)
    namespace = runpy.run_path(str(entrypoint), run_name=module_name)
    if serves_app:
        launch.assert_called_once_with()
        assert namespace["app"] is application
    else:
        launch.assert_not_called()
        assert "app" not in namespace
