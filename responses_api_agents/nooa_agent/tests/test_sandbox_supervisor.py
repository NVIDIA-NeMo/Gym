# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import math
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from responses_api_agents.nooa_agent import sandbox_supervisor


@pytest.mark.parametrize(
    "confirmed,error,exit_code", [(True, None, 0), (False, "cleanup failed", 1), (True, "launch failed", 1)]
)
def test_nooa_launcher_uses_main_reaper_without_a_second_deadline(
    monkeypatch, tmp_path: Path, confirmed: bool, error: str | None, exit_code: int
) -> None:
    receipt = {"cleanup_confirmed": confirmed, "return_code": 0, "error": error, "timed_out": False}
    reaper = MagicMock(return_value=receipt)
    monkeypatch.setattr(sandbox_supervisor.process_supervisor, "_supervise", reaper)
    assert sandbox_supervisor.supervise(tmp_path) == exit_code
    assert reaper.call_args.args == (
        [
            sys.executable,
            "-I",
            "-m",
            "responses_api_agents.nooa_agent.sandbox_entrypoint",
            str(tmp_path / "input.json"),
            str(tmp_path / "result.json"),
        ],
    )
    assert reaper.call_args.kwargs == {
        "timeout": math.inf,
        "cleanup_timeout": 5,
        "stop_path": tmp_path / "runner.stop",
    }
    assert int((tmp_path / "runner.pid").read_text()) == os.getpid()
    assert json.loads((tmp_path / "cleanup.json").read_text()) == receipt
    assert not (tmp_path / "cleanup.tmp").exists()
    assert not (tmp_path / "completion.json").exists()
