# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import Mock

import pytest

from responses_api_agents.deepseek_harness_agent import runner


@pytest.mark.parametrize("installed", [None, "0.0.0"])
def test_bootstrap_installs_the_pin_in_an_isolated_venv(tmp_path, monkeypatch, installed):
    metadata = Mock(return_value=installed) if installed else Mock(side_effect=runner.PackageNotFoundError)
    monkeypatch.setattr(runner, "version", metadata)
    execute = Mock()
    replace = Mock()
    monkeypatch.setattr(runner.subprocess, "run", execute)
    monkeypatch.setattr(runner.os, "execv", replace)
    runner.ensure_sdk(tmp_path)
    python = str(tmp_path / "venv/bin/python")
    assert execute.call_args_list[0].args[0][-2:] == ["venv", str(tmp_path / "venv")]
    install = execute.call_args_list[1].args[0]
    assert install[0] == python
    assert install[-1] == "deepseek-harness-sdk==0.1.5rc1"
    assert "--no-cache-dir" in install
    assert replace.call_args.args[0] == python


def test_bootstrap_failure_is_reported_without_starting_harness(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ensure_sdk", Mock(side_effect=RuntimeError("package index unavailable")))
    assert runner.run(tmp_path / "input.json") == 1
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["finish_reason"] == "error"
    assert result["error"] == "RuntimeError: package index unavailable"
