# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for startup validation and non-clean exit handling.

Seams:
  1. PinchBenchAgentConfig(...) — config validators catch misconfiguration at startup.
  2. agent._parse_result() — warns when multiple result files are present.
  3. agent._run_in_apptainer_direct() — non-clean exit with archive returns rc, doesn't raise.
"""

import json
import tarfile
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from responses_api_agents.pinchbench.tests.test_app import make_agent, make_config


# ---------------------------------------------------------------------------
# Seam 1 — PinchBenchAgentConfig validators
# ---------------------------------------------------------------------------


class TestWebSearchProviderValidation:
    def test_unknown_provider_is_rejected(self):
        with pytest.raises(Exception, match="web_search_provider"):
            make_config(web_search_provider="google")

    def test_tavily_selected_but_key_missing_is_rejected(self):
        with pytest.raises(ValueError, match="tavily_api_key"):
            make_config(web_search_provider="tavily")  # tavily_api_key defaults to None

    def test_brave_selected_but_key_missing_is_rejected(self):
        with pytest.raises(ValueError, match="brave_api_key"):
            make_config(web_search_provider="brave", brave_api_key=None)

    def test_tavily_with_key_is_accepted(self):
        cfg = make_config(web_search_provider="tavily", tavily_api_key="tvly-key", brave_api_key=None)
        assert cfg.web_search_provider == "tavily"

    def test_brave_with_key_is_accepted(self):
        cfg = make_config(web_search_provider="brave", brave_api_key="brave-key")
        assert cfg.web_search_provider == "brave"


class TestSandboxConfigValidation:
    def test_relative_sandbox_work_base_is_rejected(self):
        with pytest.raises(ValueError, match="absolute"):
            make_config(sandbox_work_base="relative/path")

    def test_max_tokens_exceeding_context_window_is_rejected(self):
        with pytest.raises(ValueError, match="max_tokens"):
            make_config(max_tokens=65537, context_window=65536)

    def test_max_tokens_equal_to_context_window_is_accepted(self):
        cfg = make_config(max_tokens=65536, context_window=65536)
        assert cfg.max_tokens == cfg.context_window

    def test_unknown_sandbox_provider_is_rejected(self):
        with pytest.raises(Exception, match="apptaienr"):
            make_config(sandbox_provider={"apptaienr": {}})

    def test_known_sandbox_provider_is_accepted(self):
        cfg = make_config(sandbox_provider={"apptainer": {}})
        assert "apptainer" in cfg.sandbox_provider


# ---------------------------------------------------------------------------
# Seam 2 — _parse_result: multiple result files
# ---------------------------------------------------------------------------


def _write_result(out_dir: Path, task_id: str, mean: float, filename: str = "0001_model.json") -> None:
    payload = {
        "tasks": [
            {
                "task_id": task_id,
                "grading": {
                    "runs": [{"grading_type": "automated", "breakdown": {}, "notes": ""}],
                    "mean": mean,
                },
            }
        ]
    }
    (out_dir / filename).write_text(json.dumps(payload))


def test_multiple_result_files_log_a_warning_and_still_return_a_score(tmp_path, capsys):
    _write_result(tmp_path, "task_x", 0.9, "result_a.json")
    _write_result(tmp_path, "task_x", 0.5, "result_b.json")

    result = make_agent()._parse_result("task_x", tmp_path)

    assert "multiple result JSON files" in capsys.readouterr().out
    assert result["status"] == "success"


# ---------------------------------------------------------------------------
# Seam 3 — _run_in_apptainer_direct: non-clean exit with archive present
# ---------------------------------------------------------------------------


def _write_tgz(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = b"placeholder"
    with tarfile.open(path, "w:gz") as tf:
        info = tarfile.TarInfo(name="placeholder.txt")
        info.size = len(data)
        tf.addfile(info, BytesIO(data))


def _mock_process(returncode: int) -> MagicMock:
    proc = MagicMock()
    proc.returncode = returncode
    proc.wait = AsyncMock(return_value=None)
    proc.kill = MagicMock()
    return proc


@pytest.mark.asyncio
async def test_non_clean_exit_with_archive_present_returns_rc_and_logs_warning(tmp_path, capsys):
    agent = make_agent(
        sandbox_spec={"image": "/sif/pinchbench.sif"},
        sandbox_provider={"apptainer": {"direct_exec": True}},
    )
    _write_tgz(tmp_path / "sandbox" / "out" / "out.tgz")

    with patch("asyncio.create_subprocess_exec", return_value=_mock_process(returncode=1)):
        rc = await agent._run_in_apptainer_direct("task_x", tmp_path, {"direct_exec": True})

    assert rc == 1
    out = capsys.readouterr().out
    assert "non-clean apptainer exit" in out
    assert "rc=1" in out


@pytest.mark.asyncio
async def test_clean_exit_returns_none_and_logs_no_warning(tmp_path, capsys):
    agent = make_agent(
        sandbox_spec={"image": "/sif/pinchbench.sif"},
        sandbox_provider={"apptainer": {"direct_exec": True}},
    )
    _write_tgz(tmp_path / "sandbox" / "out" / "out.tgz")

    with patch("asyncio.create_subprocess_exec", return_value=_mock_process(returncode=0)):
        rc = await agent._run_in_apptainer_direct("task_x", tmp_path, {"direct_exec": True})

    assert rc is None
    assert "non-clean" not in capsys.readouterr().out
