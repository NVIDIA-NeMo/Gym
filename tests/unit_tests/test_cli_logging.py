# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for per-server log file redirection in the CLI."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nemo_gym.cli import _run_command


class TestRunCommandLogging:
    def test_stdout_stderr_redirected_to_files(self, tmp_path: Path) -> None:
        """Subprocess stdout and stderr are written to the server log directory."""
        server_log_dir = tmp_path / "logs" / "test_server"

        proc = _run_command(
            'echo "hello stdout" && echo "hello stderr" >&2',
            working_dir_path=tmp_path,
            server_log_dir=server_log_dir,
        )
        proc.wait()

        stdout_log = server_log_dir / "stdout.log"
        stderr_log = server_log_dir / "stderr.log"
        assert stdout_log.exists(), f"Expected {stdout_log} to exist"
        assert stderr_log.exists(), f"Expected {stderr_log} to exist"
        assert "hello stdout" in stdout_log.read_text()
        assert "hello stderr" in stderr_log.read_text()

    def test_no_server_log_dir_uses_default_popen(self, tmp_path: Path) -> None:
        """Without server_log_dir, subprocess runs without file redirection (original behavior)."""
        proc = _run_command("echo ok", working_dir_path=tmp_path)
        proc.wait()
        assert proc.returncode == 0

    def test_log_directory_created_automatically(self, tmp_path: Path) -> None:
        """The server log directory is created if it does not exist."""
        server_log_dir = tmp_path / "deep" / "nested" / "svc"
        assert not server_log_dir.exists()

        proc = _run_command(
            "echo test",
            working_dir_path=tmp_path,
            server_log_dir=server_log_dir,
        )
        proc.wait()

        assert (server_log_dir / "stdout.log").exists()
        assert (server_log_dir / "stderr.log").exists()

    def test_log_files_append_across_runs(self, tmp_path: Path) -> None:
        """Successive runs append to the same log files rather than overwriting."""
        server_log_dir = tmp_path / "logs" / "svc"
        for i in range(3):
            proc = _run_command(
                f'echo "run {i}"',
                working_dir_path=tmp_path,
                server_log_dir=server_log_dir,
            )
            proc.wait()

        content = (server_log_dir / "stdout.log").read_text()
        assert "run 0" in content
        assert "run 1" in content
        assert "run 2" in content

    def test_failed_command_still_captures_output(self, tmp_path: Path) -> None:
        """Even when the command fails, output written before failure is captured."""
        server_log_dir = tmp_path / "logs" / "failing_svc"
        proc = _run_command(
            'echo "before failure" >&2 && exit 1',
            working_dir_path=tmp_path,
            server_log_dir=server_log_dir,
        )
        proc.wait()

        assert proc.returncode == 1
        stderr_content = (server_log_dir / "stderr.log").read_text()
        assert "before failure" in stderr_content


class TestRunHelperPollLogging:
    def test_poll_includes_log_paths_and_output_on_crash(self, tmp_path: Path) -> None:
        """When a server process dies, the error includes log dir path and tails both stdout and stderr."""
        from nemo_gym.cli import RunHelper

        rh = RunHelper()
        rh._log_dir = tmp_path / "logs"
        rh._head_server_thread = MagicMock(is_alive=MagicMock(return_value=True))

        # Create a mock dead process
        dead_proc = MagicMock()
        dead_proc.poll.return_value = 1
        rh._processes = {"my_server": dead_proc}

        # Create the log dir matching the flat {server_name} layout
        server_log_dir = rh._log_dir / "my_server"
        server_log_dir.mkdir(parents=True)
        (server_log_dir / "stdout.log").write_text("Starting server on port 5001\n")
        (server_log_dir / "stderr.log").write_text("ImportError: No module named 'foo'\n")

        with pytest.raises(RuntimeError) as exc_info:
            rh.poll()

        error_msg = str(exc_info.value)
        assert "my_server" in error_msg
        # Both stdout and stderr content are shown inline
        assert "Starting server on port 5001" in error_msg
        assert "ImportError: No module named 'foo'" in error_msg

    def test_poll_without_log_dir_falls_back_to_pipes(self, tmp_path: Path) -> None:
        """Without _log_dir, poll() falls back to reading from process pipes."""
        from nemo_gym.cli import RunHelper

        rh = RunHelper()
        rh._log_dir = None
        rh._head_server_thread = MagicMock(is_alive=MagicMock(return_value=True))

        dead_proc = MagicMock()
        dead_proc.poll.return_value = 1
        dead_proc.communicate.return_value = (b"some stdout", b"some stderr")
        rh._processes = {"my_server": dead_proc}

        with pytest.raises(RuntimeError, match="some stderr"):
            rh.poll()
