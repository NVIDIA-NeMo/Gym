# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from unittest.mock import MagicMock

import pytest

from nemo_gym.server_utils import ServerClient
from resources_servers.terminal_bench_2_1.app import (
    TEST_SH_PATCHES,
    VERIFIER_COMMAND,
    TerminalBench21ResourcesServer,
    TerminalBench21ResourcesServerConfig,
)


class TestApp:
    @pytest.mark.parametrize("recover", [True, False])
    def test_verifier_installer_retries_without_executing_partial_downloads(
        self, tmp_path: Path, recover: bool
    ) -> None:
        marker = tmp_path / "executed"
        attempts = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:
                pass

            def do_GET(self) -> None:
                attempts.append(self.path)
                complete = recover and len(attempts) > 1
                content = f"printf {'complete' if complete else 'partial'} >> '{marker}'\n".encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(content) + (0 if complete else 100)))
                self.end_headers()
                self.wfile.write(content)
                self.close_connection = True

        http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        worker = Thread(target=http.serve_forever, daemon=True)
        worker.start()
        config = TerminalBench21ResourcesServerConfig(
            sandbox_provider="", sandbox_config={}, host="", port=0, entrypoint="", name=""
        )
        server = TerminalBench21ResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        script = tmp_path / "test.sh"
        script.write_text("curl -LsSf https://astral.sh/uv/0.9.5/install.sh | sh\nprintf graded\n")
        try:
            with server._patch_golden_patch_solve_sh("unlisted-task", script, TEST_SH_PATCHES) as patched:
                subprocess.run(["bash", "-n", str(patched)], check=True)
                content = (
                    Path(patched)
                    .read_text()
                    .replace(
                        "https://astral.sh/uv/0.9.5/install.sh", f"http://127.0.0.1:{http.server_port}/install.sh"
                    )
                )
                # Keep the real curl retry path while avoiding production backoff in this test.
                content = content.replace("--retry-delay 2", "--retry-delay 0")
                result = subprocess.run(["bash", "-c", content], capture_output=True, text=True, timeout=20)
        finally:
            http.shutdown()
            http.server_close()
            worker.join()
        assert len(attempts) == (2 if recover else 4)
        if recover:
            assert result.returncode == 0
            assert result.stdout == "graded"
            assert marker.read_text() == "complete"
        else:
            assert result.returncode != 0
            assert result.stdout == ""
            assert not marker.exists()

    def test_verifier_receives_bootstrap_settings_and_preserves_exit_status(self, tmp_path: Path) -> None:
        script = tmp_path / "test.sh"
        script.write_text(
            'cat "$APT_CONFIG"\nprintf "uv retries=%s\\n" "$UV_HTTP_RETRIES"\nrm "$APT_CONFIG"\nexit 7\n'
        )
        result = subprocess.run(
            ["bash", "-c", VERIFIER_COMMAND.replace("/tests/test.sh", str(script))],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 7
        assert result.stdout == 'DPkg::Lock::Timeout "300";\nuv retries=8\n'
        assert result.stderr == ""

    @pytest.mark.parametrize("recover", [True, False])
    def test_verifier_retries_release_download_without_retrying_grading(self, tmp_path: Path, recover: bool) -> None:
        installed = tmp_path / "installed"
        graded = tmp_path / "graded"
        attempts = {"installer": 0, "release": 0}

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:
                pass

            def do_GET(self) -> None:
                if self.path == "/install.sh":
                    attempts["installer"] += 1
                    content = (
                        f"#!/bin/sh\nset -e\ncurl -fsS http://127.0.0.1:{http.server_port}/release.tar.gz "
                        f"> /dev/null\nprintf installed >> '{installed}'\n"
                    ).encode()
                    status = 200
                else:
                    assert self.path == "/release.tar.gz"
                    attempts["release"] += 1
                    status = 200 if recover and attempts["release"] > 1 else 504
                    content = b"release archive" if status == 200 else b"upstream unavailable"
                self.send_response(status)
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)

        http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        worker = Thread(target=http.serve_forever, daemon=True)
        worker.start()
        config = TerminalBench21ResourcesServerConfig(
            sandbox_provider="", sandbox_config={}, host="", port=0, entrypoint="", name=""
        )
        server = TerminalBench21ResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        script = tmp_path / "test.sh"
        script.write_text(
            f"set -e\ncurl -LsSf https://astral.sh/uv/0.9.5/install.sh | sh\nprintf graded >> '{graded}'\nexit 7\n"
        )
        try:
            with server._patch_golden_patch_solve_sh("unlisted-task", script, TEST_SH_PATCHES) as patched:
                content = (
                    Path(patched)
                    .read_text()
                    .replace(
                        "https://astral.sh/uv/0.9.5/install.sh", f"http://127.0.0.1:{http.server_port}/install.sh"
                    )
                )
                result = subprocess.run(
                    ["bash", "-c", content.replace("sleep 2", "sleep 0")],
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
        finally:
            http.shutdown()
            http.server_close()
            worker.join()
        assert attempts == {"installer": 1, "release": 2 if recover else 4}
        if recover:
            assert installed.read_text() == "installed"
            assert graded.read_text() == "graded"
            assert result.returncode == 7
        else:
            assert not installed.exists()
            assert not graded.exists()
            assert result.returncode == 22

    @pytest.mark.parametrize(
        ("task", "packages"),
        [("qemu-startup", "curl expect"), ("qemu-alpine-ssh", "curl sshpass"), ("code-from-image", "curl")],
    )
    def test_verifier_bootstrap_patch_preserves_grading_and_package_requirements(
        self, tmp_path: Path, task: str, packages: str
    ) -> None:
        config = TerminalBench21ResourcesServerConfig(
            sandbox_provider="", sandbox_config={}, host="", port=0, entrypoint="", name=""
        )
        server = TerminalBench21ResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        grading = "\nuvx pytest /tests/test_outputs.py\nif [ $? -eq 0 ]; then echo 1; else echo 0; fi\n"
        original = f"#!/bin/bash\napt-get update\napt-get install -y {packages}\n" + grading
        path = tmp_path / "test.sh"
        path.write_text(original)
        with server._patch_golden_patch_solve_sh(f"terminal-bench/{task}", path, TEST_SH_PATCHES) as patched:
            content = Path(patched).read_text()
            subprocess.run(["bash", "-n", str(patched)], check=True)
            assert content.endswith(grading)
            assert f"install -y {packages} || exit $?" in content
            assert "DPkg::Lock::Timeout=300" in content
            if task.startswith("qemu-"):
                assert content.count('Dir::Etc::sourcelist="$verifier_apt_sources"') == 2
                assert "https://snapshot.debian.org/archive/debian-security/20260831T235959Z/" in content
                assert "[check-valid-until=no]" in content
                assert "trusted=yes" not in content
                assert "--allow-unauthenticated" not in content
            else:
                assert "https://deb.debian.org" in content
        assert path.read_text() == original

    def test_sanity(self) -> None:
        config = TerminalBench21ResourcesServerConfig(
            sandbox_provider="",
            sandbox_config=dict(),
            host="",
            port=0,
            entrypoint="",
            name="",
        )
        TerminalBench21ResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
