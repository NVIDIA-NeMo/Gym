# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The extracted patch skips binary files and reaches the grader byte for byte."""

import asyncio
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from responses_api_agents.anyswe_agent.agent_runner import _extract_patch, _snapshot_repo
from responses_api_agents.anyswe_agent.app import AnySweAgent, AnySweInstanceConfig
from responses_api_agents.anyswe_agent.tests.test_app import _config


class TestPatchExtraction:
    @staticmethod
    def _git_repo(path: Path, files: dict[str, bytes]) -> Path:
        path.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=path, check=True)
        subprocess.run(["git", "config", "core.autocrlf", "false"], cwd=path, check=True)
        for name, content in files.items():
            (path / name).write_bytes(content)
        subprocess.run(["git", "add", "-A"], cwd=path, check=True)
        subprocess.run(
            ["git", "-c", "user.email=t@example.com", "-c", "user.name=t", "commit", "-qm", "base"],
            cwd=path,
            check=True,
        )
        return path

    def test_patch_extraction_skips_binary_files_and_keeps_bytes(self, tmp_path: Path) -> None:
        base = {
            "crlf.py": b"x = 1\r\ny = 2\r\n",
            "latin1.txt": "caf\xe9\n".encode("latin-1"),
            "picture.gif": b"GIF89a\x00\x01",
        }
        repo = self._git_repo(tmp_path / "repo", base)
        index_path = tmp_path / "baseline.index"
        baseline_tree = _snapshot_repo(repo, index_path)

        edited = {
            "crlf.py": b"x = 1\r\ny = 3\r\n",
            "latin1.txt": "caf\xe9 cr\xe8me\n".encode("latin-1"),
            "picture.gif": b"GIF89a\x00\x02\xff",
            "scratch.gif": b"GIF89a\x00\x00\x00,",
        }
        for name, content in edited.items():
            (repo / name).write_bytes(content)
        patch = _extract_patch(repo, index_path, baseline_tree)

        assert isinstance(patch, bytes)
        assert b"Binary files" not in patch and b".gif" not in patch
        # The patch applies to a clean copy of the task repo and reproduces the agent's bytes exactly.
        clean = self._git_repo(tmp_path / "clean", base)
        (tmp_path / "patch.diff").write_bytes(patch)
        subprocess.run(["git", "apply", "--verbose", str(tmp_path / "patch.diff")], cwd=clean, check=True)
        assert (clean / "crlf.py").read_bytes() == edited["crlf.py"]
        assert (clean / "latin1.txt").read_bytes() == edited["latin1.txt"]

    def test_grader_uploads_the_patch_bytes_unchanged(self, tmp_path: Path) -> None:
        uploads: dict[str, bytes] = {}
        spec_files: list[dict] = []

        class _FakeSandbox:
            def __init__(self, provider, spec) -> None:
                spec_files.append(dict(spec.files))

            async def start(self) -> None:
                pass

            async def stop(self) -> None:
                pass

            async def upload(self, local: Path, remote: str) -> None:
                uploads[remote] = Path(local).read_bytes()

            async def exec(self, command: str, **kwargs):
                return SimpleNamespace(return_code=0, stdout="PASSED tests/t.py::test_a\n", stderr="", error_type=None)

        instance = {"FAIL_TO_PASS": ["tests/t.py::test_a"]}
        params = AnySweInstanceConfig(
            **_config().model_dump(),
            run_session_id="s",
            base_results_dir=tmp_path,
            model_server_url="http://policy:8000",
            resolved_sandbox_provider={"opensandbox": {}},
            sandbox_default_metadata={},
            problem_info={"instance_id": "pkg__repo-1", "instance_dict": json.dumps(instance)},
            body={"input": "fix it", "model": "policy"},
            persistent_dir=tmp_path,
            metrics_fpath=tmp_path / "metrics.json",
            container="registry.example.com/r2e:pkg__repo-1",
        )
        diff = b"--- a/crlf.py\n+++ b/crlf.py\n@@ -1 +1 @@\n-y = 2\r\n+y = 3 # caf\xe9\r\n"

        with patch("responses_api_agents.anyswe_agent.app.AsyncSandbox", _FakeSandbox):
            resolved = asyncio.run(AnySweAgent.__new__(AnySweAgent)._grade_r2e_patch(params, diff))

        assert resolved == (True, None)
        assert uploads == {"/root/patch.diff": diff}
        assert all("/root/patch.diff" not in files for files in spec_files)
