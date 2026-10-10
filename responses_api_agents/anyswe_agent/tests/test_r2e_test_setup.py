# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""R2E-Gym grading runs the image's ``run_tests.sh`` against the tests the image keeps at ``/r2e_tests``."""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from responses_api_agents.anyswe_agent.app import AnySweAgent, AnySweInstanceConfig, _r2e_resolved
from responses_api_agents.anyswe_agent.tests.test_app import _config


# Verbatim /testbed/run_tests.sh from an R2E-Gym image (namanjain12/pillow_final:<sha>).
R2E_RUN_TESTS = (
    "PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' .venv/bin/python -W ignore -m pytest -rA r2e_tests\n"
)


def _r2e_image_layout(root: Path) -> tuple[Path, Path]:
    """A miniature R2E-Gym image: the repo with its venv and run_tests.sh, and the hidden tests at the root."""
    repo = root / "testbed"
    (repo / ".venv" / "bin").mkdir(parents=True)
    # The image's venv interpreter; a wrapper (not a symlink) so this test environment's pytest is importable.
    venv_python = repo / ".venv" / "bin" / "python"
    venv_python.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    venv_python.chmod(0o755)
    (repo / "run_tests.sh").write_text(R2E_RUN_TESTS)
    tests = root / "r2e_tests"
    tests.mkdir()
    (tests / "test_1.py").write_text("def test_fixed():\n    assert True\n")
    return repo, tests


def _run(script: str, cwd: Path) -> str:
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST_")}
    proc = subprocess.run(["bash", "-c", script], cwd=cwd, env=env, capture_output=True, text=True, timeout=120)
    return proc.stdout + proc.stderr


class TestR2ETestSetup:
    def test_run_tests_finds_the_hidden_tests_only_after_the_link(self, tmp_path: Path) -> None:
        repo, tests = _r2e_image_layout(tmp_path)
        instance = {"FAIL_TO_PASS": ["r2e_tests/test_1.py::test_fixed"], "PASS_TO_PASS": []}

        # What the grader ran before: the image's run_tests.sh straight from the repo.
        before = _run("bash run_tests.sh", repo)
        assert "file or directory not found: r2e_tests" in before
        assert _r2e_resolved(instance, before) is False

        setup = AnySweAgent._r2e_test_setup_script(str(repo), str(tests))
        after = _run(setup + "bash run_tests.sh", repo)
        assert f">>>>> Linked {tests} into {repo}" in after
        assert "PASSED r2e_tests/test_1.py::test_fixed" in after
        assert _r2e_resolved(instance, after) is True
        assert (repo / "r2e_tests").is_symlink()

    def test_leaves_tests_already_in_the_repo_alone(self, tmp_path: Path) -> None:
        repo, tests = _r2e_image_layout(tmp_path)
        (repo / "r2e_tests").mkdir()
        output = _run(AnySweAgent._r2e_test_setup_script(str(repo), str(tests)), repo)
        assert "Linked" not in output
        assert not (repo / "r2e_tests").is_symlink()

    def test_is_a_no_op_for_images_without_hidden_tests(self, tmp_path: Path) -> None:
        repo, _ = _r2e_image_layout(tmp_path)
        script = AnySweAgent._r2e_test_setup_script(str(repo), str(tmp_path / "missing"))
        proc = subprocess.run(["bash", "-c", script + "echo done"], cwd=repo, capture_output=True, text=True)
        assert proc.returncode == 0 and proc.stdout.strip() == "done"
        assert not (repo / "r2e_tests").exists()

    def test_grader_links_the_tests_before_running_them(self, tmp_path: Path) -> None:
        scripts: list[str] = []

        class _FakeSandbox:
            def __init__(self, provider, spec) -> None:
                scripts.extend(text for path, text in spec.files.items() if path.endswith("anyswe_eval.sh"))

            async def start(self) -> None:
                pass

            async def stop(self) -> None:
                pass

            async def upload(self, local: Path, remote: str) -> None:
                pass

            async def exec(self, command: str, **kwargs):
                return SimpleNamespace(return_code=0, stdout="", stderr="", error_type=None)

        params = AnySweInstanceConfig(
            **_config().model_dump(),
            run_session_id="s",
            base_results_dir=tmp_path,
            model_server_url="http://policy:8000",
            resolved_sandbox_provider={"opensandbox": {}},
            sandbox_default_metadata={},
            problem_info={"instance_id": "pkg__repo-1", "instance_dict": json.dumps({"FAIL_TO_PASS": ["t"]})},
            body={"input": "fix it", "model": "policy"},
            persistent_dir=tmp_path,
            metrics_fpath=tmp_path / "metrics.json",
            container="registry.example.com/r2e:pkg__repo-1",
        )
        with patch("responses_api_agents.anyswe_agent.app.AsyncSandbox", _FakeSandbox):
            asyncio.run(AnySweAgent.__new__(AnySweAgent)._grade_r2e_patch(params, "diff\n"))

        assert len(scripts) == 1
        script = scripts[0]
        link = script.index("ln -s /r2e_tests /testbed/r2e_tests")
        assert link < script.index("Applied Patch") < script.index("run_tests.sh")
