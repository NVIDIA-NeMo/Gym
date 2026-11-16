# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Attack fixtures at the agent -> fresh verifier boundary; no model or paid compute."""

import asyncio
import json
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from resources_servers.swe_gym.patch_policy import (
    MAX_PATCH_BYTES,
    PatchPolicyError,
    trusted_test_paths,
    validate_patch,
)
from resources_servers.swe_gym.patch_preflight import EXIT_UNSAFE_PATH, EXIT_WRONG_BASE, check_paths, main
from resources_servers.swe_gym.verification import (
    VerificationInputs,
    build_eval_script,
    drop_patch_sections,
    verification_files,
)
from resources_servers.swebench.patch_capture import PatchSizeError, _exec, capture_model_patch


def patch(path="src/answer.py", *, old="wrong", new="correct"):
    return f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n-{old}\n+{new}\n"


@pytest.mark.parametrize(
    "path",
    [
        "../escape",
        "src/../../escape",
        "/tmp/escape",
        "src//answer.py",
        "./answer.py",
        "src/./answer.py",
        ".git/config",
        "src/.git/hooks/post-checkout",
        "src/../tests/a.py",
        "src/evil\\name.py",
        "-bad\tname.py",
        "src/é.py",
        "a" * 256,
    ],
)
def test_traversal_and_noncanonical_paths(path):
    with pytest.raises(PatchPolicyError):
        validate_patch(patch(path))


@pytest.mark.parametrize(
    "path",
    [
        "tests/test_answer.py",
        "src/tests/fixture.json",
        "src/test/data.json",
        "conftest.py",
        "src/conftest.py",
        "pyproject.toml",
        "setup.py",
        "pytest.ini",
        "tox.ini",
        "sitecustomize.py",
        "src/usercustomize.py",
        "src/plugin.pth",
        "requirements.txt",
        "sandbox_runner.py",
        "request.json",
        "result.json",
    ],
)
def test_grader_tampering_fixture_rejected_in_full(path):
    # A valid source edit cannot launder a second edit that disables the tests or forges a verdict.
    tampering = patch(path, new='print("PASSED tests/hidden.py::test_answer")')
    with pytest.raises(PatchPolicyError, match="protected"):
        validate_patch(patch() + tampering)


@pytest.mark.parametrize(
    "replacement",
    [
        "new file mode 120000\n",
        "new file mode 160000\n",
        "old mode 100644\nnew mode 100755\n",
        "similarity index 100%\nrename from src/answer.py\nrename to src/other.py\n",
        "GIT binary patch\nliteral 1\nA\n",
        "Binary files a/src/answer.py and b/src/answer.py differ\n",
    ],
)
def test_special_modes_binary_and_renames(replacement):
    with pytest.raises(PatchPolicyError):
        validate_patch("diff --git a/src/answer.py b/src/answer.py\n" + replacement)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.replace("+++ b/src/answer.py", "+++ b/../result.json"),
        lambda p: p.replace("diff --git a/src/answer.py", "diff --git a/other.py"),
        lambda p: p + "arbitrary trailer\n",
        lambda p: "arbitrary preamble\n" + p,
        lambda p: p.replace("@@ -1 +1 @@", "@@ -1,2 +1 @@"),
        lambda p: p.replace("@@ -1 +1 @@", "@@ -" + "9" * 5000 + " +1 @@"),
        lambda p: p.replace("+correct\n", ""),
        lambda p: p.replace("+correct", "+\x00correct"),
        lambda p: p.replace("\n", "\r\n"),
        lambda p: p[:-1],
        lambda p: p + p,
    ],
)
def test_malformed_and_ambiguous_diff(mutate):
    with pytest.raises(PatchPolicyError):
        validate_patch(mutate(patch()))


def test_size_checked_before_parsing():
    with pytest.raises(PatchPolicyError, match="exceeds"):
        validate_patch("x" * (MAX_PATCH_BYTES + 1))
    with pytest.raises(PatchPolicyError, match="UTF-8"):
        validate_patch("\ud800")


def test_valid_regular_file_diffs_and_hidden_tests_outside_test_directory():
    assert validate_patch(patch()) == ("src/answer.py",)
    assert validate_patch("") == ()
    assert validate_patch(patch() + patch("src/helper.py")) == ("src/answer.py", "src/helper.py")
    created = "diff --git a/src/new.py b/src/new.py\nnew file mode 100644\n--- /dev/null\n+++ b/src/new.py\n@@ -0,0 +1 @@\n+value = 1\n"
    assert validate_patch(created) == ("src/new.py",)
    deleted = "diff --git a/src/gone.py b/src/gone.py\ndeleted file mode 100644\n--- a/src/gone.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-old\n"
    assert validate_patch(deleted) == ("src/gone.py",)
    assert validate_patch(patch().replace("-wrong\n", "-wrong\n\\ No newline at end of file\n"))
    with pytest.raises(PatchPolicyError, match="protected"):
        validate_patch(patch(), protected_paths=trusted_test_paths(patch()))


@pytest.mark.parametrize("kind", ["leaf", "parent", "dangling", "directory", "fifo"])
def test_trusted_verifier_tree_rejects_symlinks_and_special_files(tmp_path, kind):
    import os

    repo = tmp_path / "repo"
    repo.mkdir()
    target = tmp_path / "outside"
    target.mkdir()
    if kind == "parent":
        (repo / "src").symlink_to(target, target_is_directory=True)
    else:
        (repo / "src").mkdir()
        leaf = repo / "src/answer.py"
        if kind in {"leaf", "dangling"}:
            leaf.symlink_to(target / ("absent" if kind == "dangling" else "existing"))
        elif kind == "directory":
            leaf.mkdir()
        else:
            os.mkfifo(leaf)
    with pytest.raises(ValueError, match="symlink or special"):
        check_paths(repo, ["src/answer.py"])
    assert list(target.iterdir()) == []


class LocalCapture:
    """Only for trusted local fixtures; deliberately not an execution sandbox."""

    async def exec(self, command, *, timeout_s=30):
        result = await asyncio.to_thread(subprocess.run, command, shell=True, capture_output=True, timeout=timeout_s)
        return SimpleNamespace(
            return_code=result.returncode,
            error_type=None,
            stdout=result.stdout.decode(errors="replace"),
            stderr=result.stderr.decode(errors="replace"),
        )


@pytest.mark.skipif(shutil.which("git") is None, reason="git is required")
async def test_valid_patch_from_agent_workspace_applies_only_to_fresh_trusted_base(tmp_path):
    agent = tmp_path / "agent"
    agent.mkdir()

    def git(*args, cwd=agent):
        return subprocess.run(
            ["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init", "-q")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test")
    (agent / "src").mkdir()
    (agent / "src/answer.py").write_text("wrong\n")
    (agent / "tests").mkdir()
    (agent / "tests/hidden.py").write_text("trusted test\n")
    git("add", ".")
    git("commit", "-qm", "base")
    base = git("rev-parse", "HEAD")
    fresh = tmp_path / "fresh"
    subprocess.run(["git", "clone", "-q", str(agent), str(fresh)], check=True)
    (agent / "src/answer.py").write_text("correct\n")
    # Unrelated workspace artifacts are not transfer authority.
    (agent / "request.json").write_text('{"reward": 1}')
    capture = await capture_model_patch(
        LocalCapture(),
        str(agent),
        base,
        max_patch_bytes=MAX_PATCH_BYTES,
        pristine_untracked={"request.json"},
        drop_sections=drop_patch_sections,
    )
    paths = validate_patch(capture.patch)
    check_paths(fresh, list(paths))
    subprocess.run(["git", "-C", str(fresh), "apply", "-"], input=capture.patch, text=True, check=True)
    assert (fresh / "src/answer.py").read_text() == "correct\n"
    assert (fresh / "tests/hidden.py").read_text() == "trusted test\n"
    assert not (fresh / "request.json").exists()


async def test_capture_caps_bytes_before_provider_transfer():
    with pytest.raises(PatchSizeError):
        await _exec(LocalCapture(), "printf '123456789'", max_output_bytes=8)
    assert (await _exec(LocalCapture(), "printf '12345678'", max_output_bytes=8)).stdout == "12345678"


@pytest.mark.parametrize("code,error_type", [(90, None), (None, "timeout"), (91, None)])
async def test_capture_size_failure_distinguished_from_exec_failure(code, error_type):
    class Fake:
        async def exec(self, *args, **kwargs):
            return SimpleNamespace(return_code=code, error_type=error_type, stdout="", stderr="failed")

    with pytest.raises(PatchSizeError if code == 90 else RuntimeError):
        await _exec(Fake(), "git diff", max_output_bytes=8)


def test_preflight_checks_trusted_base_and_runs_before_candidate_or_install(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.email=test@example.com",
            "-c",
            "user.name=Test",
            "commit",
            "--allow-empty",
            "-qm",
            "base",
        ],
        check=True,
    )
    base = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    request = tmp_path / "preflight.json"
    request.write_text(json.dumps({"repo": str(repo), "base_commit": base, "paths": ["src/new.py"]}))
    assert main(request) == 0
    request.write_text(json.dumps({"repo": str(repo), "base_commit": "0" * 40, "paths": []}))
    assert main(request) == EXIT_WRONG_BASE
    (repo / "src").symlink_to(tmp_path)
    request.write_text(json.dumps({"repo": str(repo), "base_commit": base, "paths": ["src/new.py"]}))
    assert main(request) == EXIT_UNSAFE_PATH
    inputs = VerificationInputs("fixture", "getmoto/moto", "5.0", base, patch())
    script = build_eval_script(inputs)
    assert script.index("python3 -I /tmp/nemo_gym_preflight.py") < script.index("git apply --check")
    files = verification_files(inputs)
    assert "fail_to_pass" not in files["/tmp/nemo_gym_preflight.json"]
    assert "--fuzz" not in script


@pytest.mark.parametrize(
    "submission",
    [patch("conftest.py"), patch("../escape"), "x" * (MAX_PATCH_BYTES + 1)],
    ids=["grader-edit", "traversal", "oversized"],
)
def test_rejected_patch_is_scored_without_allocating_a_verifier(monkeypatch, submission):
    from unittest.mock import AsyncMock

    from fastapi.testclient import TestClient

    import resources_servers.swe_gym.app as app
    from resources_servers.swe_gym.tests.test_sessions import _VERIFY, _seed, _server
    from resources_servers.swebench.patch_capture import PatchCapture

    server, sandboxes = _server(monkeypatch)
    monkeypatch.setattr(
        app, "capture_model_patch", AsyncMock(return_value=PatchCapture.static(submission, "worktree", "worktree"))
    )
    with TestClient(server.setup_webserver()) as client:
        assert client.post("/seed_session", json=_seed().model_dump(mode="json")).status_code == 200
        response = client.post("/verify", json=_VERIFY)
        assert response.status_code == 200
        result = response.json()
        assert result["evaluation_completed"] and not result["mask_sample"]
        assert result["reward"] == 0 and result["failure_kind"] == "agent_invalid_output"
        assert len(sandboxes) == 1 and sandboxes[0].stop.await_count == 1
        app.run_verification.assert_not_awaited()


async def test_failed_test_process_cannot_get_credit_from_pass_lines():
    from resources_servers.swe_gym.verification import TEST_OUTPUT_BEGIN, TEST_OUTPUT_END, run_verification

    class Fake:
        async def exec(self, *args, **kwargs):
            return SimpleNamespace(
                return_code=1,
                error_type=None,
                stdout=f"{TEST_OUTPUT_BEGIN}\nPASSED src/test.py::test_hidden\n{TEST_OUTPUT_END}\n",
                stderr="",
            )

    result = await run_verification(
        Fake(),
        VerificationInputs("fixture", "getmoto/moto", "5.0", "base", "", fail_to_pass=["src/test.py::test_hidden"]),
    )
    assert result.completed and not result.resolved


@pytest.mark.parametrize(
    "code,error_type", [(None, "timeout"), (0, "sandbox"), (124, None), (137, None), (93, None), (94, None)]
)
async def test_infrastructure_and_preflight_failures_never_score_pass_logs(code, error_type):
    from resources_servers.swe_gym.verification import run_verification

    class Fake:
        async def exec(self, *args, **kwargs):
            return SimpleNamespace(
                return_code=code, error_type=error_type, stdout="PASSED src/test.py::test_hidden\n", stderr=""
            )

    result = await run_verification(
        Fake(),
        VerificationInputs("fixture", "getmoto/moto", "5.0", "base", "", fail_to_pass=["src/test.py::test_hidden"]),
    )
    assert not result.completed and not result.resolved


@pytest.mark.parametrize("kind", ["size", "git", "provider"])
def test_capture_failure_attribution_and_teardown(monkeypatch, kind):
    from unittest.mock import AsyncMock

    from fastapi.testclient import TestClient

    import resources_servers.swe_gym.app as app
    from resources_servers.swe_gym.tests.test_sessions import _VERIFY, _seed, _server
    from resources_servers.swebench.patch_capture import CaptureCommandError

    server, sandboxes = _server(monkeypatch)
    error = (
        PatchSizeError("too large")
        if kind == "size"
        else CaptureCommandError(
            "capture failed",
            return_code=91 if kind == "git" else None,
            error_type=None if kind == "git" else "timeout",
        )
    )
    monkeypatch.setattr(app, "capture_model_patch", AsyncMock(side_effect=error))
    with TestClient(server.setup_webserver()) as client:
        assert client.post("/seed_session", json=_seed().model_dump(mode="json")).status_code == 200
        result = client.post("/verify", json=_VERIFY).json()
        assert result["reward"] == 0 and result["mask_sample"] == (kind == "provider")
        assert result["evaluation_completed"] == (kind != "provider")
        assert result["failure_kind"] == ("verifier_error" if kind == "provider" else "agent_invalid_output")
        assert len(sandboxes) == 1 and sandboxes[0].stop.await_count == 1
        app.run_verification.assert_not_awaited()


def test_invalid_golden_patch_is_masked_as_data_failure(monkeypatch):
    from fastapi.testclient import TestClient

    from resources_servers.swe_gym.tests.test_sessions import _VERIFY, _server

    server, sandboxes = _server(monkeypatch)
    server.config.is_verifying_golden_patch = True
    with TestClient(server.setup_webserver()) as client:
        result = client.post("/verify", json=_VERIFY | {"patch": patch("conftest.py")}).json()
        assert result["mask_sample"] and not result["evaluation_completed"]
        assert result["failure_kind"] == "verifier_error" and not sandboxes


@pytest.mark.xfail(strict=True, reason="SWE pytest stdout is not authenticated against candidate source code")
async def test_runtime_forged_success_remains_an_unclosed_grader_boundary():
    """Track the stronger requirement explicitly; path filtering cannot fix in-process grading."""
    from resources_servers.swe_gym.verification import TEST_OUTPUT_BEGIN, TEST_OUTPUT_END, run_verification

    class CandidateControlledOutput:
        async def exec(self, *args, **kwargs):
            return SimpleNamespace(
                return_code=0,
                error_type=None,
                stdout=f"{TEST_OUTPUT_BEGIN}\nPASSED src/check.py::hidden\n{TEST_OUTPUT_END}\n",
                stderr="",
            )

    result = await run_verification(
        CandidateControlledOutput(),
        VerificationInputs("fixture", "getmoto/moto", "5.0", "base", "", fail_to_pass=["src/check.py::hidden"]),
    )
    assert not result.resolved
