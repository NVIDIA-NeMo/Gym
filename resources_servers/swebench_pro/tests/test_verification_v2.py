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

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from nemo_gym.sandbox.providers import SandboxExecResult
from resources_servers.swebench_pro.verification_v2 import (
    PATCH_PATH,
    SCRIPT_PATH,
    VERIFIER_FILES,
    replay_script,
    run_v2_verification,
    stopped_at_budget,
    v2_inconclusive_reason,
    verification_files,
)


PASSED_OUTPUT = json.dumps({"tests": [{"name": "test_new", "status": "PASSED"}]})


class FakeSandbox:
    """Answers the verification script and the `cat` reads of its outputs from a dict."""

    def __init__(self, files: dict[str, str], script_result: SandboxExecResult | Exception) -> None:
        self.files = files
        self.script_result = script_result
        self.commands: list[str] = []

    async def exec(self, command: str, timeout_s: int | None = None) -> SandboxExecResult:
        self.commands.append(command)
        if command.startswith("bash "):
            if isinstance(self.script_result, Exception):
                raise self.script_result
            return self.script_result
        path = command.removeprefix("cat ")
        if path in self.files:
            return SandboxExecResult(stdout=self.files[path], stderr="", return_code=0)
        return SandboxExecResult(stdout="", stderr="No such file", return_code=1)


def outputs(reward: str = "1", output_json: str | None = PASSED_OUTPUT, test_sh_log: str = "RESULT: PASSED") -> dict:
    files = {
        "/workspace/test_sh.log": test_sh_log,
        "/workspace/patch_apply_status": "0\n",
        "/logs/verifier/reward.txt": f"{reward}\n",
        "/logs/verifier/run-script-stdout.txt": "stdout",
        "/logs/verifier/run-script-stderr.txt": "stderr",
    }
    if output_json is not None:
        files["/logs/verifier/output.json"] = output_json
    return files


OK = SandboxExecResult(stdout="", stderr="", return_code=0)


@pytest.mark.asyncio
@pytest.mark.parametrize("reward,resolved", [("1", True), ("0", False)])
async def test_reward_file_decides_a_completed_verification(tmp_path: Path, reward: str, resolved: bool) -> None:
    sandbox = FakeSandbox(outputs(reward=reward), OK)

    result = await run_v2_verification(sandbox, tmp_path, timeout_s=60)

    assert result.completed is True
    assert result.resolved is resolved
    assert result.patch_applied is True
    assert result.test_results == json.loads(PASSED_OUTPUT)
    assert v2_inconclusive_reason(result) is None
    assert sandbox.commands[0] == f"bash {SCRIPT_PATH}"
    assert (tmp_path / "reward.txt").read_text() == f"{reward}\n"
    assert "STDOUT:\nstdout" in result.test_output


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "files,script_result,error,timed_out",
    [
        (outputs(reward=""), OK, "test.sh wrote no reward", False),
        (outputs(reward="0", test_sh_log="ERROR: Failed to apply test patch"), OK, "test patch did not apply", False),
        (
            outputs(),
            SandboxExecResult(stdout="", stderr="", return_code=None, error_type="unavailable"),
            "unavailable",
            False,
        ),
        (outputs(), TimeoutError("slow"), "Evaluation execution failed", True),
        (outputs(), SandboxExecResult(stdout="", stderr="", return_code=None, error_type="timeout"), "timeout", True),
        # A provider can report a command it killed as a plain exit code.
        (outputs(), SandboxExecResult(stdout="", stderr="", return_code=-1), "exited with -1", False),
        (outputs(), SandboxExecResult(stdout="no repo", stderr="", return_code=2), "exited with 2", False),
    ],
)
async def test_environment_faults_are_inconclusive(
    tmp_path: Path, files: dict, script_result: object, error: str, timed_out: bool
) -> None:
    result = await run_v2_verification(FakeSandbox(files, script_result), tmp_path, timeout_s=60)

    assert result.completed is False
    assert result.resolved is False
    assert error in result.error
    assert result.timed_out is timed_out
    assert v2_inconclusive_reason(result) is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "files,script_result,error,timed_out",
    [
        # Upstream's test.sh rewards 0 without parser output; that is its verdict.
        (outputs(output_json=None, reward="0"), OK, None, False),
        (outputs() | {"/workspace/test_sh_status": "124 60\n"}, OK, "budget", True),
        (outputs(reward="") | {"/workspace/test_sh_status": "137 90\n"}, OK, "budget", True),
        # BusyBox timeout ends test.sh with TERM (143) and leaves no reward.
        (outputs(reward="") | {"/workspace/test_sh_status": "143 61\n"}, OK, "budget", True),
    ],
)
async def test_upstream_zero_verdicts_are_final(
    tmp_path: Path, files: dict, script_result: object, error: str | None, timed_out: bool
) -> None:
    result = await run_v2_verification(FakeSandbox(files, script_result), tmp_path, timeout_s=60)

    assert result.completed is True
    assert result.resolved is False
    assert result.test_results is None
    assert result.timed_out is timed_out
    assert (error is None and result.error is None) or error in result.error
    assert v2_inconclusive_reason(result) is None


@pytest.mark.asyncio
async def test_dns_text_in_test_output_does_not_override_the_reward(tmp_path: Path) -> None:
    files = outputs(reward="0", output_json=json.dumps({"tests": []}))
    files["/logs/verifier/run-script-stderr.txt"] = (
        "dial tcp: lookup proxy.golang.org: Temporary failure in name resolution"
    )

    result = await run_v2_verification(FakeSandbox(files, OK), tmp_path, timeout_s=60)

    assert result.completed is True
    assert result.resolved is False
    assert v2_inconclusive_reason(result) is None


def test_verification_files_places_the_task_verifier_and_strips_binary_hunks() -> None:
    verifier = {name: f"contents of {name}" for name in VERIFIER_FILES}
    patch = (
        "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n"
        "diff --git a/image.png b/image.png\nGIT binary patch\nliteral 1\nA\n"
    )

    files = verification_files(verifier, patch, test_timeout_s=3000)

    assert {path: files[path] for path in files if path.startswith("/tests/")} == {
        f"/tests/{name}": f"contents of {name}" for name in VERIFIER_FILES
    }
    assert files[PATCH_PATH] == "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n"
    assert files[SCRIPT_PATH] == replay_script(3000)
    assert 'ng_timeout="timeout -k 30 3000"' in files[SCRIPT_PATH]
    assert "timeout -k" not in replay_script(None)


@pytest.mark.parametrize(
    "status_line,budget,expected",
    [
        ("124 3000", 3000, True),
        ("137 3030", 3000, True),
        ("143 3001", 3000, True),
        # An immediate failure that crosses a clock-second boundary is not a timeout.
        ("1 1", 1, False),
        ("124 2999", 3000, False),
        ("0 3000", 3000, False),
        ("124 3000", None, False),
        ("", 3000, False),
    ],
)
def test_stopped_at_budget_needs_a_timeout_status_and_the_whole_budget(
    status_line: str, budget: int | None, expected: bool
) -> None:
    assert stopped_at_budget(status_line, budget) is expected


def test_verification_files_rejects_an_incomplete_verifier() -> None:
    with pytest.raises(ValueError, match="test_patch.patch"):
        verification_files({name: "" for name in VERIFIER_FILES if name != "test_patch.patch"}, "", test_timeout_s=60)


@pytest.mark.skipif(
    shutil.which("git") is None or shutil.which("timeout", path="/usr/bin:/bin") is None,
    reason="git or timeout is not installed",
)
@pytest.mark.parametrize("patch_kind", ["clean", "fuzzy", "empty", "hang"])
def test_replay_script_applies_the_patch_like_patch_replay(tmp_path: Path, patch_kind: str) -> None:
    """Run the replay script for real, with its absolute paths moved under tmp_path."""
    repo = tmp_path / "app"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / "a.py").write_text("one\ntwo\nthree\n")
    subprocess.run(["git", "add", "a.py"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "base"], cwd=repo, check=True)
    patches = {
        "clean": "--- a/a.py\n+++ b/a.py\n@@ -1,3 +1,3 @@\n one\n-two\n+TWO\n three\n",
        "fuzzy": "--- a/a.py\n+++ b/a.py\n@@ -5,3 +5,3 @@\n one\n-two\n+TWO\n three\n",
        "empty": "",
        "hang": "--- a/a.py\n+++ b/a.py\n@@ -1,3 +1,3 @@\n one\n-two\n+TWO\n three\n",
    }
    workspace = tmp_path / "workspace"
    tests = tmp_path / "tests"
    workspace.mkdir()
    tests.mkdir()
    (workspace / "patch.diff").write_text(patches[patch_kind])
    # The reward also requires the sandbox daemon's telemetry env to be gone, as in V1.
    (tests / "test.sh").write_text(
        ("sleep 30\n" if patch_kind == "hang" else "")
        + f'grep -q TWO {repo}/a.py && [ -z "${{OTEL_SERVICE_NAME+x}}" ] && echo 1 > {tmp_path}/logs/verifier/reward.txt\n'
    )
    script = (
        replay_script(1)
        .replace("/workspace", str(workspace))
        .replace("/tests", str(tests))
        .replace("/logs/verifier", str(tmp_path / "logs" / "verifier"))
        .replace("cd /app", f"cd {repo}")
    )

    subprocess.run(
        ["bash", "-c", script],
        check=True,
        capture_output=True,
        env={"PATH": "/usr/bin:/bin", "OTEL_SERVICE_NAME": "execd"},
    )

    assert (workspace / "patch_apply_status").read_text().strip() == "0"
    assert ((tmp_path / "logs" / "verifier" / "reward.txt").is_file()) is (patch_kind in ("clean", "fuzzy"))
    status, elapsed = (workspace / "test_sh_status").read_text().split()
    assert status == {"hang": "124", "empty": "1"}.get(patch_kind, "0")
    assert stopped_at_budget(f"{status} {elapsed}", 1) is (patch_kind == "hang")
