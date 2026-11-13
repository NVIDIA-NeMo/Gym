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

import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

import pytest

from nemo_gym.sandbox import SandboxExecResult
from resources_servers.swebench_pro_v2.verification import (
    TEST_SH_EXIT_PATH,
    TEST_SH_STATUS_PATH,
    VerificationInputsV2,
    assemble_workspace_files,
    run_verification,
)


def inputs(test_sh: str = "", patch: str = "", timeout: int | None = 60) -> VerificationInputsV2:
    return VerificationInputsV2(
        instance_id="instance_example",
        base_commit="HEAD",
        patch=patch,
        run_script="",
        parser_script="",
        selected_test_files_to_run=[],
        fail_to_pass=["required"],
        pass_to_pass=[],
        test_sh=test_sh,
        test_config="{}",
        evaluation_timeout=timeout,
    )


class ArtifactSandbox:
    """Model only the provider's command/artifact boundary, using recorded verifier outputs."""

    def __init__(
        self,
        files: dict[str, str],
        execution: SandboxExecResult | Exception | None = None,
        capture_error: str | None = None,
    ) -> None:
        self.files = files
        self.execution = execution or SandboxExecResult(stdout="", stderr="", return_code=0)
        self.capture_error = capture_error

    async def exec(self, command: str, **kwargs: object) -> SandboxExecResult:
        if command.startswith("bash "):
            if isinstance(self.execution, Exception):
                raise self.execution
            return self.execution
        remote = command.removeprefix("cat ")
        if remote == self.capture_error:
            raise ConnectionError("artifact transport lost")
        return SandboxExecResult(
            stdout=self.files.get(remote, ""), stderr="", return_code=0 if remote in self.files else 1
        )


def artifacts(reward: str = "1", output: str = "") -> dict[str, str]:
    return {
        "/logs/verifier/reward.txt": reward,
        "/logs/verifier/output.json": output,
        "/workspace/patch_apply_status": "0",
        TEST_SH_STATUS_PATH: "0",
        TEST_SH_EXIT_PATH: "0",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("reward", ["0", "1"])
@pytest.mark.parametrize("output", ["", "not json", '{"tests": [{"name": "required", "status": "PASSED"}]}'])
async def test_upstream_reward_decides_the_score_independently_of_parser_output(
    tmp_path: Path, reward: str, output: str
) -> None:
    result = await run_verification(ArtifactSandbox(artifacts(reward, output)), inputs(), tmp_path, timeout_s=60)

    assert result.completed
    assert result.resolved is (reward == "1")
    assert result.patch_applied
    assert not result.timed_out
    assert (tmp_path / "reward.txt").read_text() == reward


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "execution,files,error",
    [
        (TimeoutError("RPC timeout"), artifacts(), "RPC timeout"),
        (
            SandboxExecResult(stdout="", stderr="", return_code=None, error_type="unavailable"),
            artifacts(),
            "unavailable",
        ),
        (SandboxExecResult(stdout="no repo", stderr="", return_code=2), artifacts(), "exited with 2"),
        (None, artifacts(reward=""), "no reward"),
        (
            None,
            artifacts() | {"/workspace/test_sh.log": "ERROR: Failed to apply test patch"},
            "test patch did not apply",
        ),
    ],
)
async def test_verifier_faults_remain_inconclusive(
    tmp_path: Path, execution: SandboxExecResult | Exception | None, files: dict[str, str], error: str
) -> None:
    result = await run_verification(ArtifactSandbox(files, execution), inputs(), tmp_path, timeout_s=60)

    assert not result.completed
    assert not result.resolved
    assert error in result.error


@pytest.mark.asyncio
async def test_a_lost_artifact_read_does_not_turn_a_reward_into_a_model_verdict(tmp_path: Path) -> None:
    sandbox = ArtifactSandbox(artifacts(), capture_error="/logs/verifier/reward.txt")

    result = await run_verification(sandbox, inputs(), tmp_path, timeout_s=60)

    assert not result.completed
    assert "artifact transport lost" in result.error


@pytest.mark.asyncio
@pytest.mark.parametrize("reward", ["0", "1"])
@pytest.mark.parametrize("stream", ["stdout", "stderr"])
async def test_model_test_output_cannot_reclassify_an_upstream_verdict(
    tmp_path: Path, reward: str, stream: str
) -> None:
    files = artifacts(reward=reward, output='{"tests": []}') | {
        f"/logs/verifier/run-script-{stream}.txt": (
            "ERROR: Failed to apply test patch\nTemporary failure in name resolution\n"
        )
    }

    result = await run_verification(ArtifactSandbox(files), inputs(), tmp_path, timeout_s=60)

    assert result.completed
    assert result.resolved is (reward == "1")


class LocalVerifierSandbox:
    """Run the actual replay script and artifact reads on a temporary local filesystem."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def translate(self, text: str) -> str:
        for remote in ("/workspace", "/testbed", "/app", "/tests", "/logs"):
            text = text.replace(remote, str(self.root / remote.removeprefix("/")))
        return text

    async def exec(self, command: str, **kwargs: object) -> SandboxExecResult:
        result = subprocess.run(
            self.translate(command),
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
            timeout=8,
        )
        return SandboxExecResult(stdout=result.stdout, stderr=result.stderr, return_code=result.returncode)


def local_verifier(root: Path, request: VerificationInputsV2) -> LocalVerifierSandbox:
    sandbox = LocalVerifierSandbox(root)
    repo = root / "app"
    repo.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "source.txt").write_text("broken\n")
    subprocess.run(["git", "-C", str(repo), "add", "source.txt"], check=True)
    files, _ = assemble_workspace_files(request.instance_id, None, request.patch, asdict(request))
    for name, contents in files.items():
        path = root / "workspace" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(sandbox.translate(contents).encode())
    return sandbox


@pytest.mark.asyncio
@pytest.mark.parametrize("gold", [False, True])
async def test_real_patch_replay_scores_gold_and_empty_patch_with_the_task_verifier(
    tmp_path: Path, gold: bool
) -> None:
    patch = "diff --git a/source.txt b/source.txt\n--- a/source.txt\n+++ b/source.txt\n@@ -1 +1 @@\n-broken\n+fixed\n"
    test_sh = """if grep -qx fixed source.txt; then reward=1; else reward=0; fi
printf '%s' "$reward" > /logs/verifier/reward.txt
printf '%s' '{"tests": [{"name": "required", "status": "PASSED"}]}' > /logs/verifier/output.json
"""
    request = inputs(test_sh, patch if gold else "", timeout=None)
    sandbox = local_verifier(tmp_path / "sandbox", request)

    result = await run_verification(sandbox, request, tmp_path / "evidence", timeout_s=None)

    assert result.completed
    assert result.resolved is gold
    assert result.patch_applied
    assert (tmp_path / "evidence" / "reward.txt").read_text() == str(int(gold))


@pytest.mark.asyncio
@pytest.mark.skipif(shutil.which("timeout") is None, reason="Replay requires timeout as the task images provide")
@pytest.mark.parametrize("hang", [False, True])
async def test_real_timeout_has_no_test_exit_marker_and_a_plain_124_exit_is_preserved(
    tmp_path: Path, hang: bool
) -> None:
    test_sh = "sleep 3\n" if hang else "echo 0 > /logs/verifier/reward.txt\nexit 124\n"
    request = inputs(test_sh, timeout=1)
    sandbox = local_verifier(tmp_path / "sandbox", request)

    result = await run_verification(sandbox, request, tmp_path / "evidence", timeout_s=1)

    assert result.completed
    assert not result.resolved
    assert result.timed_out is hang
    assert (tmp_path / "evidence" / "test_sh_exit_status").read_text().strip() == ("" if hang else "124")


@pytest.mark.asyncio
@pytest.mark.parametrize("reward", ["0", "1", None])
async def test_verify_stages_v2_files_and_masks_only_a_missing_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reward: str | None
) -> None:
    from types import SimpleNamespace

    from nemo_gym.client import SESSION_ID_KEY
    from nemo_gym.core import VERIFIER_ERROR
    from resources_servers.swebench_pro.tests.test_app import request_body
    from resources_servers.swebench_pro_v2.app import SWEBenchProV2VerifyRequest
    from resources_servers.swebench_pro_v2.tests.test_app import instance_request, make_server

    test_sh = "true\n" if reward is None else f"echo {reward} > /logs/verifier/reward.txt\n"
    request = inputs(test_sh, timeout=None)
    server = make_server(is_verifying_golden_patch=True, inconclusive_verification_retries=0)
    staged = []
    sandboxes = []

    async def create(body, files=None):
        staged.append(files)
        sandbox = local_verifier(tmp_path / f"sandbox-{len(staged)}", request)

        async def stop():
            sandbox.stopped = True

        sandbox.stopped = False
        sandbox.stop = stop
        sandboxes.append(sandbox)
        return sandbox

    monkeypatch.setattr(server, "_create_sandbox", create)
    body = SWEBenchProV2VerifyRequest.model_validate(
        request_body() | instance_request().model_dump() | {"patch": "", "test_sh": test_sh}
    )

    response = await server.verify(SimpleNamespace(session={SESSION_ID_KEY: "session"}), body)

    assert staged[0]["/workspace/test.sh"] == test_sh
    assert "/workspace/config.json" in staged[0]
    assert sandboxes[0].stopped
    assert response.mask_sample is (reward is None)
    assert response.evaluation_completed is (reward is not None)
    assert response.reward == float(reward == "1")
    assert response.failure_kind == (VERIFIER_ERROR if reward is None else None)
