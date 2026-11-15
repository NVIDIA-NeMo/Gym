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
"""Tests for the shared anti-cheat git-scrub helper multiple resources servers call from their
own ``seed_session`` (see anti_cheat.py's module docstring)."""

from types import SimpleNamespace

import pytest

from resources_servers.swebench import anti_cheat
from resources_servers.swebench.anti_cheat import ANTI_CHEAT_SCRIPT_FPATH, AntiCheatSetupError, apply_anti_cheat_setup


# What anti_cheat_setup.sh prints last when it scrubbed the repository.
SCRUBBED = (
    "GIT_CLEANUP base=abc commits_before=9 commits_after=3 non_ancestor_left=0 gc_rc=0 "
    "objects_before=1+2 objects_after=0+2 elapsed_s=1\n"
)


@pytest.fixture(autouse=True)
def _no_retry_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    """The scrub retries a provider error after 2 s, then 4 s; no test waits for that."""
    monkeypatch.setattr(anti_cheat, "ANTI_CHEAT_RETRY_DELAY_S", 0.0)


class _FakeSandbox:
    def __init__(self, return_code: int = 0, stdout: str = SCRUBBED) -> None:
        self.return_code = return_code
        self.stdout = stdout
        self.uploads: list[tuple[str, str]] = []
        self.exec_calls: list[dict] = []

    async def upload(self, local_path, remote_path: str) -> None:
        self.uploads.append((str(local_path), remote_path))

    async def exec(self, command: str, **kwargs):
        self.exec_calls.append({"command": command} | kwargs)
        return SimpleNamespace(return_code=self.return_code, stdout=self.stdout, stderr="")


class TestApplyAntiCheatSetup:
    @pytest.mark.asyncio
    async def test_uploads_the_shared_script_into_the_workdir(self) -> None:
        sandbox = _FakeSandbox()
        await apply_anti_cheat_setup(sandbox, "/testbed", "inst-1", "some_server")
        assert sandbox.uploads == [(str(ANTI_CHEAT_SCRIPT_FPATH), "/testbed/anti_cheat_setup.sh")]

    @pytest.mark.asyncio
    async def test_runs_it_from_the_given_workdir(self) -> None:
        sandbox = _FakeSandbox()
        await apply_anti_cheat_setup(sandbox, "/workspace/repo", "inst-1", "some_server")
        assert len(sandbox.exec_calls) == 1
        call = sandbox.exec_calls[0]
        assert call["cwd"] == "/workspace/repo"
        assert "anti_cheat_setup.sh" in call["command"]
        assert "git reset --hard" in call["command"]

    @pytest.mark.asyncio
    async def test_a_failed_scrub_does_not_raise(self) -> None:
        """Best-effort: seed_session must still hand back a working (if unscrubbed) sandbox
        rather than failing the whole rollout over a cleanup step."""
        sandbox = _FakeSandbox(return_code=1)
        await apply_anti_cheat_setup(sandbox, "/testbed", "inst-1", "some_server")  # must not raise

    @pytest.mark.asyncio
    async def test_an_upload_exception_does_not_raise(self) -> None:
        class _BrokenSandbox(_FakeSandbox):
            async def upload(self, local_path, remote_path: str) -> None:
                raise RuntimeError("boom")

        await apply_anti_cheat_setup(_BrokenSandbox(), "/testbed", "inst-1", "some_server")  # must not raise


class _FlakySandbox(_FakeSandbox):
    """Raises on the first ``failures`` execs, as a sandbox the gateway briefly could not reach does."""

    def __init__(self, failures: int, return_code: int = 0, stdout: str = SCRUBBED) -> None:
        super().__init__(return_code, stdout)
        self.failures = failures

    async def exec(self, command: str, **kwargs):
        if self.failures:
            self.failures -= 1
            self.exec_calls.append({"command": command} | kwargs)
            raise TimeoutError("502: The sandbox could not be reached")
        return await super().exec(command, **kwargs)


class TestAntiCheatRetryAndRequired:
    @pytest.mark.asyncio
    async def test_a_provider_error_is_retried_until_the_scrub_runs(self) -> None:
        sandbox = _FlakySandbox(failures=2)
        await apply_anti_cheat_setup(sandbox, "/testbed", "inst-1", "some_server", required=True)
        assert len(sandbox.exec_calls) == 3

    @pytest.mark.asyncio
    async def test_a_required_scrub_that_keeps_raising_fails(self) -> None:
        sandbox = _FlakySandbox(failures=10)
        with pytest.raises(AntiCheatSetupError, match="inst-1"):
            await apply_anti_cheat_setup(sandbox, "/testbed", "inst-1", "some_server", required=True)
        assert len(sandbox.exec_calls) == anti_cheat.ANTI_CHEAT_ATTEMPTS

    @pytest.mark.asyncio
    async def test_a_required_scrub_that_fails_raises_without_a_retry(self) -> None:
        sandbox = _FakeSandbox(return_code=1)
        with pytest.raises(AntiCheatSetupError, match="Return code: 1"):
            await apply_anti_cheat_setup(sandbox, "/testbed", "inst-1", "some_server", required=True)
        assert len(sandbox.exec_calls) == 1

    @pytest.mark.asyncio
    async def test_a_best_effort_scrub_that_keeps_raising_is_logged_not_raised(self) -> None:
        sandbox = _FlakySandbox(failures=10)
        await apply_anti_cheat_setup(sandbox, "/testbed", "inst-1", "some_server")
        assert len(sandbox.exec_calls) == anti_cheat.ANTI_CHEAT_ATTEMPTS

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "stdout",
        [
            "",
            "GIT_CLEANUP skipped reason=no_head wd=/testbed\n",
            SCRUBBED.replace("non_ancestor_left=0", "non_ancestor_left=2"),
            SCRUBBED.replace("non_ancestor_left=0", "non_ancestor_left=?"),
        ],
        ids=["no-summary", "skipped", "refs-left", "refs-uncounted"],
    )
    async def test_a_required_scrub_that_ran_and_did_not_scrub_raises(self, stdout: str) -> None:
        """The script exits 0 on these: the exit code alone would hand the agent a history that holds the fix."""
        sandbox = _FakeSandbox(stdout=stdout)
        with pytest.raises(AntiCheatSetupError, match="inst-1"):
            await apply_anti_cheat_setup(sandbox, "/testbed", "inst-1", "some_server", required=True)
        assert len(sandbox.exec_calls) == 1

    @pytest.mark.asyncio
    async def test_a_scrub_that_did_not_scrub_is_best_effort_unless_required(self) -> None:
        sandbox = _FakeSandbox(stdout="GIT_CLEANUP skipped reason=no_head wd=/testbed\n")
        await apply_anti_cheat_setup(sandbox, "/testbed", "inst-1", "some_server")  # must not raise
        assert len(sandbox.exec_calls) == 1

    @pytest.mark.asyncio
    async def test_a_required_scrub_whose_gc_failed_after_the_refs_were_removed_is_logged(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # No name reaches the commits any more; nothing raised before, so no other gc can be in the way.
        sandbox = _FakeSandbox(stdout=SCRUBBED.replace("gc_rc=0", "gc_rc=128"))
        await apply_anti_cheat_setup(sandbox, "/testbed", "inst-1", "some_server", required=True)
        assert len(sandbox.exec_calls) == 1
        assert "git gc exited 128" in capsys.readouterr().err

    @pytest.mark.asyncio
    async def test_a_gc_that_failed_after_an_attempt_raised_is_tried_again(self) -> None:
        # The attempt that raised may still be running its gc in the sandbox, and git refuses a second one.
        sandbox = _FlakySandbox(failures=1, stdout=SCRUBBED.replace("gc_rc=0", "gc_rc=128"))
        await apply_anti_cheat_setup(sandbox, "/testbed", "inst-1", "some_server", required=True)
        assert len(sandbox.exec_calls) == anti_cheat.ANTI_CHEAT_ATTEMPTS
