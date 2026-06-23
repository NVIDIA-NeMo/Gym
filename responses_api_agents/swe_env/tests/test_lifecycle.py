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

"""Sandbox lifecycle (``acquire_sandbox``) and ``verify_task`` happy/timeout/empty paths.

These tests cover always-teardown on context exit and the fresh-sandbox verify
sequence, including the resolved, empty-patch fast path, and eval-timeout cases.
"""

from __future__ import annotations

import asyncio

import pytest

import responses_api_agents.swe_env.harnesses  # noqa: F401  (register harnesses)
from nemo_gym.sandbox import SandboxExecResult, SandboxHandle, SandboxStatus
from resources_servers.swe_env.verify_task import verify_task
from responses_api_agents.swe_env.harness import SweTask
from responses_api_agents.swe_env.harnesses.swe_bench_ext import SweBenchExtHarness
from responses_api_agents.swe_env.lifecycle import acquire_sandbox


class _CountingProvider:
    """Provider instance passed directly so the test can count create/close/exec.

    Args:
        exec_sleep: Seconds to sleep inside each ``exec`` call, used to simulate a
            slow evaluation that triggers the eval timeout.
        test_output: Stdout returned for pytest commands. The trailing-status
            pytest format is the shape the test parser recognizes, and the ``.py``
            path normalizes to the F2P id in ``_task``.
    """

    name = "fake-life"

    def __init__(self, *, exec_sleep=0.0, test_output="tests/test_x.py::a PASSED\n"):
        self.create_count = 0
        self.close_count = 0
        self._exec_sleep = exec_sleep
        self._test_output = test_output

    async def create(self, spec):
        self.create_count += 1
        return SandboxHandle(
            sandbox_id=f"sb-{self.create_count}", provider_name=self.name, raw={"workdir": spec.workdir}
        )

    async def exec(self, handle, command, *, cwd=None, env=None, timeout_s=None, user=None):
        if self._exec_sleep:
            await asyncio.sleep(self._exec_sleep)
        if "pytest" in command:
            return SandboxExecResult(stdout=self._test_output, stderr="", return_code=0)
        return SandboxExecResult(stdout="", stderr="", return_code=0)

    async def upload_file(self, *a, **k):
        return None

    async def download_file(self, *a, **k):
        return None

    async def status(self, handle):
        return SandboxStatus.RUNNING

    async def close(self, handle):
        self.close_count += 1

    async def aclose(self):
        return None


def _task(**kw) -> SweTask:
    """Build a SweTask with sensible defaults, overridable per keyword.

    Args:
        **kw: Field overrides merged onto the default task fields.

    Returns:
        A SweTask configured for the swe-bench-ext benchmark.
    """
    base = dict(
        instance_id="inst-1",
        image="img:tag",
        base_commit="HEAD",
        test_command="python -m pytest -rA -q",
        model_patch="diff --git a/x b/x\n",
        test_framework="pytest",
        fail_to_pass=["tests/test_x.py::a"],
        benchmark="swe-bench-ext",
    )
    base.update(kw)
    return SweTask(**base)


# ---- acquire_sandbox: starts an env, ALWAYS stops it ------------------------


def test_acquire_sandbox_starts_and_cleans_up():
    """``acquire_sandbox`` creates one sandbox and tears it down on normal exit."""
    provider = _CountingProvider()

    async def run():
        spec = SweBenchExtHarness().build_spec(_task())
        async with acquire_sandbox(provider, spec, instance_id="inst-1") as env:
            assert env.sandbox_id is not None
        return provider.create_count, provider.close_count

    created, closed = asyncio.run(run())
    assert created == 1
    assert closed == 1  # torn down on normal exit


def test_acquire_sandbox_cleans_up_on_exception():
    """``acquire_sandbox`` tears down the sandbox even when the body raises."""
    provider = _CountingProvider()

    async def run():
        spec = SweBenchExtHarness().build_spec(_task())
        with pytest.raises(RuntimeError):
            async with acquire_sandbox(provider, spec) as env:
                assert env.sandbox_id is not None
                raise RuntimeError("boom")

    asyncio.run(run())
    assert provider.close_count == 1  # torn down even on exception


# ---- verify_task: resolved / empty-patch fast path / eval-timeout mask -------


def test_verify_task_resolved_in_fresh_sandbox():
    """``verify_task`` resolves a passing task in a freshly created sandbox."""
    provider = _CountingProvider()
    report = asyncio.run(verify_task(provider, _task()))
    assert report.resolved is True
    assert provider.create_count == 1
    assert provider.close_count == 1


def test_verify_task_empty_patch_fast_path_no_create():
    """An empty model patch short-circuits to unresolved without creating a sandbox."""
    provider = _CountingProvider()
    report = asyncio.run(verify_task(provider, _task(model_patch="")))
    assert report.patch_exists is False
    assert report.resolved is False
    assert provider.create_count == 0  # no sandbox spun up for an empty patch


def test_verify_task_eval_timeout_masks():
    """An evaluation that exceeds the eval timeout is masked as an eval_timeout error."""
    provider = _CountingProvider(exec_sleep=0.5)
    report = asyncio.run(verify_task(provider, _task(), eval_timeout_s=0.05))
    assert report.error_kind == "eval_timeout"
