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
"""A verifier that runs past its own allowance is a scored timeout; a provider timeout inside it is not."""

import asyncio
from types import SimpleNamespace

import pytest

from resources_servers.terminal_bench_4 import verifier
from resources_servers.terminal_bench_4.app import _scored_verifier_failure


class _Environment:
    """Preparation succeeds at once; the test command sleeps or raises as each case needs."""

    def __init__(self, test_command):
        self.task = SimpleNamespace(
            config=SimpleNamespace(verifier=SimpleNamespace(timeout_sec=0.05, env={}, user="root"))
        )
        self.main = object()
        self._test_command = test_command

    async def exec(self, command, **kwargs):
        if command.startswith("/tests/test.sh"):
            await self._test_command()
        return SimpleNamespace(return_code=0, error_type=None, stderr="", stdout="")


async def _runs_past_the_allowance():
    await asyncio.sleep(1.0)


async def _provider_times_out_early():
    raise TimeoutError("status polling was cancelled")


def test_a_verifier_that_runs_past_its_allowance_is_a_scored_zero(tmp_path):
    diagnostics = []
    with pytest.raises(verifier.VerifierTimeoutError):
        asyncio.run(verifier.run_verifier(_Environment(_runs_past_the_allowance), tmp_path, diagnostics))
    assert diagnostics[-1]["operation"] == "verifier_execution"
    assert _scored_verifier_failure(
        [{"subphase": "verifier_execution", "exception_type": "VerifierTimeoutError"}]
    ) == ("VerifierTimeoutError")


def test_a_provider_timeout_inside_the_allowance_stays_an_unknown_status(tmp_path):
    with pytest.raises(verifier.VerifierStatusUnknownError):
        asyncio.run(verifier.run_verifier(_Environment(_provider_times_out_early), tmp_path, []))
    assert (
        _scored_verifier_failure([{"subphase": "verifier_execution", "exception_type": "VerifierStatusUnknownError"}])
        is None
    )
