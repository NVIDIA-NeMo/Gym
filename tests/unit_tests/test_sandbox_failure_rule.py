# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
"""The sandbox rule: once the model ran a command, losing its sandbox is the model's, unless the platform shows otherwise.

A sandbox that dies, wedges, OOMs or loses its exec daemon while the model's commands run is what a model can cause
(a memory or fork bomb, ``kill -9 -1``, a command that outlives its timeout). Terminal-Bench and SWE-bench Pro score
it 0. Gym retries a sandbox failure only before the model's first command (create, start, a verifier start timeout)
or with platform evidence (``SandboxPlatformEvictionError``: an eviction, node loss or control-plane error).
"""

import pytest

from nemo_gym.failure_kinds import INFRA, MODEL, failure_cause_of, failure_cause_of_text
from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.rollout_collection import (
    NG_FAILURE_CAUSE_KEY,
    NG_FAILURE_CLASS_KEY,
    _infra_unresolved_counts,
    _masked_result_names_infra,
    is_terminal_failure,
)
from nemo_gym.sandbox.api import AsyncSandbox, SandboxSpec
from nemo_gym.sandbox.providers.base import (
    SandboxCreateError,
    SandboxOperationTimeoutError,
    SandboxPlatformEvictionError,
    SandboxPtyError,
)
from tests.unit_tests.test_sandbox import FakeSandboxProvider


class TestExecTimeSandboxErrorsAreTheModels:
    @pytest.mark.parametrize(
        "error",
        [
            SandboxOperationTimeoutError("Timed out during OpenSandbox command run after 660s; sandbox_id='x'"),
            SandboxPtyError("PTY connection closed unexpectedly (close code 1006)"),
        ],
        ids=["command-outlived-its-timeout", "shell-ended"],
    )
    def test_a_live_exec_time_sandbox_error_is_the_models(self, error: BaseException) -> None:
        assert failure_cause_of(error) == MODEL

    @pytest.mark.parametrize(
        "text",
        [
            "SandboxApiException: Failed to run command failed: Sandbox <id> has ended (OOMKilled): 1/1 observed pods",
            "SandboxTimeoutException: command timed out",
            "SandboxOperationTimeoutError: Timed out during OpenSandbox command status after 10s",
            "SandboxPtyError: PTY connection closed unexpectedly",
            "AgentInfrastructureError: mini-SWE produced no trajectory",
            "SandboxBackendUnreachableError: Sandbox was OOM-killed. OpenSandbox status: state='Error'",
        ],
    )
    def test_relayed_exec_time_sandbox_text_is_the_models(self, text: str) -> None:
        assert failure_cause_of_text(text) == MODEL

    def test_a_tb4_row_whose_agent_run_lost_its_sandbox_is_a_final_zero(self) -> None:
        row = {
            NG_FAILURE_CLASS_KEY: "infrastructure_error",
            "failure_reason": "AgentInfrastructureError: mini-SWE produced no trajectory",
            "infrastructure_error": "AgentInfrastructureError",
        }
        assert is_terminal_failure(row)

    async def test_the_exec_hard_cap_raised_inside_the_provider_is_the_models(self) -> None:
        # A command that wedges past every timeout: the provider raises a bare TimeoutError from its own module.
        import nemo_gym.sandbox.providers.opensandbox.provider as provider

        namespace = {"__name__": provider.__name__}
        exec(
            "def wedge():\n    raise TimeoutError('OpenSandbox exec exceeded hard cap of 1350s; the command wedged')",
            namespace,
        )
        with pytest.raises(TimeoutError) as caught:
            namespace["wedge"]()
        assert failure_cause_of(caught.value) == MODEL


class TestPlatformEvidenceAndCreateStayInfra:
    def test_platform_eviction_is_infra(self) -> None:
        error = SandboxPlatformEvictionError("Sandbox was evicted by the platform: reason='Evicted'")
        assert failure_cause_of(error) == INFRA
        assert failure_cause_of_text(f"SandboxPlatformEvictionError: {error}") == INFRA

    @pytest.mark.parametrize(
        "text",
        [
            "SandboxCreateError: could not create",
            "OpenSandboxCreateError: Timed out creating OpenSandbox sandbox after 1200s",
            "EnvironmentStartTimeoutError",
            "HealthcheckError",
            "AgentSetupTimeoutError",
            "VerifierStartTimeoutError",
        ],
    )
    def test_a_failure_before_the_models_first_command_stays_infra(self, text: str) -> None:
        assert failure_cause_of_text(text) == INFRA

    def test_a_live_create_error_stays_infra(self) -> None:
        assert failure_cause_of(SandboxCreateError("quota")) == INFRA

    def test_a_tb4_sandbox_create_failure_row_is_retried(self) -> None:
        row = {
            NG_FAILURE_CLASS_KEY: "infrastructure_error",
            "failure_reason": "SandboxCreateError",
            "infrastructure_error": "SandboxCreateError",
        }
        assert not is_terminal_failure(row)


class TestProviderMasksNeedAnExplicitInfraCause:
    """A ``provider_*`` mask is the producer's claim that the provider failed; only an explicit cause makes it infra."""

    @pytest.mark.parametrize("kind", ["provider_unavailable", "provider_oom_killed", "provider_quota_exhausted"])
    def test_a_provider_mask_without_a_cause_does_not_name_infra(self, kind: str) -> None:
        # A sandbox the model OOM-killed, reported as the provider's: no evidence the platform failed.
        result = {"mask_sample": True, "failure_kind": kind, "failure_reason": "Sandbox has ended (OOMKilled)"}
        assert not _masked_result_names_infra(result)

    @pytest.mark.parametrize("kind", ["provider_unavailable", "provider_oom_killed"])
    def test_a_provider_mask_with_an_explicit_infra_cause_is_retried(self, kind: str) -> None:
        result = {"mask_sample": True, "failure_kind": kind, NG_FAILURE_CAUSE_KEY: INFRA}
        assert _masked_result_names_infra(result)

    def test_other_retried_kinds_keep_naming_infra(self) -> None:
        assert _masked_result_names_infra({"mask_sample": True, "failure_kind": "judge_failed"})


class TestAMaskWithNoReasonIsUnresolvedInEvaluation:
    RESULTS = [
        {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0, "reward": 1.0},
        {TASK_INDEX_KEY_NAME: 1, ROLLOUT_INDEX_KEY_NAME: 0, "reward": 0.0, "mask_sample": True},
        {
            TASK_INDEX_KEY_NAME: 2,
            ROLLOUT_INDEX_KEY_NAME: 0,
            "reward": 0.0,
            "mask_sample": True,
            "mask_reason": "budget",
        },
    ]
    SCORED = {(0, 0), (1, 0), (2, 0)}

    def test_a_silent_mask_counts_as_unresolved_in_evaluation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("nemo_gym.rollout_collection.is_eval_mode", lambda: True)
        # The mask with no reason is owed a score it never got; the budget mask is the model's 0.
        counts = _infra_unresolved_counts({}, {}, self.SCORED, expected=3, missing=0, results=self.RESULTS)
        assert counts == (1, 3)

    def test_training_keeps_a_silent_mask_out_of_the_count(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("nemo_gym.rollout_collection.is_eval_mode", lambda: False)
        counts = _infra_unresolved_counts({}, {}, self.SCORED, expected=3, missing=0, results=self.RESULTS)
        assert counts == (0, 3)


class _FailingProvider(FakeSandboxProvider):
    """A provider whose create, upload or exec fails the way the OpenSandbox SDK does."""

    def __init__(
        self,
        *,
        create_error: BaseException | None = None,
        upload_error: BaseException | None = None,
        exec_error: BaseException | None = None,
    ) -> None:
        super().__init__()
        self._create_error, self._upload_error, self._exec_error = create_error, upload_error, exec_error

    async def create(self, spec):
        if self._create_error is not None:
            raise self._create_error
        return await super().create(spec)

    async def upload_file(self, handle, source_path, target_path):
        if self._upload_error is not None:
            raise self._upload_error
        return await super().upload_file(handle, source_path, target_path)

    async def exec(self, handle, command, **kwargs):
        if self._exec_error is not None:
            raise self._exec_error
        return await super().exec(handle, command, **kwargs)


def _sdk_error(message: str, status: int):
    sdk = pytest.importorskip("opensandbox.exceptions", reason="opensandbox SDK is not installed")
    return sdk.SandboxApiException(message, status_code=status)


class TestASandboxThatNeverStartedIsInfra:
    """Every failure inside create or start is before the model's first command, whatever the provider raised."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "failure",
        [
            {"create_error": lambda: _sdk_error("Sandbox ready timeout after create retries", 504)},
            {"create_error": lambda: SandboxOperationTimeoutError("Timed out during OpenSandbox create after 600s")},
            {"create_error": lambda: RuntimeError("provider control plane refused the request")},
            {"upload_error": lambda: _sdk_error("upload of the task files failed", 500)},
        ],
        ids=["create-504-ready-timeout", "start-timeout", "unexpected-create-error", "start-upload"],
    )
    async def test_a_create_or_start_failure_is_a_retried_create_error(self, failure: dict) -> None:
        # The SDK error is built in the test, so a missing optional SDK skips only the cases that need it.
        failure = {key: make() for key, make in failure.items()}
        sandbox = AsyncSandbox(_FailingProvider(**failure), spec=SandboxSpec(image="task", files={"/task.txt": "x"}))

        with pytest.raises(SandboxCreateError) as raised:
            await sandbox.start()

        assert failure_cause_of(raised.value) == INFRA
        assert failure_cause_of_text(f"{type(raised.value).__name__}: {raised.value}") == INFRA
        row = {NG_FAILURE_CLASS_KEY: "agent_run_error", NG_FAILURE_CAUSE_KEY: failure_cause_of(raised.value)}
        assert not is_terminal_failure(row)

    @pytest.mark.asyncio
    async def test_a_start_timeout_is_still_a_timeout_for_callers_that_catch_one(self) -> None:
        sandbox = AsyncSandbox(_FailingProvider(create_error=SandboxOperationTimeoutError("create timed out")))

        with pytest.raises(TimeoutError):
            await sandbox.start(SandboxSpec(image="task"))

    @pytest.mark.asyncio
    async def test_a_provider_create_error_passes_through_unwrapped(self) -> None:
        pytest.importorskip("opensandbox", reason="opensandbox SDK is not installed")
        from nemo_gym.sandbox.providers.opensandbox import OpenSandboxCreateError

        original = OpenSandboxCreateError("pool exhausted")
        sandbox = AsyncSandbox(_FailingProvider(create_error=original))

        with pytest.raises(OpenSandboxCreateError) as raised:
            await sandbox.start(SandboxSpec(image="task"))

        assert raised.value is original

    @pytest.mark.asyncio
    async def test_an_exec_error_after_start_is_unchanged(self) -> None:
        exec_error = _sdk_error("Failed to run command failed: execution not found", 404)
        sandbox = AsyncSandbox(_FailingProvider(exec_error=exec_error))
        await sandbox.start(SandboxSpec(image="task"))

        with pytest.raises(type(exec_error)) as raised:
            await sandbox.exec("python solve.py")

        assert raised.value is exec_error
        assert failure_cause_of(raised.value) == MODEL
