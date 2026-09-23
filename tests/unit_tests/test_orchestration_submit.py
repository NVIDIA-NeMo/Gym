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

from typing import ClassVar

import pytest
from pytest import MonkeyPatch

import nemo_gym.orchestration.submit as submit_module
from nemo_gym.orchestration.api import SlurmComputeConfig, SubmitConfig
from nemo_gym.orchestration.executors.base import BaseExecutor
from nemo_gym.orchestration.jobs import SubmissionRecord


COMPUTE = {"cluster": {"type": "slurm", "account": "my-account", "hostname": "foo"}}
SERVICE = {"container": "gym:latest", "type": "vllm", "model": "org/model"}
JOB = {"output_path": "/tmp/gym-jobs"}


def _config(benchmarks: dict) -> SubmitConfig:
    return SubmitConfig.model_validate(
        {
            "services": {"svc": SERVICE},
            "compute": COMPUTE,
            "driver": {"container": "gym:latest", "benchmarks": benchmarks},
            "job": JOB,
        }
    )


class _StubExecutor(BaseExecutor):
    """Fake executor standing in for one that hasn't implemented resumability."""

    supports_resumable: ClassVar[bool] = False
    last_run_config: ClassVar[SubmitConfig | None] = None

    def run(self, config: SubmitConfig, *, dry_run: bool = False) -> SubmissionRecord | None:
        type(self).last_run_config = config
        return None


def test_submit_raises_when_resumable_on_unsupported_executor(monkeypatch: MonkeyPatch):
    monkeypatch.setitem(submit_module._EXECUTORS, SlurmComputeConfig, _StubExecutor)
    config = _config({"gsm8k": {"resumable": True}})

    with pytest.raises(ValueError, match=r"gsm8k.*does not support resumable"):
        submit_module.submit(config)


def test_submit_lists_all_resumable_benchmarks_in_error(monkeypatch: MonkeyPatch):
    monkeypatch.setitem(submit_module._EXECUTORS, SlurmComputeConfig, _StubExecutor)
    config = _config({"gsm8k": {"resumable": True}, "mmlu": {"resumable": {"max_retries": 2}}, "other": {}})

    with pytest.raises(ValueError) as exc_info:
        submit_module.submit(config)
    assert "gsm8k" in str(exc_info.value)
    assert "mmlu" in str(exc_info.value)
    assert "other" not in str(exc_info.value)


def test_submit_proceeds_when_not_resumable_on_unsupported_executor(monkeypatch: MonkeyPatch):
    monkeypatch.setitem(submit_module._EXECUTORS, SlurmComputeConfig, _StubExecutor)
    config = _config({"gsm8k": {}})

    submit_module.submit(config)

    assert _StubExecutor.last_run_config is config


def test_submit_allows_resumable_on_supporting_executor(monkeypatch: MonkeyPatch):
    class _ResumableStubExecutor(_StubExecutor):
        supports_resumable: ClassVar[bool] = True

    monkeypatch.setitem(submit_module._EXECUTORS, SlurmComputeConfig, _ResumableStubExecutor)
    config = _config({"gsm8k": {"resumable": True}})

    submit_module.submit(config)

    assert _ResumableStubExecutor.last_run_config is config
