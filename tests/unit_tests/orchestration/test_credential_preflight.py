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


import logging
from typing import ClassVar

import pytest
import requests
from pytest import LogCaptureFixture, MonkeyPatch

import nemo_gym.orchestration.credential_preflight as preflight
import nemo_gym.orchestration.submit as submit_module
from nemo_gym.orchestration.api import SlurmComputeConfig, SubmitConfig
from nemo_gym.orchestration.credential_preflight import CredentialPreflightError, check_credentials
from nemo_gym.orchestration.executors.base import BaseExecutor
from nemo_gym.orchestration.jobs import BenchmarkJob, SubmissionRecord


JUDGE_URL = "https://judge.example.com/v1"
SECRET = "sk-test-secret"  # pragma: allowlist secret


def _config(run: dict, *, env: dict | None = None, forward_env: list[str] | None = None) -> SubmitConfig:
    return SubmitConfig.model_validate(
        {
            "services": {"svc": {"container": "gym:latest", "type": "vllm", "model": "org/model"}},
            "compute": {
                "cluster": {"type": "slurm", "account": "acct", "hostname": "host", "forward_env": forward_env or []}
            },
            "driver": {"container": "gym:latest", "env": env or {}, "benchmarks": {"bench": {"run": run}}},
            "job": {"output_path": "/tmp/gym-jobs"},
        }
    )


def _judge_run(api_key: str = "${oc.env:JUDGE_API_KEY}") -> dict:
    return {"judge_base_url": JUDGE_URL, "judge_api_key": api_key, "policy_base_url": "http://localhost:8000/v1"}


class _Response:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        self.ok = status_code < 400


@pytest.fixture
def calls(monkeypatch: MonkeyPatch) -> list[dict]:
    recorded: list[dict] = []

    def fake_get(url: str, **kwargs):
        recorded.append({"url": url, **kwargs})
        return _Response(200)

    monkeypatch.setattr(preflight.requests, "get", fake_get)
    return recorded


def test_valid_key_makes_one_authenticated_models_request(monkeypatch: MonkeyPatch, calls: list[dict]) -> None:
    monkeypatch.setenv("JUDGE_API_KEY", SECRET)
    config = _config(_judge_run(), env={"JUDGE_API_KEY": "host:JUDGE_API_KEY"})

    check_credentials(config)

    assert len(calls) == 1
    assert calls[0]["url"] == f"{JUDGE_URL}/models"
    assert calls[0]["headers"] == {"Authorization": f"Bearer {SECRET}"}
    assert calls[0]["timeout"] == preflight.PROBE_TIMEOUT_SECONDS


def test_policy_endpoint_is_not_probed(calls: list[dict]) -> None:
    check_credentials(_config({"policy_base_url": "https://policy.example.com/v1", "policy_api_key": "k"}))

    assert calls == []


def test_unset_forwarded_variable_names_the_variable_not_a_value(monkeypatch: MonkeyPatch, calls: list[dict]) -> None:
    monkeypatch.delenv("JUDGE_API_KEY", raising=False)
    config = _config(_judge_run(), forward_env=["JUDGE_API_KEY"])

    with pytest.raises(CredentialPreflightError, match="JUDGE_API_KEY") as error:
        check_credentials(config)

    assert JUDGE_URL in str(error.value)
    assert calls == []


def test_empty_variable_is_missing(monkeypatch: MonkeyPatch, calls: list[dict]) -> None:
    monkeypatch.setenv("JUDGE_API_KEY", "")
    config = _config(_judge_run(), env={"JUDGE_API_KEY": "host:JUDGE_API_KEY"})

    with pytest.raises(CredentialPreflightError, match="JUDGE_API_KEY.*unset or empty"):
        check_credentials(config)


def test_variable_the_job_never_receives_is_missing(calls: list[dict]) -> None:
    with pytest.raises(CredentialPreflightError, match="JUDGE_API_KEY.*not passed to the job"):
        check_credentials(_config(_judge_run()))


def test_variable_with_a_default_that_the_job_never_receives_is_not_probed(calls: list[dict]) -> None:
    check_credentials(_config(_judge_run("${oc.env:JUDGE_API_KEY,dummy}")))

    assert calls == []


def test_runtime_variable_unset_on_this_host_is_not_probed(monkeypatch: MonkeyPatch, calls: list[dict]) -> None:
    monkeypatch.delenv("JUDGE_API_KEY", raising=False)

    check_credentials(_config(_judge_run(), env={"JUDGE_API_KEY": "runtime:JUDGE_API_KEY"}))

    assert calls == []


@pytest.mark.parametrize("status", [401, 403])
def test_rejected_key_fails_naming_the_variable(monkeypatch: MonkeyPatch, status: int) -> None:
    monkeypatch.setenv("JUDGE_API_KEY", SECRET)
    monkeypatch.setattr(preflight.requests, "get", lambda url, **kwargs: _Response(status))
    config = _config(_judge_run(), env={"JUDGE_API_KEY": "host:JUDGE_API_KEY"})

    with pytest.raises(CredentialPreflightError) as error:
        check_credentials(config)

    assert "JUDGE_API_KEY" in str(error.value)
    assert JUDGE_URL in str(error.value)
    assert str(status) in str(error.value)
    assert SECRET not in str(error.value)


def test_literal_key_is_named_by_its_config_key(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(preflight.requests, "get", lambda url, **kwargs: _Response(401))

    with pytest.raises(CredentialPreflightError) as error:
        check_credentials(_config(_judge_run(SECRET)))

    assert "bench.run.judge_api_key" in str(error.value)
    assert SECRET not in str(error.value)


def test_network_error_only_warns(monkeypatch: MonkeyPatch, caplog: LogCaptureFixture) -> None:
    def timeout(url: str, **kwargs):
        raise requests.Timeout("timed out")

    monkeypatch.setenv("JUDGE_API_KEY", SECRET)
    monkeypatch.setattr(preflight.requests, "get", timeout)
    config = _config(_judge_run(), env={"JUDGE_API_KEY": "host:JUDGE_API_KEY"})

    with caplog.at_level(logging.WARNING):
        check_credentials(config)

    assert "could not reach" in caplog.text
    assert "JUDGE_API_KEY" in caplog.text
    assert SECRET not in caplog.text


def test_other_http_status_only_warns(monkeypatch: MonkeyPatch, caplog: LogCaptureFixture) -> None:
    monkeypatch.setenv("JUDGE_API_KEY", SECRET)
    monkeypatch.setattr(preflight.requests, "get", lambda url, **kwargs: _Response(404))
    config = _config(_judge_run(), env={"JUDGE_API_KEY": "host:JUDGE_API_KEY"})

    with caplog.at_level(logging.WARNING):
        check_credentials(config)

    assert "HTTP 404" in caplog.text


class _StubExecutor(BaseExecutor):
    ran: ClassVar[list[bool]] = []

    def run(self, config: SubmitConfig, *, dry_run: bool = False) -> SubmissionRecord | None:
        type(self).ran.append(dry_run)
        return None

    def resubmit(self, record: SubmissionRecord, benchmark: str) -> BenchmarkJob:  # pragma: no cover - unused
        raise NotImplementedError


@pytest.mark.parametrize("dry_run", [False, True])
def test_submit_fails_before_the_executor_runs(monkeypatch: MonkeyPatch, dry_run: bool) -> None:
    monkeypatch.setitem(submit_module._EXECUTORS, SlurmComputeConfig, _StubExecutor)
    monkeypatch.setattr(preflight.requests, "get", lambda url, **kwargs: _Response(401))
    monkeypatch.setenv("JUDGE_API_KEY", SECRET)
    _StubExecutor.ran.clear()
    config = _config(_judge_run(), env={"JUDGE_API_KEY": "host:JUDGE_API_KEY"})

    with pytest.raises(CredentialPreflightError, match="JUDGE_API_KEY"):
        submit_module.submit(config, dry_run=dry_run)

    assert _StubExecutor.ran == []


def test_opt_out_skips_the_check(monkeypatch: MonkeyPatch, calls: list[dict]) -> None:
    monkeypatch.setitem(submit_module._EXECUTORS, SlurmComputeConfig, _StubExecutor)
    _StubExecutor.ran.clear()

    # The variable is not passed to the job, which the check would refuse.
    submit_module.submit(_config(_judge_run()), credential_preflight=False)

    assert calls == []
    assert _StubExecutor.ran == [False]
