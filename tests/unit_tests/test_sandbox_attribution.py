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

import pytest

from nemo_gym.sandbox import attribution
from nemo_gym.sandbox.attribution import log_attribution_once, resolve_attribution, resolve_run_id


pytestmark = pytest.mark.sandbox


def test_explicit_values_win_over_env() -> None:
    resolved = resolve_attribution(
        team="cfg-team",
        user="cfg-user",
        workload="cfg-workload",
        environ={"NEMO_GYM_TEAM": "env-team", "NEMO_GYM_USER": "env-user", "NEMO_GYM_WORKLOAD": "env-workload"},
    )
    assert resolved == {"team": "cfg-team", "user": "cfg-user", "workload": "cfg-workload"}


def test_nemo_gym_env_beats_slurm_env() -> None:
    environ = {
        "NEMO_GYM_TEAM": "gym-team",
        "SLURM_JOB_ACCOUNT": "slurm-account",
        "NEMO_GYM_USER": "gym-user",
        "SLURM_JOB_USER": "slurm-user",
        "NEMO_GYM_WORKLOAD": "gym-workload",
        "SLURM_JOB_NAME": "slurm-job",
    }
    assert resolve_attribution(environ=environ) == {
        "team": "gym-team",
        "user": "gym-user",
        "workload": "gym-workload",
    }


def test_slurm_env_fallback() -> None:
    environ = {"SLURM_JOB_ACCOUNT": "account", "SLURM_JOB_USER": "slurm-user", "SLURM_JOB_NAME": "job-name"}
    assert resolve_attribution(environ=environ) == {
        "team": "account",
        "user": "slurm-user",
        "workload": "job-name",
    }


def test_workload_falls_back_to_gym_config_path() -> None:
    resolved = resolve_attribution(environ={"NEMO_GYM_CONFIG_PATH": "my_agent_server"})
    assert resolved["workload"] == "my_agent_server"


def test_slurm_job_name_beats_gym_config_path() -> None:
    environ = {"SLURM_JOB_NAME": "job-name", "NEMO_GYM_CONFIG_PATH": "my_agent_server"}
    assert resolve_attribution(environ=environ)["workload"] == "job-name"


def test_user_falls_back_to_login_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(attribution.getpass, "getuser", lambda: "login-user")
    assert resolve_attribution(environ={}) == {"user": "login-user"}


def test_root_login_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    # Containers default to root; "root" attributes the image, not a person.
    monkeypatch.setattr(attribution.getpass, "getuser", lambda: "root")
    assert resolve_attribution(environ={}) == {}


def test_root_env_user_is_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    # Only the OS-login fallback filters root; an explicit env value is deliberate.
    assert resolve_attribution(environ={"NEMO_GYM_USER": "root"}) == {"user": "root"}


def test_unresolvable_fields_are_omitted(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_no_user() -> str:
        raise OSError("no passwd entry for uid")

    monkeypatch.setattr(attribution.getpass, "getuser", raise_no_user)
    assert resolve_attribution(environ={}) == {}


def test_blank_env_and_login_values_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(attribution.getpass, "getuser", lambda: "  ")
    environ = {"NEMO_GYM_TEAM": "   ", "SLURM_JOB_ACCOUNT": "", "SLURM_JOB_NAME": "job"}
    assert resolve_attribution(environ=environ) == {"workload": "job"}


def test_environ_defaults_to_os_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEMO_GYM_TEAM", "process-env-team")
    assert resolve_attribution()["team"] == "process-env-team"


def test_run_id_explicit_wins_over_env() -> None:
    assert resolve_run_id("explicit-run", environ={"NEMO_GYM_RUN_ID": "env-run"}) == "explicit-run"


def test_run_id_env_fallback() -> None:
    assert resolve_run_id(environ={"NEMO_GYM_RUN_ID": "env-run"}) == "env-run"


def test_run_id_generated_once_per_process(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(attribution, "_process_run_id", None)
    first = resolve_run_id(environ={})
    assert first
    assert resolve_run_id(environ={}) == first


def test_log_attribution_once_logs_only_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(attribution, "_logged_attribution", False)
    with caplog.at_level(logging.INFO, logger=attribution.LOGGER.name):
        log_attribution_once({})  # empty metadata is not worth a log line
        log_attribution_once({"nemo-gym.nvidia.com/run": "run-1"})
        log_attribution_once({"nemo-gym.nvidia.com/run": "run-1"})
    matching = [record for record in caplog.records if "Sandbox attribution metadata" in record.getMessage()]
    assert len(matching) == 1
    assert "run-1" in matching[0].getMessage()
