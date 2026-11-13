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
from pathlib import Path

from omegaconf import OmegaConf

from nemo_gym.artifacts.server_logs import NO_LOG_LOCATION_REASON, server_log_locations, write_server_log_locations


def _config(**top_level) -> object:
    servers = {
        "policy_model": {"responses_api_models": {"vllm_model": {"entrypoint": "app.py", "host": "h", "port": 1}}},
        "telemetry_settings": {"exporter": "console"},  # a mapping that is no server
    }
    return OmegaConf.create({**servers, **top_level})


def test_gym_server_logs_are_listed_where_gym_tees_them(tmp_path: Path) -> None:
    locations = server_log_locations(_config(nemo_gym_log_dir=str(tmp_path / "logs"), limit=3), environ={})

    assert locations["status"] == "available"
    assert locations["gym_servers"] == {"policy_model": str(tmp_path / "logs" / "policy_model.log")}
    assert "services" not in locations


def test_a_submitted_jobs_step_logs_are_listed_for_its_own_job(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    for name in ("policy-123.log", "driver-123.log", "otel_collector-123.log", "policy-999.log"):
        (logs / name).write_text("")

    locations = server_log_locations(_config(), environ={"NEMO_GYM_SERVICE_LOG_DIR": str(logs), "SLURM_JOB_ID": "123"})

    assert locations["services"] == {
        "dir": str(logs),
        "slurm_job_id": "123",
        "file_pattern": "<step>-123.log",
        "files": {
            "driver": str(logs / "driver-123.log"),
            "otel_collector": str(logs / "otel_collector-123.log"),
            "policy": str(logs / "policy-123.log"),
        },
    }
    assert "gym_servers" not in locations


def test_a_run_with_no_known_log_location_says_so(tmp_path: Path) -> None:
    locations = server_log_locations(_config(), environ={"NEMO_GYM_SERVICE_LOG_DIR": str(tmp_path)})

    assert locations == {"schema_version": 1, "status": "unavailable", "reason": NO_LOG_LOCATION_REASON}


def test_the_file_is_written_beside_the_rollouts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("NEMO_GYM_SERVICE_LOG_DIR", raising=False)
    output = tmp_path / "results" / "rollouts.jsonl"

    path = write_server_log_locations(_config(nemo_gym_log_dir=str(tmp_path / "logs")), output)

    assert path == tmp_path / "results" / "rollouts_server_logs.json"
    assert json.loads(path.read_text())["gym_servers"]["policy_model"] == str(tmp_path / "logs" / "policy_model.log")
    assert [p.name for p in path.parent.iterdir()] == ["rollouts_server_logs.json"]


def test_a_no_serve_run_lists_the_folders_logs_only_as_candidates(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    for name in ("policy_model.log", "my_agent.log", "notes.txt"):
        (logs / name).write_text("")

    locations = server_log_locations(OmegaConf.create({"nemo_gym_log_dir": str(logs), "limit": 2}), environ={})

    # A shared folder can hold earlier runs' logs, so a no-serve run lists them as candidates only.
    assert locations["unverified_log_dir_files"] == {
        "my_agent": str(logs / "my_agent.log"),
        "policy_model": str(logs / "policy_model.log"),
    }
    assert "gym_servers" not in locations and locations["status"] == "unavailable"


def test_an_empty_log_folder_with_no_servers_is_no_location(tmp_path: Path) -> None:
    locations = server_log_locations(OmegaConf.create({"nemo_gym_log_dir": str(tmp_path)}), environ={})

    assert locations["status"] == "unavailable" and "gym_servers" not in locations


def test_earlier_runs_logs_in_a_shared_folder_are_not_this_runs_servers(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "policy_model.log").write_text("")
    (logs / "last_weeks_judge.log").write_text("")

    locations = server_log_locations(_config(nemo_gym_log_dir=str(logs)), environ={})

    assert locations["gym_servers"] == {"policy_model": str(logs / "policy_model.log")}
    assert locations["unverified_log_dir_files"] == {"last_weeks_judge": str(logs / "last_weeks_judge.log")}
