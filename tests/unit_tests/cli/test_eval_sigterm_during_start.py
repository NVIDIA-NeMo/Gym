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
"""A SIGTERM while a serving `gym eval run` is still starting its servers and collector stops the run like one later:
startup unwinds, so what it started is stopped, and the run exits 143."""

import os
import signal
import time
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

import nemo_gym.artifacts.resolved_config as resolved_config
import nemo_gym.cli.eval as cli
import nemo_gym.telemetry.setup as telemetry


class _UnhandledSigterm(Exception):
    pass


def test_a_sigterm_during_server_startup_unwinds_it_and_exits_143(tmp_path, monkeypatch):
    output = tmp_path / "output.jsonl"
    prepared = tmp_path / "output" / "preprocessed_datasets" / "train.jsonl"
    prepared.parent.mkdir(parents=True)
    prepared.write_text("{}\n")
    config = OmegaConf.create(
        {
            "output_jsonl_fpath": str(output),
            "split": "train",
            "reuse_existing_data_preparation": True,
            "disable_aggregation": True,
            "disable_health_check": True,
        }
    )
    unwound = []

    def start(_config):
        try:
            # As Slurm sends it at scancel or the time limit, while a server is still loading.
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(30)
        except KeyboardInterrupt:
            unwound.append("start")
            raise

    supervisor = MagicMock()
    supervisor.start.side_effect = start
    monkeypatch.setattr(cli, "get_global_config_dict", lambda: config)
    monkeypatch.setattr(cli, "ServerSupervisor", lambda: supervisor)
    monkeypatch.setattr(cli.GlobalConfigDictParser, "filter_for_server_instance_configs", lambda *args, **kwargs: {})
    monkeypatch.setattr(cli, "_validate_split_datasets_declared", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_terminated", False)
    monkeypatch.setattr(resolved_config, "write_resolved_config", lambda *args, **kwargs: None)
    for name in ("configure_telemetry_env", "init_telemetry", "shutdown_telemetry"):
        monkeypatch.setattr(telemetry, name, lambda *args, **kwargs: None)

    def unhandled(signum, frame):
        raise _UnhandledSigterm

    # Stands in for the default action, which would end this test process.
    previous_handler = signal.signal(signal.SIGTERM, unhandled)
    try:
        with pytest.raises(SystemExit) as stopped:
            cli.e2e_rollout_collection()
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
    assert stopped.value.code == 143
    assert unwound == ["start"]
