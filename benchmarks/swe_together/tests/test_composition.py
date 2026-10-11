# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_gym.global_config import (
    GlobalConfigDictParser,
    GlobalConfigDictParserConfig,
    taskset_environment_server_name,
)
from nemo_gym.rollout_collection import NG_ENVIRONMENT_SERVER_KEY, RolloutCollectionHelper
from resources_servers.swe_together.app import SWETConfig, SWETResourcesServer


@pytest.mark.parametrize("smoke", [False, True])
def test_documented_composition_can_dispatch_its_candidate(monkeypatch, smoke):
    monkeypatch.chdir(Path(__file__).resolve().parents[3])
    monkeypatch.delenv("SWE_TOGETHER_MODELS_CONFIG", raising=False)
    monkeypatch.delenv("NEMO_GYM_ALLOW_UNSUPPORTED_PAIRING", raising=False)
    for role in ("POLICY", "SIMULATOR", "JUDGE", "INTERACTION"):
        monkeypatch.setenv(role + "_API_KEY", "unused")
    for key, value in {
        "OPENSANDBOX_API_KEY": "unused",
        "OPENSANDBOX_SHARED_STORAGE_PATH": "/mnt/shared",
        "SWE_TOGETHER_POLICY_EGRESS_TARGET": "10.0.0.1",
        "SWE_TOGETHER_JUDGE_EGRESS_TARGET": "10.0.0.2",
    }.items():
        monkeypatch.setenv(key, value)
    paths = ["benchmarks/swe_together/opencode.yaml"]
    if smoke:
        paths.append("benchmarks/swe_together/smoke.yaml")
    paths.append("benchmarks/swe_together/opensandbox.yaml")
    config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.create({"config_paths": paths, "allow_unsupported_pairing": False}),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    environment = taskset_environment_server_name(config, "swe_together:full109")
    assert environment is not None
    RolloutCollectionHelper._validate_environment_servers([{NG_ENVIRONMENT_SERVER_KEY: environment}], config)
    resources_config = SWETConfig(
        name="swe_together_resources", **config["swe_together_resources"]["resources_servers"]["swe_together"]
    )
    server = SWETResourcesServer.model_construct(config=resources_config)
    records = [{"task_id": f"task-{index}", "reward": 1, "judge_score": 1} for index in range(109)]
    metrics = server.compute_metrics([records])
    assert metrics["planned"] == (109 if smoke else 218)
    assert metrics["missing"] == (0 if smoke else 109)
    assert metrics["complete_tasks"] == (109 if smoke else 0)
    assert metrics["upstream_compatibility_mean_zero_filled"] == (1 if smoke else 0.5)
