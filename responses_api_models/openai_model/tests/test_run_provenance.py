# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run provenance records the sampling settings this server forwards, as the server reads them."""

from pathlib import Path

import orjson
from omegaconf import OmegaConf

from nemo_gym.rollouts import provenance
from nemo_gym.utils import materialized_path_for
from responses_api_models.openai_model.app import SimpleModelServerConfig


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_a_sampling_value_forwarded_as_given_is_recorded_only_when_it_is_a_number(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(provenance, "served_models", lambda servers: {})
    config = OmegaConf.merge(
        OmegaConf.load(REPO_ROOT / "benchmarks/legal_agent_bench/model_sampling_openai.yaml"),
        {
            "lab": {"policy_temperature": "0.6", "policy_top_p": 0.95},
            "policy_model": {"responses_api_models": {"openai_model": {"request_overrides": {"seed": "7", "n": 1}}}},
        },
    )
    server = SimpleModelServerConfig.model_validate(
        {
            **OmegaConf.to_container(config.policy_model.responses_api_models.openai_model, resolve=True),
            "name": "policy_model",
            "host": "127.0.0.1",
            "port": 8000,
            "entrypoint": "app.py",
            "openai_base_url": "http://127.0.0.1:1/v1",
            "openai_api_key": "dummy",  # pragma: allowlist secret
            "openai_model": "policy",
        }
    )
    output = tmp_path / "rollouts.jsonl"
    materialized_path_for(output).write_bytes(b"")

    record = orjson.loads(provenance.write_run_provenance(output, config).read_bytes())

    assert server.extra_body["temperature"] == "0.6"
    assert server.request_overrides["seed"] == "7"
    assert record["sampling_parameters"] == {
        "policy_model.responses_api_models.openai_model.extra_body": {"top_p": 0.95},
        "policy_model.responses_api_models.openai_model.request_overrides": {"n": 1},
    }
