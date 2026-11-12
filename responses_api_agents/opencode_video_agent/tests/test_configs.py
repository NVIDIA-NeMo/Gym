# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve the actual opt-in configs without connecting to a model or sandbox."""

import json
from pathlib import Path

from omegaconf import OmegaConf

from nemo_gym.sandbox.providers.base import SandboxResources
from resources_servers.videomme_v2.app import VideoMMEV2ResourcesServerConfig
from responses_api_agents.opencode_video_agent.app import OpenCodeVideoAgentConfig
from responses_api_models.vllm_model.app import VLLMModelConfig


def test_agentic_configs_resolve_and_keep_separate_environment(tmp_path):
    root = Path(__file__).resolve().parents[3]
    cohort = tmp_path / "cohort.json"
    cohort.write_text(json.dumps({"dataset_revision": "a" * 40, "setting": "agentic", "task_ids": ["invented"]}))
    bundle = tmp_path / "runtime.tar.gz"
    bundle.write_bytes(b"fake bundle, not executed")
    values = {
        "videomme_v2_agentic_cohort_manifest": str(cohort),
        "video_context_tokens": 262144,
        "video_output_tokens": 32768,
        "video_media_profile": {"max_context_tokens": 262144, "reserved_output_tokens": 32768},
        "video_runtime_archive": str(bundle),
        "video_runtime_sha256": "a" * 64,
        "video_source_roots": [str(tmp_path)],
        "video_agent_artifacts_dir": str(tmp_path / "evidence"),
        "video_sandbox_image": "example.invalid/test-image@sha256:" + "a" * 64,
    }
    config = OmegaConf.to_container(
        OmegaConf.merge(
            OmegaConf.load(root / "resources_servers/videomme_v2/configs/videomme_v2_agentic.yaml"),
            OmegaConf.create(values),
        ),
        resolve=True,
    )
    resource = config["videomme_v2_agentic_resources"]["resources_servers"]["videomme_v2"]
    VideoMMEV2ResourcesServerConfig(name="agentic_resources", host="127.0.0.1", port=8001, **resource)
    agent = config["videomme_v2_agentic"]["responses_api_agents"]["opencode_video_agent"]
    parsed = OpenCodeVideoAgentConfig(name="agentic", host="127.0.0.1", port=8002, **agent)
    assert parsed.execution_failure_reward_zero and parsed.network_access == "model_only"
    # The sandbox resources are the provider-neutral ones every provider accepts.
    assert SandboxResources.from_mapping(parsed.sandbox_config["resources"]) == SandboxResources(
        cpu=2, memory_mib=8192
    )
    assert (
        config["videomme_v2_agentic_environment"]["environment_servers"]["legacy_agent"]["agent_server"]["name"]
        == "videomme_v2_agentic"
    )


def test_model_config_selects_optional_media_converter():
    root = Path(__file__).resolve().parents[3]
    config = OmegaConf.to_container(
        OmegaConf.merge(
            OmegaConf.load(root / "responses_api_models/vllm_video_model/configs/vllm_video_model.yaml"),
            OmegaConf.create(
                {
                    "policy_base_url": "http://127.0.0.1:9999/v1",
                    "policy_api_key": "EMPTY",
                    "policy_model_name": "invented-model",
                }
            ),
        ),
        resolve=True,
    )
    settings = config["policy_model"]["responses_api_models"]["vllm_video_model"]
    parsed = VLLMModelConfig(name="policy_model", host="127.0.0.1", port=8003, **settings)
    assert parsed.model == "invented-model" and parsed.return_token_id_information is False
