# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Judge configuration: the shipped overlays load, and removed keys do not vanish silently."""

import logging
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from resources_servers.gdpval.app import GDPValResourcesServerConfig, JudgePanelMember
from responses_api_models.openai_model.app import SimpleModelServerConfig


REPO = Path(__file__).resolve().parents[3]
BENCHMARK = REPO / "benchmarks/gdpval/config.yaml"
KIMI_OVERLAY = REPO / "resources_servers/gdpval/configs/gdpval_kimi_local_judge.yaml"
KIMI_SERVING = REPO / "responses_api_models/local_vllm_model/configs/moonshotai/Kimi-K2.6.yaml"
MINIMAX_OVERLAY = REPO / "resources_servers/gdpval/configs/gdpval_minimax_selfhosted_judge.yaml"


def _resolved(*paths: Path) -> dict:
    return OmegaConf.to_container(OmegaConf.merge(*(OmegaConf.load(path) for path in paths)), resolve=True)


def _gdpval_config(config: dict) -> GDPValResourcesServerConfig:
    return GDPValResourcesServerConfig(
        host="127.0.0.1",
        port=13000,
        name="gdpval_resources_server",
        **config["gdpval_resources_server"]["resources_servers"]["gdpval"],
    )


@pytest.fixture
def judge_env(monkeypatch):
    for variable in (
        "JUDGE_API_KEY",
        "JUDGE_BASE_URL",
        "JUDGE_GEMINI_API_KEY",
        "KIMI_CHECKPOINT_PATH",
        "MINIMAX_API_KEY",
        "MINIMAX_BASE_URL",
        "MINIMAX_MODEL",
    ):
        monkeypatch.delenv(variable, raising=False)


def test_kimi_overlay_points_the_single_judge_at_the_spawned_engine(judge_env):
    config = _resolved(BENCHMARK, KIMI_OVERLAY)
    gdpval = _gdpval_config(config)

    assert gdpval.judge_panel is None
    assert gdpval.judge_media_mode == "images_and_text"
    engine = config[gdpval.judge_model_server.name]["responses_api_models"]["local_vllm_model"]
    assert engine["model"] == "moonshotai/Kimi-K2.6"
    assert engine["vllm_serve_kwargs"]["served_model_name"] == [
        gdpval.judge_responses_create_params_overrides["model"]
    ]


def test_kimi_overlay_serves_the_standalone_kimi_settings(judge_env):
    """The overlay copies the standalone serving config; keep the two from drifting."""
    overlay = _resolved(KIMI_OVERLAY)["gdpval_judge_model_local"]["responses_api_models"]["local_vllm_model"]
    standalone = _resolved(KIMI_SERVING)["policy_model"]["responses_api_models"]["local_vllm_model"]

    for key in ("model", "uses_reasoning_parser", "vllm_serve_env_vars", "vllm_serve_kwargs"):
        assert overlay[key] == standalone[key], key


def test_minimax_overlay_repoints_the_default_judge_proxy(judge_env, monkeypatch):
    monkeypatch.setenv("MINIMAX_BASE_URL", "http://judge.invalid:5000/v1")
    monkeypatch.setenv("MINIMAX_MODEL", "minimax-m3")
    config = _resolved(BENCHMARK, MINIMAX_OVERLAY)
    gdpval = _gdpval_config(config)

    assert gdpval.judge_panel is None
    assert gdpval.judge_media_mode == "images_and_text"
    assert (gdpval.judge_handles_audio, gdpval.judge_handles_video) == (False, True)
    proxy = SimpleModelServerConfig(
        host="127.0.0.1",
        port=12000,
        name=gdpval.judge_model_server.name,
        **config[gdpval.judge_model_server.name]["responses_api_models"]["openai_model"],
    )
    assert proxy.openai_base_url == "http://judge.invalid:5000/v1"
    assert proxy.openai_model == gdpval.judge_responses_create_params_overrides["model"] == "minimax-m3"


@pytest.mark.parametrize("legacy", [True, False])
def test_removed_handles_audio_video_flag_is_migrated(legacy, caplog):
    with caplog.at_level(logging.WARNING, logger="resources_servers.gdpval.app"):
        member = JudgePanelMember(name="gemini", handles_audio_video=legacy)

    assert (member.handles_audio, member.handles_video) == (legacy, legacy)
    assert "handles_audio_video is deprecated" in caplog.text


def test_removed_handles_audio_video_flag_is_migrated_inside_the_server_config():
    config = GDPValResourcesServerConfig(
        host="127.0.0.1",
        port=13000,
        name="gdpval_resources_server",
        entrypoint="app.py",
        judge_model_server={"type": "responses_api_models", "name": "gdpval_judge_model"},
        judge_panel=[{"name": "gemini", "handles_audio_video": True}, {"name": "gpt"}],
    )

    assert [(m.handles_audio, m.handles_video) for m in config.judge_panel] == [(True, True), (False, False)]


@pytest.mark.parametrize("new_key", ["handles_audio", "handles_video"])
def test_mixing_removed_and_new_av_flags_fails(new_key):
    with pytest.raises(ValueError, match="handles_audio_video"):
        JudgePanelMember(name="gemini", handles_audio_video=True, **{new_key: False})
