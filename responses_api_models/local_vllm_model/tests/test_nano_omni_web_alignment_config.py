# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf
from omegaconf.errors import InterpolationResolutionError


CONFIG = (
    Path(__file__).parents[1] / "configs" / "nvidia" / "Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16-alignment.yaml"
)
PROFILE = "Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16-alignment"
PUBLIC_MODEL = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"
PUBLIC_REVISION = "24e67ea000b7c2837fc8f9488aa2008524fac8ba"  # pragma: allowlist secret


def _config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))[PROFILE]["responses_api_models"]["local_vllm_model"]


def test_public_nano_omni_profile_pins_web_alignment_runtime() -> None:
    config = _config()
    serve = config["vllm_serve_kwargs"]

    assert config["model"] == PUBLIC_MODEL
    assert serve["revision"] == PUBLIC_REVISION
    assert config["extra_body"] == {
        "temperature": 0.1,
        "top_p": 0.95,
        "max_output_tokens": 16384,
        "chat_template_kwargs": {"truncate_history_thinking": False},
    }
    assert serve["tensor_parallel_size"] == 8
    assert serve["data_parallel_size"] == 1
    assert serve["max_model_len"] == 128000
    assert serve["max_num_seqs"] == 32
    assert serve["reasoning_parser"] == "nano_v3"


def test_public_compatibility_assets_are_pinned_without_cluster_paths() -> None:
    text = CONFIG.read_text(encoding="utf-8")
    serve = _config()["vllm_serve_kwargs"]

    assert serve["tokenizer"] == PUBLIC_MODEL
    assert serve["tokenizer_revision"] == PUBLIC_REVISION
    assert "chat_template" not in serve
    assert serve["tool_call_parser"] == "qwen3_coder"
    assert serve["reasoning_parser_plugin"] == "${oc.env:NANO_V3_REASONING_PARSER_PLUGIN}"
    assert "???" not in text
    assert "/lustre/" not in text
    assert "/home/" not in text
    assert "/Users/" not in text


def test_public_profile_requires_explicit_reasoning_parser_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NANO_V3_REASONING_PARSER_PLUGIN", raising=False)
    config = OmegaConf.load(CONFIG)

    with pytest.raises(InterpolationResolutionError, match="NANO_V3_REASONING_PARSER_PLUGIN"):
        OmegaConf.to_container(config, resolve=True)

    parser_path = "/runtime-assets/nano_v3_reasoning_parser.py"
    monkeypatch.setenv("NANO_V3_REASONING_PARSER_PLUGIN", parser_path)
    config = OmegaConf.load(CONFIG)
    resolved = OmegaConf.to_container(config, resolve=True)
    profile = resolved[PROFILE]["responses_api_models"]["local_vllm_model"]
    assert profile["vllm_serve_kwargs"]["reasoning_parser_plugin"] == parser_path
