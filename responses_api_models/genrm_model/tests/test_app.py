# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for GenRM model."""

from responses_api_models.genrm_model.app import GenRMModelConfig


class TestGenRMModelConfig:
    """Test GenRM model configuration."""

    def test_config_defaults(self):
        """GenRMModelConfig has supports_principle_role and inherits local vLLM fields."""
        config = GenRMModelConfig(
            host="localhost",
            port=8000,
            entrypoint="app.py",
            name="test_genrm_model",
            model="test-model",
            return_token_id_information=False,
            uses_reasoning_parser=False,
            vllm_serve_kwargs={"tensor_parallel_size": 1, "data_parallel_size": 2},
            vllm_serve_env_vars={},
        )
        assert config.supports_principle_role is True

    def test_config_supports_principle_role_override(self):
        """supports_principle_role can be set to False."""
        config = GenRMModelConfig(
            host="localhost",
            port=8000,
            entrypoint="app.py",
            name="test_genrm_model",
            model="test-model",
            return_token_id_information=False,
            uses_reasoning_parser=False,
            vllm_serve_kwargs={"tensor_parallel_size": 1, "data_parallel_size": 2},
            vllm_serve_env_vars={},
            supports_principle_role=False,
        )
        assert config.supports_principle_role is False
