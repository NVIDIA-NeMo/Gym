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
"""SPEED-Bench serving defaults are applied to resolved model bindings only."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig


@pytest.fixture
def resolve(monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[2])

    def parse(overrides=None, *, benchmark="qualitative", model_name="policy_model", model_type="local_vllm_model"):
        options = {"entrypoint": "app.py"}
        if model_type == "local_vllm_model":
            options.update(model="test-model", vllm_serve_kwargs={})
        model = {"responses_api_models": {model_type: options}}
        initial = OmegaConf.create(
            {
                "config_paths": [f"benchmarks/speed-bench/config_{benchmark}.yaml"],
                "model_template": model,
                model_name: {"_inherit_from": "model_template"},
                "unrelated_model": model,
                f"speed_bench_{benchmark}_simple_agent": {
                    "responses_api_agents": {"speed_bench_agent": {"model_server": {"name": model_name}}},
                },
            }
        )
        return GlobalConfigDictParser().parse(
            GlobalConfigDictParserConfig(
                skip_load_from_cli=True,
                skip_load_from_dotenv=True,
                offline=True,
                initial_global_config_dict=OmegaConf.merge(initial, overrides or {}),
            )
        )

    return parse


@pytest.mark.parametrize("benchmark", ["qualitative", "throughput_2k"])
@pytest.mark.parametrize("model_name", ["policy_model", "custom_backend"])
def test_resolved_binding_enables_metrics_only_on_selected_model(resolve, benchmark, model_name):
    config = resolve(benchmark=benchmark, model_name=model_name)
    model = config[model_name].responses_api_models.local_vllm_model
    assert model.vllm_serve_kwargs.per_request_spec_decode_metrics == "summary"
    assert (
        "per_request_spec_decode_metrics"
        not in config.unrelated_model.responses_api_models.local_vllm_model.vllm_serve_kwargs
    )
    assert "model_template" not in config
    assert "policy_base_url" not in config


@pytest.mark.parametrize("level", ["summary", "detailed", "none"])
def test_explicit_metric_level_takes_precedence(resolve, level):
    config = resolve(
        {
            "policy_model": {
                "responses_api_models": {
                    "local_vllm_model": {
                        "vllm_serve_kwargs": {"per_request_spec_decode_metrics": level},
                    }
                }
            }
        }
    )
    assert (
        config.policy_model.responses_api_models.local_vllm_model.vllm_serve_kwargs.per_request_spec_decode_metrics
        == level
    )


def test_prometheus_mode_does_not_enable_response_metrics(resolve):
    config = resolve(
        {
            "speed_bench_qualitative_resources_server": {
                "resources_servers": {
                    "speed_bench": {
                        "metrics_source": "prometheus",
                        "vllm_base_url": "http://example.test:8000/v1",
                    }
                }
            }
        }
    )
    assert (
        "per_request_spec_decode_metrics"
        not in config.policy_model.responses_api_models.local_vllm_model.vllm_serve_kwargs
    )


@pytest.mark.parametrize("base_url", ["http://example.test:8000/v1", ["http://example.test:8000/v1"]])
def test_local_wrapper_using_external_endpoint_is_unchanged(resolve, base_url):
    config = resolve({"policy_model": {"responses_api_models": {"local_vllm_model": {"base_url": base_url}}}})
    assert (
        "per_request_spec_decode_metrics"
        not in config.policy_model.responses_api_models.local_vllm_model.vllm_serve_kwargs
    )


@pytest.mark.parametrize("model_type", ["vllm_model", "openai_model", "dummy_model"])
def test_other_model_adapters_do_not_gain_local_serving_options(resolve, model_type):
    config = resolve(model_type=model_type)
    model = config.policy_model.responses_api_models[model_type]
    assert "vllm_serve_kwargs" not in model


def test_other_benchmark_does_not_enable_metrics(resolve):
    config = resolve(
        {
            "config_paths": ["benchmarks/aime25/config.yaml"],
            "speed_bench_qualitative_simple_agent": {"_delete_key": "responses_api_agents"},
        }
    )
    assert (
        "per_request_spec_decode_metrics"
        not in config.policy_model.responses_api_models.local_vllm_model.vllm_serve_kwargs
    )
