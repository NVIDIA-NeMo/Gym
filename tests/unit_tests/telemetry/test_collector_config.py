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
import os

import pytest
import yaml
from pydantic import ValidationError

from nemo_gym.server_runtime.supervisor.collector import render_local_collector_config
from nemo_gym.telemetry.collector import NODE_METRICS_FILTER
from nemo_gym.telemetry.config import LOCAL_NODE_METRICS, CollectorConfig, TelemetryConfig
from nemo_gym.telemetry.setup import configure_telemetry_env


def _render(remote_export=False, **fields):
    config = CollectorConfig(**fields)
    return yaml.safe_load(
        render_local_collector_config(config, run_id="run-1", file_dir="/scratch/otel", remote_export=remote_export)
    )


def _attributes(doc):
    return {a["key"]: a["value"] for a in doc["processors"]["resource/local"]["attributes"]}


class TestTelemetryConfigWithACollector:
    def test_the_block_alone_sets_no_destination(self, monkeypatch):
        for name in ("OTEL_EXPORTER_OTLP_ENDPOINT", "OTEL_EXPORTER_OTLP_PROTOCOL"):
            monkeypatch.setenv(name, "")
            monkeypatch.delenv(name)
        monkeypatch.setenv("NEMO_GYM_OTEL_RUN_ID", "run-1")
        config = TelemetryConfig(enabled=True, collector={})
        assert (config.otlp_endpoint, config.otlp_protocol) == (None, None)
        assert configure_telemetry_env(config) == "run-1"
        assert "OTEL_EXPORTER_OTLP_ENDPOINT" not in os.environ


class TestCollectorConfigValidation:
    def test_defaults(self):
        config = CollectorConfig()
        assert (config.binary, config.remote_export, config.scrape_targets) == ("otelcol-contrib", False, {})
        assert config.local_node_metrics == list(LOCAL_NODE_METRICS)
        assert (config.gpu_metrics_port, config.node_metrics_port) == (None, None)

    @pytest.mark.parametrize(
        "fields, message",
        [
            ({"scrape_targets": {"policy": "http://localhost:8000"}}, "is not host:port"),
            ({"scrape_targets": {"policy": "localhost"}}, "is not host:port"),
            ({"resource_attributes": {"run_id": "x"}}, "run_id are set by Gym"),
            ({"remote_export": True, "endpoint": "https://otlp.example.com"}, "remote_export needs service_name"),
            (
                {"remote_export": True, "endpoint": " ", "service_name": ""},
                "remote_export needs endpoint and service_name",
            ),
            ({"remote_export": True, "endpoint": "otlp.example.com", "service_name": "s"}, "must be an http:// or"),
            ({"remote_export": True, "endpoint": "https://", "service_name": "s"}, "with a host"),
            ({"otlp_http_port": 4319}, "must differ"),
            ({"token_env": "NOT-A-NAME"}, "not a valid environment variable name"),
            ({"local_node_metrics_extra": ["node_("]}, "not a valid regular expression"),
            ({"unknown": 1}, "Extra inputs are not permitted"),
        ],
    )
    def test_rejects(self, fields, message):
        with pytest.raises(ValidationError, match=message):
            CollectorConfig(**fields)


class TestRenderLocalCollectorConfig:
    def test_listens_on_localhost_only(self):
        doc = _render(otlp_grpc_port=5317, otlp_http_port=5318, health_port=5319)
        protocols = doc["receivers"]["otlp"]["protocols"]
        assert protocols == {"grpc": {"endpoint": "127.0.0.1:5317"}, "http": {"endpoint": "127.0.0.1:5318"}}
        assert doc["extensions"]["health_check"]["endpoint"] == "127.0.0.1:5319"

    def test_scrapes_the_named_targets_and_the_node_exporters_when_set(self):
        doc = _render(scrape_targets={"policy": "localhost:8000"}, gpu_metrics_port=9400, node_metrics_port=9100)
        jobs = {
            job["job_name"]: job["static_configs"]
            for job in doc["receivers"]["prometheus"]["config"]["scrape_configs"]
        }
        assert jobs == {
            "gym-vllm/policy": [{"targets": ["localhost:8000"], "labels": {"gym_service": "policy"}}],
            "dcgm": [{"targets": ["localhost:9400"]}],
            "node": [{"targets": ["localhost:9100"]}],
        }
        assert NODE_METRICS_FILTER in doc["service"]["pipelines"]["metrics/local"]["processors"]

    def test_without_targets_there_is_no_prometheus_receiver_and_no_node_filter(self):
        doc = _render()
        assert "prometheus" not in doc["receivers"]
        assert doc["service"]["pipelines"]["metrics/local"]["receivers"] == ["otlp"]
        assert NODE_METRICS_FILTER not in doc["processors"]

    def test_writes_every_signal_into_the_scratch_directory(self):
        exporters = _render()["exporters"]
        assert {name: exporter["path"] for name, exporter in exporters.items()} == {
            "file/metrics": "/scratch/otel/metrics.jsonl",
            "file/traces": "/scratch/otel/traces.jsonl",
            "file/logs": "/scratch/otel/logs.jsonl",
        }

    def test_local_only_has_no_managed_export(self):
        doc = _render(remote_export=False, endpoint="https://otlp.example.com", service_name="svc")
        assert "otlp_http/managed" not in doc["exporters"]
        assert set(doc["service"]["pipelines"]) == {"metrics/local", "traces/local", "logs/local"}

    def test_remote_export_sends_the_token_from_its_variable(self):
        doc = _render(
            remote_export=True, endpoint="https://otlp.example.com", service_name="svc", token_env="MY_TOKEN"
        )
        managed = doc["exporters"]["otlp_http/managed"]
        assert managed == {
            "endpoint": "https://otlp.example.com",
            "headers": {"Authorization": "Bearer ${env:MY_TOKEN}"},
        }
        assert "Authorization" not in _attributes(doc)

    def test_stamps_the_run_and_the_users_attributes(self, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "12345")
        attributes = _attributes(_render(resource_attributes={"benchmark": "aime", "model": "m-${x}"}))
        assert attributes["run_id"] == "run-1"
        assert attributes["slurm_job_id"] == "12345"
        assert attributes["benchmark"] == "aime"
        # The collector would expand `${x}`; written as `$${x}` it arrives as given.
        assert attributes["model"] == "m-$${x}"
        assert {"user", "host"} <= attributes.keys()

    def test_the_users_attributes_do_not_replace_a_producers_own(self):
        doc = _render(service_name="svc", resource_attributes={"service.instance.id": "x"})
        actions = {a["key"]: a["action"] for a in doc["processors"]["resource/local"]["attributes"]}
        assert (actions["service.instance.id"], actions["service.name"], actions["run_id"]) == (
            "insert",
            "upsert",
            "upsert",
        )

    def test_outside_slurm_there_is_no_job_id(self, monkeypatch):
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        assert "slurm_job_id" not in _attributes(_render())

    def test_copies_rollout_ids_out_of_engine_spans(self):
        doc = _render()
        assert "transform/engine_rollout_id" in doc["service"]["pipelines"]["traces/local"]["processors"]
