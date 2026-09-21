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

"""The OpenTelemetry collector that runs beside every benchmark job.

One collector per benchmark job, in its own `srun --overlap` step like any other service. It
scrapes each model service's Prometheus `/metrics` on localhost, accepts OTLP from the job's own
processes, stamps the resource attributes the shared dashboards filter on (`user`, `run_id`,
`slurm_job_id`, plus `benchmark`, `cluster`, `model`), and exports to the configured OTLP/HTTP
endpoint and to `<job dir>/otel/*.jsonl` at the same time.

The ingest token travels twice, as the HTTP `Authorization` header and as an `Authorization`
resource attribute: a routing proxy in front of the backend resolves the tenant from the
attribute and silently drops payloads without it while still answering 200.
"""

import getpass
import os
from pathlib import Path

import yaml

from nemo_gym.orchestration.api import BaseModelServiceConfig, SubmitConfig


COLLECTOR_SERVICE_NAME = "otel_collector"
COLLECTOR_DIR = "otel"
COLLECTOR_CONFIG_NAME = "collector.yaml"
COLLECTOR_HEALTH_PORT = 13133
OTLP_GRPC_PORT = 4317
OTLP_HTTP_PORT = 4318
# Gym's optional-dependency group that brings nemo-lens; installed in the driver when observability
# is active so Gym's own servers emit into the collector.
GYM_TELEMETRY_EXTRA = "telemetry"


def driver_telemetry_env(gym_job_id: str, span_groups: str) -> dict[str, str]:
    """Environment that switches on Gym's Lens instrumentation and points it at the collector.

    Gym reads these in every server process (`NEMO_GYM_OTEL_*` are Gym's own, the `OTEL_*` ones
    are the SDK's); an explicit value in `driver.env` wins over these.
    """
    return {
        "NEMO_GYM_OTEL_ENABLED": "1",
        "NEMO_GYM_OTEL_RUN_ID": gym_job_id,
        "NEMO_GYM_OTEL_SPAN_GROUPS": span_groups,
        "OTEL_EXPORTER_OTLP_ENDPOINT": f"http://localhost:{OTLP_HTTP_PORT}",
        "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
    }


# Seconds the collector keeps running after the driver exits, so one more scrape sees the final
# counters before it is asked to flush and stop.
FINAL_SCRAPE_GRACE_SECONDS = 20
SHUTDOWN_WAIT_SECONDS = 30


def scrape_targets(config: SubmitConfig) -> dict[str, int]:
    """Service name to serving port for every service that exposes a model (and so `/metrics`)."""
    return {
        name: service.port for name, service in config.services.items() if isinstance(service, BaseModelServiceConfig)
    }


def otel_active(config: SubmitConfig) -> bool:
    """Whether a collector step is added to this job: enabled, and there is something to scrape."""
    return config.otel.enabled and bool(scrape_targets(config))


def validate_destination(config: SubmitConfig) -> None:
    """An enabled collector needs somewhere to send to; a bare default config has none."""
    missing = [k for k in ("endpoint", "service_name") if getattr(config.otel, k) is None]
    if missing:
        raise ValueError(
            f"The OTel collector is enabled but otel.{' and otel.'.join(missing)} is not set. "
            "Set them for this deployment, or set `otel.enabled: false`."
        )


def resolve_token(config: SubmitConfig) -> str:
    """The ingest token from the submitting environment; a missing one fails the submit."""
    name = config.otel.token_env
    token = os.environ.get(name)
    if not token:
        raise ValueError(
            f"The OTel collector is enabled but {name!r} is not set in the submitting shell's environment. "
            f"Export the ingest token as {name}, or set `otel.enabled: false`."
        )
    return token


def collector_config_path(remote_bench_dir: Path) -> Path:
    return remote_bench_dir / COLLECTOR_DIR / COLLECTOR_CONFIG_NAME


def _policy_model_name(config: SubmitConfig) -> str | None:
    if config.driver.policy_model is None:
        return None
    service = config.services[config.driver.policy_model]
    if not isinstance(service, BaseModelServiceConfig):
        return None
    return service.served_model_name or service.model


def render_collector_config(config: SubmitConfig, benchmark_name: str, remote_bench_dir: Path) -> str:
    """The collector's YAML for one benchmark job.

    `run_id` is the submission's gym job id, which is the job directory's parent by construction
    (`<output_path>/<gym_job_id>/<benchmark>`); `cluster` is the sole compute key, the same value
    `SubmissionRecord.cluster` records. Values only known inside the job (`SLURM_JOB_ID`, the
    token) are left as `${env:...}` for the collector to expand at startup.
    """
    obs = config.otel
    otel_dir = remote_bench_dir / COLLECTOR_DIR
    token = f"${{env:{obs.token_env}}}"
    interval = f"{obs.scrape_interval_seconds}s"

    # The scrape job name becomes the scraped data's `service.name`, which `transform/identity`
    # below turns into its display name; Lens-instrumented Gym servers arrive with their own.
    scrape_configs = [
        {
            "job_name": f"{obs.component}/{name}",
            "scrape_interval": interval,
            "static_configs": [{"targets": [f"localhost:{port}"], "labels": {"gym_service": name}}],
        }
        for name, port in scrape_targets(config).items()
    ]
    # Node exporters keep their conventional job names so the standard DCGM / node_exporter
    # dashboards' queries apply unchanged; DCGM already labels every sample with `hpc_job`.
    for job_name, port in (("dcgm", obs.gpu_metrics_port), ("node", obs.node_metrics_port)):
        if port is not None:
            scrape_configs.append(
                {
                    "job_name": job_name,
                    "scrape_interval": interval,
                    "static_configs": [{"targets": [f"localhost:{port}"]}],
                }
            )
    # Every producer's own `service.name` is kept as the display identity, then `service.name`
    # itself is overwritten with the routing identity the backend expects (see `resource` below).
    keep_display_name = (
        'set(resource.attributes["service.name.override"], resource.attributes["service.name"]) '
        'where resource.attributes["service.name.override"] == nil and resource.attributes["service.name"] != nil'
    )
    identity = {
        f"{signal}_statements": [{"context": "resource", "statements": [keep_display_name]}]
        for signal in ("metric", "trace", "log")
    }
    # vLLM names its metrics `vllm:<name>`; the shared dashboards, and Prometheus convention, use
    # `vllm_<name>`, and the backend keeps whatever name arrives. Renamed after parsing so counters
    # and histograms keep their types (Prometheus relabelling would make them untyped). `$$` escapes
    # the collector's own `${...}` expansion so the regexp groups reach OTTL intact.
    rename_colon_metrics = {
        "metric_statements": [
            {
                "context": "metric",
                "statements": ['replace_pattern(metric.name, "^([^:]+):(.+)$", "$${1}_$${2}")'],
            }
        ]
    }

    attributes = [
        ("service.name", obs.service_name, "upsert"),
        ("Authorization", token, "upsert"),
        ("user", getpass.getuser(), "upsert"),
        ("run_id", remote_bench_dir.parent.name, "upsert"),
        ("slurm_job_id", "${env:SLURM_JOB_ID}", "upsert"),
        ("benchmark", benchmark_name, "upsert"),
        ("cluster", next(iter(config.compute)), "upsert"),
    ]
    model = _policy_model_name(config)
    if model:
        attributes.append(("model", model, "upsert"))
    resource_actions = [{"key": k, "value": v, "action": a} for k, v, a in attributes]
    # `${env:SLURM_JOB_ID}` expands to a bare number, which the collector types as an int;
    # dashboards match it as a label string.
    resource_actions.append({"key": "slurm_job_id", "action": "convert", "converted_type": "string"})

    doc = {
        "extensions": {"health_check": {"endpoint": f"0.0.0.0:{COLLECTOR_HEALTH_PORT}"}},
        "receivers": {
            "prometheus": {"config": {"scrape_configs": scrape_configs}},
            "otlp": {
                "protocols": {
                    "grpc": {"endpoint": f"0.0.0.0:{OTLP_GRPC_PORT}"},
                    "http": {"endpoint": f"0.0.0.0:{OTLP_HTTP_PORT}"},
                }
            },
        },
        "processors": {
            "batch": {},
            "resource": {"attributes": resource_actions},
            "transform/identity": identity,
            "transform/metric_names": rename_colon_metrics,
        },
        "exporters": {
            "otlp_http/managed": {"endpoint": obs.endpoint, "headers": {"Authorization": f"Bearer {token}"}},
            "file/metrics": {"path": str(otel_dir / "metrics.jsonl")},
            "file/traces": {"path": str(otel_dir / "traces.jsonl")},
            "file/logs": {"path": str(otel_dir / "logs.jsonl")},
        },
        "service": {
            "extensions": ["health_check"],
            # otlphttp says nothing about a 2xx at info level; debug is the only way to see
            # from the log that exports are leaving at all.
            "telemetry": {"logs": {"level": "debug"}},
            "pipelines": {
                "metrics": {
                    "receivers": ["prometheus", "otlp"],
                    "processors": ["transform/metric_names", "transform/identity", "resource", "batch"],
                    "exporters": ["otlp_http/managed", "file/metrics"],
                },
                "traces": {
                    "receivers": ["otlp"],
                    "processors": ["transform/identity", "resource", "batch"],
                    "exporters": ["otlp_http/managed", "file/traces"],
                },
                "logs": {
                    "receivers": ["otlp"],
                    "processors": ["transform/identity", "resource", "batch"],
                    "exporters": ["otlp_http/managed", "file/logs"],
                },
            },
        },
    }
    return yaml.safe_dump(doc, sort_keys=False)
