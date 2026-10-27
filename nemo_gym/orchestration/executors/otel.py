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
endpoint (when remote export is enabled) and to `<job dir>/otel/*.jsonl`.

The ingest token travels twice, as the HTTP `Authorization` header and as an `Authorization`
resource attribute: a routing proxy in front of the backend resolves the tenant from the
attribute and silently drops payloads without it while still answering 200.
"""

import getpass
import logging
import os
from pathlib import Path

import yaml

from nemo_gym.orchestration.api import BaseModelServiceConfig, SubmitConfig
from nemo_gym.telemetry.endpoints import ROLLOUT_ID_ATTRIBUTE


logger = logging.getLogger(__name__)


COLLECTOR_SERVICE_NAME = "otel_collector"
COLLECTOR_DIR = "otel"
COLLECTOR_CONFIG_NAME = "collector.yaml"
COLLECTOR_HEALTH_PORT = 13133
OTLP_GRPC_PORT = 4317
OTLP_HTTP_PORT = 4318
GYM_TELEMETRY_EXTRA = "telemetry"


def driver_telemetry_env(gym_job_id: str, span_groups: str, *, logs: bool = True) -> dict[str, str]:
    """Environment that switches on Gym's Lens instrumentation and points it at the collector.

    Gym reads these in every server process (`NEMO_GYM_OTEL_*` are Gym's own, the `OTEL_*` ones
    are the SDK's); an explicit value in `driver.env` wins over these.
    """
    return {
        "NEMO_GYM_OTEL_ENABLED": "1",
        "NEMO_GYM_OTEL_RUN_ID": gym_job_id,
        "NEMO_GYM_OTEL_SPAN_GROUPS": span_groups,
        "NEMO_GYM_OTEL_LOGS_ENABLED": "1" if logs else "0",
        "OTEL_EXPORTER_OTLP_ENDPOINT": f"http://localhost:{OTLP_HTTP_PORT}",
        "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
    }


VLLM_TRACES_FLAG = "--otlp-traces-endpoint"


def vllm_traces_flag(host: str = "localhost") -> str:
    """The `vllm serve` flag that sends one span per request to the collector's OTLP/HTTP receiver."""
    return f"{VLLM_TRACES_FLAG} http://{host}:{OTLP_HTTP_PORT}/v1/traces"


def vllm_traces_env(component: str, service_name: str) -> dict[str, str]:
    """Environment for a traced vLLM service.

    vLLM picks the exporter by `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL` and defaults to gRPC, so the
    protocol has to match the HTTP endpoint in the flag. vLLM sets no `service.name` of its own; the
    collector copies the one given here into `service.name.override`, so the spans carry the same
    display identity as the service's scraped metrics (`<component>/<service>`).
    """
    return {
        "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL": "http/protobuf",
        "OTEL_SERVICE_NAME": f"{component}/{service_name}",
    }


# Seconds the collector keeps running after the driver exits, so one more scrape sees the final
# counters before it is asked to flush and stop.
FINAL_SCRAPE_GRACE_SECONDS = 20
SHUTDOWN_WAIT_SECONDS = 30


def scrape_targets(config: SubmitConfig) -> dict[str, int]:
    """Service name to serving port for every service that exposes a model (and so `/metrics`)."""
    return {
        name: service.port
        for name, service in config.deployed_services.items()
        if isinstance(service, BaseModelServiceConfig)
    }


def otel_active(config: SubmitConfig) -> bool:
    """Enabled is enough: Gym's own servers push telemetry even when there is no local model to scrape."""
    return config.otel.enabled


def gym_telemetry_active(config: SubmitConfig) -> bool:
    """Whether the driver is instrumented with nemo-lens and pointed at the collector."""
    return config.otel.enabled and config.otel.gym_telemetry


def validate_gym_telemetry(config: SubmitConfig) -> None:
    if gym_telemetry_active(config) and config.driver.gym_install is None:
        raise ValueError(
            "otel.gym_telemetry is on but driver.gym_install is not set, so the driver cannot install "
            "Gym's telemetry extra and no Gym metrics or traces would be exported. Set driver.gym_install "
            "(repo, ref), or set `otel.gym_telemetry: false` to run the collector with engine and node "
            "metrics only."
        )


def validate_destination(config: SubmitConfig) -> None:
    """Remote export needs a destination; local collection does not."""
    if not (otel_active(config) and config.otel.remote_export):
        return
    missing = [k for k in ("endpoint", "service_name") if getattr(config.otel, k) is None]
    if missing:
        raise ValueError(
            f"The OTel collector is enabled but otel.{' and otel.'.join(missing)} is not set. "
            "Set them for this deployment, or set `otel.remote_export: false` for local collection only."
        )


def resolve_token(config: SubmitConfig) -> str | None:
    """Resolve the remote ingest token, switching to local collection with a warning if absent."""
    if not (otel_active(config) and config.otel.remote_export):
        return None
    name = config.otel.token_env
    token = os.environ.get(name)
    if not token:
        config.otel.remote_export = False
        logger.warning(
            "%r is missing or empty in the submitting shell; setting otel.remote_export=false. "
            "Telemetry will still be written to the job's otel/ directory.",
            name,
        )
    return token or None


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

    # The job name is what scraped data carries as `service.name`, and so becomes its display identity.
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
    # Each producer's own `service.name` survives as the display identity before `resource` overwrites it.
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

    # vLLM records Gym's `X-Request-Id: <rollout id>-<8 hex>` as `gen_ai.request.id` (prefixed `chatcmpl-`), the
    # only place the engine span names its rollout. Aggregators group on whole attributes, so copy it into the
    # key every other Gym span uses. Request ids that are not Gym's (a plain uuid) do not match.
    gym_engine_request_id = "^(?:chatcmpl-)?(.+)-[0-9a-f]{8}$$"
    engine_rollout_id = {
        "trace_statements": [
            {
                "context": "span",
                "statements": [
                    # Derive in the span-local cache, write only while absent: the pinned collector has no converter
                    # form of `replace_pattern`, and an id the span already carries must stay as it is.
                    'set(cache["engine_rollout_id"], attributes["gen_ai.request.id"]) '
                    f'where name == "llm_request" and attributes["{ROLLOUT_ID_ATTRIBUTE}"] == nil '
                    'and attributes["gen_ai.request.id"] != nil '
                    f'and IsMatch(attributes["gen_ai.request.id"], "{gym_engine_request_id}")',
                    f'replace_pattern(cache["engine_rollout_id"], "{gym_engine_request_id}", "$${{1}}") '
                    'where cache["engine_rollout_id"] != nil',
                    f'set(attributes["{ROLLOUT_ID_ATTRIBUTE}"], cache["engine_rollout_id"]) '
                    f'where attributes["{ROLLOUT_ID_ATTRIBUTE}"] == nil and cache["engine_rollout_id"] != nil',
                ],
            }
        ]
    }

    attributes = [
        *([("service.name", obs.service_name, "upsert")] if obs.service_name is not None else []),
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
    # The token rides only on the managed export, never into the job-directory copies.
    managed_actions = [{"key": "Authorization", "value": token, "action": "upsert"}, *resource_actions]

    # Operations that exist only as spans (sandbox start/exec, model calls) still get latency and count series.
    span_metrics = {
        "histogram": {
            "explicit": {"buckets": ["250ms", "1s", "2s", "5s", "10s", "30s", "60s", "120s", "300s", "600s", "1800s"]}
        },
        "dimensions": [{"name": "service.name.override"}, {"name": "nemo.gym.sandbox.provider"}],
        "metrics_flush_interval": f"{obs.scrape_interval_seconds}s",
    }

    metric_receivers = (["prometheus"] if scrape_configs else []) + ["otlp", "span_metrics"]
    trace_transforms = ["transform/identity"] + (["transform/engine_rollout_id"] if obs.vllm_traces else [])
    doc = {
        "extensions": {"health_check": {"endpoint": f"0.0.0.0:{COLLECTOR_HEALTH_PORT}"}},
        "connectors": {"span_metrics": span_metrics},
        "receivers": {
            **({"prometheus": {"config": {"scrape_configs": scrape_configs}}} if scrape_configs else {}),
            "otlp": {
                "protocols": {
                    "grpc": {"endpoint": f"0.0.0.0:{OTLP_GRPC_PORT}"},
                    "http": {"endpoint": f"0.0.0.0:{OTLP_HTTP_PORT}"},
                }
            },
        },
        "processors": {
            "batch": {},
            "resource/managed": {"attributes": managed_actions},
            "resource/local": {"attributes": resource_actions},
            "transform/identity": identity,
            "transform/metric_names": rename_colon_metrics,
            **({"transform/engine_rollout_id": engine_rollout_id} if obs.vllm_traces else {}),
        },
        "exporters": {
            "otlp_http/managed": {"endpoint": obs.endpoint, "headers": {"Authorization": f"Bearer {token}"}},
            "file/metrics": {"path": str(otel_dir / "metrics.jsonl")},
            "file/traces": {"path": str(otel_dir / "traces.jsonl")},
            "file/logs": {"path": str(otel_dir / "logs.jsonl")},
        },
        "service": {
            "extensions": ["health_check"],
            # debug would print every batch, token included, into the collector log.
            "telemetry": {"logs": {"level": "info"}},
            "pipelines": {
                "metrics": {
                    "receivers": metric_receivers,
                    "processors": ["transform/metric_names", "transform/identity", "resource/managed", "batch"],
                    "exporters": ["otlp_http/managed"],
                },
                "metrics/local": {
                    "receivers": metric_receivers,
                    "processors": ["transform/metric_names", "transform/identity", "resource/local", "batch"],
                    "exporters": ["file/metrics"],
                },
                "traces": {
                    "receivers": ["otlp"],
                    "processors": [*trace_transforms, "resource/managed", "batch"],
                    "exporters": ["otlp_http/managed"],
                },
                "traces/local": {
                    "receivers": ["otlp"],
                    "processors": [*trace_transforms, "resource/local", "batch"],
                    "exporters": ["file/traces", "span_metrics"],
                },
                "logs": {
                    "receivers": ["otlp"],
                    "processors": ["transform/identity", "resource/managed", "batch"],
                    "exporters": ["otlp_http/managed"],
                },
                "logs/local": {
                    "receivers": ["otlp"],
                    "processors": ["transform/identity", "resource/local", "batch"],
                    "exporters": ["file/logs"],
                },
            },
        },
    }
    if not obs.remote_export:
        del doc["processors"]["resource/managed"]
        del doc["exporters"]["otlp_http/managed"]
        for signal in ("metrics", "traces", "logs"):
            del doc["service"]["pipelines"][signal]
    return yaml.safe_dump(doc, sort_keys=False)
