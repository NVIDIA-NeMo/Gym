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
"""The OpenTelemetry Collector's configuration, shared by every launcher that starts one.

`gym eval submit` runs a collector beside each Slurm job (`nemo_gym.orchestration.executors.otel`), and
`gym env start` runs one beside its servers when `telemetry.collector` is set
(`nemo_gym.server_runtime.supervisor.collector`). Both get the same receivers, transforms and pipelines from
`collector_document`, and differ only in what they scrape and the resource attributes they stamp.
"""

from collections.abc import Sequence

from nemo_gym.telemetry.endpoints import ROLLOUT_ID_ATTRIBUTE


# Below the clusters' ephemeral range (ip_local_port_range 9000-65000): the collector's default
# 13133 lies inside it, so an outgoing connection on the node can hold it and abort startup.
COLLECTOR_HEALTH_PORT = 4319
OTLP_GRPC_PORT = 4317
OTLP_HTTP_PORT = 4318
COLLECTOR_FILE_SIGNALS = ("metrics", "traces", "logs")

NODE_METRICS_FILTER = "filter/local_node_metrics"
# The scrape job name, which the prometheus receiver sets as `service.name` on every node's series.
NODE_SCRAPE_JOB = "node"


def _ottl_string(value: str) -> str:
    """`value` as an OTTL string literal inside the collector's YAML: `$$` stops the collector's own `${...}` expansion."""
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("$", "$$") + '"'


def node_metrics_filter(keep: Sequence[str] | None, extra: Sequence[str] = ()) -> dict | None:
    """The `filter` processor that drops node_exporter metrics outside `keep` and `extra` from the local file.

    Keyed on the scrape job rather than a name prefix, so it covers every family node_exporter serves (including
    textfile collectors with their own prefixes) on every scraped node, and leaves DCGM, engine and Gym metrics alone.
    `keep=None` keeps every node_exporter metric, so there is no filter.
    """
    if keep is None:
        return None
    pattern = "^(?:" + "|".join(f"(?:{p})" for p in [*keep, *extra]) + ")$"
    condition = (
        f'resource.attributes["service.name"] == {_ottl_string(NODE_SCRAPE_JOB)} '
        f"and not IsMatch(metric.name, {_ottl_string(pattern)})"
    )
    return {"metrics": {"metric": [condition]}}


def collector_document(
    *,
    scrape_configs: list[dict],
    resource_actions: list[dict],
    file_dir: str,
    remote_export: bool,
    endpoint: str | None,
    token_env: str,
    scrape_interval_seconds: int,
    engine_rollout_ids: bool,
    node_filter: dict | None,
    otlp_grpc_port: int = OTLP_GRPC_PORT,
    otlp_http_port: int = OTLP_HTTP_PORT,
    health_port: int = COLLECTOR_HEALTH_PORT,
    listen_host: str = "0.0.0.0",
) -> dict:
    """The collector's configuration, as the document its YAML is dumped from.

    `scrape_configs` are Prometheus scrape jobs (none: no prometheus receiver), `resource_actions` the
    `resource` processor actions stamped on everything, and `file_dir` where `{metrics,traces,logs}.jsonl`
    are written. `engine_rollout_ids` copies the rollout id out of vLLM's request spans. The receivers and the
    health check listen on `listen_host`: every interface by default, for processes on other nodes.
    """
    token = f"${{env:{token_env}}}"
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

    # The token rides only on the managed export, never into the local copies.
    managed_actions = [{"key": "Authorization", "value": token, "action": "upsert"}, *resource_actions]

    # Operations that exist only as spans (sandbox start/exec, model calls) still get latency and count series.
    # Exported remotely only: locally the same calls and durations are in `traces.jsonl`.
    span_metrics = {
        "histogram": {
            "explicit": {"buckets": ["250ms", "1s", "2s", "5s", "10s", "30s", "60s", "120s", "300s", "600s", "1800s"]}
        },
        "dimensions": [{"name": "service.name.override"}, {"name": "nemo.gym.sandbox.provider"}],
        "metrics_flush_interval": f"{scrape_interval_seconds}s",
    }

    local_receivers = (["prometheus"] if scrape_configs else []) + ["otlp"]
    local_filters = [NODE_METRICS_FILTER] if node_filter else []
    trace_transforms = ["transform/identity"] + (["transform/engine_rollout_id"] if engine_rollout_ids else [])
    doc = {
        "extensions": {"health_check": {"endpoint": f"{listen_host}:{health_port}"}},
        "connectors": {"span_metrics": span_metrics},
        "receivers": {
            **({"prometheus": {"config": {"scrape_configs": scrape_configs}}} if scrape_configs else {}),
            "otlp": {
                "protocols": {
                    "grpc": {"endpoint": f"{listen_host}:{otlp_grpc_port}"},
                    "http": {"endpoint": f"{listen_host}:{otlp_http_port}"},
                }
            },
        },
        "processors": {
            "batch": {},
            "resource/managed": {"attributes": managed_actions},
            "resource/local": {"attributes": resource_actions},
            "transform/identity": identity,
            "transform/metric_names": rename_colon_metrics,
            **({"transform/engine_rollout_id": engine_rollout_id} if engine_rollout_ids else {}),
            **({NODE_METRICS_FILTER: node_filter} if node_filter else {}),
        },
        "exporters": {
            "otlp_http/managed": {"endpoint": endpoint, "headers": {"Authorization": f"Bearer {token}"}},
            **{f"file/{signal}": {"path": f"{file_dir}/{signal}.jsonl"} for signal in COLLECTOR_FILE_SIGNALS},
        },
        "service": {
            "extensions": ["health_check"],
            # debug would print every batch, token included, into the collector log.
            "telemetry": {"logs": {"level": "info"}},
            "pipelines": {
                "metrics": {
                    "receivers": [*local_receivers, "span_metrics"],
                    "processors": ["transform/metric_names", "transform/identity", "resource/managed", "batch"],
                    "exporters": ["otlp_http/managed"],
                },
                # Filtered before `resource/local` overwrites `service.name`, while it still names the scrape job.
                "metrics/local": {
                    "receivers": local_receivers,
                    "processors": [
                        "transform/metric_names",
                        "transform/identity",
                        *local_filters,
                        "resource/local",
                        "batch",
                    ],
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
    if not remote_export:
        del doc["processors"]["resource/managed"]
        del doc["exporters"]["otlp_http/managed"]
        for signal in ("metrics", "traces", "logs"):
            del doc["service"]["pipelines"][signal]
        # Span metrics feed only the remote export; a connector nothing receives from fails collector startup.
        del doc["connectors"]
        doc["service"]["pipelines"]["traces/local"]["exporters"].remove("span_metrics")
    return doc
