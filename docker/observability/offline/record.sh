#!/usr/bin/env bash
# Start a standalone, record-only OTel Collector -- no Grafana/Prometheus/Tempo needed.
# Writes every signal Gym sends to ./data/{traces,metrics}.jsonl on the host.
#
# Runs on alternate ports (14317/14318) so it can coexist with the live stack's collector
# (which owns 4317/4318) if that happens to be running too -- point Gym at *this* port.
#
# Usage:
#   bash docker/observability/offline/record.sh start
#   gym eval run ... --config configs/telemetry_grafana.yaml ++telemetry.otlp_endpoint=http://localhost:14317 ...
#   bash docker/observability/offline/record.sh stop
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CONTAINER_NAME="nemo-gym-otel-recorder"

case "${1:-}" in
  start)
    mkdir -p "$HERE/data"
    chmod 777 "$HERE/data"  # the collector image runs as a non-root uid
    docker run -d --name "$CONTAINER_NAME" \
      -p 14317:4317 -p 14318:4318 \
      -v "$HERE/otel-collector-record.yaml:/etc/otel-collector-config.yaml:ro" \
      -v "$HERE/data:/data" \
      otel/opentelemetry-collector-contrib:0.114.0 \
      --config=/etc/otel-collector-config.yaml
    echo "Recording collector up. Point Gym at: OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:14317"
    echo "(or ++telemetry.otlp_endpoint=http://localhost:14317 on your gym command)"
    ;;
  stop)
    docker stop "$CONTAINER_NAME" >/dev/null && docker rm "$CONTAINER_NAME" >/dev/null
    echo "Recording collector stopped. Data in $HERE/data/{traces,metrics}.jsonl"
    ;;
  *)
    echo "Usage: $0 {start|stop}" >&2
    exit 1
    ;;
esac
