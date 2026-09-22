#!/usr/bin/env bash
# Replay files recorded by record.sh into the live Prometheus + Tempo (the ones Grafana's
# dashboards already read from) -- so an eval run captured while the observability stack
# was down becomes visible in the same Grafana dashboards after the fact.
#
# Requires: the main stack (docker/observability/docker-compose.yml) already running.
#
# How this works: the live otel-collector is briefly stopped and swapped for a one-shot
# replay collector *reusing its service name* on the observability_default network, so
# Prometheus's existing static scrape target ("otel-collector:8889") picks it up with no
# config changes. It sits for 20s (two scrape intervals) so Prometheus has time to pull
# the replayed metrics, then the live collector is restarted.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
COMPOSE_DIR="$(cd "$HERE/.." && pwd)"
NETWORK="observability_default"
REPLAY_CONTAINER="nemo-gym-otel-replay"

if ! docker network inspect "$NETWORK" >/dev/null 2>&1; then
  echo "Network $NETWORK not found -- start the main stack first:" >&2
  echo "  docker compose -f $COMPOSE_DIR/docker-compose.yml up -d" >&2
  exit 1
fi

echo "Stopping the live otel-collector (briefly -- live telemetry export pauses during replay)..."
docker compose -f "$COMPOSE_DIR/docker-compose.yml" stop otel-collector

docker run -d --name "$REPLAY_CONTAINER" \
  --network "$NETWORK" --network-alias otel-collector \
  -v "$HERE/otel-collector-replay.yaml:/etc/otel-collector-config.yaml:ro" \
  -v "$HERE/data:/data:ro" \
  otel/opentelemetry-collector-contrib:0.114.0 \
  --config=/etc/otel-collector-config.yaml

echo "Replaying $HERE/data/{traces,metrics}.jsonl -- waiting 20s for Prometheus to scrape..."
sleep 20

docker stop "$REPLAY_CONTAINER" >/dev/null && docker rm "$REPLAY_CONTAINER" >/dev/null
echo "Restarting the live otel-collector..."
docker compose -f "$COMPOSE_DIR/docker-compose.yml" start otel-collector

echo "Done. Traces are in Tempo immediately (push-based); metrics were scraped into Prometheus."
echo "Open Grafana: http://localhost:3000/d/nemo-gym-all-metrics"
