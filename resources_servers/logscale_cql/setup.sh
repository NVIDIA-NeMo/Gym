#!/usr/bin/env bash
#
# LogScale Gym – one-command setup
#
#   1. Start the LogScale container (humio/humio-single-node-demo)
#   2. Wait until it is healthy
#   3. Discover the sandbox repository and its ingest token
#   4. Ingest synthetic_falcon_events.csv
#   5. Verify with a count() query
#
# Usage:
#   bash setup.sh                          # defaults
#   bash setup.sh --url http://host:8080   # custom LogScale URL
#   bash setup.sh --skip-container         # skip docker, just ingest + verify
#   bash setup.sh --skip-ingest            # start container only, no CSV ingest

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOGSCALE_URL="${LOGSCALE_URL:-http://localhost:8080}"
DATASETS_DIR="$SCRIPT_DIR/data/datasets_v2"
INGEST_SCRIPT="$SCRIPT_DIR/data/ingest_datasets.py"
SKIP_CONTAINER=false
SKIP_INGEST=false
HUMIO_LICENSE="${HUMIO_LICENSE:-eyJ0eXAiOiJKV1QiLCJhbGciOiJFUzUxMiJ9.eyJpc09lbSI6ZmFsc2UsImF1ZCI6Ikh1bWlvLWxpY2Vuc2UtY2hlY2siLCJzdWIiOiJOdmlkaWFTaGFyZWRSZXNlYXJjaCIsInVpZCI6IjVOS1ZjTEJFRFB3cVVIR0ciLCJtYXhVc2VycyI6MTAwMCwiYWxsb3dTQUFTIjpmYWxzZSwibWF4Q29yZXMiOjk5OTk5OSwidmFsaWRVbnRpbCI6MTgwMzgxOTYwMCwiZXhwIjoxODY2NDcwMzQ5LCJpc1RyaWFsIjpmYWxzZSwiaWF0IjoxNzcxODYyMzQ5LCJtYXhJbmdlc3RHYlBlckRheSI6NjR9.Ad0BFe5rCCOKEo0WFYNdsLyFB7iOSrNGRMIMLQppmoFBU1xACh3V5P_XKmVV1NqK0PGZ4FeODQjuGkLHteWWVsywASCz8C-HYpw9kgRPjbocuikn1p0YuzoxTf53OA0-MD0HH_RafmJ98E2CXlXRkvNlLUB3LsgbayP2PEhebhVMfqu4}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --url)            LOGSCALE_URL="$2"; shift 2 ;;
    --skip-container) SKIP_CONTAINER=true; shift ;;
    --skip-ingest)    SKIP_INGEST=true; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

echo "=== LogScale Gym Setup ==="
echo "  URL:       $LOGSCALE_URL"
echo "  Datasets:  $DATASETS_DIR"
echo "  Runtime:   $CONTAINER_RUNTIME"
echo ""

# ── 1. Start the container ─────────────────────────────────────────
# Supports Docker (default) or Apptainer (set CONTAINER_RUNTIME=apptainer)
CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-apptainer}"

start_container_docker() {
  if docker ps --format '{{.Names}}' | grep -q '^humio$'; then
    echo "[OK] Container 'humio' is already running."
    return
  fi
  if docker ps -a --format '{{.Names}}' | grep -q '^humio$'; then
    echo "[..] Starting stopped container 'humio'..."
    docker start humio
  else
    echo "[..] Creating and starting container 'humio' (Docker)..."
    docker run -d \
      -p 8080:8080 \
      -p 9200:9200 \
      --name=humio \
      --ulimit="nofile=250000:250000" \
      --stop-timeout 300 \
      -e AUTHENTICATION_METHOD=none \
      -e ELASTIC_PORT=9200 \
      -e "PUBLIC_URL=$LOGSCALE_URL" \
      -e KAFKA_SERVERS=127.0.0.1:9092 \
      -e HUMIO_PORT=8080 \
      -e "HUMIO_LICENSE=${HUMIO_LICENSE:-}" \
      humio/humio-single-node-demo
  fi
}

start_container_apptainer() {
  if curl -sf "$LOGSCALE_URL/api/v1/status" > /dev/null 2>&1; then
    echo "[OK] LogScale is already running at $LOGSCALE_URL."
    return
  fi
  echo "[..] Starting LogScale container (Apptainer)..."
  # Uses host networking (no --net) to avoid iptables/CNI dependency.
  # Port 8080/9200 bind directly on the host.
  #
  # NOTE: We use `apptainer run` (not `instance start`) because `instance start`
  # does NOT execute the Docker CMD/ENTRYPOINT. The `--env` flags also don't
  # propagate to child processes (supervisord -> humio), so we set env vars via
  # the APPTAINERENV_ prefix which Apptainer injects into the container process.
  export APPTAINERENV_AUTHENTICATION_METHOD=none
  export APPTAINERENV_ELASTIC_PORT=9200
  export APPTAINERENV_PUBLIC_URL="$LOGSCALE_URL"
  export APPTAINERENV_KAFKA_SERVERS=127.0.0.1:9092
  export APPTAINERENV_HUMIO_PORT=8080
  export APPTAINERENV_AUTO_UPDATE_MAXMIND=false

  nohup apptainer run \
    --writable-tmpfs \
    ${GEOLITE_CITY_PATH:+--bind "$GEOLITE_CITY_PATH:/data/humio-data/IpLocationDb.mmdb"} \
    ${GEOLITE_ASN_PATH:+--bind "$GEOLITE_ASN_PATH:/data/humio-data/IpAsnDb.mmdb"} \
    docker://humio/humio-single-node-demo \
    > /tmp/logscale_apptainer.log 2>&1 &

  echo "  Apptainer PID: $!"
  echo "  Log: /tmp/logscale_apptainer.log"
}

start_container() {
  case "$CONTAINER_RUNTIME" in
    apptainer) start_container_apptainer ;;
    docker)    start_container_docker ;;
    *)         echo "[FAIL] Unknown CONTAINER_RUNTIME=$CONTAINER_RUNTIME (use docker or apptainer)"; exit 1 ;;
  esac
}

if [ "$SKIP_CONTAINER" = false ]; then
  start_container
fi

# ── 2. Wait for healthy ────────────────────────────────────────────
# First run with Apptainer pulls + converts the Docker image (~5-10 min),
# then Kafka + Humio need ~30s to start. Allow up to 15 min total.
echo "[..] Waiting for LogScale at $LOGSCALE_URL ..."
retries=180
for i in $(seq 1 $retries); do
  if curl -sf "$LOGSCALE_URL/api/v1/status" > /dev/null 2>&1; then
    echo "[OK] LogScale is healthy (attempt $i/$retries)."
    break
  fi
  if [ "$i" -eq "$retries" ]; then
    echo "[FAIL] LogScale did not become healthy after $((retries * 5))s."
    echo "  Check /tmp/logscale_apptainer.log for Apptainer errors."
    exit 1
  fi
  sleep 5
done

# ── 2b. Install license key (MUST happen before any GraphQL queries) ─
# Newer LogScale versions require a valid license before the GraphQL API
# accepts queries, even with AUTHENTICATION_METHOD=none.
if [ -n "${HUMIO_LICENSE:-}" ]; then
  echo "[..] Installing license key..."
  python3 -c "
import requests, time
for attempt in range(5):
    try:
        resp = requests.post(
            '$LOGSCALE_URL/graphql',
            json={
                'query': 'mutation(\$l: String!) { updateLicenseKey(license: \$l) { expiresAt } }',
                'variables': {'l': '$HUMIO_LICENSE'},
            },
            headers={'Content-Type': 'application/json'}, timeout=10,
        )
        data = resp.json()
        if data.get('data'):
            print('  License expires:', data['data']['updateLicenseKey']['expiresAt'])
            break
        else:
            print(f'  Attempt {attempt+1}: {resp.text[:100]}')
    except Exception as e:
        print(f'  Attempt {attempt+1}: {e}')
    time.sleep(3)
"
fi

# ── 3. Discover sandbox repo + ingest token ────────────────────────
echo "[..] Discovering sandbox repository..."
REPO=$(python3 -c "
import requests, json, time
for attempt in range(5):
    try:
        data = requests.post(
            '$LOGSCALE_URL/graphql',
            json={'query': '{ searchDomains { name } }'},
            headers={'Content-Type': 'application/json'}, timeout=10,
        ).json().get('data')
        if data:
            domains = data['searchDomains']
            sandboxes = [d['name'] for d in domains if d['name'].startswith('sandbox_')]
            print(sandboxes[0] if sandboxes else domains[0]['name'])
            break
    except Exception:
        pass
    time.sleep(3)
")
echo "  Repository: $REPO"

echo "[..] Fetching ingest token..."
INGEST_TOKEN=$(python3 -c "
import requests, json
query = '{ searchDomain(name: \"$REPO\") { ... on Repository { ingestTokens { token } } } }'
resp = requests.post(
    '$LOGSCALE_URL/graphql',
    json={'query': query},
    headers={'Content-Type': 'application/json'}, timeout=10,
).json()
print(resp['data']['searchDomain']['ingestTokens'][0]['token'])
")
echo "  Ingest token: ${INGEST_TOKEN:0:12}..."

# ── 4. Ingest datasets (skipped with --skip-ingest) ──────────────
if [ "$SKIP_INGEST" = false ]; then
  if [ ! -d "$DATASETS_DIR" ]; then
    echo "[WARN] Datasets dir not found: $DATASETS_DIR (skipping ingest)"
    echo "  Download from GitLab registry: datasets_merged.zip on logscale_cql v0.0.2"
  elif [ ! -f "$INGEST_SCRIPT" ]; then
    echo "[WARN] Ingest script not found: $INGEST_SCRIPT (skipping ingest)"
  else
    echo "[..] Ingesting datasets from $DATASETS_DIR ..."
    python3 "$INGEST_SCRIPT" \
      --datasets-dir "$DATASETS_DIR" \
      --logscale-url "$LOGSCALE_URL"
  fi
else
  echo "[OK] Skipping ingest (--skip-ingest)"
fi

# ── 5. Verify ───────────────────────────────────────────────────────
echo "[..] Verifying with count() query..."
python3 -c "
import requests, time, json

repo = '$REPO'
url  = '$LOGSCALE_URL'

job = requests.post(
    f'{url}/api/v1/repositories/{repo}/queryjobs',
    json={'queryString': 'count()', 'start': '7days', 'isLive': False},
    headers={'Content-Type': 'application/json', 'Accept': 'application/json'},
).json()
job_id = job.get('id')

for _ in range(30):
    r = requests.get(
        f'{url}/api/v1/repositories/{repo}/queryjobs/{job_id}',
        headers={'Accept': 'application/json'},
    ).json()
    if r.get('done'):
        events = r.get('events', [])
        count = events[0].get('_count', 0) if events else 0
        print(f'  count() = {count}')
        break
    time.sleep(0.5)
"

echo ""
echo "=== Setup complete ==="
echo "  LogScale URL:  $LOGSCALE_URL"
echo "  Repository:    $REPO"
echo ""
echo "Start the Gym resources server:"
echo "  python app.py"
