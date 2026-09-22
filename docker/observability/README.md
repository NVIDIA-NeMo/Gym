# Local Grafana stack for NeMo Gym telemetry

`nemo_gym/telemetry/` (see its `README.md`) exports OpenTelemetry traces and metrics over
OTLP. This directory is a local backend for that: an OTel Collector fans the data out to
Prometheus (metrics) and Tempo (traces), and Grafana is provisioned with both datasources
plus a starter dashboard.

```
Gym servers --OTLP--> otel-collector --> Prometheus (metrics)  --> Grafana
                                    \--> Tempo (traces)         -/
```

## 1. Start the stack

```bash
docker compose -f docker/observability/docker-compose.yml up -d
```

Verify: `curl http://localhost:9090/-/healthy`, `curl http://localhost:3200/status/version`,
`curl http://localhost:3000/api/health`.

If a container exits immediately with `permission denied` reading its config file, your
umask is stripping the world-read bit from files this tool wrote (common on shared/managed
hosts). Fix once with:

```bash
find docker/observability -type d -exec chmod 755 {} \;
find docker/observability -type f -exec chmod 644 {} \;
docker compose -f docker/observability/docker-compose.yml restart
```

## 2. Point Gym at the collector

```bash
uv sync --extra dev --extra telemetry
```

Then either pass `configs/telemetry_grafana.yaml` on the command line, e.g.:

```bash
gym env start --resources-server <your-benchmark> --model-type vllm_model \
    --config-path configs/telemetry_grafana.yaml
```

or export the equivalent env vars directly (env vars always win over the YAML block):

```bash
export NEMO_GYM_OTEL_ENABLED=1
export NEMO_GYM_OTEL_SPAN_GROUPS=all   # includes `sandbox` spans; `per_rollout` omits them
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
export OTEL_EXPORTER_OTLP_PROTOCOL=grpc
export NEMO_GYM_OTEL_CPU_SAMPLING_ENABLED=1
export NEMO_GYM_OTEL_MEMORY_SAMPLING_ENABLED=1
```

## 3. Open Grafana

`http://localhost:3000` — anonymous access is enabled for local dev (`docker-compose.yml`
sets `GF_AUTH_ANONYMOUS_ORG_ROLE=Admin`; do not reuse this compose file outside a local
sandbox). The **NeMo Gym / NeMo Gym Overview** dashboard is pre-provisioned with rollout
throughput/duration, sandbox startup, concurrency queue-wait, model-call duration, CPU,
verify success rate, and HTTP retry/timeout panels. For per-rollout traces, use **Explore →
Tempo** and search by `service.name` (e.g. `nemo-gym/pinchbench`) or by `nemo.gym.rollout.id`
/ `nemo.gym.run.id` span attributes.

Metric names carry a `_milliseconds` suffix on every `_ms`-named histogram (e.g.
`gym_rollout_duration_ms_milliseconds_bucket`) — that comes from the OTel Collector's
Prometheus exporter expanding the `ms` unit; it's not a Gym naming bug. If you add panels,
check `http://localhost:9090/api/v1/label/__name__/values` for the real name first.

## 4. Run PinchBench against it

PinchBench needs its own per-task sandbox image and model/judge credentials — build the
image first (see `benchmarks/pinchbench/README.md`), then:

```bash
gym eval run --benchmark pinchbench --model-type vllm_model \
    --config-path configs/telemetry_grafana.yaml \
    +sandbox_image=<pinchbench.sif | docker://pinchbench-openclaw:latest> \
    +model_base_url=<endpoint/v1> +model_api_key=<key> +model_name=<model> \
    +judge_model=<judge> +judge_base_url=<endpoint/v1> +judge_api_key=<key> \
    +brave_api_key=<key>
```

**Caveat specific to PinchBench:** its agent config sets `model_server: null` — OpenClaw,
running inside the per-task sandbox, calls the policy model directly rather than through
Gym's model server. Rollout duration, sandbox start/exec spans, verify spans, and
concurrency are all visible; **policy-model call latency is not**, because that call never
passes through a Gym-instrumented process. Fixing this (propagating trace/rollout context
into the sandbox so OpenClaw's outbound model call joins the trace) is a separate,
unimplemented piece of work — see the `gym.sandbox.*` spans for what PinchBench *does*
give you today.

## 5. Tear down

```bash
docker compose -f docker/observability/docker-compose.yml down -v
```

`-v` also drops Prometheus/Tempo/Grafana's local volumes (none are declared here — data
lives only in the containers' writable layers — so a plain `down` already discards
everything; `-v` is a no-op unless you later add named volumes).
