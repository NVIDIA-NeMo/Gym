# `nemo_gym.telemetry`

Optional OpenTelemetry instrumentation for NeMo Gym, built on
[nemo-lens](https://github.com/NVIDIA-NeMo/Lens).

User-facing documentation lives in
[`fern/versions/latest/pages/observability/`](../../fern/versions/latest/pages/observability).
This file is the design note for people changing the module.

## What it is for

One rollout crosses three Gym processes — the agent server calls the model server, both
call the resources server — and until now nothing tied those hops together. The goal of
this module is that **one rollout produces one trace**, with correct parent/child edges
across process boundaries.

Everything else here is supporting work for that.

## Why Gym's shape forced a different design from NeMo-RL's

| | Megatron-LM | NeMo-RL | NeMo Gym |
| --- | --- | --- | --- |
| Process model | one process tree | Ray driver + actors | **N independent FastAPI processes** |
| How settings reach a worker | in-process | Ray `runtime_env` | **`os.environ` snapshotted by `Popen`** |
| Default `export_strategy` | `single_rank` | `single_rank` | **`all_ranks`** |
| The hard part | where to put a span | driver/worker split | **cross-process context propagation** |

Gym's servers share no memory and inherit no handle. The orchestrator
(`gym env start` / `gym env test`) translates the `telemetry:` config block into
`NEMO_GYM_OTEL_*` environment variables *before* spawning anything, and each server
process reads that environment back. That is the whole of the coordination.

`all_ranks` is the default because each Gym server is rank 0 of its own world of 1. A
rank-based filter would either silence all of them or none, and any silenced process is a
hole in the middle of a distributed trace.

## Files

| File | Role |
| --- | --- |
| `config.py` | `TelemetryConfig` — the `telemetry:` block. Imports pydantic only. |
| `setup.py` | Lifecycle: `configure_telemetry_env` (orchestrator), `init_telemetry` (per server process), `get_telemetry`, `shutdown_telemetry`. |
| `span_groups.py` | `GymSpanGroup` — Gym's groups and the `default` / `per_rollout` / `all` presets. |
| `metrics.py` | Wrapper over nemo-lens's `gym.*` instruments, with a stated position on each. |
| `_fallbacks.py` | The single import point for instrumentation primitives. |

## Two rules worth knowing before you edit

**1. Gate at the call site, before `managed_span`.**

```python
if is_span_group_enabled(GymSpanGroup.VERIFY):
    with managed_span(GymSpanGroup.VERIFY, "gym.verify", **attrs):
        ...
```

`managed_span` has its own internal gate, so the outer `if` looks redundant. It is not.
Entering a disabled `managed_span` still builds and drives a `@contextmanager` generator,
and `managed_span` also resolves its gate through a function-local import. Checking the
group first skips both.

Gym serves at 16k+ concurrency, so per-request sites are hot paths. Attribute dicts and
f-strings go *inside* the gate too, never in the arguments of a call above it.
`tests/unit_tests/telemetry/test_overhead.py` guards the disabled-path cost.

**2. `_fallbacks.py` is one of four synchronised copies.**

`kb/knowledge/conventions/fallback-sync.md`: nemo-lens's no-op surface exists in
`nemo/lens/fallbacks.py` plus one `_fallbacks.py` per consumer. When a signature changes
in lens, change it here in the same PR — a drift only breaks the configuration where lens
is *absent*, which is the one nobody runs by accident.
`tests/unit_tests/telemetry/test_fallbacks.py` compares the two parameter-for-parameter.

Note this file differs from NeMo-RL's: it resolves to the **real** `nemo.lens.helpers`
implementations when lens is installed, so a call site needs one import rather than its
own `try/except ImportError`. RL's re-exports `nemo.lens.fallbacks` (the no-ops) in both
cases, which is why every RL call site carries a second try/except.

## Metrics: read `metrics.py` before adding one

nemo-lens's `record_gym_metrics` records **without attributes** at the pinned commit.
`gym.server.request_duration_ms` is therefore deliberately unused — undimensioned, it
would merge every endpoint of every server type into one histogram. The FastAPI
instrumentor's `http.server.request.duration`, dimensioned by route/method/status, is used
instead. `gym.servers.active` is a gauge and is written by the orchestrator only.

Reward and accuracy numbers do **not** belong here. They are experiment telemetry (W&B's
job), not application telemetry — see
`kb/knowledge/concepts/application-vs-experiment-telemetry.md`. Gym's existing
`AggregateMetricsMixin` stays where it is.

## Testing

```bash
pytest tests/unit_tests/telemetry/ -x
```

The load-bearing test is `test_propagation.py`: two Gym apps, an in-memory span exporter,
one request through `server_utils.request()`, asserting a single trace id and a correct
parent/child edge across the boundary. If you change one thing about this module, make
sure that test still fails when propagation breaks.
