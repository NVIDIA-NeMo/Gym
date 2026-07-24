<!-- SPDX-License-Identifier: Apache-2.0 -->
# OpenAir Congestion Resource Server

`openair_congestion` is a multi-turn NeMo Gym environment for 5G RAN congestion control. On each turn, an agent reads cell and UE KPIs and must emit exactly one of eight network-control tool calls. The environment validates the call, applies a deterministic synthetic network transition, returns the next KPIs, and computes an auditable reward.

The default replay environment is fully contained in this directory. It needs neither a 5G lab nor a GPU. Live OpenAirInterface control is intentionally outside the scope of this contribution.

For a guided build-and-validation walkthrough, see the
[5G congestion-control environment tutorial](https://docs.nvidia.com/nemo/gym/main/environment-tutorials/openair-congestion).
For the code-level claim and verification map, see
[REVIEWER-EVIDENCE.md](REVIEWER-EVIDENCE.md).

## Agent-environment contract

Each task supplies the agent with:

1. A system instruction describing the 5G operator role.
2. Eight tool schemas.
3. A user observation containing current cell and UE KPIs.

The agent must return exactly one tool call:

| Tool | Required arguments | Effect |
|---|---|---|
| `set_scheduler_policy` | `cell_id`, `policy` in `{PF, RR, MaxCI}` | Select a per-cell MAC scheduler. |
| `set_prb_cap` | `cell_id`, `target`, `target_id`, `max_prb` | Cap PRBs for a UE or slice. |
| `set_mcs_bounds` | `cell_id`, `mcs_min`, `mcs_max`, `target_bler` | Bound link adaptation. |
| `set_qos_weights` | `cell_id`, `weights` | Change per-5QI scheduling weights. |
| `set_admission_policy` | `cell_id`, `accept_threshold_pct`, `slice_reservation` | Change RRC admission policy. |
| `set_handover_trigger` | `cell_id`, `a3_offset_db`, `ttt_ms` | Change the A3 handover trigger. |
| `set_ul_power_control` | `cell_id`, `p0_dbm`, `alpha` | Change uplink fractional power control. |
| `noop` | none | Leave controls unchanged for one step. |

Tool argument schemas come from `openair_congestion.tools.TOOL_SCHEMA_BY_NAME`. The guardrail rejects unavailable or out-of-range actions and applies the configured rejection cost without crashing the episode. Missing, malformed, unknown, or multiple tool calls violate the protocol: the episode terminates, receives the finite negative `protocol_violation_penalty`, and releases its session immediately.

Replay actions are persistent absolute setpoints rather than fixed tool-name bonuses. Parameter changes therefore produce distinct deterministic transitions, while reapplying an identical setpoint is idempotent. The replay admission-control ledger keeps requested, admitted, delivered, denied, and forcibly terminated service consistent with the emitted UE topology. Non-empty slice reservations are rejected because the bundled synthetic topology does not model slices.

## Reward and verification

The environment itself is the verifier. There is no LLM judge. For each accepted or guardrail-rejected transition, `openair_congestion.rewards.compute_breakdown` reports the selected reward version, total, measurements, service accounting, and every contributing term:

- changes in SLA violations, delivered throughput, and Jain fairness;
- current SLA, PRB, access, fairness, and buffer pressure;
- optional action magnitude cost; and
- the illegal-action rejection cost.

The objective can be read as congestion cost: a clean steady transition is `0`, and persistent congestion contributes negative level costs. A transition that materially improves KPIs can receive positive delta credit. Therefore, compare episode returns only when reward profile, backend, horizon, and task manifest are identical.

Reward selection is versioned and shared by live and replay paths:

- T2 episodes use `openair_t2_v3` with explicit requested/admitted/delivered service, denial, and forced-termination accounting.
- Connected T1 runner episodes preserve `openair_v1` with the runner's `0.08` PRB-pressure threshold.
- Standalone T1, T3, and replay-tier episodes preserve frozen `openair_v1` with a `0.85` PRB-pressure threshold.

The unnormalized `delta_sla` term remains part of frozen V1. T2 uses the V3 service objective rather than changing V1 in place.

The handwritten relief policy is an example policy, baseline, and possible SFT-data generator. It is not the verifier. Model evaluation should compare `noop`, handwritten expert, SFT, and GRPO on the same fixed tasks and report per-term rewards and uncertainty.

## Supported backends

| Backend | Action affects next state? | Training use | Description |
|---|---:|---|---|
| `replay` (default) | Yes | Supported | Deterministic parameter-aware synthetic setpoints, service accounting, guardrails, and reward computation. Identical task inputs and action sequences reproduce identical episodes. |
| `dataset_replay` | No | Diagnostics only | Replays recorded observations and recomputes guardrail/reward information. Because the selected action cannot change the prerecorded next state, this backend must not be used for policy training or model-quality claims. |

Select the backend in `configs/openair_congestion.yaml` or with `OPENAIR_CONGESTION_BACKEND`. Selecting `oai_collector` fails at startup with an explicit unsupported-backend error; no live OAI implementation is represented as complete here.

Episode slots are finite. Repeated reset, normal termination, truncation, protocol failure, and explicit close release state immediately. A hard client or process crash cannot call `/close`, so an inactive session is reclaimed on a later reset after the configurable `session_ttl_s` lease (one hour by default).

## Quick start

From a NeMo Gym checkout:

```bash
uv sync --all-extras --all-groups
source .venv/bin/activate
```

Run the self-contained scripted client over the real FastAPI reset/step/close surface:

```bash
python resources_servers/openair_congestion/client.py
```

Validate configuration and the resource-server package:

```bash
gym env validate \
  --resources-server openair_congestion \
  --model-type openai_model \
  --model gpt-4.1-2025-04-14 \
  --model-url https://api.openai.com/v1 \
  --model-api-key "$OPENAI_API_KEY"

gym env test --resources-server openair_congestion
```

To start the resource, shared Gymnasium agent, and an OpenAI-compatible model server:

```bash
gym env start \
  --resources-server openair_congestion \
  --model-type openai_model \
  --model gpt-4.1-2025-04-14 \
  --model-url https://api.openai.com/v1 \
  --model-api-key "$OPENAI_API_KEY"
```

Then collect five model-driven rollouts from another activated terminal:

```bash
gym eval run --no-serve \
  --agent openair_congestion_gymnasium_agent \
  --input resources_servers/openair_congestion/data/example.jsonl \
  --output results/openair_congestion_rollouts.jsonl \
  --limit 5 \
  --num-repeats 1
```

## Checked-in example evidence

The contribution includes the required five-row example set and validated artifacts:

- `data/example.jsonl`: task inputs and tool schemas;
- `data/example_metrics.json`: NeMo Gym example-validation metrics; and
- `data/example_rollouts.jsonl`: five full scripted trajectories generated through the actual server API.

The checked-in rollouts prove deterministic resource-server wiring, lifecycle behavior, reward-profile/service-accounting provenance, and bounded completion. They are labeled `resource_server_wiring_not_model_quality` and are not evidence that an SFT or GRPO checkpoint is better than a baseline.

`difficulty`, `regime_mix`, and `scenario_id` remain task/evaluator metadata. They are intentionally omitted from rendered telemetry and neutral example prompts so a model must act from the KPI state rather than generator labels.

Regenerate the rollouts and verify byte-for-byte determinism:

```bash
python resources_servers/openair_congestion/generate_example_rollouts.py
shasum -a 256 resources_servers/openair_congestion/data/example_rollouts.jsonl
```

Regenerate NeMo Gym validation metrics in a temporary directory:

```bash
validation_dir=$(mktemp -d)
gym dataset collate \
  --config resources_servers/openair_congestion/configs/openair_congestion.yaml \
  --output-dir "$validation_dir" \
  --mode example_validation
cp "$validation_dir/example_metrics.json" \
  resources_servers/openair_congestion/data/example_metrics.json
```

## Dataset replay

`dataset_replay` accepts JSONL KPI snapshots, GRPO rollout traces, or the CSV snapshot adapter. It validates every row at startup and reports the file, line, episode, field, and offending value for malformed data. Cell and UE identifiers must be unique, topology counts must agree exactly, and CSV `step`, `cell_id`, and `ue_id` values must be integral rather than silently truncated.

### KPI snapshots

Each row is one timestep. Rows with the same `episode_id` form an episode and need at least two observations. Required data are `cells[]`, `cells[].prb_util_dl_p50`, a non-empty `cells[].ues[]`, and `cells[].ues[].delivered_mbps`. Optional KPI fields pass through; missing fields are synthesized to the canonical observation shape.

```json
{"episode_id":"run_a","step":0,"cells":[{"cell_id":0,"prb_util_dl_p50":0.55,"ues":[{"ue_id":0,"offered_mbps":20.0,"delivered_mbps":18.0,"bler":0.05,"sinr_db":12.0}]}]}
```

See `data/fixtures/sample_provided.jsonl`.

### Rollout traces

Trace rows use `reward_measurements` to reconstruct aggregate state. `aggregate_delivered_mbps` and `n_ues` are required. Recorded actions and rewards are not trusted: the current policy supplies the action, and the environment recomputes guardrail and reward output. See `data/fixtures/sample_trace.jsonl`.

Set the backend and file in the resource-server config:

```yaml
backend: dataset_replay
dataset_path: data/fixtures/sample_provided.jsonl
cell_capacity_mbps: 60.0
```

Task `scenario_id` must match a dataset episode key or be omitted for deterministic seed-based selection.

## Environment-quality checks

Run the offline scripted capability sweep:

```bash
python resources_servers/openair_congestion/model_sweep.py
```

The CI gate expects the known-policy ordering `relief > noop > random-valid > catastrophic`. Optional OpenAI-compatible models can be added with:

```bash
python resources_servers/openair_congestion/model_sweep.py \
  --models resources_servers/openair_congestion/sweep_models.example.json \
  --out results/openair_congestion_sweep.json
```

Non-compliance exploratory sweeps default to a zero failure ceiling. If a
bounded amount of model-protocol failure is acceptable for exploration, set it
explicitly (for example, `--max-failure-rate 0.01`). Failed episodes are
reported but never enter the paired return comparison; only prompt/repeat keys
usable for every declared model are compared.

For the contribution-guide minimum profile, declare at least a smaller and a frontier-equivalent model with unique increasing `capability_rank` values in the model spec, then run:

```bash
python resources_servers/openair_congestion/model_sweep.py \
  --compliance-profile \
  --concurrency 16 \
  --models resources_servers/openair_congestion/sweep_models.example.json \
  --out results/openair_congestion_model_profile.json
```

That mode requires at least two distinct real model identities with unique predeclared capability ranks; it cannot pass by running scripted anchors alone. Before expansion it verifies that the checked-in manifest contains each of the five supported regimes exactly once and that each row's `regime_mix` is the matching one-hot label. It then expands those rows into 500 deterministic prompts and requests 16 responses per prompt. The report includes return quantiles, per-regime returns, per-tool call/rejection/mean-reward metrics, episode-return correlations, infrastructure and parsing failures, per-episode pair keys/usability, the scripted ordering gate, and the predeclared small-to-frontier ordering gate.

A compliance model-capability profile is evaluable only when every declared model completes the same prompt/repeat pairs with zero infrastructure errors, parse failures, and structurally invalid calls. `--max-failure-rate` must remain zero in compliance mode. Each adjacent higher-capability model must have a positive paired mean-return delta and a deterministic 95% bootstrap interval whose lower bound is above zero. Unusable episodes never contribute to the ordering calculation. Unparseable output still receives the environment's terminal protocol penalty for operational accounting; it is not silently converted to `noop` or treated as valid capability evidence.

No full real-model compliance receipt is checked into this contribution. One
strict Qwen3-1.7B/Qwen3-8B run is summarized in
[REVIEWER-EVIDENCE.md](REVIEWER-EVIDENCE.md), but it was not evaluable and 8B
is not a frontier model. Run and attach a passing report from the intended
small and frontier endpoints before claiming the model-capability gate has
passed.

Generate a deterministic, derived-oracle single-intervention benchmark:

```bash
python resources_servers/openair_congestion/golden_set.py \
  --out results/openair_congestion_golden_set.jsonl
```

The golden labels come from exhaustive evaluation of a finite action grid under deterministic dynamics, not from a human or model judge.

## Tests

```bash
pytest resources_servers/openair_congestion/tests -q
pytest resources_servers/gymnasium/tests/test_app.py \
  responses_api_agents/gymnasium_agent/tests/test_app.py -q
```

The suite covers schema and configuration validation, deterministic replay, action effects, guardrails, reward ordering and decomposition, dataset ingestion, session cleanup, HTTP behavior, checked-in artifacts, the capability sweep, and golden-set self-validation.

## Current limitations

- Replay dynamics are deterministic synthetic approximations, not measurements from a live OAI/FlexRIC deployment.
- Parameter sensitivity and reward ordering establish a controlled test environment, not physical simulator fidelity.
- The optional external `congestion_gen` package improves scenario generation but is not included or required; when absent, the bundled deterministic fallback is used.
- Dataset replay is non-causal and diagnostics-only.
- This contribution includes no live OAI actuator/exporter backend.
- Checked-in scripted rollouts do not establish SFT or GRPO model quality.
- A passing small-to-frontier compliance report remains required before claiming the model-capability gate passed.

## License

Apache-2.0. All telemetry shipped with the offline backends is synthetic benchmark data.
