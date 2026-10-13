<!-- SPDX-License-Identifier: Apache-2.0 -->
# OpenAir Congestion Resource Server

`openair_congestion` is a multi-turn NeMo Gym environment for 5G RAN
congestion control. On every turn, a policy reads cell and UE KPIs and emits
exactly one of eight bounded tool calls. The resource server validates the
call, the causal `replay` backend applies a deterministic synthetic transition,
and the environment computes a decomposed KPI reward without an LLM judge.

The default `replay` path is self-contained: it needs neither a 5G lab nor a
GPU. It is a controlled training and evaluation environment, not a live
OpenAirInterface or FlexRIC deployment.

For the code-level verification map, see
[REVIEWER-EVIDENCE.md](REVIEWER-EVIDENCE.md).

## Component map

| Component | Responsibility |
|---|---|
| Model or policy | Reads rendered KPI telemetry and eight tool schemas, then emits one tool call. |
| Gymnasium agent | Runs the reset/model-step loop and conditionally sends `/close` when the loop exits if the server advertises explicit-close support. |
| Resource server | Owns session and episode state and enforces the one-call protocol. |
| Guardrail | Validates tool names, arguments, topology references, and safety bounds. |
| `replay` backend | Applies causal, persistent synthetic setpoints with modeled parameter effects. |
| Verifier | `compute_breakdown` scores KPI changes and rejected actions programmatically. |
| `dataset_replay` backend | Replays recorded next states for ingestion and reward diagnostics only. |

## Agent-environment contract

Each task supplies a system instruction, the tool schemas, and an observation
containing current cell and UE KPIs. The policy must return exactly one call:

| Tool | Required arguments | Synthetic control |
|---|---|---|
| `set_scheduler_policy` | `cell_id`, `policy` in `{PF, RR, MaxCI}` | Select a per-cell scheduler. |
| `set_prb_cap` | `cell_id`, `target`, `target_id`, `max_prb` | Cap PRBs for an observed UE. |
| `set_mcs_bounds` | `cell_id`, `mcs_min`, `mcs_max`, `target_bler` | Bound link adaptation. |
| `set_qos_weights` | `cell_id`, `weights` | Change per-5QI scheduling weights. |
| `set_admission_policy` | `cell_id`, `accept_threshold_pct`, empty `slice_reservation` | Change admission threshold; slices are not modeled. |
| `set_handover_trigger` | `cell_id`, `a3_offset_db`, `ttt_ms` | Change the A3 handover trigger. |
| `set_ul_power_control` | `cell_id`, `p0_dbm`, `alpha` | Change uplink power control. |
| `noop` | none | Keep current setpoints for one step. |

The authoritative schemas live in `openair_congestion/tools.py`. Missing,
malformed, unknown, or multiple calls advance as `noop` and add the finite
`protocol_violation_penalty`. A well-formed unsafe call is rejected by the
guardrail and scored as a rejected transition.

Accepted replay actions are persistent absolute setpoints. Supported setpoint
changes can alter later synthetic KPIs; reapplying the same setpoint is
idempotent. Empty slice reservations are accepted, but non-empty reservations
are rejected because the bundled topology does not model slices. Synthetic
delivered throughput is capped at each cell's configured capacity; when scaling
is needed, delivery-derived buffer, PDB, fairness, and SLA fields are recomputed.

Difficulty, regime, and scenario labels remain evaluator metadata. They are
not rendered into the policy's KPI message.

## Reward and verification

The environment itself is the verifier. For each backend step accepted or
rejected by the guardrail,
`openair_congestion/rewards.py::compute_breakdown` returns:

- the `openair_v1` reward version;
- raw KPI measurements;
- each weighted reward term; and
- the scalar total used by evaluation or training.

Protocol violations produce the same backend transition as `noop` plus the
configured negative surcharge, so malformed output cannot end a costly episode
early or outscore the equivalent valid `noop` trajectory.

The terms cover changes in SLA violations, delivered throughput, and Jain
fairness; current SLA, PRB, access, fairness, and buffer pressure; optional
action magnitude; and illegal-action rejection cost. A clean steady
transition is `0`, persistent congestion contributes negative level costs,
and a material improvement can receive positive delta credit.

Compare returns only when task manifest, backend, reward version, horizon, and
decoding settings are identical. The handwritten relief policy is a scripted
baseline, not the verifier or learned-policy evidence.

## Backends

| Backend | Action changes the next state? | Use |
|---|---:|---|
| `replay` (default) | Yes | Causal synthetic development, evaluation, and training. |
| `dataset_replay` | No | Recorded-data ingestion and reward/contract diagnostics. |

The bundled sampler behind `replay` deliberately creates medium/high-difficulty
overload. Its five regime names drive distinct synthetic pressure patterns:
offered-load pressure (`prb_exhaustion`), higher-load burst snapshots
(`bursty`), SINR/BLER impairment (`interference`), access pressure
(`prach_storm`), and heterogeneous 5QI demand (`qos_competition`). These are
deterministic benchmark dynamics, not claims of live-network fidelity.

`dataset_replay` returns the prerecorded next observation regardless of the
current policy's action. Its metadata therefore reports
`training_usable: false` and `diagnostic_only: true`; it must not be used for
on-policy GRPO or model-quality claims.

Episode slots are finite. Reset, normal completion, truncation, protocol
failure, and explicit close release state immediately. A hard client crash
cannot call `/close`, so an inactive session is reclaimed on a later reset
after `session_ttl_s` (one hour by default).

## Quick start

From a NeMo Gym checkout:

```bash
uv sync --all-extras --all-groups
source .venv/bin/activate
```

Run one scripted episode over the actual FastAPI reset/step/close surface:

```bash
python resources_servers/openair_congestion/client.py
```

Validate and test the package:

```bash
gym env validate \
  --resources-server openair_congestion \
  --model-type openai_model \
  --model gpt-4.1-2025-04-14 \
  --model-url https://api.openai.com/v1 \
  --model-api-key "$OPENAI_API_KEY"

gym env test --resources-server openair_congestion
```

Start the resource server, shared Gymnasium agent, and an OpenAI-compatible
policy server:

```bash
gym env start \
  --resources-server openair_congestion \
  --model-type openai_model \
  --model gpt-4.1-2025-04-14 \
  --model-url https://api.openai.com/v1 \
  --model-api-key "$OPENAI_API_KEY"
```

From another activated terminal, collect repeated policy rollouts and profile
their rewards with the standard Gym workflow:

```bash
gym eval run --no-serve \
  --agent openair_congestion_gymnasium_agent \
  --input resources_servers/openair_congestion/data/example.jsonl \
  --output results/openair_congestion_rollouts.jsonl \
  --limit 5 \
  --num-repeats 2

gym eval profile \
  --inputs results/openair_congestion_rollouts_materialized_inputs.jsonl \
  --rollouts results/openair_congestion_rollouts.jsonl
```

These commands evaluate a hosted policy; they do not train it.

## Checked-in and derived evidence

The package includes:

- `data/example.jsonl`: five task inputs and tool schemas;
- `data/example_metrics.json`: NeMo Gym example-validation metrics; and
- `data/example_rollouts.jsonl`: five scripted trajectories through the real
  resource-server API.

The checked-in rollouts provide reviewable scripted records of wiring,
lifecycle behavior, reward decomposition, and bounded completion. They are
labeled `resource_server_wiring_not_model_quality` and do not establish SFT
or GRPO quality.

Regenerate them with:

```bash
python resources_servers/openair_congestion/generate_example_rollouts.py
```

Generate the finite-grid single-intervention golden set with:

```bash
python resources_servers/openair_congestion/golden_set.py \
  --out results/openair_congestion_golden_set.jsonl
```

For each deterministic decision state, the script exhaustively evaluates a
bounded action grid, applies one candidate, and then coasts with `noop`. The
result is a reproducible reward-and-dynamics sanity oracle for that grid and
horizon. It is not a universal multi-step optimum or real-model evidence.

## Diagnose recorded data with `dataset_replay`

`dataset_replay` accepts nested KPI-snapshot JSONL. It validates every input at
startup with file, line, episode, field, and offending-value provenance.

### KPI snapshot format

Each row is one timestep. Rows sharing an `episode_id` form an episode and
need at least two observations. Required data are a non-empty `cells` list,
`cells[].prb_util_dl_p50`, a non-empty `cells[].ues` list, and
`cells[].ues[].delivered_mbps`. Optional KPI fields pass through; missing
fields are synthesized to the canonical observation shape.

Set `kpi_source_mode` explicitly for measured snapshots. If omitted, it
defaults to `replay` and the observation is stamped synthetic.

```json
{"episode_id":"run_a","step":0,"recorded_action":{"name":"noop","arguments":{}},"cells":[{"cell_id":0,"prb_util_dl_p50":0.55,"ues":[{"ue_id":0,"offered_mbps":20.0,"delivered_mbps":18.0,"bler":0.05,"sinr_db":12.0}]}]}
```

See `data/fixtures/sample_provided.jsonl`.

When `step` is present, every row in that episode must provide a unique integer;
the loader sorts those rows by `step`. An optional `recorded_action` is
validated and returned as diagnostic metadata; it does not
control the prerecorded next state. Stored scalar rewards are ignored. The
backend recomputes reward from the served before/after observations, the
current evaluation action, guardrail result, and configured reward contract.

To use checked-in or custom JSONL, set `backend: dataset_replay` and
`dataset_path` in `configs/openair_congestion.yaml`, then use the same
validation, start, and evaluation commands shown above. The workflow validates
ingestion and reward diagnostics; it cannot supply the counterfactual next
state for a different action and remains diagnostic-only.

## Extend the environment

When adding a KPI:

1. Add and validate it in `openair_congestion/schemas.py`.
2. Parse or derive it in `dataset_backend.py` and record honest provenance.
3. Populate it in `openair_congestion/replay_env.py`.
4. Render it only if the policy should observe it.
5. If it changes scoring, add an auditable term and version the reward contract.
6. Update fixtures, documentation, and focused tests.

When adding a tool:

1. Define its OpenAI function schema and bounds in `openair_congestion/tools.py`.
2. Add topology and safety checks in `openair_congestion/guardrail.py`.
3. Give it deterministic, parameter-sensitive, persistent replay effects, or
   reject it when the synthetic topology cannot represent it honestly.
4. Update prompts, examples, fixtures, baselines, and focused tests.

## Training and checkpoint evaluation

Use causal `replay` for GRPO. The model, tokenizer, optimizer, and SFT/GRPO
settings belong in the NeMo RL training YAML, not the resource-server YAML.
Set the base model or checkpoint in the NeMo RL model section and use
disjoint training and evaluation manifests. This package does not ship a
validated OpenAir-specific NeMo RL job YAML; use the current NeMo RL GRPO
tutorial as the schema authority.

Evaluate a trained checkpoint by serving it through an OpenAI-compatible model
endpoint and rerunning the `gym env start`, `gym eval run`, and
`gym eval profile` workflow above. Some checkpoints need a model-specific chat
template or tool-call parser. Keep the task rows, backend, reward version,
horizon, decoding settings, and repeat count fixed when comparing policies.

## Tests

```bash
pytest resources_servers/openair_congestion/tests -q
pytest responses_api_agents/gymnasium_agent/tests/test_app.py -q
```

The suite covers configuration and schemas, representative deterministic
action effects, guardrails, reward ordering and decomposition, dataset
ingestion, session cleanup, HTTP behavior, checked-in artifacts, and
golden-set self-validation.

## Limitations

- Replay dynamics are deterministic synthetic approximations, not live
  OAI/FlexRIC measurements or physical-simulator fidelity.
- Dataset replay is non-causal and diagnostic-only.
- The finite-grid golden set is a scripted sanity oracle, not a universal
  optimum or evidence of policy quality.
- Checked-in scripted rollouts do not establish SFT or GRPO quality.
- Hosted-model evaluation and NeMo RL training are optional empirical work,
  separate from resource-server correctness.

## License

Apache-2.0. All telemetry shipped with the offline backends is synthetic
benchmark data.
