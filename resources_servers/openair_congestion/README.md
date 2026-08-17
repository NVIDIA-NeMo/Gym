<!-- SPDX-License-Identifier: Apache-2.0 -->
# OpenAir Congestion Resource Server

`openair_congestion` gives NeMo Gym a multi-turn environment for 5G RAN
congestion control. On each turn, a policy reads cell and UE KPIs and returns
exactly one of eight bounded tool calls. The resource server validates the
call, and the causal `replay` backend applies a deterministic synthetic
transition. The environment then scores the KPI changes with a decomposed
reward. No LLM judge is involved.

The default `replay` path runs without a 5G lab or GPU. It is meant for
controlled training and evaluation and does not connect to a live
OpenAirInterface or FlexRIC deployment.

For a guided walkthrough, see the
[5G congestion-control environment tutorial](https://docs.nvidia.com/nemo/gym/main/environment-tutorials/openair-congestion).
For the code-level verification map, see
[REVIEWER-EVIDENCE.md](REVIEWER-EVIDENCE.md).

## Component map

| Component | Responsibility |
|---|---|
| Model or policy | Reads the rendered KPI telemetry and eight tool schemas, then returns one tool call. |
| Gymnasium agent | Runs the reset/model-step loop. When the server advertises explicit-close support, the agent sends `/close` as the loop exits. |
| Resource server | Tracks session and episode state and enforces the one-call protocol. |
| Guardrail | Checks tool names, arguments, topology references, and safety bounds. |
| `replay` backend | Applies causal, persistent synthetic setpoints with modeled parameter effects. |
| Verifier | Uses `compute_breakdown` to score KPI changes and rejected actions. |
| `dataset_replay` backend | Replays recorded next states for ingestion and reward diagnostics only. |

## Agent-environment contract

Each task gives the policy a system instruction, the tool schemas, and an
observation with the current cell and UE KPIs. The policy must return exactly
one call:

| Tool | Required arguments | Synthetic control |
|---|---|---|
| `set_scheduler_policy` | `cell_id`, `policy` in `{PF, RR, MaxCI}` | Select a per-cell scheduler. |
| `set_prb_cap` | `cell_id`, `target`, `target_id`, `max_prb` | Cap an observed UE when the floor and full-reassignment checks pass. |
| `set_mcs_bounds` | `cell_id`, `mcs_min`, `mcs_max`, `target_bler` | Bound link adaptation. |
| `set_qos_weights` | `cell_id`, `weights` | Change per-5QI scheduling weights. |
| `set_admission_policy` | `cell_id`, `accept_threshold_pct`, empty `slice_reservation` | Replay accepts only topology-neutral 100% admission; slices are not modeled. |
| `set_handover_trigger` | `cell_id`, `a3_offset_db`, `ttt_ms` | Change the A3 handover trigger. |
| `set_ul_power_control` | `cell_id`, `p0_dbm`, `alpha` | Change uplink power control. |
| `noop` | none | Keep current setpoints for one step. |

The schemas in `openair_congestion/tools.py` are authoritative. A response with
no call, a malformed or unknown call, or multiple calls violates the protocol.
The server advances one backend step with `noop` and adds the bounded negative
`protocol_violation_penalty`; the episode then continues unless its normal step
budget or backend horizon has been reached. This prevents malformed output
from earning a better return by ending a congested episode early. The
guardrail rejects a well-formed but unsafe call, and the verifier scores it as
a rejected transition. The fallback is causal in synthetic `replay`; in
`dataset_replay` it advances the recorded diagnostic sequence.

Supported replay controls are deterministic, parameter-aware, persistent
setpoints; reapplying the same setpoint is idempotent. Traffic-shedding controls
that the synthetic state cannot score honestly fail closed. Replay rejects
admission thresholds below 100%, non-empty slice reservations, and PRB caps
below `ceil(273 / active UEs)`. A 100% admission setting is accepted but leaves
the topology unchanged. An at/above-floor PRB cap is also rejected if other
observed UEs cannot absorb all displaced throughput. A persistent cap is
suspended on any later transition without enough reassignment headroom. Lower
settings need persistent denied-demand and per-UE PRB accounting, which this
backend does not model.

Difficulty, regime, and scenario labels stay on the evaluator side. They do
not appear in the policy's KPI message.

## Reward and verification

The environment itself is the verifier. For each accepted action or guardrail
rejection, `openair_congestion/rewards.py::compute_breakdown` returns:

- the `openair_v1` reward version;
- raw KPI measurements;
- each weighted reward term; and
- the scalar total used by evaluation or training.

The malformed call itself never reaches the backend. Its `noop` fallback does,
so the turn receives that backend's inaction reward plus the configured
protocol surcharge. The response records both pieces in the reward breakdown.

The reward tracks changes in SLA violations, delivered throughput, and Jain
fairness. It also accounts for current SLA, PRB, access, fairness, and buffer
pressure, along with optional action magnitude and illegal-action rejection
cost. A clean steady transition is `0`, persistent congestion contributes
negative level costs, and a material improvement can receive positive delta
credit.

Only compare returns when the task manifest, backend, reward version, horizon,
and decoding settings match. The handwritten relief policy is a scripted
baseline. It is neither the verifier nor evidence of a learned policy.

## Backends

| Backend | Action changes the next state? | Use |
|---|---:|---|
| `replay` (default) | Yes | Causal synthetic development, evaluation, and training. |
| `dataset_replay` | No | Recorded-data ingestion and reward/contract diagnostics. |

The bundled `replay` sampler intentionally creates medium/high-difficulty
overload. Its five regimes produce different synthetic pressure patterns:
offered-load pressure (`prb_exhaustion`), higher-load burst snapshots
(`bursty`), SINR/BLER impairment (`interference`), access pressure
(`prach_storm`), and heterogeneous 5QI demand (`qos_competition`). These are
deterministic benchmark dynamics, not claims of live-network fidelity.

Synthetic replay enforces one shared throughput capacity per cell after both
the baseline transition and any accepted action effect. Aggregate delivered
throughput cannot exceed that capacity or UE offered load, and the emitted
buffer, packet-delay, SLA, UE-count, and Jain-fairness fields are recomputed
from the final delivered values. The current generated scenarios use 60 Mbps
per cell. This is an accounting invariant, not a claim that the coarse
synthetic dynamics reproduce a physical RAN.

`dataset_replay` returns the prerecorded next observation regardless of which
action the current policy chooses. Its metadata therefore reports
`training_usable: false` and `diagnostic_only: true`; it must not be used for
on-policy GRPO or model-quality claims.

Episode slots are finite. A reset, normal completion, truncation, or explicit
close releases state immediately. A protocol violation consumes a penalized
`noop` turn and keeps the same episode alive unless that turn reaches its normal
limit. A client that crashes cannot call `/close`. In that case, a later reset
reclaims the inactive session after `session_ttl_s` (one hour by default).

## Quick start

From the root of a NeMo Gym checkout:

```bash
uv venv --python 3.13.14
source .venv/bin/activate
uv sync --extra dev
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

In another activated terminal, collect repeated policy rollouts and profile
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

These commands evaluate a hosted policy. They do not train or update it.

## Checked-in and derived evidence

The repository includes these checked-in files:

- `data/example.jsonl`: five task inputs and tool schemas;
- `data/example_metrics.json`: NeMo Gym example-validation metrics; and
- `data/example_rollouts.jsonl`: five scripted trajectories through the
  resource server's reset/step/close contract.

The scripted rollouts make it possible to review the wiring, lifecycle
behavior, reward decomposition, and bounded completion directly. They are
labeled `resource_server_wiring_not_model_quality` and do not establish SFT or
GRPO quality.

Regenerate them with:

```bash
python resources_servers/openair_congestion/generate_example_rollouts.py
```

Generate the finite-grid single-intervention golden set with:

```bash
python resources_servers/openair_congestion/golden_set.py \
  --out results/openair_congestion_golden_set.jsonl
```

For each deterministic decision state, the script evaluates every action in a
bounded grid, applies one candidate, and then coasts with `noop`. This produces
a reproducible reward-and-dynamics sanity oracle for that grid and horizon. It
is not a universal multi-step optimum or real-model evidence.

## Diagnose recorded data with `dataset_replay`

`dataset_replay` accepts nested KPI-snapshot JSONL. At startup, it validates
every input and reports actionable source context (including file and row or
episode details where applicable).

### KPI snapshot format

Each row represents one timestep. Rows with the same `episode_id` form an
episode and must include at least two observations. Every row needs a
non-empty `cells` list, `cells[].prb_util_dl_p50`, a non-empty `cells[].ues`
list, and `cells[].ues[].delivered_mbps`. Rows labeled `measured` or `recorded`
must also provide `cells[].rrc_connected_ues` and each UE's `bler` and
`sinr_db`; the loader will not invent measured exporter values. Every field
labeled `derived` in the KPI provenance contract is recomputed from its named
source fields. A supplied derived value must match that canonical result within
an absolute tolerance of `1e-6` (with no relative tolerance) or ingestion fails
with file, line, field, supplied value, and expected value context.

Set `kpi_source_mode` explicitly when the snapshots are measured. If the field
is missing, it defaults to `replay`, and the observation is marked synthetic.

```json
{"episode_id":"run_a","step":0,"kpi_source_mode":"measured","recorded_action":{"name":"noop","arguments":{}},"cells":[{"cell_id":0,"prb_util_dl_p50":0.55,"rrc_connected_ues":1,"ues":[{"ue_id":0,"offered_mbps":20.0,"delivered_mbps":18.0,"bler":0.05,"sinr_db":12.0}]}]}
```

See `data/fixtures/sample_provided.jsonl`.

If an episode uses `step`, every row in that episode must provide a unique
integer. The loader sorts the rows by that value; after sorting, `t_s` must be
nondecreasing within the episode. An optional
`recorded_action` is validated and returned as diagnostic metadata, but it
never controls the prerecorded next state. The backend ignores stored scalar
rewards and recomputes reward from the served before/after observations, the
current evaluation action, guardrail result, and configured reward contract.

For the exact checked-in and custom-JSONL YAML and run commands, follow the
[recorded-data section of the Fern tutorial](https://docs.nvidia.com/nemo/gym/main/environment-tutorials/openair-congestion#5-choose-the-right-replay-backend).
This workflow validates ingestion and reward diagnostics. It cannot supply a
counterfactual next state for a different action, so it remains diagnostic
only.

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

Use causal `replay` for GRPO. Configure the model, tokenizer, optimizer, and
SFT/GRPO settings in the NeMo RL training YAML, not the resource-server YAML.
Set the base model or checkpoint in the NeMo RL model section, and keep the
training and evaluation manifests separate. This package does not include a
validated OpenAir-specific NeMo RL job YAML. Use the current NeMo RL GRPO
tutorial as the schema authority.

The [Fern tutorial](https://docs.nvidia.com/nemo/gym/main/environment-tutorials/openair-congestion)
provides the exact NeMo RL Gym configuration and local-vLLM checkpoint
evaluation commands. Some checkpoints need a model-specific chat template or
tool-call parser. Keep the task rows, backend, reward version, horizon,
decoding settings, and repeat count fixed when comparing policies.

## Tests

```bash
pytest resources_servers/openair_congestion/tests -q
pytest responses_api_agents/gymnasium_agent/tests/test_app.py -q
```

The tests cover configuration and schemas, representative deterministic action
effects, guardrails, reward ordering and decomposition, dataset ingestion,
session cleanup, HTTP behavior, checked-in artifacts, and golden-set
self-validation.

## Limitations

- Replay dynamics are deterministic synthetic approximations, not live
  OAI/FlexRIC measurements or physical-simulator fidelity.
- Dataset replay is non-causal and diagnostic-only.
- The finite-grid golden set is a scripted sanity oracle, not a universal
  optimum or evidence of policy quality.
- Checked-in scripted rollouts do not establish SFT or GRPO quality.
- A bounded real-model rollout and manual trace inspection are required by the
  repository quality bar before merge, separately from resource-server unit
  correctness. Fresh NeMo RL training remains follow-up empirical evidence and
  is not claimed here.

## License

Apache-2.0. All telemetry shipped with the offline backends is synthetic
benchmark data.
