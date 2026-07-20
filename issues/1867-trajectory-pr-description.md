## Summary

Extends Gym's native Responses contract so that **the agent's `NeMoGymResponse` is itself the trajectory** — telemetry rides on the contract, nothing is stored twice, and no sidecar object exists. Following the same extension pattern as the `*ForTraining` token-ID variants: `response.output` becomes the episode's lossless item list where model-produced items are `*WithAgentTelemetry` variants tagged with `agent_step_no` (which model call produced them) and `function_call_output` items carry `execution` (independent per-call timing, errors, provider metadata); the `Response` gains optional `generations[]` (per-model-call native usage + provider identity) and `agent_telemetry` (provenance, run totals, dropped-event accounting). Model servers leave everything `None`; plain payloads validate to the plain classes unchanged. The `claude_code_agent` is the reference producer: it harvests the session transcript Claude Code already writes to its ephemeral `CLAUDE_CONFIG_DIR/projects/` (previously deleted unread) and parses it once.

Addresses the telemetry requirements of #1867. Complements (does not conflict with) #1715: the schema is agent-generic and lives in core, but no existing model-server or agent code paths change; other harnesses adopt it by writing a thin adapter over `TrajectoryBuilder`.

## Terminology

Two words are used precisely throughout this PR (and in the schema):

- **Agent step** — one interaction with the environment through the model: a single LLM generation together with the orchestration of the tool calls it issued and their outputs. One iteration of an agent loop is one agent step; `simple_agent`'s `max_steps` counts exactly these. (Providers differ: Claude Code calls these "turns" — its `num_turns`/`--max-turns` count model calls; the trajectory normalizes that to `num_agent_steps`.)
- **Turn** — a full cycle of control: from the user handing input to the agent until the agent hands control back, all tool calls orchestrated and their outputs fed back through the model. A turn contains one or more agent steps. Gym's single-task rollouts are one-turn episodes; multi-turn environments (user simulators, e.g. tau2) have several, delimited in the trajectory by `user_message` steps.

## Why the existing contracts aren't enough

`NeMoGymResponseCreateParamsNonStreaming` and `NeMoGymResponse` are Gym's strict mirrors of the OpenAI Responses API wire contract: one request (`input`: an item list) → one response (`output`: an item list, plus a single `usage`). Gym serves this contract at **two altitudes** with different semantics:

- **Model server** `/v1/responses`: one exchange = one model generation. The response envelope (`id`, `model`, `created_at`) and `usage` describe a single sampling.
- **Agent server** `/v1/responses` (**agent-as-model**: the agent server registers the identical route with the identical request/response types as the model server — `base_responses_api_agent.py:49` vs `base_responses_api_model.py:51` — so callers cannot distinguish them, and an agent can serve as the "model" of another harness): `simple_agent.responses()` accepts the same create params, runs a *whole turn* inside — N agent steps: model calls with tool executions between them — and returns one `Response` whose `output` is the in-order concatenation of every step's items. This mirrors OpenAI's own **hosted-tools** semantics (one `Response` containing `web_search_call`-style items from a server-side loop, one aggregate usage): the Responses contract is *designed* to hide an internal loop behind one exchange.

To be precise about what "flattening" means here: the episode is naturally a *list of lists* — `[[step 1's items], [step 2's items], …]`, each inner list from one generation with its own envelope, usage, and timing. The aggregation flattens it in the structural sense (`[[a,b],[c]] → [a,b,c]`): **every item survives, intact and in order; the partition into steps — and everything attached to it — does not.** The partition is not even recoverable afterwards: a generation may emit `message + calls` together and adjacent generations may both emit messages, so the inner-list boundaries can't be reconstructed from the flat sequence alone. (One genuine content-level merge does happen on main: the Claude Code path inlines reasoning into `<think>` tags inside message text — which is why the trajectory keeps reasoning as native items and moves the inlining into the derived view.)

So the flattening is a legitimate use of the contract, not an abuse — and that is precisely the problem for telemetry. Hiding internal structure is acceptable when the loop runs behind someone else's API; **Gym owns this loop**. In `simple_agent.responses()`, every step's `NeMoGymResponse` is validated in scope — each with its own `id`, `model`, and usage — and then the aggregation point discards them: `model_response.output = new_outputs` returns the *last* step's envelope carrying the full flat item list and a running usage sum (zeroing `cached_tokens`/`reasoning_tokens`, a `TODO` in the code). `run()` stores that flattened response in the rollout row via the verify response, and the collector stamps task/rollout identity (`task_idx`, `rollout_idx`, `agent_ref`) on it. The per-step structure isn't *unavailable* today — **it is in hand at the aggregation point and thrown away there.** (`claude_code_agent` produces the same artifact shape, with the loop running inside the CLI instead.)

What survives is genuinely good: a linear, append-only record of *what was said*, in native item types (which the trajectory keeps unchanged). What the aggregation discards is everything about *how the turn executed*.

### Running example

A weather assistant with a `get_weather` tool — **two turns, two agent steps each**:

```
TURN 1                                   TURN 2
user: "Weather in Paris?"                user: "And in Berlin?"
  step 1: LLM → get_weather(Paris)         step 3: LLM → get_weather(Berlin)
          env → "72°F, sunny"                      env → "65°F, rain"
  step 2: LLM → "It's 72°F and sunny."     step 4: LLM → "Berlin is 65°F, rainy."
  ── control back to user ──               ── control back to user ──
```

**What the model server receives per step** (the append-only recurrence, `s_{k+1} = s_k ⊕ generation_k ⊕ outputs_k`, re-materialized in full each call and continuing across the turn boundary):

```
step 1 input: [u1]
step 2 input: [u1, call₁, out₁]
step 3 input: [u1, call₁, out₁, msg₁, u2]
step 4 input: [u1, call₁, out₁, msg₁, u2, call₂, out₂]
```

**What today's rollout row records** for this episode — and note the sharpest problem first: **there is no single answer.** A multi-turn episode has no defined record shape on main; each harness family produces a different one:

- `simple_agent` **cannot represent the episode**: one `responses()` call is one turn (the loop's break at `msg₁` *is* the end of turn 1), and nothing drives a second turn — the row would contain turn 1 only.
- A Claude-Code-style flat parse yields the sketch below — with `u2` **absent entirely** (mid-episode user input is never an output item).
- tau2 (the one real multi-turn benchmark) records the full dialogue but in a third shape: most of it hand-rewritten into `responses_create_params.input`, the rest in `response.output`, plus bolt-on `num_agent_calls` and min/mean/max usage aggregates on the verify response because per-step stats have no standard slot.

Taking the middle case as the sketch (illustrative numbers):

```jsonc
"response": {
  "id": "resp_D", "created_at": <t_final>,                 // step 4's envelope; steps 1–3's are gone
  "output": [call₁, out₁, msg₁, call₂, out₂, msg₂],        // one flat list; u2 is not in the record at all
  "usage": {"input_tokens": 1036, "output_tokens": 60,     // 4 steps summed
            "input_tokens_details": {"cached_tokens": 0}}  // zeroed by the aggregation (TODO in code)
}
```

**What this PR records** — the same content **once**, in a lossless `response.output` whose items carry their own telemetry, plus per-generation records on the Response (abridged):

```jsonc
"response": {
  "output": [                                           // every item exactly once, tagged with its step
    {"type": "function_call", "agent_step_no": 1, "call_id": "c1", ...},          // call₁
    {"type": "function_call_output", "agent_step_no": 1, "call_id": "c1",
     "output": "72°F, sunny", "execution": {"duration_ms": 532.0, ...}},          // out₁ + its timing
    {"type": "message", "agent_step_no": 2, ...},                                 // msg₁ → control → user
    {"type": "message", "role": "user", "content": "And in Berlin?"},             // u2 IS in the record
    {"type": "function_call", "agent_step_no": 3, "call_id": "c2", ...},          // u1 stays in
    {"type": "function_call_output", "agent_step_no": 3, "call_id": "c2",         //   create_params.input
     "output": "65°F, rain", "execution": {"duration_ms": 498.0, ...}},           //   (never repeated)
    {"type": "message", "agent_step_no": 4, ...}                                  // msg₂
  ],
  "usage": {"input_tokens": 1036, "input_tokens_details": {"cached_tokens": 420}, ...},   // fixed totals
  "generations": [                                      // per model call, on the Response
    {"agent_step_no": 1, "stop_reason": "tool_use", "response_id": "resp_A", "request_id": "req_A",
     "usage": {"input_tokens": 210, "output_tokens": 18, "input_tokens_details": {"cached_tokens": 0}, ...}},
    {"agent_step_no": 2, "stop_reason": "end_turn", "usage": {"…cached_tokens": 180}},
    {"agent_step_no": 3, "stop_reason": "tool_use", "usage": {"…cached_tokens": 240}},
    {"agent_step_no": 4, "stop_reason": "end_turn"}
  ],
  "agent_telemetry": {"source": "transcript", "num_agent_steps": 4, ...}
}
```

Turns are derivable from the output itself (`role: "user"` items / `end_turn` generations); `reconstruct_model_input(response.output, agent_step_no=3, base_input=[u1])` returns exactly step 3's wire input above. Note this is the **first well-defined multi-turn record shape**: the same structure for a `simple_agent`-style loop, a Claude-Code-style session, or a tau2-style dialogue — replacing the three incompatible per-harness accidents above.

### What a consumer can and cannot answer

| Question (against the example) | Today's rollout row | `Trajectory` (this PR) |
|---|---|---|
| Where does step 2 end and turn 2 begin? | Discarded at the aggregation point; guessable at best from item-type transitions (`out₁` is agent-inserted, so a boundary *probably* follows) — and `u2`'s position is a per-benchmark convention | Explicit: generation #2 has `stop_reason: "end_turn"`; `u2` is a `role: "user"` item in the content plane |
| What did step 3 cost, including cache hits? | Unanswerable: one usage (1036/60) summed over all four steps, `cached_tokens` zeroed | `generations[2].usage`: native `NeMoGymResponseUsage` incl. `cached_tokens: 240`, deduplicated per API message; raw provider usage verbatim beside it |
| What exact context did step 3's model call see? | Unrecoverable without boundary guesses; the naive fix (record every step's request) stores the growing prefix 4× — O(n²) | `reconstruct_model_input(output, trajectory, agent_step_no=3, base_input=[u1])` → `[u1, call₁, out₁, msg₁, u2]` — computed from the single stored copy |
| How long did `get_weather(Berlin)` take; did it fail? | Nothing recorded (`function_call_output` carries only `call_id`/`output`/`status`) | Tool execution `c2`: `started_at`/`ended_at`/`duration_ms: 498.0`, `error` — independent per call even when one generation issues several in parallel |
| Which model call was `resp_C` (provider-side debugging)? | The envelope keeps only step 4's `id` | Each generation record carries its `response_id`/`request_id` |
| If context was compacted before step 3, what did it actually see? | Misrepresented: the flat list silently contains items step 3 never saw | The summary is a `NeMoGymContextBoundaryMessage` item in the output; reconstruction restarts there |
| Did the run drop anything (subagents, orphan tool results)? | Silent | `dropped_records` counters, plus `num_agent_steps`/`duration_ms`/`total_cost_usd` for cross-checking |
| Were two calls issued together, and did they run concurrently? | Not representable (see the parallel-calls case below — the flat parses don't even agree on ordering) | Same generation range ⇔ same generation; overlapping tool-execution intervals ⇔ concurrent execution |

### Parallel tool calls: the case that discriminates every alternative

Let one generation issue **two calls at once** — `get_weather(Paris)` and `get_weather(Berlin)` in a single agent step. There are four separable facts to record: **pairing** (which output belongs to which call), **issuance grouping** (both calls came from *one* generation), **ordering semantics** (issue order vs. completion order), and **execution timing** (did they actually overlap). Each record shape handles them differently:

**`simple_agent` (main)** — calls land in issue order, outputs appended after:

```jsonc
"output": [..., call_paris, call_berlin, out_paris, out_berlin, ...]
```

Pairing survives (`call_id`); grouping is only *inferable* from adjacency; no timing. And a fact the record cannot show: `simple_agent` executes "parallel" calls **serially** — the tool loop awaits each POST one at a time (`app.py:130-162`).

**Claude-Code-style flat parse (main)** — each call is emitted only when its result arrives, so the record is in **completion order**:

```jsonc
"output": [..., call_berlin, out_berlin, call_paris, out_paris, ...]   // Berlin finished first
```

Issue order is gone, simultaneous issuance is unmarked, and a call whose result never arrives is dropped. Note the two flat parses don't even agree with each other on ordering — another instance of the record shape being a per-harness accident.

**tau2 (main)** — the chat source *does* group the calls (one assistant message holds `tool_calls=[paris, berlin]`), but the conversion to Responses items flattens the group into adjacency; no timing.

**This PR** — all four facts, explicit, without copying an item:

```yaml
# content plane (response.output):
output: [call_paris, call_berlin,       # issue order (as sampled)
         out_paris, out_berlin]         # arrival order (as observed); pairing via call_id

# on the items themselves (all tagged agent_step_no: 1 — grouping: same tag ⇔ same generation):
#   out_paris.execution:  {started_at: t₀, ended_at: t₀ + 1561ms}
#   out_berlin.execution: {started_at: t₀, ended_at: t₀ + 1615ms}             # overlapping ⇒ concurrent
# on the Response:
generations:
  - {agent_step_no: 1, response_id: resp_A, usage: {...}}
```

This is validated on a real transcript: one generation, two `Bash` calls, 1561.0 ms / 1615.0 ms overlapping intervals. It also unlocks a diagnostic no flat record can express: run the same episode through `simple_agent` with an in-process adapter and the `execution` intervals would be *non-overlapping* — exposing that the harness serializes parallel calls.

**Model-server capture (#1715)** — sees the generation emit `[call_paris, call_berlin]` (issuance grouping preserved) but **never sees execution**: no outputs, no timing — tools run in the harness. The cleanest illustration of why the two capture layers are complementary: the model server knows what was *asked*; only the harness trajectory knows what *happened*.

One-line summary: `call_id` pairing survives every alternative; **issuance grouping, ordering semantics, and concurrency survive only in the trajectory.**

(The same structure was validated on a real Claude Code transcript — including one generation issuing two parallel `Bash` calls timed independently at 1561 ms / 1615 ms, and a per-step cache ramp `0 → 22,951 → 29,567` cached tokens that the summed usage provably erases; see Testing.)

### Design references: what was borrowed from the OpenAI Agents SDK and ATIF

#1867 allows "ATIF, Responses-plus-telemetry, or Gym-native with exporters" as the format choice; this PR is **Responses-plus-telemetry**, using both prior designs as references. To make the comparison explicit:

**From the OpenAI Agents SDK** (an in-memory framework, not a wire format):

- **Items/spans separation** — their conversation `RunItem`s wrap raw Responses items while telemetry lives in tracing spans. We keep the conceptual split but **anchor it in the contract**: our items *are* Gym's validated native types (no `raw_item` wrapper), the function-span data rides each `function_call_output` item as `execution` (`started_at`/`ended_at`/`duration_ms`/`error`), and the generation-span data is the `generations[]` records on the Response — persisted, rather than exported to a tracing backend and lost to the record.
- **Per-model-call boundaries** — their `RunResult.raw_responses` keeps one `ModelResponse` per call, but only in memory. Our generation records are that structure made durable (as index ranges over the output, so nothing is copied).
- **Input reconstruction** — their `to_input_list()` rebuilds only the *next* call's input. Our `reconstruct_model_input(..., agent_step_no=k)` rebuilds *any* step's input, compaction-aware.
- **Deliberately not taken**: the SDK's classes. They are runtime dataclasses holding live `Agent` references and framework lifecycle — not schema-validatable, not serializable, and a framework dependency Gym doesn't need.

**From ATIF** (harbor RFC 0001, a stored-trajectory format):

- **Delta/append-only representation** instead of re-storing the full request per model call — realized here as per-item `agent_step_no` tags over the append-only output, leaner than ATIF's step objects (no content is repeated at all; the initial prompt lives only in `responses_create_params.input`).
- **The context-boundary convention** for compaction, with defined reconstruction semantics (a P0 in the issue discussion).
- **Call/observation correlation and execution metadata** — ATIF's `source_call_id` + observation `extra`; ours ride the `function_call_output` item itself (`call_id` pairing, `execution.extra`).
- **Deliberately not taken**: ATIF as the native format. Content stays in Gym's own Responses types — indeed the content *is* the scored `response.output` — and an ATIF exporter stays mechanical if ever needed.

### Where should the telemetry extend the existing contract? Three candidate surfaces

Nothing prevents adding this telemetry as an extension of the contracts Gym already has — additive optional fields don't break substitutability, and the `*ForTraining` items are precedent for extending the wire mirrors. The design decision is **which surface to extend**. There are three candidates; **this PR picks the first** (evolved: telemetry on the items and the Response, with zero content duplication), and the third is #1715's (complementary, not competing). Earlier drafts used the second; the decisive argument for the first is that once the telemetry is item-anchored and duplication-free, riding the contract means it survives every validation hop (verify requests, resources servers, rollout rows) with no side-channel, and composed agents (agent-as-policy) carry it through automatically.

**Option 1 — a field on `NeMoGymResponse`** (the agent/model impersonation surface):

```yaml
response:                                  # NeMoGymResponse + one new optional field
  id: resp_D
  model: claude-sonnet-4-6
  output: [call₁, out₁, msg₁, call₂, out₂, msg₂]   # unchanged flat items
  usage: {input_tokens: 1036, output_tokens: 60}    # unchanged aggregate
  trajectory:                              # NEW — populated by agents, always null from models
    schema_version: "1.0"
    steps: [...]
```

**Option 2 — an item-free overlay on the rollout record** (`BaseVerifyResponse` → one rollout JSONL row; an earlier draft of this PR — superseded):

```yaml
# one rollout row = the verify response, stamped by the collector
_ng_task_index: 7
_ng_rollout_index: 0
responses_create_params: {input: [u1], tools: [...]}        # the initial prompt lives ONLY here
response:                                                   # the content plane — every item exactly once
  id: resp_D
  output: [call₁, out₁, msg₁, u2, call₂, out₂, msg₂]        # lossless: native reasoning items, issue order,
  usage: {input_tokens: 1036, ...}                          #   unresolved calls kept, user messages recorded
reward: 1.0
trajectory:                                # NEW — annotations only; contains NO items
  schema_version: "1.0"
  source: transcript                       # provenance: transcript | stream_json | in_process
  num_agent_steps: 4
  usage: {input_tokens: 1036, input_tokens_details: {cached_tokens: 420}, ...}
  generations:
    - {agent_step_no: 1, output_start: 0, output_end: 2,    # → [call₁, out₁], by reference
       stop_reason: tool_use, response_id: resp_A, request_id: req_A,
       usage: {input_tokens: 210, output_tokens: 18, ...}, provider_usage: {...}}
    - {agent_step_no: 2, output_start: 2, output_end: 3, stop_reason: end_turn, ...}
  tool_executions:
    - {call_id: c1, started_at: ..., ended_at: ..., duration_ms: 532.0}
  context_boundaries: []
```

**Option 3 — capture at the model server, per generation** (#1715's altitude; the training half):

```yaml
# one record per model call, captured server-side where tokens exist, keyed for a later join
step_record:
  correlation: {rollout: hdr-abc123, request_id: req_C}    # e.g. via ANTHROPIC_CUSTOM_HEADERS
  request: {input: [...], temperature: 0.7, top_p: 0.95}   # true rendered request + sampling params
  response: {output: [...], usage: {...}}
  prompt_token_ids: [...]
  generation_token_ids: [...]
  generation_log_probs: [...]
```

| | **Option 1**: field on `NeMoGymResponse` | **Option 2 (this PR)**: field on the rollout record | **Option 3 (#1715)**: model-server capture |
|---|---|---|---|
| Surface / altitude | The impersonation surface between model and agent altitudes | The episode record (verify response → rollout row) | The per-generation boundary, where tokens exist |
| Data shape it fits | Episode-shaped data on a per-call type — mismatch | Episode-shaped: agent-step ranges, tool executions, compaction, dropped-records | Generation-shaped: token IDs, logprobs, true prompt, sampling params |
| Hot-path cost | Small — telemetry is per-item tags and per-call records, no content duplication | Small, but the sidecar must be re-attached at every hop and stripped by any validator that doesn't know it | Server-side store; response payloads untouched |
| Model-server semantics | Field is dead at the model altitude; the mirror stops being OpenAI-shaped | Untouched | Untouched |
| Schema versioning | Telemetry evolution pinned to the response type every server signature depends on | Independent `schema_version` | Independent capture format |
| Composition (agent-as-policy) | Inner trajectories ride through hops automatically — the one real advantage | Per-level records, linked by generation-span `response_id`/`request_id` | Joins per level via correlation keys |
| Verdict | **Chosen** — the item-anchored form has no duplication, survives every hop by construction, and follows the `*ForTraining` precedent; the residual cost (fields dead at the model altitude) is accepted and documented | Superseded — a sidecar must be re-attached at every hop and can drift from the response it describes | Complementary — supplies the token layer this PR deliberately doesn't |

The rule generalizing the choice: **put each fact at the altitude where it is born, on the contract that already lives there** — item-shaped facts (which step produced me; how my tool execution went) ride the items, exactly like `*ForTraining` token IDs; generation-shaped facts (per-call usage/identity/stop semantics) ride the Response beside the aggregate `usage` they decompose; run-shaped facts ride `agent_telemetry`; token IDs stay at the model server (Option 3) until joined. Items only carry facts that are 1:1 with the item — per-generation facts are keyed one level up (`agent_step_no` is the foreign key), because generations can exist without items (a call killed mid-step) and group facts stamped on members invite repetition and inconsistency.

## Capture model: one contract, two ownership modes

Gym will never own every agent loop — and the contract must not care. There are two capture modes producing the **same schema**:

| | White-box: Gym owns the loop | Black-box: Gym wraps an external harness |
|---|---|---|
| Examples | `simple_agent`, browsecomp, finance, gymnasium agents | `claude_code_agent` (this PR), OpenCode, mini-swe-agent |
| How the trajectory is produced | Drive `TrajectoryBuilder` inside `responses()` as the loop executes | An adapter reconstructs it post-hoc from the harness's artifacts (transcripts, stream events, SDK items) |
| Model-call boundaries | Exact — each per-call `Response` is in hand | Exact *if* artifacts carry per-call identity (Claude Code transcripts do: `message.id`, `requestId`); degraded fallback otherwise |
| Per-call usage | Exact | As recorded by the harness (Claude Code: per-message usage incl. cache detail) |
| Tool timing | **Measured** around the tool invocation | **Reconstructed** from record timestamps (`started_at` ≈ issuing record's write time; `completed_at` exact) |
| True prompt visibility | Full — Gym composes the model request | Partial — the harness composes prompts internally; only the model server sees the final request (→ #1715) |
| Token IDs (training) | Via the #1715 model-server join | Same join, provided the harness is routed through Gym's model server |

What makes the fidelity difference legible to consumers, instead of a silent quality lottery:

- **`source`** labels provenance (`"transcript"` vs `"stream_json"`, later `"in_process"`) — the "dialect/stage" label the issue discussion asks for.
- **`null` means not observable from this source** — never approximated, never fabricated (e.g. no timing in the stream-json fallback rather than step-boundary guesses).
- **`dropped_records`** counts events seen but not represented; its absence means nothing was dropped.
- **Same schema, same validation** in both modes — fidelity differences are data (nullable fields, provenance), never a schema fork.

The deeper structure behind both modes: an **agent is a harness applied to a policy, closed under composition** — `Agent = Harness(Policy)` serves the same `create params → Response` contract as the policy itself (OpenAI's hosted-tools flattening semantics), so agents can serve as the policy of more powerful agents. The `Response` is the (lossy-by-design) *composition interface*; the `Trajectory` is the structure-preserving record of *one harness level*; a composed agent's full observability is a tree of trajectories linked by generation-span identity (`response_id`/`request_id`), which is also how subagent support will land. See `issues/agent-composition-nomenclature.md` for the full nomenclature and its contract anchoring.

## Relationship to the existing `parse_stream_json` path

The agent already parsed Claude Code's stream-json stdout ([`parse_stream_json`, app.py:71 on main](https://github.com/NVIDIA-NeMo/Gym/blob/main/responses_api_agents/claude_code_agent/app.py#L71)) into Responses output items. That path could not simply *become* the trajectory:

1. **Its input doesn't contain the required data.** Stream-json stdout carries no timestamps, no `requestId`, and no `toolUseResult` execution metadata — those exist only in the on-disk transcript records, which is precisely the #1867 gap (tool timing, per-call identity). The transcript is the trajectory's primary source; stdout events are the degraded fallback.
2. **It is intentionally lossy in ways the trajectory must not be.** It inlines thinking into `<think>` tags inside message text (vs a native `reasoning` item), drops a `tool_use` whose result never arrives (vs orphan accounting in `dropped_records`), collapses per-model-call boundaries into one flat list, and has no representation for compaction, errors, or per-call usage.

Instead, this PR **replaces it with a single parse producing the telemetry-bearing Response**: `build_trajectory()` returns `(output_items, generations, agent_telemetry)` — the lossless, tagged item list destined for `response.output` plus the Response-level records. `parse_stream_json` keeps its signature as a thin wrapper; the old hand-rolled event loop is deleted, and response and telemetry cannot drift apart because they are one object.

**The response contract changes deliberately** (this agent is unbaselined; the change is the point of the Responses-centric design):

- Reasoning is a native `reasoning` item instead of `<think>`-inlined message text (the one genuine content-level merge on main, removed).
- Function calls appear at their **issue positions** (main emitted them at result-completion positions), with outputs at arrival positions — pairing stays via `call_id`.
- A call whose result never arrived is **kept** (main silently dropped it); mid-episode user messages are recorded as items (main omitted them).
- Usage accounting is fixed: main double-counted (`result` event totals summed on top of per-assistant usage) and hardcoded `cached_tokens` to 0.
- The content plane is built from the transcript when available (stream-json stdout as fallback) — so after a timeout, the partial transcript surfaces as a partial response instead of an empty padded one.

Verifiers reading the final assistant message are unaffected; consumers that parsed `<think>` tags out of message text should read native `reasoning` items instead.

## What's in the PR

- **`nemo_gym/openai_utils.py`** — the contract extension: `*WithAgentTelemetry` item variants (`agent_step_no` tag; `execution: NeMoGymToolExecution` on tool outputs), `NeMoGymContextBoundaryMessage` (compaction marker), and two optional `NeMoGymResponse` fields — `generations: list[NeMoGymGeneration]` and `agent_telemetry: NeMoGymAgentTelemetry`. Variants have required discriminating fields and sit first in the item union, so plain payloads deterministically validate to the plain classes (full core suite: 1,141 tests, zero regressions).
- **`nemo_gym/trajectory.py`** — `TrajectoryBuilder` (`build()` → `(output_items, generations, agent_telemetry)`; owns agent-step numbering/tags, model-call dedup by `response_id`, call/execution correlation, orphan handling), `usage_from_provider()`/`summed_usage()`, `reconstruct_model_input()`/`to_response_create_params()`/`agent_step_slices()` (tag-based, compaction-aware).
- **`responses_api_agents/claude_code_agent/trajectory.py`** — Claude Code adapter: parses transcript records (timestamps, `requestId`, `toolUseResult`, `sourceToolAssistantUUID`, `isCompactSummary`, sidechains) and stream-json events (fallback; missing telemetry stays `null`, never fabricated), drives the builder.
- **`responses_api_agents/claude_code_agent/app.py`** — harvests `projects/*/*.jsonl` from the per-run config dir before cleanup (including after timeouts), builds the telemetry-bearing Response in one parse (`capture_trajectory: true` knob), fixes the two latent usage bugs (double-counted totals; `cached_tokens` hardcoded to 0), and deliberately upgrades `response.output` to the lossless form (see reconciliation section). The verify response carries no extra field — **the response is the trajectory**.
- Docs in the agent README (schema, reconstruction semantics, adapter how-to) and config YAML.

## #1867 acceptance criteria

- ✅ Standard schema for per-model-call token stats and response metadata (native `NeMoGymResponseUsage` on `generations[]`, provider identity per call)
- ✅ Prompt/completion/total/cached tokens; raw provider usage verbatim covers non-native fields (e.g. `cache_creation_input_tokens`); reasoning tokens 0 for Anthropic (not reported)
- ✅ Task/rollout identity via the surrounding rollout row; agent-step identity via `agent_step_no`/`request_id`/`response_id`; timestamps per step
- ✅ Reconstructable model-visible history, delta-encoded, compaction-aware, resolving to native create params
- ✅ Tool observations with output, status/error, start/end timestamps, duration — independent per parallel call
- ✅ Versioned + validatable; dropped events counted, no silent omissions

## Testing

- `tests/unit_tests/test_trajectory.py` — 26 tests on the builder and helpers, including the load-bearing contract proof: telemetry-tagged items, `generations`, and `agent_telemetry` survive `NeMoGymResponse.model_dump → model_validate` round trips, while plain model-server payloads validate to the plain classes with telemetry `None`
- Full core unit suite: **1,141 tests pass** — the `openai_utils` contract extension causes zero regressions
- Agent suite: 68 tests pass — `parse_stream_json` tests encode the lossless response contract (native reasoning items, issue-order parallel calls with per-item tags, unresolved calls kept)
- Validated against real Claude Code transcripts: item tags `[1,1,1,1,1,2,2,3,∅,∅]` over a 10-item output, two parallel `Bash` calls from one generation timed independently on their own output items (1561.0 ms / 1615.0 ms), the per-step cache ramp (`cached_tokens` 0 → 22,951 → 29,567) in `generations[].usage`, the initial prompt correctly absent, and the full Response round-tripping through the contract unchanged

## Training path: relationship to #1715 and retokenization drift

**This trajectory is eval/observability-only by design — it does not yet support training the policy model.** No `prompt_token_ids` / `generation_token_ids` / `generation_log_probs` are captured, for two reasons: the Claude Code surface doesn't expose them (the transcript and stream-json record *text*; `/v1/messages` returns no token IDs), and token capture belongs at a different layer (below). The contract is deliberately ready, though: `response.output` uses the `NeMoGymResponseInputItem` union, which already includes the `*ForTraining` variants (`TokenIDLogProbMixin`) — an output carrying token IDs validates today unchanged, and each item's `agent_step_no` tag says exactly which model call it belongs to. What's missing is a source, not a slot.

**Do not train from this trajectory's text — retokenization drift is real** (see [vLLM's Agent Lightning post](https://vllm.ai/blog/2025-10-22-agent-lightning)). Re-tokenizing recorded text does not reproduce the sampled token IDs: BPE merge boundaries shift across concatenation points, chat-template re-application differs from what the server rendered, and tool-call/reasoning serialization round-trips differently. RL importance ratios (PPO/GRPO) need logprobs of the *exact sampled tokens*; drifted tokens make "on-policy" data silently off-policy. Main's parse was its own cautionary example: it inlined reasoning as `<think>…</think>` text — a re-serialization the policy never sampled. The lossless content plane removes exactly that class of drift source.

**The composition with #1715 (model-server capture) is the training path** — the two capture layers are complementary halves of one record, split the way Agent Lightning splits it (tokens captured at the serving layer, semantics reconstructed around them):

| | Model-server capture (#1715) | Agent Harness trajectory (this PR) |
|---|---|---|
| Sees | The true model boundary: post-conversion request (Messages→chat template), sampled **token IDs + logprobs**, per-call usage and identity | Rollout semantics: agent-step structure, tool timing/errors, compaction boundaries, reward linkage via the verify response |
| Blind to | Rollout/task attribution, tool execution, verifier outcome | Token IDs, the rendered prompt, applied sampling params |

The join key already exists on this PR's side: each **`NeMoGymGeneration`'s `request_id`/`response_id`**. The missing third piece is small: stamp rollout identity onto the server side — for the Claude Code path the model server is already in the loop when `model_server` is set, and Claude Code supports `ANTHROPIC_CUSTOM_HEADERS`, so `_run_claude_code` can inject a per-rollout correlation header that the server's `StepRecord`s carry. A merge step then grafts captured token IDs into the items tagged with the matching agent step (upgrading them to `*ForTraining` variants in place), with the telemetry contributing what tokens alone can't: agent-step tags and tool-output items for loss masking, compaction-aware context reconstruction, and reward for credit assignment. No retokenization anywhere — tokens flow from sampling time to training verbatim; a cheap validator (retokenize, diff against captured IDs, alert on mismatch) turns residual drift from a silent bias into a monitored invariant.

## Limitations / follow-ups

- `started_at` for a tool call is the issuing assistant record's write time (execution start isn't observable without hooks, which the bare runtime avoids); `completed_at` is exact.
- Subagent (sidechain) trajectories are counted but not represented — matches #1867's out-of-scope list; natural follow-up issue.
- Only the Claude Code harness emits trajectories so far. For in-process loops like `simple_agent`, the adapter is even simpler than Claude Code's: drive `TrajectoryBuilder` **inside `responses()`**, where every per-call `NeMoGymResponse` is already in hand (exact boundaries, real per-call usage) and tool timing can be *measured* around the resources-server calls rather than reconstructed from artifacts. External-harness agents follow the Claude Code adapter pattern; #1715's model-server capture supplies the token layer either way.
- Training enablement (see the training-path section): per-rollout correlation header injection, the `StepRecord` ⋈ `agent_step` merge step, and a retokenization-drift validator.

