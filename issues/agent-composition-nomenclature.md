# Agent Composition: Nomenclature and Data Contracts

**Status:** design note / RFC seed
**Related:** #1867 (standardized trajectories), #1715 (model-server capture), environment-unification RFC, agent-env boundary exploration

## Motivation

Gym's architecture already behaves as if agents were composable, but the concepts are implicit and the
nomenclature is inconsistent ("model", "model server", "agent", "harness", "policy" are used
interchangeably in configs, docs, and code). This note fixes the nomenclature by anchoring each concept
in the data contract it serves — because the contracts, not the class hierarchy, are what make
composition work.

## The three concepts

### Policy model

Anything that serves the model contract:

```
NeMoGymResponseCreateParamsNonStreaming  →  NeMoGymResponse
```

with **single-generation semantics**: the response is one sampling of the LLM — `usage` describes one
model call, every `output` item was produced by that call, and (via the model server) token IDs and
logprobs exist for it. Gym's model servers (`vllm_model`, `openai_model`, …) are policy models.

### Harness

A harness is **not a wire contract — it is a transformer**:

```
Harness : (Policy, Tools/Environment)  →  Agent
```

It owns everything between two model contracts: prompt/context composition, tool dispatch and result
formatting, control flow (retries, loops, subagents), context management (compaction), and stopping
conditions. The harness is never independently addressable — you interact with it only through the
agent it produces. It may be in-process Python (`simple_agent`'s `responses()` loop) or an external
black-box binary (`claude` CLI, OpenCode); ownership of the loop is an implementation detail (see
"Capture modes" in the #1867 trajectory PR).

### Agent

An agent is a harness applied to a policy, exposed behind **the same contract as the policy**:

```
Agent = Harness(Policy) :  NeMoGymResponseCreateParamsNonStreaming  →  NeMoGymResponse
```

but with **flattened-episode semantics**: `output` interleaves items from many generations plus tool
results, and `usage` is an aggregate. This is not a Gym invention — it is exactly the OpenAI
Responses API's hosted-tools semantics (one `Response` containing `web_search_call`-style items from a
server-side loop). The contract is *designed* to hide an internal loop behind one exchange.

## The closure property

Because Agent and Policy serve the same wire contract, harnesses compose:

```
Agent₂ = Harness₂(Agent₁) = Harness₂(Harness₁(Policy))
```

An agent can be the "policy" of a more powerful agent. The type system of the composition is the
`create params → Response` interface; closure is what lets you extend a harness (add planning, add
review loops, add tool layers) and get back something that still plugs in anywhere a model does.

Gym instances of this today:

- `simple_agent` takes a `model_server` ref — the loop's inner calls go to whatever serves
  `/v1/responses`. Point it at another *agent* server and you have `Harness₂(Agent₁)` with zero code.
- `claude_code_agent` composes an external harness over any Gym model server (every model server
  exposes `/v1/messages` by mapping Messages ↔ Responses around its own backend) — a black-box harness
  closed over the same interface.
- Claude Code's internal subagents (sidechains) are the same pattern one level down, inside the
  harness.

## What composition hides — and which contract restores it

Each application of a harness **flattens the level below**: level N's whole episode is level N+1's
"single model call." This is the source of the telemetry gap #1867 addresses, restated structurally:

| Concept | Contract | Role in composition |
|---|---|---|
| Policy model | `create params → Response` (single generation) | leaf |
| Harness | transformer, no wire contract | edge |
| Agent | `create params → Response` (flattened episode) | node — same interface as leaf, hence closure |
| Environment / verifier | `run request → verify response` (reward) | closes an agent over a task distribution |
| **Trajectory** (#1867) | versioned record of **one harness level**: steps = the level's model calls + tool observations, spans = its telemetry | structure-preserving observation of one node |
| Trajectory tree (future) | trajectories linked across levels | full observation of a composed agent |

Key structural fact: in a composed agent, an `agent_turn`'s generation span at level N carries the
`response_id` of a call that, at level N+1, is an **entire trajectory**. The generation span's
identity fields are therefore not just debugging metadata — they are the **edges of the composition
tree**. Subagent/sidechain support (out of scope in #1867, counted in `dropped_records`) is exactly
"record the child trajectory and link it by that identity"; ATIF's subtrajectory references are the
same idea.

Training follows the same geometry: rewards attach at the root (verify response), behavior is recorded
per node (trajectory), but **tokens exist only at the leaf** — the policy model is the only level
where token IDs/logprobs are real. That is why token capture belongs to the model server (#1715) and
reaches the trajectory by a join on generation-span identity, never by retokenizing any level's text.

## The RL reading

The engineering algebra above has an exact RL semantics, and several design decisions only make full
sense under it.

### Two MDPs, one boundary question

- **Token-level MDP** — where training happens. State = the exact token prefix the model server
  rendered; action = next token; policy π = the LLM. *Everything else is environment*: the harness,
  the tools, the verifier, sibling agents. This is Agent Lightning's cut, and it is why token capture
  belongs at the model server (#1715): that layer records `(s, a, log π(a|s))` at the only altitude
  where those objects exist. Retokenization drift is, in these terms, **corruption of the action
  record** — you train against actions the policy never took.
- **Turn-level MDP** — what the Trajectory records. State = model-visible item history; action = one
  complete generation (an `agent_turn`'s policy-produced items); transition = tool execution plus
  harness context management; reward = terminal, from the verifier. `reconstruct_model_input()` is
  state reconstruction; steps are transitions.

  The agent loop in `responses()` **is this MDP unrolled**: each `while True` iteration is one
  (state, action, transition) tuple — `input = body.input + new_outputs` re-materializes `s_t`, the
  single model call samples `a_t ~ π(·|s_t)` (one generation, even when it contains several parallel
  `function_call`s), the tool-execution block is the environment transition producing observations
  `o_t`, and the state recurrence is append-only: `s_{t+1} = s_t ⊕ a_t ⊕ o_t` (compaction replaces
  this recurrence with `s_{t+1} = summarize(s_t) ⊕ …`, hence `context_boundary`). An action is one
  generation and *not* the whole episode because the policy-gradient objective factorizes per
  generation — `Σ_t ∇log π(a_t|s_t)·A_t` — and intermediate states contain environment-injected
  tokens the policy never sampled; the flattened `Response.output` looks like one action but is not
  factorizable without the turn boundaries. Note "action" is level-relative: from one level above
  (agent-as-policy), the *entire* episode is a single temporally extended action — see the options
  reading below. Harness-side error handling (e.g. injecting an invalid-arguments
  `function_call_output`) is an environment transition at this altitude, not a policy action.

The perennial "is the harness part of the agent or the environment?" question has no fixed answer
because it depends on which MDP you fix: **at the token level the harness is environment; at the
system level it is agent.** Both are true simultaneously. (This maps one-to-one onto the
policy/system/hierarchical boundary cuts in the agent-env boundary research note — the cuts are MDP
altitudes, not competing opinions.)

### The harness is an observation function — and that has training consequences

At the turn level, the harness *is* `O(s)`: it renders state into the prompt (system prompt, tool
schemas, compaction summaries) and interprets generations into effects. A policy trained on rollouts
under harness H is optimized for `O_H`, not for the task in the abstract. Swapping harnesses is a
**distribution shift**; cross-harness eval deltas can be dialect-fitting rather than capability
(the GEPA/RR-1 finding, in MDP terms). Recording the true model-visible input per turn is what makes
this detectable at all.

### Composition = options (hierarchical RL)

`Agent-as-policy` is precisely a **temporally extended action**: when harness₂ "calls its policy" and
that policy is agent₁, the call is an option invocation whose internal trajectory lives one level
down. The trajectory tree is therefore the hierarchical-RL data structure: generation-span identity
edges are option invocations, subagent/sidechain trajectories are option-internal episodes, and
hierarchical credit assignment traverses exactly those edges. "Closed under composition" is the
engineering name for "options are actions."

### Credit assignment and the loss mask are derivable, not chosen

Reward enters at the root (verify response) and must flow: episode → turns → tokens. The step
structure gives each stage its natural granularity: episode-relative advantages (GRPO-style) need only
the root reward; turn-level credit (process rewards, per-turn advantages) needs `agent_turn`
boundaries; token-level gradients need the #1715 join. And the loss mask falls out mechanically from
item provenance: policy-produced items (`message`, `reasoning`, `function_call`) are actions —
loss-bearing; `function_call_output` items are environment observations — masked. No per-benchmark
masking heuristics.

### Compaction is a POMDP event, not bookkeeping

Post-compaction, the state is *generated by the environment* (a summarizer), not derived by
concatenation. Without a `context_boundary`, the recorded "state" for later turns is a lie, and any
importance weight computed against it conditions the policy on tokens it never saw. This is why
boundary/compaction semantics were a P0 in the #1867 discussion, not a nice-to-have.

### What the behavior policy spec still lacks

Off-policy corrections need the **complete behavior policy specification** per model call, not just
its outputs: sampling parameters (temperature, top_p, penalties), the true rendered prompt, and the
sampled token IDs with logprobs. All three exist only at the model-server altitude. Follow-up: when
the #1715 join lands, the generation span should carry the applied sampling params alongside token
identity — without them, recorded rollouts under- specify the behavior policy and silently bias
off-policy reuse.

## Nomenclature to adopt

- **Policy model** (or just *policy* in training contexts): serves single-generation Responses.
  Avoid calling agents "models" even when they serve the model interface — say "agent-as-policy".
- **Harness**: the transformer. "Agent harness" in docs should mean this, never the agent server
  process itself.
- **Agent**: `Harness(Policy)` behind the Responses contract, flattened-episode semantics.
- **Agent-as-policy**: an agent used as the inner policy of another harness (the closure property in
  use).
- **Level**: one harness application in a composed agent; trajectories are per-level.
- **Flattening**: the projection an agent's `Response` performs on its internal episode
  (`to_response_output()` is this projection made explicit — trajectory → flat output).

## Implications / recommendations

1. **Docs & configs**: rename ambiguous uses ("model" for agent endpoints) using the glossary above;
   `model_server` refs that may point at agents could be documented as `policy` refs.
2. **Trajectory follow-up**: subagent trajectories = child trajectories linked by generation-span
   `response_id`/`request_id`; add a `parent` identity block to the schema when that lands
   (schema-versioned change).
3. **Composition tests**: a two-level fixture (`simple_agent` over `simple_agent` over a mock model)
   would pin the closure property and the flattening semantics as executable documentation.
4. **`source` per level**: in a trajectory tree, each level labels its own capture provenance
   (`in_process`, `transcript`, …) — fidelity remains per-node data, never a schema fork.
5. **Behavior-policy completeness (RL)**: when #1715's join lands, generation spans should carry the
   applied sampling params (temperature, top_p, …) alongside token IDs/logprobs — recorded rollouts
   otherwise under-specify the behavior policy for off-policy reuse.
6. **Loss-mask helper (RL)**: add a core utility deriving the training mask from item provenance
   (policy items loss-bearing, `function_call_output` masked) so masking is mechanical per the
   turn-level MDP, not per-benchmark convention.
7. **Harness-shift audits (RL)**: with true model-visible inputs recorded per turn, cross-harness
   comparisons can separate dialect-fitting from capability — worth a standard analysis recipe once
   two harnesses emit trajectories for the same tasks.
