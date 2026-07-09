## Summary

Prototype [PR](https://github.com/NVIDIA-NeMo/Gym/pull/1962)

NeMo Gym could benefit from a shared abstraction for **Agent Architecture Components**: behavior-shaping modules around the policy that can be varied independently of the Task and may be optimized without retraining policy weights.

In code these are **`AgentModule`s** with versioned **`AgentModuleRef`** provenance. Each module has a cognitive capability **`type`** discriminator (`working_memory`, `skill_library`, `planning`, …). More types may be added over time.

Today these concerns are spread across:

- `nemo_gym/prompt.py` — run-level prompt templates materialized into `responses_create_params.input`
- `nemo_gym/skills.py` — run-level `skills.path` with `skills_ref` provenance ([#1256](https://github.com/NVIDIA-NeMo/Gym/issues/1256))
- Agent server config wiring (`model_server`, `resources_server`, `max_steps`, tools, etc.)
- Per-agent ad hoc state (memory, playbooks, tracing, permissions)
- Rollout rows (`responses_create_params`, `agent_ref`, verifier fields) plus runtime stamps (`_ng_task_index`, `skills_ref`, …)
- Agent `run()` composes the rollout around Agent `responses()` (module activation, `/v1/responses`, verify, adaptation) without a shared module lifecycle

This issue proposes **`AgentModule`**, a **provenance contract** for optimizable artifacts, and vocabulary aligned with common agent frameworks — while preserving Gym's Agent / Environment separation.

### Terminology

| Term | Use for |
|------|---------|
| **`AgentModule`** | Code lifecycle (`activate`, `adapt`, `module_refs`) |
| **`type`** | Cognitive capability discriminator: `working_memory`, `skill_library`, `long_term_memory`, `planning`, `reasoning`, `control`, … |
| **`AgentModuleRef`** | Versioned provenance stamp on rollout results |
| **Run-level** | Selected on `gym eval run` (`prompt_config`, `skills.path`) — not a dataset-row field |
| **Environment** | Resources server only (`seed_session`, tools, `verify`, reward) |

Notes:

- Avoid **environment-level** for skills/prompts; that collides with Gym's formal Environment.
- Prompts are `working_memory`; skills are `skill_library`; playbooks are usually `long_term_memory`.

## Background

### Why this matters now

Several workflows are pushing on the same boundary:

| Workflow | What varies | Current integration |
|----------|-------------|---------------------|
| GEPA / DSPy prompt optimization ([#1551](https://github.com/NVIDIA-NeMo/Gym/pull/1551)) | system/user prompt | external script calls Agent `/run` as black-box objective |
| ACE / TALES playbook learning ([#1706](https://github.com/NVIDIA-NeMo/Gym/pull/1706)) | playbook + memory | in-process state inside a custom Agent harness |
| Skill evaluation ([#1256](https://github.com/NVIDIA-NeMo/Gym/issues/1256)) | skill library directory | `skills.path` run knob + `skills_ref` |
| Policy LLM training | model weights | rollout JSONL trajectories |
| Trajectory logging ([#1867](https://github.com/NVIDIA-NeMo/Gym/issues/1867)) | per-step evidence | model/tool/usage telemetry |

These all need:

1. A stable place to declare behavior-shaping inputs
2. Provenance on every rollout (which artifact version was active?)
3. A trajectory contract modules can adapt from for optimization
4. Clear separation from Environment reward/verification and rollout collection concerns

### Related Gym work

- [#1256: agent skill evaluation infrastructure](https://github.com/NVIDIA-NeMo/Gym/issues/1256) — skills as a **run-level** knob; `skills_ref` provenance
- [#1867: standardized trajectories format](https://github.com/NVIDIA-NeMo/Gym/issues/1867) — reconstructable trajectories, tool timing, per-model-call stats
- `issues/agent-module-e2e-stress-test.md` — GPQA slice with `prompt_artifact` / `gepa_prompt` module scaffolds
---

## Problem

### 1. No common boundary for non-policy cognitive capabilities

Working memory, skill libraries, long-term memory/playbooks, planning, reasoning strategies, and control policies all shape Agent behavior. Today their concrete implementations are scattered across prompt templates (`prompt_config`), skill directories (`skills.path`), per-agent harness code, and ad hoc config fields.

### 2. `run()` wraps `responses()`, but modules need a shared lifecycle

`SimpleResponsesAPIAgent` exposes two abstract endpoints:

```python
# nemo_gym/base_responses_api_agent.py
async def responses(...) -> NeMoGymResponse      # Agent behavior loop
async def run(...) -> BaseVerifyResponse         # Agent-server rollout endpoint
```

In `simple_agent`, the split is sharp:

**`responses()` / `/v1/responses` = Agent behavior loop**

- Normalize `input`
- Call policy model server
- Route function calls to resources server
- Append tool outputs, accumulate usage, enforce `max_steps`
- Return model-visible trajectory fragment

**`run()` / `/run` = Agent-server rollout endpoint around that loop**

```text
simple_agent.run(row)
  -> resources_server /seed_session     # Environment
  -> AgentModule.activate(...)
  -> self /v1/responses                 # Agent behavior
  -> resources_server /verify           # Environment
  -> AgentModule.adapt(...)
  -> return BaseVerifyResponse          # rollout record
```

Rollout collection already calls `/run`; this work should not require moving seed/verify or `/run` responsibilities to a separate execution abstraction. Instead, Agent modules provide a common way for each Agent server's existing `/run` path to expose behavior-shaping inputs, adaptation events, and provenance.

### 3. Prompt and skills are parallel but not unified

Both are run-level knobs applied to skill/prompt-agnostic datasets:

- `prompt.py` — builds `responses_create_params.input` from YAML templates at rollout time
- `skills.py` — loads skill directories, stamps `skills_ref` with content hash

They share the pattern "vary behavior without changing the dataset" but use different contracts and provenance shapes.

### 4. Policy training vs module optimization are conflated in rollout data

Rollouts must answer:

- Which prompt/skills/playbook/memory snapshot was active?
- Which model checkpoint and module versions were used?
- Was reward attributable to the policy or to a module artifact change?

Without module refs, ablations and optimizer loops are hard to compare fairly.

### 5. Framework vocabulary is fragmented

External agent frameworks converge on similar layers but use different names. Gym should document a mapping, not blindly adopt one SDK's API.

---

## External framework research (terminology map)

### OpenAI Agents SDK

[openai/openai-agents-python](https://github.com/openai/openai-agents-python/)

| Concept | Role | Gym mapping |
|---------|------|-------------|
| Agent | instructions + tools + guardrails + handoffs + model | Agent harness + modules |
| Tools | function/MCP/hosted tools | Environment tools + Agent tool policy |
| Guardrails | input/output validation | Control module |
| Handoffs / agents-as-tools | multi-agent delegation | Workflow module |
| Sessions | conversation history | State module |
| Tracing | run observability | Observation module; ties to [#1867](https://github.com/NVIDIA-NeMo/Gym/issues/1867) |
| Sandbox agents | filesystem workspace | Capability / runtime module |

Core insight: **Agent = configured loop around a model**, not just the model call.

### Claude Agent SDK

[Agent SDK overview](https://code.claude.com/docs/en/agent-sdk/overview)

| Concept | Role | Gym mapping |
|---------|------|-------------|
| Built-in tools | Read/Edit/Bash/Glob/Grep/… | Capability module (native runtime) |
| Hooks | PreToolUse, PostToolUse, Stop, SessionStart, … | Control + observation hooks on Agent loop |
| Subagents | delegated specialists | Workflow module |
| MCP | external tool servers | Capability module |
| Permissions | allowed_tools, permission_mode | Control module |
| Sessions | resume/fork context | State module |
| Skills | `.claude/skills/*/SKILL.md` | `skill_library` module ([#1256](https://github.com/NVIDIA-NeMo/Gym/issues/1256)) |
| Memory | CLAUDE.md project context | Context module |
| Plugins | bundled extensions | Capability/workflow module |

Core insight: **most complexity lives around a simple model/tool loop** — permissions, context, extensibility, session storage.

### Microsoft Agent Framework

[microsoft/agent-framework](https://github.com/microsoft/agent-framework)

| Concept | Role | Gym mapping |
|---------|------|-------------|
| Middleware | request/response pipelines | Agent module `activate` chain |
| Workflows | graph orchestration (sequential, concurrent, handoff, group) | Workflow module / Agent harness |
| Checkpointing / HITL / time-travel | durable workflows | Agent harness + state modules |
| Declarative agents | YAML-defined agents | Agent config + modules |
| Agent Skills | domain knowledge bases | `skill_library` module |
| Observability | OpenTelemetry | Observation module |
| AF Labs | benchmarking, RL | Optimizer drivers over Gym rollouts |

Core insight: **separate agent definition from orchestration graph from observability**.

### LangGraph + LangSmith

[LangSmith for LangGraph](https://info.langchain.com/langsmith/for-langgraph)

| Concept | Role | Gym mapping |
|---------|------|-------------|
| Graph nodes/edges | workflow structure | Workflow module / Agent harness |
| State transitions | per-node state | State module + trajectory events |
| Traces | node/tool visibility | [#1867](https://github.com/NVIDIA-NeMo/Gym/issues/1867) trajectory contract |
| Offline eval | dataset regression testing | `gym eval run` + aggregate metrics |
| Version comparison | prompt/graph A/B | module refs + optimizer runs |
| Production → eval loop | traces become test cases | rollout JSONL + annotation |

Core insight: **build / test / deploy / monitor / improve** requires stable traces and versioned artifacts.

### Google ADK

[ADK Gemini models](https://adk.dev/agents/models/google-gemini/)

| Concept | Role | Gym mapping |
|---------|------|-------------|
| LlmAgent + model config | model routing | Model module |
| Tools | function/MCP/OpenAPI tools | Capability module |
| Artifacts | agent-owned outputs | `AgentModuleRef` |
| Sessions / memory | cross-turn state | State module |
| Evaluation / optimization | offline improvement | Optimizer over modules + policy |

### Claude Code design space (paper)

[Dive into Claude Code (arXiv:2604.14228)](https://arxiv.org/abs/2604.14228)

The paper argues most agent systems share:

- A **simple while-loop**: model → tools → repeat
- **Surrounding systems**: permissions, context compaction, extensibility (MCP, plugins, skills, hooks), subagent delegation, session storage

Relevant design principles for Gym:

| Paper theme | Gym implication |
|-------------|-----------------|
| Human decision authority | Control modules; eval semantics documented |
| Safety / permissions | tool policy modules; admissible-command settings explicit |
| Reliable execution | rollout collection retries, trajectory fidelity ([#1867](https://github.com/NVIDIA-NeMo/Gym/issues/1867)) |
| Capability amplification | skills, tools, MCP, subagents as modules |
Core insight: **most complexity lives around a simple model/tool loop** — permissions, context, extensibility, session storage ([arXiv:2604.14228](https://arxiv.org/abs/2604.14228)).

---

## Concrete abstractions to refit Gym toward

The sections above are terminology maps. This section extracts **reusable architectural primitives** from cognitive programming, agentic control, and memory frameworks — the kinds of abstractions Gym should adopt (or explicitly map to) when generalizing beyond `prompt.py` / `skills.py`.

### Layer model (synthesis)

Most frameworks decompose into the same stack:

```text
┌─────────────────────────────────────────────────────────────┐
│ Optimizer / Trainer (GEPA, ACE, eval driver)                  │  ← changes artifacts or weights
├─────────────────────────────────────────────────────────────┤
│ Rollout collection / runner                                  │  ← starts runs, persists outputs
├─────────────────────────────────────────────────────────────┤
│ Agent harness loop (ReAct, reflection, plan-execute)          │  ← model ↔ tools until stop
├─────────────────────────────────────────────────────────────┤
│ Agent modules (working_memory, skill_library, memory, control)│ ← behavior-shaping capabilities
├─────────────────────────────────────────────────────────────┤
│ Policy model + auxiliary model configs                      │  ← trainable or swappable LLMs
├─────────────────────────────────────────────────────────────┤
│ Environment (tools, state, verify, reward)                  │  ← Gym resources servers
└─────────────────────────────────────────────────────────────┘
```

Gym's gap: layers 3–4 are collapsed into per-agent `app.py` files and a few rollout knobs (`prompt_config`, `skills.path`). The proposal is to make layer 4 first-class without changing the existing rollout collection driver or `/run` contract.

---

### A. Cognitive / program-synthesis layer (DSPy)

[DSPy](https://dspy.ai/) treats LLM programs as **composable modules** optimized by **teleprompters**, not hand-tuned prompt strings.

| DSPy primitive | What it is | Gym refit target |
|----------------|------------|------------------|
| **Signature** | Typed I/O contract for a step (`question → answer`) | `responses_create_params` schema + verifier fields; benchmark row shape |
| **Module** | Composable unit (`Predict`, `ChainOfThought`, `ReAct`, custom `Module`) | Agent harness pattern (simple_agent loop, langgraph graphs) |
| **Predictor** | One optimizable LM call inside a module (instruction + demos) | **Prompt module** — the unit GEPA actually mutates |
| **Demonstration** | Few-shot example attached to a predictor | `responses_create_params.input` prefix examples; materialized training rows |
| **Teleprompter / Optimizer** | `BootstrapFewShot`, `MIPROv2`, `GEPA` — compiles program against metric | **Optimizer driver** (external script or `GepaPromptModule`) — not part of rollout hot path |
| **Metric + feedback** | Scalar score + optional natural-language feedback | Environment `verify` reward + rich verifier metadata for reflection |
| **Trace** | Full execution path per predictor | [#1867](https://github.com/NVIDIA-NeMo/Gym/issues/1867) trajectory; GEPA needs predictor-scoped trace slices |
| **Pareto frontier** (GEPA) | Candidate pool across eval instances | Optimizer state, not Agent runtime |

**Refit for `prompt.py`:** today a prompt is a YAML template applied through run-level `prompt_config`. DSPy suggests promoting it to a working-memory module with:

- `signature` (which row fields fill which slots)
- `instruction` (optimizable text)
- `demonstrations[]` (optional few-shot blocks)
- `name` (which step in a multi-module program this prompt feeds)

GEPA does not optimize "the whole agent" — it optimizes **named predictors** using trace-local feedback. Gym should allow multiple prompt artifacts per Agent (system vs user vs reflection vs curator).

**Refit for PR #1551:** `optimize.py` black-boxes `/run`. Better shape:

```text
Teleprompter (GEPA driver)
  -> for each candidate prompt artifact:
       rollout collection runs dataset with candidate Agent modules
       -> trajectories + rewards + textual feedback
     -> reflect -> propose delta -> update prompt artifact
```

---

### B. Agentic control patterns (reasoning loops)

From [ReAct](https://agentpatterns.ai/agent-design/react-pattern/), [LangGraph tutorials](https://docs.langchain.com/oss/python/langgraph/graph-api), and Gym's own `langgraph_agent` implementations.

| Pattern | Control flow | Gym example today | Module / harness implication |
|---------|--------------|-------------------|------------------------------|
| **ReAct** | Thought → Action → Observation loop until answer or budget | `simple_agent.responses()` | Default **policy harness**; `max_steps` = step budget |
| **Plan-and-Execute** | Planner emits multi-step plan; executor runs steps | `rewoo_agent` (plan → worker → solve) | **Workflow module** separate from policy |
| **Reflection / Reflexion** | Generate → critique → revise | `reflection_agent` (generate → reflect loop) | Reasoning loop + auxiliary model config + scratchpad state |
| **ReWOO** | Plan with variable substitution, execute without re-planning | `rewoo_agent` | Planner artifact + deterministic executor |
| **Tool-use loop** | Model emits function calls; runtime executes | All resources-server agents | **Tool router** in harness; tools owned by Environment |

**Key abstraction: Scratchpad vs State vs Messages**

- ReAct **scratchpad** = growing Thought/Action/Observation trace (working memory)
- LangGraph **State** = typed dict with **reducers** (`add_messages`, `operator.add`) merging node outputs
- Gym **`NeMoGymResponse.output`** = append-only model-visible items (function calls, messages, tool outputs)

Gym should treat the scratchpad as a first-class **Trajectory** object ([#1867](https://github.com/NVIDIA-NeMo/Gym/issues/1867)), not only the final `response` blob. Harness pattern (ReAct vs reflection vs ReWOO) is **Agent-owned**.

**Refit for `simple_agent.responses()`:** document it as Gym's default **ReAct-with-tool-calls harness** — not the definition of "Agent."

---

### C. Memory architectures (Letta / LangGraph)

Two complementary models appear across [Letta/MemGPT](https://www.letta.com/blog/agent-memory/) and [LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence):

| Tier | Letta (MemGPT) | LangGraph | Gym refit |
|------|----------------|-----------|-----------|
| **Working / core** | In-context memory blocks (persona, task) | Graph state keys (messages, custom fields) | `responses_create_params.input` + injected module context |
| **Thread / recall** | Searchable conversation history | **Checkpointer** (thread-scoped snapshots) | Session cookies + rollout trajectory per task |
| **Long-term / archival** | Vector/graph store, agent retrieves via tools | **Store** (cross-thread key-value) | Playbook, skill library, external memory index |
| **Compaction** | Agent edits/summarizes blocks | Context pruning events ([#1867](https://github.com/NVIDIA-NeMo/Gym/issues/1867)) | Future **context_engineering** module |

**Letta insight:** agent **self-edits** memory via tools (`memory_insert`, `memory_replace`) — memory updates are Agent actions.

**LangGraph insight:** separate **short-term** (checkpointer / thread) from **long-term** (store / cross-thread). Reducers define how concurrent node updates merge.

**Refit recommendations:**

- **`working_memory` module** ≈ core context block (prompt-backed, static or optimizable)
- **`skill_library` module** ≈ skill packages / reusable operational procedures ([#1256](https://github.com/NVIDIA-NeMo/Gym/issues/1256))
- **`long_term_memory` module** ≈ playbooks, recall, archival stores, summaries, with explicit `lifecycle` and `retrieval_policy`
- Environment session state (cookies, `seed_session`) ≠ Agent memory — keep boundary clear

---

### D. ACE three-role context engineering

[ACE (Agentic Context Engineering)](https://ace-agent.github.io/) is a concrete playbook-optimization architecture beyond "store a string prompt."

| Role | Responsibility | Optimizes | Gym module type |
|------|----------------|-----------|-----------------|
| **Generator** | Run task with current playbook in context | — (uses playbook) | Policy harness + `playbook` context module |
| **Reflector** | Review trajectory; score bullets helpful/harmful; extract lessons | — (produces feedback) | Auxiliary model config |
| **Curator** | Apply **delta updates** (add/edit/prune bullets); deterministic merge | Playbook artifact | `long_term_memory` module `adapt()` + `AgentUpdateEvent` |

**Playbook bullet schema** (addressable, incrementally mergeable):

```text
[str-00001] helpful=5 harmful=0 :: Always verify data types before processing
```

**Design rules ACE enforces (worth adopting):**

- No monolithic playbook rewrites (prevents **context collapse**)
- Generator only proposes candidates; Reflector only scores; Curator only applies add/edit/remove
- Whole-playbook regeneration is forbidden during optimization

**Refit for PR #1706:** move in-process playbook dict to a **`PlaybookModule`** with:

- `module_refs()` → content hash of bullet set
- `lifecycle` → explicit (`per_rollout` | `per_run` | `persistent`)
- `adapt(TrajectoryEvent)` → curator deltas after terminal verify
- separate **reasoner** (policy model server) from **reflector** (auxiliary model config)

---

### E. Policy weights vs module artifacts

| What gets optimized | Mechanism | Gym artifact |
|--------------------|-----------|--------------|
| Policy LLM weights | Weight training on token-level trajectories ([#1867](https://github.com/NVIDIA-NeMo/Gym/issues/1867), [#1715](https://github.com/NVIDIA-NeMo/Gym/pull/1715)) | Model checkpoint |
| Prompt / playbook / skills | GEPA/ACE/skill search on same trajectories | `AgentModuleRef` + `AgentUpdateEvent` |
| Environment / task | Benchmark design, verifier | Dataset + resources server config |

Rollout rows should carry **both** `model_checkpoint_ref` (if any) and `agent_module_refs[]` so trainers and module optimizers do not attribute gains incorrectly.

### F. OpenAI Agents SDK — execution and safety primitives

Beyond the high-level table, these are concrete classes worth mapping:

| Primitive | Behavior | Gym module mapping |
|-----------|----------|-------------------|
| **Runner** | Owns turn loop, tool execution, handoffs, max_turns | Agent harness / runner |
| **RunContextWrapper** | Dependency injection + usage aggregation | `AgentContext` passed to modules |
| **AgentHooks / RunHooks** | Lifecycle callbacks (on_tool_end, on_handoff, …) | Module `adapt()` + tracing ([#1867](https://github.com/NVIDIA-NeMo/Gym/issues/1867)) |
| **Guardrails** | Parallel input/output validation; tripwires | **Control module**; distinct from Environment reward |
| **Handoffs** | Delegate to specialist agent with filtered context | **Workflow module** |
| **Sessions** | `previous_response_id` continuity | State module + checkpointer semantics |

**Insight:** Guardrails are **not** the same as Environment verification. They are pre/post conditions on Agent I/O. Gym should not conflate verifier reward with guardrail tripwires.

---

### G. Refit map: today's Gym → target module types

| Today in Gym | Framework analog | Target `AgentModule` type | Notes |
|--------------|------------------|---------------------------|-------|
| `prompt.py` / `prompt_config` | DSPy Predictor instruction + Signature | `working_memory` | Use module `name`, demos, optimizable instruction hash |
| `skills.py` / `skills.path` | Claude Skills / MAF Skills / Letta skills | `skill_library` | Already has `skills_ref` hash ([#1256](https://github.com/NVIDIA-NeMo/Gym/issues/1256)) |
| `responses_create_params` | DSPy Signature I/O + tool schema | `request_template` (optional) | Separate task input from Agent knobs |
| `simple_agent.responses()` | ReAct harness | *harness*, not a module | Default tool loop |
| `langgraph_agent/*` | LangGraph StateGraph patterns | `workflow` + `state` modules | Reflection = critic role; ReWOO = planner |
| In-agent playbook (PR #1706) | ACE playbook | `long_term_memory` | Delta updates, bullet IDs, lifecycle |
| Haystack / vector memory (PR #1706) | Letta archival / LangGraph Store | `memory` | Retrieval policy + module ref |
| `model_server` ref | DSPy LM / ADK model | Agent config | Trainable |
| Direct NIM reflector calls | DSPy reflection LM / ACE Reflector | Agent/module config | Config-driven, not hardcoded |
| `max_steps` | ReAct step budget | `control.budget` | Part of harness config |
| Tool routing in harness | ReAct Action + MCP | `tool_policy` | Allowed tools, forced tool choice |
| `aggregate_metrics` | LangSmith eval | Eval / rollout collection | Not Agent module |
| Rollout JSONL | LangSmith traces | Rollout collection output contract | Must include module refs |

---

### H. Module config catalog

Discriminated `AgentModuleConfig` types (extensible):

```yaml
- type: working_memory
- type: skill_library
- type: long_term_memory
- type: planning
- type: reasoning
- type: control
```

---

### I. Implications for `AgentModule` interface (revised)

The minimal interface should align with patterns above:

```python
class AgentModule:
    name: str
    type: str  # from catalog in §H

    async def activate(self, ctx: AgentContext) -> AgentContext:
        """Inject context, stage files, set tool policy, configure runtime inputs."""
        return ctx

    async def adapt(self, event: TrajectoryEvent) -> list[AgentUpdateEvent]:
        """Post-step or post-rollout: ACE curator, GEPA feedback buffer, memory write."""
        return []

    def module_refs(self) -> list[AgentModuleRef]:
        """Provenance stamp for this rollout."""
        return []
```

**Extensions by module class (optional hooks, not all modules implement):**

| Hook | Used by |
|------|---------|
| `activate(ctx)` | working_memory, skill_library, long_term_memory, planning, control |
| `adapt(event)` | long_term_memory/playbook (ACE), working_memory (GEPA buffer), long_term_memory (memory writes) |
| `on_tool_result(event)` | guardrails, tracing |
| `merge_delta(update: AgentUpdateEvent)` | playbook curator (deterministic, may be non-LLM) |
| `materialize()` | skills (`stage_skills`), playbook export |

**TrajectoryEvent kinds** should cover ReAct steps and ACE/GEPA needs:

```text
model_call | tool_call | tool_result | env_step | verify | terminated | truncated | guardrail_trip | custom
```

This subsumes the earlier `TrajectoryStep` sketch and aligns with [#1867](https://github.com/NVIDIA-NeMo/Gym/issues/1867).

---

### J. Research deliverable (suggested follow-up doc)

Before implementation, produce a short **Agent Framework Abstraction Catalog** (`docs/agent-framework-abstractions.md` or issue appendix) with:

1. Layer model (§ above)
2. Per-framework primitive tables (DSPy, LangGraph, OpenAI Agents, ACE, Claude Code paper)
3. Gym refit map (§G)
4. Module config catalog (§H)
5. Explicit non-goals (Gym is not a skill runtime, not a LangGraph port, not a DSPy clone)

This issue's implementation phases should reference that catalog as the naming authority.

---

## Proposed taxonomy: module types

See **§H** for the type catalog and **§ Concrete abstractions to refit Gym toward** for framework mappings.

| `type` | Role (summary) | Optimizable? |
|--------|------------------|--------------|
| `working_memory` | Active context for this rollout: prompt-backed instructions, scratchpad, compacted context | Yes (GEPA) |
| `skill_library` | Skill packages / reusable operational procedures | Yes (skill eval) |
| `long_term_memory` | Playbooks, persistent recall, archival stores, summaries | Yes (ACE, memory tuning) |
| `planning` | Plan templates, decomposition, workflows | Sometimes |
| `reasoning` | Deliberation strategies, reflection / critique loops | Sometimes |
| `control` | Tool policy, guardrails, budgets, permissions | Sometimes |

This list is intentionally open; more module types may be added over time.

`skills.path` is **run-level** (rollout collection stamps `skills_ref`); the Agent applies skills during `/run`.

Every module should answer:

1. Does it affect model-visible context?
2. Does it affect available actions/tools?
3. Does it affect control flow?
4. Does it persist state?
5. Can it update from trajectories?
6. Can it be optimized independently of policy weights?
7. How is its version recorded in rollout provenance?
8. What is its reset boundary (per rollout, per run, persistent)?

---

## Core proposal

### Agent = policy + Agent modules + harness

```text
Rollout collection — starts `/run` requests, records results
Agent              — policy loop + modules + harness
Environment        — task state, tools/steps, verify, reward, metrics
Model Server       — inference backend for policy (and optionally auxiliary roles)
```

Distinguish three optimization modes:

| Mode | Target | Evidence |
|------|--------|----------|
| **Policy optimization** | LLM weights | trajectories with token IDs ([#1867](https://github.com/NVIDIA-NeMo/Gym/issues/1867), [#1715](https://github.com/NVIDIA-NeMo/Gym/pull/1715)) |
| **Module optimization** | prompts, skills, playbooks, memory, routing | trajectories + reward + `AgentModuleRef` / update events |
| **Environment evaluation** | task difficulty, verifier | reward, metrics; modules held constant |

### AgentModule interface (design probe — not final API)

See **§I** for extended hooks (`on_tool_result`, `merge_delta`, `TrajectoryEvent` kinds) and alignment with DSPy/ACE/Letta patterns.

Chosen to cover the minimal lifecycle: **shape context → act → adapt from evidence → expose provenance**.

```python
class AgentModule:
    name: str
    type: str

    async def activate(self, context: AgentContext) -> AgentContext:
        """Before Agent acts: inject prompt, stage skills, attach memory, set tool policy, etc."""
        return context

    async def adapt(self, event: TrajectoryEvent) -> list[AgentUpdateEvent]:
        """After evidence exists: GEPA reflection, ACE curation, memory writes, etc."""
        return []

    def module_refs(self) -> list[AgentModuleRef]:
        """Provenance for this rollout."""
        return []
```

**Why `activate` not `apply_to_request`:** modules may do more than edit `responses_create_params` — stage filesystem skills, configure MCP, set permissions, or apply tool policy.

**Why `adapt` returns update events:** optimizers need structured before/after module transitions, not silent in-place mutation.

**Why `module_refs`:** generalizes `skills_ref` to prompts, playbooks, memory snapshots, tool policies, adapters.

### AgentModuleRef

```json
{
  "type": "working_memory | skill_library | long_term_memory | planning | reasoning | control | …",
  "name": "answer_format_prompt",
  "hash": "sha256-prefix",
  "path": "optional path",
  "uri": "optional://artifact/location"
}
```

`skills_ref` remains as a backwards-compatible companion for `skill_library` modules with `path`, `hash`, `skills[]`.

### AgentUpdateEvent

```json
{
  "module": {"type": "working_memory", "name": "answer_format_prompt"},
  "update_type": "replace | append | prune | reweight | checkpoint",
  "before_ref": {"hash": "..."},
  "after_ref": {"hash": "..."},
  "reason": "optional rationale for optimizers (e.g. GEPA reflection)",
  "trigger": {"task_index": 0, "rollout_index": 0, "step_id": "..."}
}
```

### TrajectoryEvent

Build on [#1867](https://github.com/NVIDIA-NeMo/Gym/issues/1867). Modules need:

- model input/output (or deltas)
- tool calls + observations + timing
- environment step/verify results
- reward, usage, errors
- active `module_refs` at step time

---

## Mapping today's Gym pieces

### `prompt.py` → `working_memory` module

Run-level `prompt_config` on `gym eval run` materializes `responses_create_params.input` from row fields. Mutually exclusive with pre-populated `input` in dataset rows.

```yaml
modules:
  - type: working_memory
    name: answer_format_prompt
    config:
      path: benchmarks/prompts/eval/aai/mcq-4choices.yaml
    lifecycle: static
```

Preserve: prompt sweeps without re-preparing datasets (`gym dataset render` for training materialization).

### `skills.py` → `skill_library` module

[#1256](https://github.com/NVIDIA-NeMo/Gym/issues/1256): Gym is **not** the skill runtime. Gym stages skills and records `skills_ref` (path + content hash + metadata). Agent native discovery (Claude Code, Codex, Read/Glob tools) loads them.

```yaml
modules:
  - type: skill_library
    name: baseline_skills
    config:
      path: skills/variant_a/
      discovery:
        mode: native_agent_runtime   # stage_skills() for Claude Code; etc.
    lifecycle: static              # or mutable for optimizers
```

### Dataset rows vs run-level knobs

**Dataset row (JSONL)** — task + routing + env verifier payload:

| Field | Role |
|-------|------|
| `responses_create_params` | model request body (often partial; prompt may build `input`) |
| `agent_ref` | which Agent runtime instance (`name` = top-level config key) |
| env fields (`expected_answer`, `options`, `verifier_metadata`, …) | Environment seed/verify |
| `_ng_task_index`, `_ng_rollout_index` | injected by rollout collection |
| `skills_ref` | injected when `skills.path` set |

**Not in dataset rows (by design):** `prompt_config`, `skills.path`, `responses_create_params` CLI overrides — run-level knobs for A/B without duplicating datasets.

**Config `datasets:` entry** — metadata (name, type, jsonl_fpath, source, license), not the row schema itself.

### Config: runtime instance name vs inner template key

```yaml
gpqa_mcqa_simple_agent:          # runtime instance name (routing, agent_ref, NEMO_GYM_CONFIG_PATH)
  responses_api_agents:
    simple_agent:                # inner template key (often matches folder; can differ)
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: gpqa_mcqa_resources_server
```

Outer name = process identity. Inner name = config template slot. Often equal by convention; GPQA shows when they diverge.

### Rollout collection semantics (definition)

**Run-level** = selected by `gym eval run` and handled by the existing rollout collection driver, not by dataset rows:

- load/merge config, iterate dataset, repeats, concurrency
- apply run-level overrides such as `prompt_config`, `skills.path`, and response-create-param overrides
- stamp indices and `skills_ref`
- call the Agent server's `/run`
- persist JSONL, failure sidecars, resume data, and aggregate metrics

Agent modules do not require a new execution path. The intended phase-1 flow is the current flow with module hooks inside the Agent server's `/run`:

```text
RolloutCollectionHelper
  -> Agent /run(row)
       -> Environment.seed_session(row)
       -> AgentModule.activate(row)          # working_memory, skill_library, …
       -> Agent.responses(...) or harness loop
       -> Environment.verify(row + response)
       -> AgentModule.adapt(trajectory_event)   # optional online updates
       -> return verify response + module_refs + update_events
  -> persist rollout JSONL and metrics
```

`prompt_config` and `skills.path` can be migrated gradually by normalizing them into Agent module config before `/run`, but `/run` remains the stable executable surface.

---

## Optimizer fit

### GEPA / DSPy ([#1551](https://github.com/NVIDIA-NeMo/Gym/pull/1551))

- **Target:** `working_memory` module
- **Today:** external script treats `/run` as black-box prompt→reward
- **Target:** GEPA adapts from `TrajectoryEvent`s with verifier feedback; updates prompt-backed working memory; rollouts carry a `working_memory` module ref
- **Gap:** held-out eval splits, rich failure feedback, don't swallow 500s as reward=0

### ACE / TALES ([#1706](https://github.com/NVIDIA-NeMo/Gym/pull/1706))

- **Target:** `long_term_memory` modules
- **Today:** in-process playbook; synthetic final response; direct NIM calls
- **Target:** reasoner via Model Server; full trajectory in rollout; playbook lifecycle explicit (per-episode vs cross-episode documented)
- **Gap:** `AcePlaybookModule` scaffold exists in design notes; needs real update algorithm + reset policy

### Skill optimization ([#1256](https://github.com/NVIDIA-NeMo/Gym/issues/1256))

- **Static:** compare `skills.path` variants; diff by `skills_ref.hash`
- **Adaptive:** optimizer mutates skill dirs in place; hash distinguishes versions at same path
- **Levels:** content quality, selection quality, library composition, generalization (per issue #1256 table)

### Policy LLM training

- Consume standardized trajectories ([#1867](https://github.com/NVIDIA-NeMo/Gym/issues/1867))
- Include active `module_refs` in training rows for attribution
- Do not conflate module artifact changes with policy checkpoint versions

---

## Config sketch

```yaml
gpqa_mcqa_simple_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: gpqa_mcqa_resources_server
      model_server:
        type: responses_api_models
        name: policy_model
      modules:
        - type: working_memory
          name: answer_format_prompt
          config:
            path: benchmarks/prompts/eval/aai/mcq-4choices.yaml
          lifecycle: static

        - type: skill_library
          name: skills
          config:
            path: skills/gpqa_variant_a
          lifecycle: static
```

Run-level overrides remain for backwards compatibility:

```bash
gym eval run \
  +agent_name=gpqa_mcqa_simple_agent \
  +prompt_config=benchmarks/prompts/eval/aai/mcq-4choices.yaml \
  +skills.path=skills/variant_a/
```

CLI normalizes these into module configs internally.

---

## Design questions

1. Should modules be declared only on Agent config, or also as top-level run overrides compiled into modules?
2. Should `skills_ref` become a strict subset of `agent_module_refs[]`, or stay parallel during migration?
3. Where do mutable module updates run — inside Agent `/run`, via `/adapt`, or only in external optimizer drivers?
4. Concurrency: cross-episode playbook persistence + parallel rollouts — what locking/snapshot semantics?
5. Native harness agents (Claude SDK, OpenAI Agents, LangGraph): how do they export module refs without Gym owning their runtime?
---

## Suggested implementation plan

### Phase 1 — Provenance (low risk)

- [ ] Define `AgentModuleRef` pydantic model
- [ ] Add `agent_module_refs` to rollout output schema
- [ ] Map existing `skills_ref` → module ref
- [ ] When `prompt_config` used, emit a `working_memory` module ref (hash of template file)
- [ ] Document in rollout JSONL schema

### Phase 2 — Module config normalization

- [ ] `AgentModuleConfig` discriminated union: `working_memory`, `skill_library`, `planning`, `reasoning`, `control`
- [ ] Compile `prompt_config` + `skills.path` CLI knobs into modules
- [ ] Agent config optional `modules:` list
- [ ] Backwards compatible: no modules = current behavior

### Phase 3 — Agent hooks

- [ ] `AgentModule.activate` / `adapt` / `module_refs` on Agent base or mixin
- [ ] `simple_agent`: apply working_memory module before model loop; optional adapt on verify
- [ ] `stage_skills` integration for skill-capable agents

### Phase 4 — Mutable modules + lifecycle

- [ ] `AgentUpdateEvent` schema
- [ ] Lifecycle: `per_rollout | per_run | persistent | external`
- [ ] Concurrency rules for persistent modules
- [ ] ACE / GEPA modules as reference implementations

## Out of scope

- Rewriting all Agents to modules in one pass
- Moving seed/verify or `/run` responsibilities out of Agent servers
- Adopting OpenAI / Claude / MAF / ADK APIs as Gym's native runtime
- Gym as universal skill runtime ([#1256](https://github.com/NVIDIA-NeMo/Gym/issues/1256) explicitly out of scope)
- Full trajectory schema (owned by [#1867](https://github.com/NVIDIA-NeMo/Gym/issues/1867))
- Policy weight training implementation
- `ng_eval_skills` multi-variant orchestration CLI (compare runs manually first)

---

## Acceptance criteria

- [ ] Documented Agent module taxonomy and framework terminology map (this issue + `fern/` or `docs/` follow-up)
- [ ] `AgentModuleRef` on rollout rows; `skills_ref` mappable to it
- [ ] `prompt_config` rollouts emit working_memory module provenance
- [ ] Agent config supports `modules:` for at least `working_memory` and `skill_library`
- [ ] Clear written boundary: `/v1/responses` = Agent behavior; `/run` = existing Agent-server rollout surface with module hooks
- [ ] Design supports static module A/B (skills, prompts) and mutable optimization (GEPA, ACE) with update events
- [ ] Policy training rows can include active module refs without conflating module versions with model checkpoints
- [ ] Works for `simple_agent` first; guidance for native-harness agents documented

---

## References

- [#1256 — agent skill evaluation infrastructure](https://github.com/NVIDIA-NeMo/Gym/issues/1256)
- [#1867 — standardized trajectories format](https://github.com/NVIDIA-NeMo/Gym/issues/1867)
- [#1551 — DSPy GEPA recipe](https://github.com/NVIDIA-NeMo/Gym/pull/1551)
- [#1706 — ACE + Haystack TALES agent](https://github.com/NVIDIA-NeMo/Gym/pull/1706)
- [OpenAI Agents SDK](https://github.com/openai/openai-agents-python/)
- [Claude Agent SDK overview](https://code.claude.com/docs/en/agent-sdk/overview)
- [Microsoft Agent Framework](https://github.com/microsoft/agent-framework)
- [LangSmith for LangGraph](https://info.langchain.com/langsmith/for-langgraph)
- [Google ADK — Gemini models](https://adk.dev/agents/models/google-gemini/)
- [Dive into Claude Code (arXiv:2604.14228)](https://arxiv.org/abs/2604.14228)
- Gym code: `nemo_gym/prompt.py`, `nemo_gym/skills.py`, `nemo_gym/base_responses_api_agent.py`, `responses_api_agents/simple_agent/app.py`, `nemo_gym/rollout_collection.py`
