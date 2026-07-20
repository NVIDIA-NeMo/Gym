## Summary

Converge on **system-architecture contracts** for NeMo Gym so environments match how RL systems are modeled, and **benchmarks and agent servers compose by addition** — not via per-pair wrappers like `anyswe`.

Three roles on three orthogonal axes:

- **Environment** — MDP authority (dynamics contract, action/observation semantics, reward mechanism). Usually realized by one **resources server** (`seed_session → SessionDescriptor`, `verify`, optional MCP tools / `step`, per-episode state), but can be **formed from several** — e.g. a task/verify RS plus a sandbox broker RS, or a facade RS that delegates. **Fixed across all tasks in a benchmark family.**
- **Task** — one **problem instance** sampled from a task distribution: initial observation (prompt), privileged grading metadata, and world-init hints for that episode. **Varies per dataset row.** In RL terms: \(\tau \sim p(\tau)\), then run one episode in environment \(\mathcal{E}\). SWE-bench Verified is one Environment (`swe_bench`) and ~500 Tasks (`django__django-13741`, …).
- **Agent Server** — hosts an **Agent**: policy model, tools, and orchestration (planning, value/reward models, nested control, blackbox CLIs, …). Implemented as a **`responses_api_agents/` server** (e.g. `claude_code_agent`, `simple_agent`). **Independent of which Environment** it is wired to in YAML. Distinct server type from Environment and Sandbox — not a resources server.
- **Sandbox** — substrate and isolation. The `nemo_gym/sandbox/` provider seam today; optionally a **sandbox resources server (broker)** that holds live handles and hands out serializable references.

**Environment ≠ Task.** Conflating them leads to per-instance resources servers, grading code inside agent servers (#1738), or implicit task blobs with no stable identity on the wire. **Task** should be a first-class value (typed, identified, passed through `seed_session` / `verify`); **SessionDescriptor** is the Environment's response to accepting a task — episode context, not the task itself.

**Environment ≠ Benchmark.** An **Environment** is executable software (resources server + verifier + interaction contract) — you can train or dev on it. A **Benchmark** is a **repeatable eval product** built on top: fixed task split, metric, comparison protocol, and baselines (leaderboard claims). Not every Environment is a benchmark; one Environment can support many benchmarks. Gym's `verified: true` on a resources server marks a **benchmark-grade** Environment configuration (baselined, protocol locked) — not the Environment concept itself.

The motivating mismatch: PR #1738 parks SWE **grading and world spec inside the agent server** (`responses_api_agents/swe_env/`, inline `verify_task`, no `resources_servers/swe_bench/`). The target restores the Environment as a first-class resources server while keeping **blackbox agent servers like `claude_code_agent` where they belong** — connected to the Environment through an **interface-agnostic `seed_session` descriptor**, not an `anyswe`-style wrapper.

## Motivation / context

- Parent: #1249 (decouple SWE infra from agent harnesses).
- PR #1738 converges `anyswe` onto SWE grading code but **grades inline in the agent server**, **deletes `resources_servers/swe_env/`**, and keeps that code under `responses_api_agents/swe_env/` instead of the Environment RS. That dissolves the Environment server and the `verified:` marker.
- Complementary (out of scope here): ansubramania's *Blackbox agent integration* design doc — training/RL-data half (token-id capture, trajectory stitching, capture store).

## Key design decisions

- **Task is a first-class RL object, not “just a dataset row.”** A **Task** is everything needed to instantiate one episode: public initial observation, privileged verifier metadata, and world-init fields. A **dataset** is a batch of tasks; an **episode / rollout** is one agent attempt at one task; a **task instance** (Gym glossary) is one of \(K\) stochastic attempts at the same task (pass@\(k\), GRPO groups). The platform should carry stable task identity (`task_id` / `instance_id`, benchmark key) on every rollout — not only implicit fields inside JSONL.
- **Environment defines rules; Task defines the problem.** Same Environment serves all tasks in a benchmark family. Do not create one resources server per task instance.
- **Benchmark fixes the measurement; Environment provides the engine.** A benchmark selects *which* tasks (dataset + split), *which* metric (e.g. `% resolved`, pass@1), and *which* comparison protocol (harness + model held constant). The Environment implements how episodes run and score. Example: `swe_bench` = Environment; *SWE-bench Verified* (500-instance test split, resolve rate) = benchmark built on it.
- **Not every Environment is a benchmark.** Training environments, harness-dev sandboxes, and curated eval subsets may share the same RS without a published leaderboard protocol. Conversely, multiple named benchmarks (Verified, Lite, Multilingual) can share one Environment by swapping **tasks** (dataset) and harness keys — no new RS per publication.
- **`verified:` marks benchmark readiness, not Environment existence.** `verified: false` = Environment works, not yet baselined; `verified: true` = gold-patch baseline + reviewed protocol suitable for comparable eval claims.
- **Agent servers stay under `responses_api_agents/`.** Blackbox CLIs (Claude Code, Codex, …) are hosted by agent servers (e.g. `claude_code_agent`). They connect to an Environment via **`seed_session` / `verify`**, not by being re-hosted inside a generic run-in-box wrapper agent.
- **`seed_session` returns a SessionDescriptor** — placement topology, optional `SandboxSpec` / box reference, optional MCP connection info, optional model egress hints. The agent server binds to the Environment from this descriptor; it does not hardcode docker, OpenSandbox, or SWE-bench wiring.
- **Only data crosses HTTP.** Sandbox **spec** and (when brokered) a **box reference/token** — never a live handle (`SandboxHandle.raw`).
- **Environment = semantics + substrate (when the task has a world).** The resources server owns reward and task definition; the sandbox realizes state and transitions (repo/filesystem for SWE). Reward stays in `verify`; the box carries no reward authority.
- **Hermetic grading (the twin).** `verify()` always grades in its **own fresh box**, independent of where the agent server ran the episode.
- **Composition, not a cross-product.** A run names agent server × environment × sandbox backend × topology; combinations add. Drop the `anyswe` wrapper-per-(agent × benchmark) pattern.

### Sandboxing topologies

| Topology | What's isolated | Typical attachment | Example |
|---|---|---|---|
| **A — None** | nothing | agent server on host; terminal `verify` | math, MCQ |
| **B — Env-sandboxed** | env world/state | agent server outside; **MCP tools** into env box | MCP weather, env-owned DB |
| **C — Agent-in-env** | world + agent execution, one box | agent server **in-box** per descriptor | SWE-bench + Claude Code |
| **D — Whole interaction** | full episode boundary | outer orchestrator | untrusted composite agent |

Reference pair: **`claude_code_agent` + `swe_bench`** — Environment returns topology **C** + per-instance `SandboxSpec`; agent server runs Claude Code inside the box, then POSTs to `/verify`. Same agent server + **`example_mcp_weather`** demonstrates topology **B** (MCP from descriptor).

## Tracked work items

- [ ] **First-class Task contract** — document and stabilize the Task value object (identity, public vs privileged fields, benchmark key). Dataset JSONL maps to Task; Environment methods take Task (or equivalent wire shape), not ad-hoc merged metadata. SWE prototype: `SweTask` in `resources_servers/swe_bench/harness.py`; generalize cross-benchmark naming in a follow-up.
- [ ] **SessionDescriptor contract** — extend `seed_session` response: `placement.topology`, optional `sandbox.{spec, ref}`, optional `mcp`, optional `egress`. Document agent-server placement bindings (host / MCP / in-box). Descriptor = Environment's answer to “task accepted; here is episode context,” distinct from the Task payload.
- [ ] **`resources_servers/swe_bench`** — Environment RS: `build_spec` from task row, `seed_session`, `verify` (fresh eval box). Recover `verified:` marker and `/verify` endpoint. Baseline on gold patches → `verified: true` to graduate **Environment → benchmark-grade eval** for SWE-bench Verified.
- [ ] **Relocate SWE domain code to Environment RS** — move harnesses, parsing, `verify_task`, etc. out of `responses_api_agents/swe_env/` into **`resources_servers/swe_bench/`** (private modules: `harnesses/`, `parsing/`, `verify_task.py`, …). No top-level `nemo_gym/swe/` unless a second RS or non-HTTP consumer appears later.
- [ ] **In-box binding in agent server** — topology C in `claude_code_agent` first: acquire box via `nemo_gym/sandbox`, apply descriptor egress, exec agent, pass harvest to `/verify`. Agent-specific (CLI vs loopback HTTP differs per agent). Extract shared helpers into `nemo_gym/sandbox/` only if a second agent duplicates the same pattern — no separate `nemo_gym/placement/` package for the prototype.
- [ ] **Sandbox broker RS (optional)** — `resources_servers/sandbox/` (or equivalent): hold handles, expose acquire/exec/upload/download/release by reference; backends = local docker/apptainer + remote OpenSandbox/cloud. Client-side `RemoteSandboxProvider` implementing `SandboxProvider`.
- [ ] **Configuration & placement** — document WHO/WHAT/WHERE: env supplies world spec; agent server declares intrinsic defaults; driver sets deployment policy (backend, topology override, limits). Precedence: run config > agent defaults; world spec = env ⊕ agent default.
- [ ] **Topology D / nesting** — nested-container capability as broker/provider property validated against spec.
- [ ] **Publish design note in `fern/`** after team review.

## Open questions

- **Cross-benchmark `Task` schema** — one portable Gym type vs per-Environment task structs (`SweTask`, …) with a shared minimal base (id, benchmark, initial_observation, verifier_metadata)?
- **Benchmark packaging** — is a benchmark always (Environment RS + frozen JSONL + eval config + metric), or a first-class registry beyond `verified:`?
- **`step()` vs terminal-only?** First-class `step()` for env-driven benchmarks, or terminal `verify` + side channels? Blackbox in-box agent servers force terminal grading; native agent servers may use `step`.
- **Box lifecycle owner for topology C with a broker.** Environment-owned state-box + agent server drives by reference (RL-canonical) vs agent-owned box (today's #1738, no broker). Prefer env-owned when broker exists; agent-owned as no-broker fallback.
- **Sandbox broker failure domain** — orphan reaping, TTL keyed by rollout id, reconnect if consumer dies mid-rollout.
- **Two-box cost** — working + eval boxes at scale; pool/image cache on broker.
- **One rollout identity** — session cookie, MCP token, box ref, env state, reaper TTL as one first-class object.
- **`anyswe` fate** — stepping-stone only; dissolve into `swe_bench` RS + descriptor-driven `claude_code_agent` (and other agent servers), not the long-term home.

## Out of scope

Training / RL-data half: model-call recording under rollout join key, dialect conversion, token-id/logprob capture, capture store, trajectory stitching — see *Blackbox agent integration*. Plugs in after these boundaries exist.

## Related

#1249 (parent), #1738 (convergence PR), #1677 / #1572 (swe_env), #1682 (MCP RS), #1377 (sandbox provider factory), #1707 (provider config decoupling).
