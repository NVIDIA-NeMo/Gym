---
name: env-onboarding-team
description: >
  Agent-team-based, eval-first onboarding of any merged NeMo Gym environment
  (community-contributed or internal). Spawns specialized agents (environment
  code auditor, paper-baseline researcher, eval dataset designer, baseline runner,
  multi-model reward profiler, discriminativeness analyst, expert-review curator,
  devil's advocate) that
  coordinate via the shared Task list and direct messaging. The team produces the
  onboarding package: task inventory, integration-gap report, tiered eval datasets
  (smoke / eval_v0 / eval_full), a paper-baseline reproduction experiment plan,
  baseline artifacts with W&B-tracked runs (a runs table linking every experiment
  to its wandb URL), and a targeted expert-review package — the evidence needed
  to flip `verified: false → true`. Triggered by: "team onboard environment",
  "eval-first onboarding", "onboard <server> with agents", "build eval set for
  <server> with the team".
argument-hint: "<resources-server-name> [target-model-endpoint]"
allowed-tools:
  - AskUserQuestion
  - Bash
  - Read
  - Write
  - Edit
  - Glob
  - Grep
  - Agent
  - TeamCreate
  - TeamDelete
  - SendMessage
  - TaskCreate
  - TaskList
  - TaskGet
  - TaskUpdate
  - TaskStop
  - WebFetch
  - WebSearch
---

# Eval-First Environment Onboarding — Agent Team

Onboard a NeMo Gym environment to the eval-first quality bar using a coordinated
team of specialized agents.

**Principle**: contributors provide representative eval task sets *before* the
environment is used for training. An environment is "onboarded" when it ships a
curated eval dataset with difficulty/edge-case coverage, a defensible metric
contract, frozen baseline artifacts, and evidence that the Gym integration
faithfully reproduces the canonical benchmark. Manual expert review is still
required — this skill's job is to make that review *targeted and cheap*, not to
replace it.

> For reviewing a PR that adds an environment, use `review-pr-team`. This skill
> is for onboarding an environment already merged (or checked out locally): it
> produces datasets, experiments, and artifacts, not review comments.

## Coordination model (this harness)

### Side-panel layout (preferred when available)

If `TeamCreate` appears in your available tools, use it — each teammate opens in
its own tmux pane (leader left, agents right). Requirements are already in this
repo's `.claude/settings.json` (`agentTeams: true`, `teammateMode: "tmux"`).

- **Spawn** each agent with `TeamCreate` (one call per agent), full prompt as
  `initialMessage`.
- **Communicate** via `SendMessage` (by name or ID). Findings arrive as
  `SendMessage` from teammate to leader.
- **Tear down** with `TeamDelete` after collation (Phase 6).

### Background-task fallback (when TeamCreate is unavailable)

Fall back to the `Agent` tool: spawn each wave's agents in a single message with
multiple `Agent` calls (`run_in_background: true`, `subagent_type:
"general-purpose"`, unique names). Continue/question a spawned agent with
`SendMessage` — its context is preserved. If a completion notification arrives
without the findings content, `SendMessage` the agent asking it to resend.

### Common to both modes

- Maintain the shared task list with `TaskCreate`/`TaskUpdate`/`TaskList` so the
  user can follow progress. The leader owns orchestration; agents do NOT poll.
- **Sandbox**: pass `dangerouslyDisableSandbox: true` on Bash calls that need
  network (`git clone`, `curl`, `uv sync`, model endpoints) or where bubblewrap
  fails (`bwrap: loopback: ...`).

## Epistemic provenance rule (applies to EVERY agent and the final report)

Every claim in agent findings and the final onboarding report MUST carry one of
four provenance tags. This prevents design placeholders from being read as
measurements:

| Tag | Meaning | Allowed source |
|-----|---------|----------------|
| `MEASURED` | Number produced by a run in this engagement | rollout/profile artifacts on disk, cite the file |
| `STRUCTURAL` | Fact read from code | `file:line` in this repo or the pinned upstream dep |
| `PRIOR` | Claim from a paper/external source | a reference **fetched and verified this session** (arXiv API, repo README at a pinned ref) — NEVER from model memory |
| `PLACEHOLDER` | A role in the design, no data behind it | e.g. "the target model" before any sweep has run |

Hard rules: no statement about how a specific model (Nemotron, a frontier model,
anything) performs on this environment without `MEASURED` or `PRIOR` backing.
"Task X is easy/hard for the target model" is `PLACEHOLDER` until the baseline
sweep runs. Verify arXiv IDs via
`curl -sL "https://export.arxiv.org/api/query?search_query=..."` (https — http
301s to an empty body); never cite paper IDs or numbers from memory alone.

---

## Phase 0: Parse & Validate

Extract `$SERVER` (resources server name) and optional `$TARGET_ENDPOINT` from
`$ARGUMENTS`. Validate:

```bash
ls resources_servers/$SERVER/  # must contain app.py + configs/
```

If missing, list `resources_servers/` and ask the user which to onboard.

Determine `$MODE`:
- **full** — a usable model endpoint exists (`$TARGET_ENDPOINT` given, or
  `env.yaml`/config points at a reachable endpoint, or `gym env status` shows a
  live model server). Baseline runs happen in this engagement.
- **design-only** — no endpoint. The team produces datasets + experiment plans
  with exact commands; runs happen later. All performance-dependent selections
  are marked provisional.

Also record `$GPU_AVAILABLE` (`nvidia-smi` check) — some model servers need it.

Record `$WANDB_AVAILABLE`: W&B tracking is built into Gym's global config — when
top-level `wandb_project`, `wandb_name`, and `wandb_api_key` are set (env.yaml or
`++wandb_*=` overrides), every `gym eval run` invocation inits a W&B run
automatically (`nemo_gym/global_config.py`, `WANDBConfig`) and logs aggregate/key
metrics, rollout tables, and profiling histograms. Check whether `wandb_api_key`
is configured (do NOT print it). In full mode with no key, ask the user whether
to proceed untracked or provide one — **onboarding runs should be W&B-tracked**:
the run links are the durable, shareable evidence attached to the environment.

---

## Phase 1: Context

1. Read `resources_servers/$SERVER/`: `app.py`, `configs/*.yaml`, `README.md`,
   `requirements.txt`, `data/*` (note any existing `example*.jsonl` and
   `*_aggregate_metrics.json` — these model the artifact pattern to extend).
2. Read `CLAUDE.md` (root) and Glob `.claude/skills/*/SKILL.md`; read at least
   `nemo-gym-reward-profiling` (profiling workflow this skill builds on),
   `nemo-gym-pivot-datasets` (discriminativeness/mixed-reward selection logic),
   and `add-benchmark` if present. Do not hardcode other skill names.
3. Identify the **upstream dependency**: pinned git ref / package in
   `requirements.txt`. The auditor will clone it — the pin is what makes the
   audit reproducible.
4. Identify the **model panel surface**: the repo-root `env.yaml` holds the
   policy endpoint secrets (`policy_base_url`, `policy_api_key`,
   `policy_model_name`, usually with commented-out alternate models) consumed
   by the model server configs via `${policy_*}` interpolation (e.g.
   `responses_api_models/openai_model/configs/openai_model.yaml`). NEVER print
   or copy secret values — reference them only as `${policy_api_key}`-style
   interpolations or `++key=<from env.yaml>` placeholders. **RECOMMENDED
   endpoints: NVIDIA inference endpoints** — build.nvidia.com
   (`https://integrate.api.nvidia.com/v1`; one `nvapi-` key serves the whole
   catalog: Nemotron, gpt-oss, deepseek, qwen, ...) or the internal NVIDIA
   inference hub — because switching panel members is then a single
   `++policy_model_name=<model>` override per invocation against one key. The
   model server reads `policy_*` at startup, so each panel member is its own
   server start + run cycle. List candidate panel members (configured +
   commented alternates in env.yaml) as `$MODEL_PANEL` and confirm the
   selection with the user in the Phase 1 briefing.
5. Post a short plain-language briefing to the user before spawning agents:
   what the environment is, what it wraps, dataset row schema, `$MODE`,
   `$MODEL_PANEL`, and the planned waves.
6. Create the shared task list (Phase 2 tasks) with `TaskCreate`.

---

## Phase 2: Tasks & Waves

**Wave 1 (parallel, no blockers):**

| Task | Owner | Description |
|------|-------|-------------|
| `audit-environment` | `env-auditor` | Code-level task inventory + integration-gap checklist against server AND pinned upstream. |
| `research-paper-baselines` | `paper-researcher` | Locate + verify canonical paper/repo; extract protocol, metric definitions, baseline tables; draft the reproduction experiment matrix. |

**Wave 2 (blocked by BOTH Wave 1 tasks):**

| Task | Owner | Description |
|------|-------|-------------|
| `design-eval-datasets` | `dataset-designer` | Tiered datasets (smoke/eval_v0/eval_full) + metric contract + row schema, built on the audit and the paper protocol. |
| `curate-review-package` | `review-curator` | Expert review rubric + trajectory sampling plan + anomaly triggers (can start from the audit; folds in run artifacts later if full mode). |

**Wave 3 (full mode only; blocked by `design-eval-datasets`):**

| Task | Owner | Description |
|------|-------|-------------|
| `run-baselines` | `baseline-runner` | Smoke run, then candidate sweep (`num_repeats >= 4`) + `gym eval profile`; then the feasible slice of the paper-reproduction matrix. |
| `profile-rewards` | `reward-profiler` | (blocked by `run-baselines`) Reward profiling across a multi-model panel ($MODEL_PANEL); validates the reward signal itself. |
| `analyze-discriminativeness` | `discriminativeness-analyst` | (blocked by `run-baselines` AND `profile-rewards`) Classify tasks saturated/discriminative/floored; finalize eval_v0 membership; compare repro results to paper numbers. |

In design-only mode, skip Wave 3; `dataset-designer` marks eval_v0 membership
`PROVISIONAL` and the deliverables include the exact commands for the deferred
sweep AND the deferred multi-model reward-profiling panel (model list, per-model
override cycle, wandb naming).

**Wave 4 (blocked by all prior waves):**

| Task | Owner | Description |
|------|-------|-------------|
| `challenge-package` | `devil-advocate` | Stress-test the whole onboarding package. Waits for the leader to push consolidated findings via SendMessage. |

**Wave 5 (leader):** `collate-onboarding` — merge, apply verdicts, preview to
user, write deliverables on approval.

Spawn Wave 1 immediately; spawn later waves as their blockers complete (push the
predecessor findings into each new agent's prompt — agents do not poll).

---

## Agent Prompt Specifications

### Common preamble (include in EVERY agent prompt)

```
You are a member of the eval-first onboarding team for the NeMo Gym environment
`resources_servers/$SERVER` (repo: NVIDIA-NeMo/Gym conventions apply; read
CLAUDE.md first).

Mode: $MODE (full = baseline runs happen now; design-only = plans + commands only).
Environment briefing: $BRIEFING
Upstream dependency: $UPSTREAM (pinned ref from requirements.txt)

Instructions:
- Claim your task with TaskUpdate(status="in_progress"); mark completed when done.
- PROVENANCE RULE (mandatory): tag every claim MEASURED / STRUCTURAL / PRIOR /
  PLACEHOLDER. STRUCTURAL needs file:line. PRIOR needs a reference you fetched
  and verified THIS session (arxiv API over https, or repo file at a pinned ref)
  — never from memory. No model-performance claims without MEASURED or PRIOR.
- Prefer Read/Glob/Grep for local lookups. Use Bash with
  dangerouslyDisableSandbox: true when you need network.
- Report ALL findings — never cap or summarize away items.
- When done: TaskUpdate(completed) and SendMessage your findings to the leader.
  Your final message IS your findings (raw structured data, not prose for a
  human).
- Answer questions from other agents via SendMessage.
```

### `env-auditor` (Wave 1)

```
You are the environment code auditor. The README is a claim; the code is the
fact. Audit BOTH the Gym server (resources_servers/$SERVER/) and the pinned
upstream dependency (shallow-clone it at the pinned ref into /tmp, inspect,
delete afterwards).

Deliverable 1 — TASK INVENTORY (STRUCTURAL, exhaustive):
- Enumerate every task/game/variation the integration can address: counts per
  framework/category, the exact selector fields in a dataset row (e.g.
  framework/task_no/split/seed), and how each selector maps to upstream content.
- Task-index stability: is task_no derived from a sorted list, a dict/JSON
  ordering, or a glob? Note anything that could silently reorder between
  upstream versions (a dict-ordered registry means task_no is only stable while
  the pin holds — say so).
- Built-in difficulty gradients (difficulty params, lengths, seen/unseen splits)
  and what each concretely changes.

Deliverable 2 — INTEGRATION-GAP CHECKLIST. Check each item and report
pass/fail/N-A with file:line evidence:
1. SPLIT MECHANICS: does the split/train-test selector actually change behavior
   for EVERY task family? Hunt for silent no-ops — e.g.
   `getattr(module, "train_environments", None) or module.environments`
   falls back to the full list when the attribute doesn't exist upstream. For
   each family state where the split is enforced: server, upstream, or ONLY the
   dataset (row-level task_no/seed choice). Dataset-enforced splits must be
   called out — they are the eval author's responsibility.
2. REWARD COMPARABILITY: raw vs normalized; per-family reward ranges (0/1?
   0-100? hundreds with large negative death penalties?); cumulative-score
   deltas vs per-step rewards; is a pooled mean across rows meaningful, or must
   the metric be normalized per task (score/max_score) and macro-averaged per
   family? Does the env already expose max_score/won in info?
3. EPISODE/STEP CAPS: default max steps vs what tasks plausibly need vs the
   canonical paper's protocol. Flag defaults that make truncation dominate.
4. SEED SEMANTICS: does seed select CONTENT (game file, variation, split
   membership) or only stochasticity? Per family.
5. DIFFICULTY/OBSERVABILITY KNOBS: hints, admissible/valid-action exposure,
   oracle info in observations. Which setting is the honest eval; which is a
   diagnostic variant.
6. REPRODUCIBILITY: pinned refs, runtime downloads (what changes if the URL
   content changes?), nondeterminism sources.
7. GYM CONVENTIONS: async /run, state isolation per session, graceful error
   handling, aiohttp-not-httpx (per CLAUDE.md) — flag violations for follow-up
   but do not fix.

Deliverable 3 — recommended small server fixes (do NOT implement): e.g. surface
normalized_score/won in step info, enforce or loudly document split handling.
```

### `paper-researcher` (Wave 1)

```
You are the paper-baseline researcher. The onboarding package must include
experiments that reproduce the baseline performance reported in the
environment's canonical paper — reproducing published numbers through the Gym
integration is the harness-parity test: a discrepancy is an integration bug
until proven otherwise.

Tasks:
1. LOCATE + VERIFY the canonical reference(s): the paper for the suite and, if
   relevant, papers for constituent benchmarks. Resolve via the arXiv API
   (https) and the upstream repo README at the pinned ref. NEVER cite an ID,
   table, or number from memory — fetch it. If a paper cannot be fetched,
   report that explicitly rather than substituting recollection.
2. EXTRACT THE PROTOCOL (all PRIOR, with source links): step caps, observation
   settings (e.g. admissible commands on/off), prompts/agent scaffold, models
   evaluated, temperature/sampling, number of episodes/seeds, and the exact
   metric definitions (normalized score, success rate, aggregation method).
3. EXTRACT BASELINE NUMBERS: per-family (and per-task where published) results
   for each model the paper evaluated. Build a machine-readable table.
4. DRAFT THE REPRODUCTION MATRIX: which paper model(s) are reachable from our
   endpoints (or nearest available proxy — mark proxies clearly), which settings
   must match the paper exactly, expected values, and an acceptance tolerance
   justified from the paper's own reported variance (fallback: a stated
   tolerance like ±5 normalized points, labeled as our choice). Include the
   exact `gym env start` / `gym eval run` / `gym eval profile` commands and the
   config overrides needed to match the protocol.
5. GAP LIST: paper protocol features the Gym integration cannot currently match
   (send to env-auditor via SendMessage to cross-check), so the repro plan says
   "matches paper except X" honestly.
```

### `dataset-designer` (Wave 2 — receives both Wave 1 reports in its prompt)

```
You are the eval dataset designer. Build the tiered eval task sets from the
auditor's inventory and the paper protocol. Design against the QUALITY BAR:

1. COVERAGE OF CAPABILITY AXES: name the distinct skills the domain exercises
   and ensure each has rows. Task count is not coverage.
2. DIFFICULTY GRADIENT WITH HEADROOM AT BOTH ENDS: rows the target model should
   pass reliably (regression detectors), rows it should fail but a stronger
   reference passes (progress measures — "stronger passes" separates hard from
   broken), and an explicit hard tail (longevity). Use the environment's
   built-in gradients (difficulty levels, sequence lengths, seen/unseen)
   deliberately. Until the sweep runs, all per-model assignments are
   PLACEHOLDER; tier by structure and PRIOR paper results, and say so.
3. DISCRIMINATIVE UNDER REPEATS: the sweep (k>=4 rollouts/task) will classify
   tasks saturated/discriminative/floored; design v0 to be re-selectable from
   eval_full without changing schema. Keep anchor tasks at the extremes; weight
   toward mixed-pass-rate tasks (same principle as pivot selection in
   nemo-gym-pivot-datasets).
4. CONTAMINATION HYGIENE: enforce upstream train/test splits AT THE ROW LEVEL
   wherever the auditor found split handling is dataset-enforced (compute the
   correct task_no/seed sets programmatically, never by hand). Flag
   internet-contaminated content (e.g. classic games with published
   walkthroughs) as its own reported slice, never silently pooled.
5. REPRODUCIBILITY: every row pins seed, step cap, system prompt; declare
   temperature in the protocol. Nothing left to runner defaults.
6. METRIC CONTRACT: headline = normalized score + success rate, reported
   per-family plus an equal-weight macro-average across families (never a
   pooled raw-reward mean across incomparable reward scales). Secondary
   diagnostics: truncation rate, invalid-action rate, steps-to-completion,
   tokens/episode.
7. SLICING METADATA on every row: family, capability tag, difficulty tier,
   split provenance, contamination flag — so gym eval profile output pivots
   without external joins.
8. SIZING: eval_v0 ~50-100 tasks x 4 repeats is the quality bar; justify your
   number against cost. No silent caps — document every exclusion.

Deliverables:
- Tier definitions: smoke (~5 rows, wiring only, may reuse existing example
  data), eval_v0 (curated), eval_full (complete inventory usable for
  re-selection and periodic deep runs).
- A GENERATOR SCRIPT design (or the script itself if the leader asks): emits
  the JSONLs from the upstream task lists programmatically, including the
  split-aware index computation. Hand-written task_no lists are a defect.
- The row schema (Responses-API format per CLAUDE.md: responses_create_params
  + selector fields + agent_ref + metadata) with one fully worked example row.
- The metric contract as a README-ready section.
```

### `review-curator` (Wave 2)

```
You are the expert-review curator. Manual review by domain experts is required
before an environment is trusted; your job is to make that review targeted and
cheap instead of exhaustive.

Deliverables:
1. REVIEW RUBRIC (short, concrete questions): does the reward reflect genuine
   task progress? is failure attributable to the model rather than the harness
   (parsing, truncation, env error)? is the observation text intact and
   readable? does the trajectory show the environment state actually changing
   in response to actions? is the task winnable under the configured step cap?
2. SAMPLING PLAN: a stratified sample (~10 trajectories: per family, mix of
   pass/fail, include the extremes) for expert eyes, with instructions for
   viewing them in the rollout viewer (tools/rollout_viewer) if present in the
   repo — verify it exists before citing it.
3. ANOMALY TRIGGERS — every one of these gets expert review, not sampling:
   large-magnitude negative rewards; tasks where the STRONGEST reference model
   floors (broken-vs-hard adjudication); high invalid-action-rate episodes
   (count parser rejections in observations); zero-variance-at-zero task groups;
   repro-matrix cells outside tolerance.
4. In full mode (after run-baselines), fill the plan with actual rollout files
   and anomaly instances (MEASURED). In design-only mode ship the plan with
   placeholders and the commands that will populate it.
```

### `baseline-runner` (Wave 3, full mode only)

```
You are the baseline runner. Produce the MEASURED artifacts.

Workflow (per nemo-gym-reward-profiling):
1. SMOKE: gym env start for $SERVER + the configured model server; run the
   smoke tier; confirm rewards/terminations look sane before spending on the
   sweep. Set RAY_TMPDIR=/tmp on long paths.
2. SWEEP: gym eval run --no-serve over eval_full (or the widest affordable
   slice — document any truncation loudly) with num_repeats>=4 for the target
   model, and at least 1-2 repeats for one stronger reference model if an
   endpoint is available. Then gym eval profile.
3. REPRO MATRIX: run the feasible cells of the paper-reproduction matrix with
   the paper's exact protocol settings.
4. Artifacts under resources_servers/$SERVER/data/ (or a results/ dir the
   leader specifies): rollouts, materialized inputs, aggregate metrics,
   profiling output. Report file paths + row counts + any failed/partial runs
   honestly (use ++allow_partial_rollouts=True only when noted in findings).

W&B TRACKING ($WANDB_AVAILABLE=true — otherwise note every run as UNTRACKED):
- One CLI invocation = one W&B run (Gym inits it at config parse when
  wandb_project/wandb_name/wandb_api_key are set). Run each experiment cell as
  its own invocation with `++wandb_project=<project>` and a naming convention
  that makes runs self-describing and greppable:
  `++wandb_name=onboard-$SERVER/<purpose>-<tier>-<model>-r<num_repeats>`
  where <purpose> is smoke | sweep | repro-<cell>.
- Capture the run URL printed at init for EVERY run. A MEASURED claim cites BOTH
  the local artifact path AND the wandb run URL.
- Massive-rollout caveat: `upload_rollouts_to_wandb` defaults to true; set
  `++upload_rollouts_to_wandb=False` only when uploads are prohibitive, and say
  so in findings (metrics still log; only the rollout table is skipped).
- Deliver a RUNS TABLE: one row per run — purpose, dataset tier, model,
  num_repeats, key metrics (per-family normalized score / macro-average),
  local artifact path, wandb URL, tracked/untracked.

Never edit datasets or configs to make numbers look better; report mismatches.
```

### `reward-profiler` (Wave 3, full mode only, blocked by run-baselines)

```
You are the multi-model reward profiler. Your lens is the REWARD SIGNAL itself,
not any single model's score: a trustworthy environment must produce rewards
that discriminate between models of different strength and use their range
meaningfully. Follow the nemo-gym-reward-profiling skill's workflow.

MODEL PANEL: profile eval_v0 (or the affordable slice — disclose any
truncation) across the panel $MODEL_PANEL — at least 3 models of clearly
different expected strength (small / medium / frontier-class). RECOMMENDED
source: NVIDIA inference endpoints (build.nvidia.com /
integrate.api.nvidia.com, or the internal inference hub) — one key serves the
catalog, so each panel member is just a ++policy_model_name override. Reuse
the baseline-runner's target-model artifacts as one panel member; never re-run
what already exists.

MECHANICS:
- Endpoint credentials come from the repo-root env.yaml (policy_base_url /
  policy_api_key / policy_model_name and its commented alternates). NEVER
  print, log, or copy secret values — reference them as ${policy_api_key}
  interpolations or "<from env.yaml>" placeholders in anything you write.
- One panel member = one cycle: start the model server with
  ++policy_model_name=<model> (plus base_url/api_key interpolation changes if
  switching provider), gym eval run --no-serve with num_repeats >= 4, then
  gym eval profile. RAY_TMPDIR=/tmp on long paths.
- W&B-track every run using the runner's naming convention with purpose
  `profile-<model>`; capture each run URL.
- Coordinate with baseline-runner via SendMessage so you never start servers
  on the same ports concurrently.

ANALYSIS (all MEASURED; per family AND macro-averaged per the metric contract):
1. STRENGTH MONOTONICITY: do stronger panel members score higher? An inversion
   is a red flag — verifier bug, reward-hacking surface, or a genuinely
   surprising result; report it either way with example trajectories.
2. RANGE USAGE: rewards pinned at 0 or max; degenerate all-equal task groups;
   within-group variance (rollout_infos keyed by _ng_task_index /
   _ng_rollout_index).
3. REWARD PATHOLOGIES: large-magnitude negative outliers; reward earned under
   truncation vs termination; tasks where reward disagrees with success/won
   signals.
4. CROSS-MODEL DISCRIMINATION TABLE: per-task pass rate by panel member — send
   to discriminativeness-analyst (tasks that separate the panel are prime
   eval_v0 candidates) and to review-curator (pathologies become anomaly
   entries for expert review).
Deliver a runs-table fragment (purpose, tier, model, repeats, key metrics,
artifact paths, wandb URL) covering every run you performed.
```

### `discriminativeness-analyst` (Wave 3, blocked by run-baselines AND profile-rewards)

```
You are the discriminativeness analyst. Consume the profiling artifacts
(group_level_metrics / rollout_infos keyed by _ng_task_index and
_ng_rollout_index).

1. Classify every task for the target model: SATURATED (all repeats pass),
   FLOORED (all fail), DISCRIMINATIVE (mixed). All MEASURED.
2. Cross-tab against the designer's difficulty tiers — where structure-based
   tiers disagree with measured pass rates, flag for the review-curator.
3. Finalize eval_v0 membership: keep coverage constraints, keep a few saturated
   anchors (regression detection) and the hard tail, weight the middle toward
   DISCRIMINATIVE tasks. Fold in the reward-profiler's cross-model table —
   tasks that separate the model panel are prime keeps; tasks no panel member
   moves on are floored-or-broken candidates. Document every task dropped from
   v0 and why.
4. REPRO COMPARISON: per-cell table of Gym-measured vs paper-reported numbers
   with the tolerance verdict. Out-of-tolerance cells are integration bugs
   until proven otherwise — send them to env-auditor and review-curator.
5. Report per-family normalized-score summaries + the macro-average, never a
   pooled raw mean.
```

### `devil-advocate` (Wave 4)

```
You are the devil's advocate. Wait for the leader to send the consolidated
package via SendMessage — do NOT poll.

Challenge, with up to 2 rounds per agent via SendMessage:
1. PROVENANCE: every untagged or mis-tagged claim. Any model-performance claim
   without MEASURED/PRIOR backing gets DISPUTED. Any PRIOR without a fetched
   reference gets DISPUTED.
2. SPLIT ENFORCEMENT: independently re-read the code paths the auditor cited.
   If the dataset relies on row-level split enforcement, verify the designer's
   generator logic actually computes the right index sets.
3. METRIC COMPARABILITY: would the headline number change its meaning if the
   family mix changed? Is anything pooled across incomparable reward scales?
4. ROW VALIDITY: do sample rows actually resolve against the server config
   (selector fields, agent_ref wiring, defaults fallback)?
5. REPRO PLAN HONESTY: are proxy models labeled? Is the tolerance justified or
   invented? Are protocol mismatches disclosed?
5a. RUN TRACEABILITY: does every MEASURED number trace to a runs-table row with
   a wandb URL (or explicit UNTRACKED marker)? Spot-check that cited wandb runs
   exist and their logged config matches the claimed protocol (model, repeats,
   step cap) — a link whose config disagrees with the claim gets DISPUTED.
6. REVIEW BURDEN: is the expert package genuinely smaller than reviewing
   everything, and does it still catch the failure modes the rubric names?
Render per-finding verdicts: CONFIRMED / DISPUTED (why) / DOWNGRADED.
```

---

## Phase 3: Collation (leader)

- When each wave completes, push its findings into the next wave's prompts (and
  to `devil-advocate` at the end as one consolidated SendMessage, grouped by
  agent). Broker challenge rounds: nudge a target agent when the devil's
  advocate DMs it and it may be idle.
- Apply verdicts: drop DISPUTED, adjust DOWNGRADED.
- Assemble the **onboarding report** with these sections, every claim carrying
  its provenance tag: (1) environment briefing + task inventory table;
  (2) integration-gap findings + recommended server fixes; (3) eval quality-bar
  assessment + tier definitions + metric contract; (4) paper-baseline
  reproduction plan and (full mode) results table; (4a) the RUNS TABLE — every
  run of this engagement with its wandb URL (or an explicit UNTRACKED marker),
  so each number in the report traces to a link; (4b) reward-validity findings
  from the multi-model panel — strength monotonicity, range usage, pathologies,
  and the cross-model discrimination table; (5) discriminativeness
  classification and eval_v0 membership (or PROVISIONAL marking);
  (6) expert-review package; (7) `verified` flag recommendation — flip to
  `true` only when the evidence in (4)-(6) exists and an expert has signed off.

## Phase 4: Preview & Confirm

Show the user the full report plus the exact file list to be written:

- `resources_servers/$SERVER/data/{smoke,eval_v0,eval_full}.jsonl` (+ generator
  script under `scripts/` in the server dir)
- baseline/repro artifacts (full mode)
- README section (task inventory, metric contract, split-enforcement caveats,
  repro protocol, and a **Baseline runs** table with the wandb links — the
  runs stay associated with the environment in-repo, not only in the report)
- expert-review package (rubric + sample manifest)

Then `AskUserQuestion`: **(1) Write all deliverables**, **(2) Discuss
individually**, **(3) Report only — write nothing**. Never commit; leave
staging/committing to the user (new datasets will also trigger the repo's
pre-commit metadata hooks — mention this).

## Phase 5: Write Deliverables (on approval)

Write approved files. Run the dataset generator and validate: row counts per
tier, JSON-parse every row, spot-check selector resolution against the server
config. Report what was written, with per-tier row counts, and restate the two
or three highest-priority follow-ups (typically: server fixes from the audit,
the deferred sweep in design-only mode, expert sign-off).

## Phase 6: Teardown

`TeamDelete` each teammate (or `SendMessage` shutdown_request + `TaskStop` in
fallback mode). Remove any /tmp upstream clones.
