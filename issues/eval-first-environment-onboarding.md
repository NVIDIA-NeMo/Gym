<!-- Proposed issue title: [RFC] Eval-first environment onboarding: a quality bar for `verified` environments -->

## Summary

NeMo Gym's environment catalog is growing fast (104 resources servers, with accelerating community contributions — e.g. TALES [#1556](https://github.com/NVIDIA-NeMo/Gym/pull/1556), PinchBench [#1810](https://github.com/NVIDIA-NeMo/Gym/pull/1810)), and environments feed directly into training. This RFC proposes an **eval-first onboarding bar**: before an environment is used for training, its contributors provide a **representative eval task set** — with difficulty/edge-case coverage and baseline evidence — that shows how target models actually perform in that domain.

Today the repo has the *hooks* for this but not the *bar*: every new server gets `verified: false` (pre-commit), CI validates data pre-merge ([#2023](https://github.com/NVIDIA-NeMo/Gym/pull/2023)) and requires 5 example rows ([#2025](https://github.com/NVIDIA-NeMo/Gym/pull/2025)) — but what it takes to flip `verified: true`, and what that flag gates, is undocumented and unenforced.

This RFC proposes (1) an eval dataset contract, (2) a quality-bar checklist, (3) graded verification levels, (4) a split between automated validation and expert review, and (5) a standard Environment Report artifact. It is deliberately a **requirements-gathering RFC**: the Open Questions section is the point — please react there.

Prior art: OpenEnv RFC-008 ([huggingface/openenv#778](https://github.com/huggingface/openenv/issues/778)) proposes an automated quality bar for environment submissions (learnability, reward integrity, observability, reproducibility, resources, security). We borrow its strongest ideas and adapt them to Gym's post-merge, training-oriented context. A pilot onboarding exercise on `resources_servers/tales` informed the concrete criteria below.

## Terminology

| Term | Use for |
|------|---------|
| **Eval task set** | A curated `TaskSet(role: eval)` (vocabulary per the environment-unification RFC, `issues/unify-environment-abstraction.md`) — not the 5-row `example.jsonl` wiring check |
| **Quality bar** | The checklist an eval task set + its evidence must satisfy |
| **Verification level** | Graded status (L0–L3) an environment holds; proposed successor/refinement of the boolean `verified` flag |
| **Harness parity** | Evidence that the Gym integration reproduces the canonical benchmark's published numbers under the paper's protocol |
| **Reward integrity** | Floor/ceiling evidence: degenerate policies score at floor, reference solutions score at max |
| **Resolution** | The minimum score difference the eval can detect given task count × repeats × measured variance |
| **Baseline** | A measured result for a **(model, harness/prompt scaffold, protocol)** triple — never a model alone |

## Background

### Why this matters now

| Pressure | Consequence without a bar |
|----------|---------------------------|
| Community environments merge with wiring-level checks only | `verified: false` accumulates indefinitely; nobody knows what "done" means |
| Environments feed RL training | Broken/gameable verifiers produce reward-hacked checkpoints; discovering this *during* a training run is the most expensive possible time |
| "How does Nemotron perform in domain X?" is the recurring question | Without a metric contract and pinned harness, per-environment numbers are incomparable and unanswerable in aggregate |
| Expert review is the scarcest resource | Unstructured review of 104+ environments doesn't scale; experts burn time on checks a script could run |

### What exists today

- `verified: false` auto-added to new server configs by the `add-verified-flag` pre-commit hook; `verified: true` informally means "baselined and reviewed" — criteria undocumented.
- Pre-merge CI: data validation for changed servers ([#2023](https://github.com/NVIDIA-NeMo/Gym/pull/2023)), 5 example rows ([#2025](https://github.com/NVIDIA-NeMo/Gym/pull/2025)). These check *wiring*, not *eval quality*.
- `gym eval run` / `gym eval profile` + W&B integration produce exactly the artifacts a baseline needs — but nothing requires or standardizes them per environment.

### What the TALES pilot taught us

A pilot onboarding of `resources_servers/tales` (5 text-adventure frameworks, canonical paper arXiv:2504.14128) surfaced failure modes that generalize:

1. **Metric comparability is the first bug you hit.** Per-family reward scales differed by orders of magnitude (0–1 vs. cumulative scores in the hundreds); a pooled raw-reward mean was meaningless. Environments need an explicit metric contract (normalized per-task, macro-averaged per family).
2. **Split enforcement is often the dataset author's job.** Upstream train/test split machinery silently no-ops for some task families; only row-level task selection enforces it. This must be audited per environment, not assumed.
3. **Paper parity catches integration bugs.** Reproducing published numbers through the Gym harness found a wrong metric definition and a task-population mismatch (27 vs. 54 tasks) that code review missed.
4. **The easy case is unrepresentative.** TALES has a paper, ground-truth walkthroughs, upstream splits, and programmatic rewards. Most community environments will have none of these — the playbook needs paths for the messy cases (see Open Questions).

## Problem

1. **`verified` is boolean and undefined.** There is no documented set of criteria, evidence, or sign-off that flips it — so it can neither be requested from contributors nor audited later.
2. **Example rows ≠ eval set.** The 5-row requirement checks that the server runs; it says nothing about capability coverage, difficulty distribution, contamination, or whether the tasks discriminate between models.
3. **Reward/verifier integrity is unchecked.** Nothing requires evidence that a no-op or degenerate policy scores at floor, that a reference solution scores at max, or that stronger models outscore weaker ones. These are the cheapest, highest-yield checks (per RFC-008's reward-integrity panel) and they catch exactly the bugs that poison training.
4. **Baselines are not comparable or durable.** Scores depend on the prompt scaffold and agent loop, which are not pinned; results are not tied to (dataset hash × server version × upstream pin × protocol), so any of these changing silently invalidates them.
5. **Verification rots.** Even a well-verified environment drifts: upstream deps bump, server fixes land, model panels change. Without re-validation triggers, `verified: true` decays into noise.
6. **Expert review is unbounded and unmeasured.** The goal is to *reduce* expert burden with playbooks/tooling — but nothing measures expert-minutes-per-environment or time-to-verified, so we can't tell whether tooling helps.

## Proposal (v0 — for discussion)

### P1. Eval dataset contract

Each onboarded environment ships, under `resources_servers/<server>/data/`:

| Tier | Size | Purpose |
|------|------|---------|
| `smoke.jsonl` | ~5 rows | wiring check; runs in CI (exists today as example rows) |
| `eval_v0.jsonl` | ~50–100 tasks × ≥4 repeats | the curated headline eval: capability coverage + difficulty gradient + discriminative middle |
| `eval_full.jsonl` | full clean inventory | re-selection pool and periodic deep runs |
| `parity.jsonl` (optional) | protocol-exact | harness-parity vs. published numbers, when a canonical benchmark exists |

Contract requirements:

- **Generated, not hand-written**: a deterministic `scripts/generate_eval_datasets.py` emits the JSONLs from upstream task inventories (split-aware index computation included). Hand-curated task ID lists are a defect.
- **Slicing metadata on every row**: task family, capability tag, difficulty tier, split provenance, contamination flag — so `gym eval profile` output pivots without external joins.
- **Everything pinned per row**: seed, step cap, system prompt; sampling params declared in the protocol. Nothing left to runner defaults.
- **Metric contract in the README**: headline = normalized score + success rate, per-family plus equal-weight macro-average. Never a pooled raw mean across incomparable reward scales. Secondary diagnostics: truncation rate, invalid-action rate, steps-to-completion, tokens/episode.

### P2. Quality-bar checklist

An eval task set meets the bar when it has:

**Coverage & difficulty**
- Named capability axes for the domain, each with rows (task count ≠ coverage).
- Difficulty gradient with headroom at both ends: saturated anchors (regression detectors), a discriminative middle, an explicit hard tail. "A stronger model passes it" is what separates *hard* from *broken*.

**Discrimination**
- k≥4 repeats per task; tasks classified saturated / discriminative / floored for the target model; `eval_v0` weighted toward discriminative tasks, with anchors kept at the extremes.

**Contamination & reproducibility**
- Upstream train/test splits enforced at the row level where the server doesn't enforce them; contaminated content (e.g. tasks with published solutions in pretraining corpora) is a *reported slice*, never silently pooled.
- Upstream dependency SHA-pinned; dataset regenerable bit-for-bit from the generator.

**Reward integrity** (adapted from OpenEnv RFC-008; zero or near-zero LLM cost)
- **Floor**: null/random/degenerate policies score at floor across the task set.
- **Ceiling**: a reference solution (walkthrough, oracle, gold trajectory) replayed through the *Gym harness* scores at max — this validates the entire reward pipeline with no model in the loop.
- **Monotonicity**: across a small model panel of clearly different strength, stronger models score higher; inversions are treated as verifier bugs until explained.
- **No leakage**: observations don't contain the answer; oracle/diagnostic knobs (e.g. admissible-commands exposure) are declared diagnostic-only.

**Statistical resolution**
- The report states the eval's minimum detectable difference from measured variance ("resolves ~±X points at n×k"). Without this, no one can tell whether a model delta is signal.

**Harness parity** (when a canonical benchmark exists)
- Gym-measured numbers for a paper-evaluated model, under the paper's protocol, within a stated tolerance justified by the paper's own variance. Out-of-tolerance cells are integration bugs until proven otherwise.

### P3. Verification levels

Replace (or refine) the boolean `verified` with graded levels:

| Level | Meaning | Evidence | Gate for |
|-------|---------|----------|----------|
| **L0 merged** | wiring works | example rows + CI data validation (today's pre-merge bar) | inclusion in the repo |
| **L1 validated** | eval set meets the deterministic bar | dataset contract + automated checks pass (see P4) | listing as an eval benchmark |
| **L2 baselined** | measured evidence exists | baseline runs (W&B-tracked), reward-integrity results, resolution statement | eval results quoted in reports |
| **L3 verified** | expert signed off | review package + domain-expert sign-off | **training use** |

Boolean back-compat: `verified: true` ⇔ L3. Pre-merge bar stays at L0 — the contribution barrier does not rise; the *training* bar becomes explicit.

### P4. Enforcement split: automated vs. judged

The core cost-reduction move: **everything deterministic moves out of expert review into code.**

- **`gym env validate <server>` (new)**: row schema, per-row metadata completeness, generator reproducibility, seed determinism, split-index correctness, reward-range sanity, null-policy floor, oracle-ceiling replay (when a reference solution exists), truncation-dominance check. CI-runnable; extends the existing pre-merge data validation.
- **Agent/skill tooling (maintainer-side)**: the judgment prep — upstream audit, paper protocol extraction, difficulty tiering, discriminativeness analysis, review-package curation. Reduces expert work; doesn't replace it.
- **Expert review (human)**: scoped to what only a domain expert can judge — does the reward reflect genuine task progress, are failures attributable to the model rather than the harness, is the task set representative of the domain. Anomaly-triggered (strongest-model-floors, out-of-tolerance parity cells, reward pathologies) plus a small stratified trajectory sample, instead of exhaustive review.

### P5. Environment Report

Each onboarded environment gets a standard, **machine-readable** report (e.g. `report.yaml` + a rendered docs page): task inventory, metric contract, quality-bar checklist results, baseline runs table (model × dataset × repeats × metrics × W&B link), reward-integrity results, resolution statement, verification level, sign-off record. Machine-readability is what makes "how does Nemotron perform across domains" answerable as a dashboard query instead of a doc-reading exercise.

## Open Questions — requirements we need to gather

This is the section to react to. Each question names its stakeholders.

**Q1 — Who is the playbook's user?** (env contributors? Gym maintainers? an internal onboarding team with agent tooling?) The answer changes the primary artifact: a public contributor spec + self-serve `gym env validate`, vs. an internal expert-accelerator. Community contributors don't have internal endpoints, W&B projects, or agent harnesses. *Stakeholders: maintainers, community contributors.*

**Q2 — Where is the bar enforced?** Pre-merge (raises contribution barrier, keeps catalog clean — OpenEnv's model) vs. post-merge levels (low barrier, explicit training gate — the P3 proposal). Is L0-at-merge / L3-for-training the right stance? *Stakeholders: maintainers, training teams.*

**Q3 — What exactly does training use gate on?** L3 strictly? L2 with maintainer approval? Who is allowed to consume an L1 environment for experimental RL, and how is that tracked? *Stakeholders: training teams.*

**Q4 — What replaces harness parity when there's no paper?** Most community environments have no canonical benchmark. Candidate substitutes: oracle-ceiling replay + null-policy floor + cross-model monotonicity as the integrity evidence, plus contributor-attested domain representativeness. Is that sufficient for L2? *Stakeholders: maintainers, domain experts.*

**Q5 — What's the bar for LLM-judged verifiers?** For rubric/LLM-judge rewards, dataset composition matters less than verifier validity: human-agreement rate on a labeled sample? adversarial gaming probes? judge-model pinning? This RFC's checklist is programmatic-reward-shaped; the LLM-judge path needs its own criteria. *Stakeholders: verifier authors, training teams.*

**Q6 — What statistical resolution do we require?** What effect size must an eval detect (e.g. a 3-point Nemotron improvement)? That determines minimum task counts × repeats, and therefore cost. Do we mandate a resolution statement only, or a minimum resolution? *Stakeholders: eval consumers, training teams.*

**Q7 — What is the canonical baseline model panel?** Which models (and which Nemotron checkpoints/endpoints) constitute the small/medium/frontier panel for monotonicity and baselines, so reports are comparable across environments? Who maintains that list as models rotate? *Stakeholders: maintainers, model teams.*

**Q8 — What triggers re-validation?** Upstream pin bumps, server code changes, dataset regeneration, panel rotation? Does a level *decay* (L3 → L2) on drift, and is that automated (CI detects the hash change) or manual? *Stakeholders: maintainers.*

**Q9 — Data governance for community eval sets?** Licensing/redistribution rights for task content, PII, copyrighted material (e.g. commercial game text). Minimum: a provenance/licensing declaration in the contract. Enough? *Stakeholders: maintainers, legal.*

**Q10 — How do we measure expert burden?** The stated goal is reducing it. Candidate metrics: expert-minutes per environment, time-to-L3, anomaly-triggered review count vs. total tasks. What do we instrument from environment #2 onward? *Stakeholders: domain experts, maintainers.*

**Q11 — How does this land relative to the environment-unification RFC?** The natural home for the dataset contract is `TaskSet(role: eval)` in the unified `env.yaml` manifest. Do we spec against that target now, or against today's `resources_servers/<server>/data/` layout with a migration note? *Stakeholders: maintainers.*

## Non-goals

- **Sandboxing/security redesign** — RFC-008 has a security pillar; here we assume Gym's existing review + isolation story and only note the boundary.
- **Training-algorithm specifics** — this RFC gates *whether* an environment is training-ready, not how it's trained on.
- **Replacing human review** — the explicit goal is targeted, cheap expert review, not zero expert review.
- **Retroactive mass re-verification** — the 104 existing servers migrate opportunistically (when touched, or when a training team wants them), not in one campaign.

## Rollout sketch

1. **Phase 0 (this RFC)**: converge on Q1–Q11; lock the contract and levels.
2. **Phase 1**: publish the contributor-facing playbook (docs + PR/issue template), implement `gym env validate` on top of the existing data-validation CI.
3. **Phase 2**: pilot on two environments — one friendly (canonical paper + programmatic reward, e.g. TALES) and one messy (no paper and/or LLM-judge verifier) — and instrument expert burden (Q10).
4. **Phase 3**: wire verification levels into config/registry, add drift-triggered re-validation, stand up the cross-environment report dashboard.
