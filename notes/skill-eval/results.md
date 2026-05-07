# Skill-eval results

Internal note. **Period:** ~2 days · **Owner:** Lawrence Lane (llane@nvidia.com) · **Status:** v8 complete, ongoing.

Companion to [`harness.md`](harness.md) (build + operate the harness).

## TL;DR

Built a three-server NeMo Gym harness to grade `.claude/skills/*/SKILL.md` content via paired rollouts, then used it to audit eight skills and empirically validate three content edits.

The first six iterations were confounded — the "control arm" was reading the skill payload off disk on turn 1 (first SKILL.md, then references after we removed SKILL.md). Only `gym-run` had an uncontaminated measurement by accident, because it's the only skill with no `references/` directory.

After fixing contamination (2×2 over skill-in-prompt × references-on-disk) and surfacing multi-axis deltas (reward / tool calls / tokens), the real picture: **skills differ by pattern, not quality**. Some teach accuracy; some teach efficiency invisibly; some compete with their own references; some are redundant once docs exist.

Three predicted prescriptions, all landed with clean provenance attribution:

| edit | predicted | observed | verdict |
|---|---|---|---|
| `gym-profile`: inline confirming fields into patterns table | flip realistic from −0.107 | **+0.040** (Δ +0.147) | hit |
| `gym-data`: replace ceiling-clipped scenarios adversarially | drop docs-only below 0.95 | 0.97 → **0.88**; realistic +0.093 | hit |
| `gym-review`: shrink SKILL.md 110 → 53 lines | realistic flat, efficiency flat | flat ✓ — but standalone collapsed +0.298 → +0.012 | caveated hit |

The harness isn't really a skill-eval tool. It's a doc-quality instrument: give it a markdown file plus scenarios testing specific JTBDs, and it tells you whether the doc teaches, competes with its own references, is redundant, or is in a measurement dead-zone.

## v8 scoreboard

n=5, 480 rollouts, Opus 4.7 policy + judge via NVIDIA inference API. Bold = effect outside ~±0.10 noise floor at n=5 per-cell.

| skill | skill \| refs=T | skill \| refs=F | refs \| skill=T | refs \| skill=F |
|---|---|---|---|---|
| **gym-run** | **+0.380** | **+0.411** | −0.058 | −0.027 |
| **add-benchmark** | **+0.162** | **+0.328** | +0.083 | **+0.249** |
| **gym-debug** | **+0.133** | **+0.187** | +0.027 | +0.080 |
| gym-data | +0.093 | +0.053 | +0.067 | +0.027 |
| gym-scaffold-agent | +0.053 | **+0.333** | +0.093 | **+0.373** |
| gym-review | +0.048 | +0.012 | **+0.601** | **+0.565** |
| gym-profile | +0.040 | **+0.309** | +0.024 | **+0.293** |
| gym-config | +0.027 | **+0.133** | −0.022 | +0.084 |

The four columns:

- **`skill | refs=T`** (skill+docs − docs-only) — realistic-deployment value of the skill overlay. The number that matters for shipping.
- **`skill | refs=F`** (skill-only − blind) — skill as a standalone doc.
- **`refs | skill=T`** (skill+docs − skill-only) — do refs still matter when the skill is prompted?
- **`refs | skill=F`** (docs-only − blind) — marginal value of references alone.

### Δtools on the realistic contrast

Every skill reduces tool calls when added on top of references, even when the accuracy delta is small:

| skill | Δtools (skill \| refs=T) |
|---|---|
| gym-review | −4.73 |
| gym-debug | −4.13 |
| gym-data | −3.00 |
| gym-profile | −2.00 |
| gym-run | −1.67 |
| gym-config | −1.60 |
| add-benchmark | −1.13 |
| gym-scaffold-agent | −1.00 |

Efficiency is the most robust multi-skill pattern across runs. At deployment scale, the skills save real latency / cost per invocation regardless of whether they materially move accuracy. **Always read both axes** — a skill that doesn't touch reward but cuts 4 tool calls per rollout is real value.

## Per-skill audit

### gym-run — load-bearing keeper

Realistic +0.38, standalone +0.41. The only skill without a `references/` directory, so its v6 "+0.487" wasn't contaminated to begin with — and it survives the noise floor cleanly across v6→v7→v8. sc1 is the strongest single signal in the sprint: blind=0.00, docs=0.00 — the model literally cannot do this task without the skill.

No edits needed. This is the template for what skill-only value looks like when the content is genuine operational procedure.

### add-benchmark — keep; sc3 ceiling-clipped

Realistic +0.16, standalone +0.33, refs-alone +0.25. Skill and references both carry signal; skill adds genuine value on top. sc3 (the httpx-wrapper question) is solved by references alone (docs=1.00, blind=0.03) — clean example of content that migrated entirely into the reference. Keep; consider an sc4 testing JTBDs neither the SKILL.md nor existing scenarios cover.

### gym-config — keep; 2/3 scenarios too easy

Realistic +0.03, standalone +0.13. Driven entirely by sc2 (combined-reward config validation, where the validation checklist forces field-name recall). sc1 and sc3 are ceiling-clipped (docs=1.00). Source: `.claude/skills/gym-config/evals/evals.json` needs harder scenarios — bad interpolation, circular refs, `${oc.env}` misuse, duplicate instance names across composed configs.

### gym-debug — solid; references carry most of the load

Realistic +0.13, standalone +0.19. sc1 is where the skill adds real value (the `check_twice_swap` + positional-bias diagnosis); sc2 / sc3 are ceiling-clipped with refs (the diagnostic-fields catalog handles them directly). Δtools −4.13 is the strongest in the tree — the skill teaches the model to skip exploratory commands.

### gym-review — shrunk; realistic preserved, standalone collapsed

v7 → v8: SKILL.md shrunk from 110 to 53 lines (commit `8fdcdb2c`). Kept how-to-invoke `scripts/review.py`, severity meanings, the five judgment checks the script can't do, cross-references to `references/anti-patterns.md`. Dropped the full BLOCK / WARN tables (redundant with references), the report template, the verbose judgment list.

| measure | v7 | v8 | observation |
|---|---|---|---|
| `skill \| refs=T` (realistic) | +0.029 | +0.048 | preserved |
| Δtools (skill+docs) | −4.87 | −4.73 | preserved |
| `skill \| refs=F` (standalone) | +0.298 | **+0.012** | collapsed |

**Verdict:** context-dependent. If the deployment model assumes readers always have the repo checked out, the shrink is a pure win. If there's a use case where the skill pack is loaded *without* its references, the new SKILL.md is insufficient. Product call, not a measurement call.

#### Before/after — what moved out

*Kept (load-bearing for realistic deployment):* invocation paragraph, severity definitions, judgment-bullet list, cross-refs, an affirmative *"if the script is quiet, the code is clean — say so"* sentence.

*Dropped (moved into the references, which are seeded into the workspace):* full BLOCK rules table, full WARN rules table, verbose per-rule descriptions, review report template.

The shrink works in the realistic cell because `scripts/review.py` emits each finding's rule name inline, and the references already explain each rule. What broke the skill-only cell is that *when references aren't on disk*, neither are the rule explanations — and the SKILL.md no longer carries the tables to fall back on.

### gym-profile — narrate-to-references prescription validated

v7 → v8: rewrote the patterns table to inline confirming field names in prose (commit `3f3a3300`). Promoted `pass_threshold` from a command flag to a named concept subsection.

| measure | v7 | v8 | observation |
|---|---|---|---|
| `skill \| refs=T` (realistic) | **−0.107** | **+0.040** | flipped from net-negative to slightly-positive (Δ +0.147) |
| `skill \| refs=F` (standalone) | +0.278 | +0.309 | small strengthening |
| Per-scenario, sc2 with-skill | 0.72 | 0.96 | +0.24 |
| Per-scenario, sc3 with-skill | 0.84 | 0.96 | +0.12 |

Provenance diff: `md` only (clean single-skill attribution).

#### Before/after — the patterns table

*Before (v7) — two columns, confirming field buried in a code block elsewhere:*

```
| Thinking model scores lower than instruct
  | `reasoning_format_violation_rate` may be high — check if thinking tags
     are being stripped before answer extraction |
```

*After (v8) — three columns, confirming field named in prose:*

```
| Thinking model scores below instruct on the same tasks
  | Think-block stripping failure — `<think>` tokens leak into extraction
  | `reasoning_format_violation_rate` (should be low; if high, read
     `extracted_model_code` for those tasks and look for literal `<think>`) |
```

Same factual content. The v8 version forces the pattern → cause → confirming-field chain to complete in prose, with a closing rule: *"when you cite a pattern, also cite the specific field you checked."*

The `pass_threshold` change has the same shape: in v7 it appeared once as a flag on an `ng_reward_profile` command line; in v8 it has a standalone subsection titled *"pass_threshold — the knob that controls how partial credit counts"* with worked examples. The model now names the knob when reasoning about partial-reward scenarios.

**The general lesson** — Diátaxis-flavored: how-to-mode skills (commands and recipes) fail on assertions that test conceptual noun recall, because the recipe path never reaches the noun in prose. Either rewrite the how-to to narrate to the references at decision points, or expect the skill to be redundant with whatever explanation lives in references/.

### gym-scaffold-agent — content gap remains

Realistic +0.05 (within noise), standalone +0.33. Skill is useful in isolation but doesn't add much on top of references. Sc3 is mildly negative (skill+docs=0.84, docs-only=0.79) — the skill's pattern catalog biases the model toward finding issues even when the code is clean. **Real content gap:** the skill is scoped tightly to RL training agents (cookie propagation, token-ID accumulation, httpx → aiohttp). Non-RL agent patterns (evaluation, orchestration, wrappers) aren't covered. We hit this directly when building `skill_eval_agent` for this sprint and got nothing useful from the skill. Not yet edited.

### gym-data — adversarial scenarios opened headroom

v7 → v8: replaced all three scenarios with adversarial versions (commit `5e84dd08`).

| measure | v7 | v8 | observation |
|---|---|---|---|
| `skill \| refs=T` (realistic) | +0.013 | **+0.093** | now measurable |
| `docs-only` cell | 0.97 | 0.88 | dropped below 0.95 — measurement dead-zone gone |
| `blind` cell | 0.96 | 0.85 | new scenarios are harder but model priors are still strong |

Provenance diff: `evals+fx` only.

#### Before/after — scenario prompts

*sc1 — format validation becomes schema audit:*

> **Before:** *"Validate the tool-calling dataset at evals/files/sample_tool_calling.jsonl for a search+calculation benchmark."* — well-formed fixture, correct answer is "yes, valid." Trivial.
>
> **After:** *"Audit the tool-calling dataset at evals/files/sample_tool_schema_bugs.jsonl. For each entry that would fail OpenAI's function-tool schema or is internally inconsistent, identify the entry and state exactly what's wrong."* — fixture has four planted bugs (required-field not in properties, `parallel_tool_calls` vs `expected_tool_calls` mismatch, missing `function` wrapping, single-tool with `parallel_tool_calls=true`) plus one clean line.

*sc2 — format check becomes semantic audit:*

> **Before:** *"Check evals/files/sample_bad_data.jsonl for data quality issues before I upload it."* — three format-level issues (missing `verifier_metadata`, missing `input`, literal answer leakage). Obvious.
>
> **After:** *"Audit the benchmark at evals/files/sample_mislabeled_benchmark.jsonl. Some `expected_answer` values are wrong for the user's question. Identify every entry where the gold label is semantically incorrect — the schema is fine."* — seven entries; three have wrong gold labels (capital of Australia "Sydney" → Canberra; leap year 365 → 366; gold symbol "Go" → Au). Domain judgment, not schema validation.

*sc3 — copy-the-schema becomes extend-the-schema:*

> **Before:** *"Generate 5 example entries for a SQL benchmark based on the schema in evals/files/sample_sql_benchmark.jsonl."* — fill-in-the-blank.
>
> **After:** *"Generate 3 entries for a multi-turn customer-support benchmark with `expected_tool_sequence`, `forbidden_sequence`, and `partial_credit`, exercising three branching scenarios: (a) email-only refund, (b) abuse attempt, (c) ambiguous-amount partial refund."* — requires reading the schema, applying constraints, and coherently extending it.

**Lesson** — the "before" versions were solvable by pattern-matching the fixture; the "after" versions require applying real judgment. The same shape applies to any ceiling-clipped scenario: move from validating *format* to testing *understanding*.

## NeMo Gym sharp edges

Concrete, reproducible, cheap-to-fix. The most useful output of the dogfood for the framework team:

1. `ng_run` runs `python app.py`, not `python -m`. Relative imports (`from .schemas import X`) break; must use absolute imports from project root.
2. Trailing slashes in `policy_base_url` produce double-slash 404s on some providers.
3. Some `/v1/responses` providers return `object: "chat.completion"`; `NeMoGymResponse` validation fails on the literal. Normalize in the model server (`responses_api_models/openai_model/app.py`).
4. OpenAI's `FunctionToolParam` requires explicit `strict: False` or validation fails at the model server.
5. Host NeMo Gym `.venv/bin` leaks into subprocess PATH via inherited env; rollouts can see `ng_*` binaries, Ray sockets, HF/MLflow credentials. We strip in `_build_sandbox_env`.
6. Sandbox PATH strip removes the `python` alias on macOS — rollouts silently failed with empty output until we added a workspace-local `python → python3` symlink.
7. Rollout JSONL only persists the *final turn's* token usage. Multi-turn tool loops lose intermediate-turn tokens.
8. Workspace cleanup must be inside a `finally` block — tool failures, judge failures, cancellation all leak tmpdirs otherwise.

Methodology gotchas (likely generalize past skill eval):

9. **Every artifact seeded into a workspace contaminates the control arm.** SKILL.md peek 100% pre-fix; references peek 100% pre-second-fix. Apply the same gating discipline anywhere the harness ships material adjacent to the thing being measured.
10. **Content-hash provenance is only as clean as its boundary.** Hashes stamped at JSONL-build time can lie if server code is edited between build and `ng_collect_rollouts`. Want rollout-time re-hashing eventually.
11. **"Same-sha" is necessary but not sufficient.** Model and judge endpoints' temperature-0 stochasticity moves deltas by up to 0.20 per cell on bit-identical runs. Provenance is a cache-hit check, not a p-value.

## Upstream candidates

Two with existing demand. Two parked.

**Worth absorbing:**

1. **Assertion-grade LLM-as-judge as a shared base class.** Three existing in-tree implementations: `skill_judge`, `code_gen`, `equivalence_llm_judge`. Common shape is `(response, tool_calls, assertions[]) → grades[]`. A shared base class saves the next team re-inventing it.
2. **Per-turn token aggregation in the agent base class.** Today only the final response's `usage` reaches the output JSONL. Every multi-turn agent eventually wants this.

**Parked:**

3. *Paired-arm pattern as a framework primitive.* Real idea, but N=1 customer (us). Wait for a second team to want it.
4. *Framework-level provenance stamping.* Same — useful in theory, unclear demand. The narrow version (a `ng_stamp_provenance` helper that hashes Hydra-config-referenced files into a sidecar) is small and PR-able if anyone wants it.

## Open methodology questions

- **Judge drift at temperature=0 is the dominant noise source.** n=5 gives detectable effects ~±0.10 per cell / ~±0.05 skill-level. A calibration run at n=20 on one cell would pin down the actual detectable effect size.
- **The "skill-only" cell is a useful diagnostic but a weird deployment target.** `gym-review` showed that optimizing for the realistic cell can be invisible on the standalone cell. Be clear which cell is the target before shipping a SKILL.md change.
- **Doc-eval as a category.** This methodology generalizes to any markdown-plus-scenarios evaluation. We've only tested it on SKILL.md files; testing on a doc page would validate that the methodology isn't skill-specific.

## Retraction log (claims that didn't survive)

- ~~"Every skill reduces tool calls by 0.8–4.8 per rollout."~~ Replaced by the v8 `skill | refs=T` contrast: every skill reduces tool calls, magnitudes −1.00 to −4.73.
- ~~"gym-profile is actively misleading."~~ Replaced by "competes with its own references when both are present" → retired in v8 after the patterns-table rewrite flipped realistic to slightly-positive.
- ~~"gym-review's SKILL.md is dead weight."~~ Nuanced in v8: dead weight for realistic deployment, load-bearing for skill-only deployment.
- *Shape-probe null result* (bullet vs checkbox vs heading): withdrawn pending rerun on clean harness. v7 suggests effect sizes are below the n=5 floor regardless.
