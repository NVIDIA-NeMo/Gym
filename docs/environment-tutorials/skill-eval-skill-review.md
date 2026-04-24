# Skill-by-skill audit — v7 prescriptions, v8 outcomes

Per-skill verdicts, root-cause hypotheses, and prescriptive file-level changes based on the v7 4-cell rollout. Each entry cites the four v7 Δreward effects, the per-scenario breakdown, and specific source files to inspect or edit. Three prescriptions were implemented between v7 and v8; those entries include a **v8 outcome** block showing whether the predicted effect landed.

The four effects (see `skill-eval-scoreboard.md` for definitions):

- **`skill | refs=T`** (realistic deployment)
- **`skill | refs=F`** (standalone value)
- **`refs | skill=F`** (marginal value of references alone)
- **`Δtools (skill | refs=T)`** (efficiency axis)

Bold = effect outside noise floor (~±0.10 at n=5 per-cell).

---

## `gym-run` — load-bearing, keep as-is

| `skill | refs=T` | `skill | refs=F` | `refs | skill=F` | Δtools |
|---|---|---|---|
| **+0.436** | **+0.480** | −0.013 | −1.87 |

Per-scenario (`skill | refs=T`): sc1 **+0.97**, sc2 **+0.24**, sc3 +0.10.

**Verdict:** the one unambiguous keeper. Large positive on both realistic and standalone contrasts, no references directory to contaminate, sc1 is literally "the model cannot do this task without the skill" (blind=0.00, docs=0.00).

**Why it works:** `gym-run`'s scenarios test specific `ng_run` / `ng_status` / `ng_collect_rollouts` workflow knowledge that the model has no priors for. The SKILL.md walks through env.yaml setup, server health checks, and smoke-test diagnostics — concrete operational knowledge, not explainable from first principles.

**Source:**
- `.claude/skills/gym-run/SKILL.md` (205 lines) — no changes needed.
- No `references/` directory. Consider adding one only if scenarios expand.

**Gaps:** none identified. If anything, this skill is a template for what "skill-only" value looks like when the content is a genuine operational procedure.

---

## `add-benchmark` — keep; scenario 3 is ceiling-clipped

| `skill | refs=T` | `skill | refs=F` | `refs | skill=F` | Δtools |
|---|---|---|---|
| **+0.141** | **+0.446** | **+0.313** | −1.27 |

Per-scenario (`skill | refs=T`): sc1 **+0.14**, sc2 **+0.28**, sc3 +0.00.

**Verdict:** skill and references both carry real signal; the skill adds genuine value on top of a strong reference doc. sc3 (httpx wrapper) is effectively solved by references alone (docs=1.00, blind=0.03) — a clean example of content that migrated entirely into the reference.

**Why it works:** the skill covers the full workflow (data prep → server scaffold → config → baseline), and the references in `patterns.md` go deep on wrapper patterns. They're complementary, not competing.

**Source:**
- `.claude/skills/add-benchmark/SKILL.md` (257 lines) — no changes needed.
- `.claude/skills/add-benchmark/references/patterns.md` (2,541 words — largest in the tree) — clearly doing real work.

**Gaps:**
- sc3 is ceiling-clipped by references. Either write a harder wrapper scenario (e.g. library that uses a httpx transport *and* has its own scoring library *and* emits non-binary rewards) to recover discriminating power, or accept that this scenario just validates the reference content.
- Consider an sc4 testing the "convert from a source repo's format" workflow — that's operational knowledge neither the SKILL.md nor current scenarios exercise.

---

## `gym-config` — holds; 2/3 scenarios are too easy

| `skill | refs=T` | `skill | refs=F` | `refs | skill=F` | Δtools |
|---|---|---|---|
| **+0.111** | **+0.173** | +0.062 | −1.60 |

Per-scenario (`skill | refs=T`): sc1 −0.07, sc2 **+0.40**, sc3 +0.00.

**Verdict:** meaningful value but driven almost entirely by sc2. sc1 and sc3 are ceiling-clipped with refs (docs=1.00), masking any signal.

**Why sc2 works:** validating a combined-reward config requires naming specific fields (`reward_if_quality_low` etc.) in prose. The SKILL.md's validation checklist pushes the model toward reading the file and citing exact fields. Reference content alone (config-patterns.md) doesn't prompt this behavior as reliably.

**Source:**
- `.claude/skills/gym-config/SKILL.md` (219 lines) — no changes needed; the "Before you answer" checklist is doing real work.
- `.claude/skills/gym-config/references/config-patterns.md` (818 words) — likely appropriate coverage for sc1/sc3's simpler validations.

**Gaps:**
- **`.claude/skills/gym-config/evals/evals.json` sc1 and sc3 are too easy.** Add more adversarial scenarios: bad interpolation, circular references, `${oc.env:...}` misuse, duplicate instance names across composed configs. This is where the skill-content quality can't be measured, not where the skill is bad.
- The `fixtures_sha` between different gym-config scenarios would tell us whether the fixture files themselves are the bottleneck. They likely are — probably simple YAMLs with one obvious issue.

---

## `gym-debug` — holds; references carry most of the load

| `skill | refs=T` | `skill | refs=F` | `refs | skill=F` | Δtools |
|---|---|---|---|
| +0.080 | **+0.267** | **+0.187** | **−4.33** |

Per-scenario (`skill | refs=T`): sc1 **+0.24**, sc2 +0.00, sc3 +0.00.

**Verdict:** sc1 is where the skill adds realistic value; sc2/sc3 are ceiling-clipped with refs (docs=1.00). Tool-call reduction is the strongest in the tree — the skill teaches the model to skip exploratory commands.

**Why sc1 works but sc2/sc3 don't:** sc1 is about `check_twice_swap` + positional-bias diagnosis, which requires recalling *multiple interacting concepts* — the skill's structured walk-through primes the model for that. sc2 (think-block interference) and sc3 (combined reward) are "identify the diagnostic field from the rollout JSON" tasks, which the references handle directly.

**Source:**
- `.claude/skills/gym-debug/SKILL.md` (123 lines) — keep.
- `.claude/skills/gym-debug/references/diagnostic-fields.md` and `error-patterns.md` (1,481 words combined) — doing strong work on sc2/sc3.

**Gaps:**
- Scenarios are fixture-driven, and the fixtures contain the diagnostic fields the assertions test for. So the model can reason from the fixture even without docs. Consider sc4/sc5 that test diagnosis *without* the specific field being pre-named in the fixture — e.g., "rewards are 0.5 and the rollout is truncated; is this a judge fallback or a model budget issue?" requires integrating multiple signals.

---

## `gym-review` — delete or dramatically shrink SKILL.md

| `skill | refs=T` | `skill | refs=F` | `refs | skill=F` | Δtools |
|---|---|---|---|
| +0.029 | **+0.298** | **+0.602** | **−4.87** |

Per-scenario (`skill | refs=T`): sc1 +0.00, sc2 +0.03, sc3 +0.06.

**Verdict:** **the skill is almost entirely redundant with its references.** `refs | skill=F` = +0.602 is the largest reference-alone signal in the tree; `skill | refs=T` is essentially zero across every scenario.

**Why:** `gym-review` ships a `scripts/review.py` that the model runs, plus two strong reference docs (`anti-patterns.md`, `fix-patterns.md`, 1,291 words combined). The script does most of the work by producing a structured report; references tell the model what each finding means. SKILL.md adds nothing on top.

**Source:**
- `.claude/skills/gym-review/SKILL.md` (110 lines) — **shrink to ~20–30 lines.** Current SKILL.md duplicates content that lives in the references and the script. Keep only: how to invoke `review.py`, how to interpret its output categories, and when to bypass it.
- `.claude/skills/gym-review/references/anti-patterns.md` and `fix-patterns.md` — **keep; these are load-bearing.**
- `.claude/skills/gym-review/scripts/review.py` — doing the heavy lifting.

**Gaps:**
- The measurable skill value is the tool-call efficiency (−4.87 per rollout). That suggests the SKILL.md is teaching the model to *trust the script* and not explore independently. Which is useful! But we're paying for 110 lines of SKILL.md to teach "run the script" — the same thing could be a 5-line paragraph in the references.

**v8 outcome (prescription implemented in `8fdcdb2c`):** shrunk SKILL.md from 110 → 53 lines. Kept how-to-invoke, severity meanings, affirmative "if clean, approve" framing, and cross-references to anti-patterns.md / fix-patterns.md.

- v8 `skill | refs=T` = +0.048 (v7 +0.029, change within noise — **realistic value preserved**).
- v8 Δtools on skill+docs = −4.73 (v7 −4.87, efficiency **preserved**).
- v8 `skill | refs=F` = **+0.012** (v7 +0.298, collapse of **−0.286** — standalone value lost).
- Provenance diff: `md` only. Clean single-skill attribution.

**Status:** **context-dependent.** Keep the shrink if the deployment model assumes readers always have the repo checked out (then it's a pure win — same realistic effect, shorter doc, same efficiency). Revert or partially-restore if there's a "skill-only, no references" use case to support (e.g., plugin-style delivery of only SKILL.md). This is a product call, not a measurement call.

**Before/after — what moved out vs stayed:**

*Kept (load-bearing for the realistic deployment):*

- Step 1: how to invoke `scripts/review.py` (one paragraph).
- Step 2: what BLOCK vs WARN severity means (two sentences each).
- Step 3: the five judgment checks the script can't do (bulleted).
- Step 4: a one-line "report the findings" framing, with an affirmative *"if the script is quiet, the code is clean — say so"* sentence added.
- Cross-references to `references/anti-patterns.md` and `references/fix-patterns.md`.

*Dropped (moved into the references, which are already seeded into the workspace):*

- The full BLOCK rules table (`httpx-usage`, `ray-get-async`, `missing-semaphore`, etc. with per-rule descriptions).
- The full WARN rules table.
- The verbose numbered "Apply judgment" list — kept as bullets, context now lives in the references.
- The review report template.

The shrink worked because `scripts/review.py` emits each finding with its rule name inline, so the model can look up the rule in `anti-patterns.md` on demand rather than carrying the full table in context. What broke the skill-only cell is that *when references aren't on disk*, neither is `anti-patterns.md` — and the SKILL.md no longer has the tables to fall back on. The shrink is deployment-model-dependent: a clean win when refs are guaranteed present, a regression when they aren't.

---

## `gym-scaffold-agent` — competes with refs on sc3

| `skill | refs=T` | `skill | refs=F` | `refs | skill=F` | Δtools |
|---|---|---|---|
| −0.040 | **+0.213** | **+0.333** | −0.73 |

Per-scenario (`skill | refs=T`): sc1 −0.04, sc2 +0.00, sc3 −0.08.

**Verdict:** within noise on the realistic contrast. Standalone value is real (+0.213), but the refs do more alone (+0.333), and adding the skill on top slightly hurts on sc3.

**Why sc3 is slightly negative:** sc3 is the "clean code, should be approved" scenario. In v7, with skill prompted + refs on disk, the model tends to flag non-issues — the skill's pattern-catalog structure biases it toward finding problems even when none exist. Without the skill (docs-only), the model is more willing to say "looks fine."

**Source:**
- `.claude/skills/gym-scaffold-agent/SKILL.md` (179 lines) — needs trimming. The RL-centric content (cookie propagation, token accumulation, httpx→aiohttp) is correct but over-dominant in the narrative.
- `.claude/skills/gym-scaffold-agent/references/agent-patterns.md` (776 words) — carries most of the value.

**Gaps:**
- **SKILL.md lacks an "when the code is clean, say so" framing.** The pattern-catalog structure teaches the model to search for issues. Add an explicit "Approval heuristic" section that tells the model how to recognize correct code and how to write a clean-bill-of-health review.
- **Not enough coverage of non-RL agent patterns.** The skill is scoped tightly to RL training agents (cookie propagation, token IDs for RLVR). But many agent-server use cases are non-RL (evaluation, orchestration) and the skill has nothing to say. This is real content gap — the author of this sprint's `skill_eval_agent` got no value from the skill.

---

## `gym-profile` — rewrite SKILL.md to narrate to references

| `skill | refs=T` | `skill | refs=F` | `refs | skill=F` | Δtools |
|---|---|---|---|
| **−0.107** | **+0.278** | **+0.342** | −2.87 |

Per-scenario (`skill | refs=T`): sc1 +0.00, sc2 **−0.16**, sc3 **−0.16**.

**Verdict:** **the clearest "skill competes with references" pattern in the sprint.** Useful standalone, but adding it on top of refs consistently costs ~0.16 on non-ceiling scenarios.

**Why:** the SKILL.md is structured as a how-to (Steps 1–5, commands with flags). The references (`metrics-guide.md`) are structured as explanation (named sections per concept: `pass_threshold`, variance, suspicious patterns). For sc2/sc3 assertions that test *conceptual noun recall* (e.g. "mentions `pass_threshold`", "recommends checking `extracted_model_code`"), the without-skill model reasons from first principles + references and surfaces the noun correctly. The with-skill model follows the SKILL.md's recipe structure and never reaches the point of naming the specific noun — it cites the pattern and shows a generic code snippet.

**Source:**
- `.claude/skills/gym-profile/SKILL.md` (148 lines) — **needs targeted rewrite.** Specifically:
  - Line ~67 (patterns table): inline the confirming field names into the narrative. Current: `"reasoning_format_violation_rate may be high — check if thinking tags are being stripped"`. Should be: `"reasoning_format_violation_rate is high → read extracted_model_code to confirm <think> tokens leaked into extraction"`.
  - Elevate `pass_threshold` from a command flag (line ~37) into its own explained subsection.
  - Every pattern → cause chain needs a specific field-name handoff in prose, not via a code snippet.
- `.claude/skills/gym-profile/references/metrics-guide.md` (716 words) — **keep; this is the better doc for these JTBDs.**

**Gaps:**
- This is a Diátaxis-mode mismatch: how-to (SKILL.md) doing explanation's job (conceptual noun recall). The references are explanation-mode and do this correctly. Rewriting SKILL.md to cross-reference `metrics-guide.md` at each decision point — rather than duplicating content in recipe form — should flip `skill | refs=T` to positive.

**v8 outcome (prescription implemented in `3f3a3300`):** prescription landed. Patterns table expanded to three columns (trigger / cause / confirm-by-reading), with specific field names in the "confirm" column. `pass_threshold` promoted to a named concept subsection.

- v8 `skill | refs=T` = **+0.040** (v7 −0.107, change **+0.147** — outside noise floor).
- v8 `skill | refs=F` = +0.309 (v7 +0.278, small strengthening within noise).
- Provenance diff: `md` only. Clean single-skill attribution.
- Per-scenario: sc2 went from with=0.72/without=0.88 to with=0.96/without=0.92; sc3 from with=0.84/without=1.00 to with=0.96/without=1.00.

**Status:** keep. The "narrate to references" prescription is empirically validated on this skill.

**Before/after — the patterns table:**

*Before (v7) — two-column table, confirming field buried in a code block elsewhere:*

```
| Thinking model scores lower than instruct | `reasoning_format_violation_rate` may be high
                                             — check if thinking tags are being stripped
                                             before answer extraction |
```

*After (v8) — three-column table with the confirming field in prose:*

```
| Thinking model scores below instruct on the same tasks
  | Think-block stripping failure — `<think>` tokens are leaking into the extraction regex
  | `reasoning_format_violation_rate` (should be low; if high, read `extracted_model_code`
     for those tasks and look for literal `<think>` or `</think>` in the extracted string) |
```

Same factual content. The v8 version forces the pattern → cause → **confirming field by name** chain to complete in prose, and adds a closing rule after the table: *"when you cite a pattern from this table, also cite the specific field you checked."*

The `pass_threshold` change is structurally similar: in v7 it appeared once as a flag on an `ng_reward_profile` command (`+pass_threshold=1.0`); in v8 it has a standalone subsection titled *"pass_threshold — the knob that controls how partial credit counts"* with worked examples. The model now names the knob when reasoning about partial-reward scenarios.

---

## `gym-data` — ceiling-clipped; scenarios need rewriting

| `skill | refs=T` | `skill | refs=F` | `refs | skill=F` | Δtools |
|---|---|---|---|
| +0.013 | +0.040 | +0.013 | **−3.27** |

Per-scenario (`skill | refs=T`): all near-ceiling (blind=0.88–1.00, docs/skill/both all at 0.96–1.00).

**Verdict:** cannot be measured. Every cell of every scenario is ≥ 0.96. Tool-call reduction suggests the skill teaches something (−3.27 calls per rollout), but the accuracy dimension is invisible.

**Why:** scenarios are "validate this dataset" tasks where the problems are obvious on inspection (sc1: well-formed tool-calling data; sc2: bad data; sc3: schema extension). Modern frontier models solve these from first principles.

**Source:**
- `.claude/skills/gym-data/SKILL.md` (200 lines) — can't evaluate without harder scenarios.
- `.claude/skills/gym-data/references/schema.md` (743 words) — can't evaluate.
- `.claude/skills/gym-data/evals/evals.json` — **this is the file that needs work.**

**Gaps:**
- **Rewrite all three scenarios to be adversarial.** Examples:
  - sc1 replacement: a dataset where tool-calling messages have subtly malformed `function.arguments` (quoted JSON string that should be an object), where the right answer requires knowing the OpenAI schema edge case.
  - sc2 replacement: a dataset where the labels look correct but are semantically wrong (e.g., labeled "safe" but content is clearly unsafe) — tests whether the model applies judgment, not just format validation.
  - sc3 replacement: generate examples for a benchmark with a non-obvious schema (multi-turn with branching tool calls; verifier_metadata with nested expected-answer structures).
- Until scenarios move, this skill is in a measurement dead zone regardless of whether its content is good.

**v8 outcome (prescription implemented in `5e84dd08`):** all three scenarios replaced with adversarial versions. sc1 = 5-entry tool-call schema audit with 4 planted bugs. sc2 = 7-entry semantic mislabel check with 3 wrong gold answers. sc3 = multi-turn branching schema-extension task.

- v8 `skill | refs=T` = **+0.093** (v7 +0.013, change +0.080 — just outside noise floor, **now measurable**).
- v8 cell means: blind=0.85, docs-only=0.88, skill-only=0.91, skill+docs=0.97 (v7 all 0.96–1.00).
- `docs-only` dropped below 0.95 — measurement dead-zone is gone.
- Provenance diff: `evals+fx` only. Clean single-skill attribution.

**Status:** scenarios now interpretable. `blind` only dropped to 0.85 — the new scenarios are harder but still within the model's priors' reach. Even-more-adversarial scenarios could sharpen the measurement further, but the skill is no longer in a measurement dead zone.

**Before/after — scenario prompts:**

*sc1 — format validation becomes schema audit:*

> **Before (v7):** *"Validate the tool-calling dataset at evals/files/sample_tool_calling.jsonl for a search+calculation benchmark."*
> — fixture is well-formed; correct answer is "yes, valid." Trivial for a frontier model from priors alone.
>
> **After (v8):** *"Audit the tool-calling dataset at evals/files/sample_tool_schema_bugs.jsonl. For each entry that would fail OpenAI's function-tool schema or is internally inconsistent, identify the entry and state exactly what's wrong."*
> — fixture has four planted bugs (required-field not in properties, `parallel_tool_calls` vs `expected_tool_calls` mismatch, missing `function` wrapping, single-tool with `parallel_tool_calls=true`) plus one clean line. Requires reading each entry and applying schema knowledge.

*sc2 — format check becomes semantic audit:*

> **Before:** *"Check evals/files/sample_bad_data.jsonl for data quality issues before I upload it."*
> — three format-level issues (missing `verifier_metadata`, missing `input`, literal answer leakage in the prompt). Obvious.
>
> **After:** *"Audit the benchmark at evals/files/sample_mislabeled_benchmark.jsonl. Some `expected_answer` values are wrong for the user's question. Identify every entry where the gold label is semantically incorrect — the schema is fine."*
> — seven entries, three with wrong gold labels: capital of Australia = "Sydney" (should be Canberra), leap year = 365 (should be 366), gold symbol = "Go" (should be Au). Model must apply domain judgment, not just schema validation.

*sc3 — copy-the-schema becomes extend-the-schema:*

> **Before:** *"Generate 5 example entries for a SQL benchmark based on the schema in evals/files/sample_sql_benchmark.jsonl."*
> — fill-in-the-blank; the model copies the schema and generates five plausible SQL questions. Near-trivial.
>
> **After:** *"Here is one entry of a multi-turn customer-support benchmark … Generate 3 additional entries that follow the same schema and exercise three distinct branching scenarios: (a) a legitimate refund that requires looking up account first because the user only gave an email, (b) an abuse attempt where the model should refuse, and (c) a partial refund where the amount is ambiguous."*
> — the base schema has `expected_tool_sequence`, `forbidden_sequence`, `partial_credit`, and branching logic. Requires reading the schema, understanding the constraints, and generating coherent new branches.

In every case the "before" version could be solved by pattern-matching the fixture format; the "after" version requires reading the contents and applying real judgment. The scenarios moved from validating *format* to testing *understanding*.

---

## Cross-cutting patterns and prescriptions

**Where SKILL.md is redundant with references (shrink SKILL.md):**
- `gym-review` — **implemented v7→v8.** Realistic value preserved, standalone collapsed (see v8 outcome block). Context-dependent win.

**Where SKILL.md competes with references (rewrite SKILL.md to narrate to refs):**
- `gym-profile` — **implemented v7→v8.** Prescription validated; realistic flipped from −0.107 to +0.040.
- `gym-scaffold-agent` — milder case; not yet edited.

**Where scenarios are ceiling-clipped (rewrite evals.json):**
- `gym-data` — **implemented v7→v8.** Measurable headroom recovered (docs-only 0.97 → 0.88).
- `gym-config` (sc1, sc3) — not yet edited.
- `gym-debug` (sc2, sc3) — not yet edited.
- `add-benchmark` (sc3 is actually fine — solved by refs — but consider a new scenario that needs unique skill knowledge)

**Where the skill has real content gaps (add new material):**
- `gym-scaffold-agent` — missing non-RL agent patterns (evaluation, orchestration). Still open.
- `gym-scaffold-agent` — missing "how to recognize correct code" heuristic. Still open.

**Where the skill is load-bearing as-written:**
- `gym-run` (keep)
- `add-benchmark` (keep)
- `gym-config` (keep, but write harder scenarios)

### v7 → v8 results on the three prescriptions

| skill | prescription | predicted direction | observed | verdict |
|---|---|---|---|---|
| `gym-profile` | rewrite patterns table inline | `skill \| refs=T` → 0 or positive | −0.107 → **+0.040** | **hit** |
| `gym-review` | shrink SKILL.md | `skill \| refs=T` flat, no regression | +0.029 → +0.048 (flat) | realistic hit; standalone collapse |
| `gym-data` | adversarial scenarios | `docs-only` < 0.95 | 0.97 → **0.88** | hit |

Two outright hits, one caveated hit. The framework for turning v7 cell-level observations into single-skill editable prescriptions generated three testable predictions, all of which landed in the predicted direction with clean provenance attribution. The caveat on gym-review (standalone value lost) was a surprise — the audit predicted the shrink would be a clean win and it isn't on every cell.

### Remaining prescriptions (not yet implemented)

1. `gym-scaffold-agent` — add non-RL agent pattern coverage. Expected: `skill | refs=T` moves from −0.040 to at least +0.05 on scenarios that test the new material. Requires writing new SKILL.md content AND at least one new scenario that tests it.
2. `gym-config` sc1/sc3 — harder scenarios. Expected: `docs-only` drops below 0.95 on those scenarios, revealing whether the skill adds value there.
3. `gym-debug` sc2/sc3 — harder scenarios, same pattern as gym-config.

## Prerequisites for trusting these recommendations

v8 confirmed two of three prescriptions outright. Before shipping further edits:

1. **Calibration run at n=20** on one cell (e.g., `gym-profile sc2 skill+docs`) to pin down detectable effect size. v8 same-all drifts of −0.084 (gym-config) and +0.093 (gym-scaffold-agent) show that n=5 can't resolve effects near the noise floor.
2. **One edit at a time** — validated in v7→v8: three skills edited, three clean `md` / `evals+fx` provenance tags on exactly those skills, `same-all` on every other skill. The discipline works; keep doing it.
3. **Post-edit diff must show the right provenance tag.** If the prescription is "rewrite SKILL.md for gym-X," the diff should flag `md` as the change for exactly that skill and `—` (same-all) for every other skill. Anything else is a confound.
4. **On shrinks specifically**, measure the standalone contrast too. gym-review showed that a SKILL.md shrink can preserve realistic value while collapsing standalone value — if both cells matter for your deployment model, check both before shipping.
