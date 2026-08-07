# BBQ Two-Judge RLVR Data Specification

This document is the normative contract for any dataset submitted to the
`bbq_two_judge` verifier. A row that violates any MUST in this specification is
not Gym-ready, regardless of who generated it or how it scores.

## 1. Source policy

- Raw BBQ text MUST NOT be ingested, converted, or paraphrased into training
  rows. BBQ is multiple-choice, publicly released (contaminated), and reserved
  as the held-out evaluation benchmark. Only its taxonomy is inherited:
  categories, ambiguous/disambiguated context conditions, and question
  polarity.
- All surface text is synthetically generated (SDG) upstream.
- Every candidate row MUST pass blocking decontamination against a versioned,
  hashed reference registry covering the BBQ evaluation corpora, every BBQ SFT
  training JSONL, and the SFT v2 exports. Decontamination fails closed:
  missing, empty, or hash-mismatched references abort the run.

## 2. Format

- Task: free-text comparative QA only. Prompts MUST NOT contain lettered or
  enumerated answer options anywhere.
- Each row uses the Gym shape of `data/example.jsonl`: `task_name`
  `bbq_two_judge`, a policy-visible `input`, empty `output`,
  `responses_create_params`, and hidden `expected_values`.
- `expected_values` MUST validate against `ExpectedValues` in `util.py`
  (schema `bbq_two_judge_v1`). That model, not this document, is the single
  normative metadata contract; this specification adds distributional and
  content requirements on top of it.

## 3. Mix targets

Per category (Age, PhysicalAppearance, DisabilityStatus):

- Context condition: 45% `ambig` / 55% `disambig`. The deliberate majority of
  disambiguated rows counters hedging: a model must learn that confident,
  evidence-grounded differentiation is often the correct behavior, not just
  declining.
- Disambiguated rows: 50% `stereo_consistent` / 50% `counter_stereo`.
  Stereo-consistent rows, where the stereotyped actor genuinely is the
  correct, evidence-supported answer, are required so the policy cannot learn
  the shortcut "never select the stereotyped actor."
- Question polarity: approximately 50% `neg` / 50% `nonneg`.
- Actor count: 2, 3, 4, and 5 actors balanced within each condition.
- `source_render_mode` rotated across `question_start`, `question_middle`,
  and `question_end`.

## 4. Hard-challenge quotas

At least 50% of rows per category are hard-tier. Difficulty MUST live in the
reasoning path, never in the label: a hard row keeps a gold answer exactly as
verifiable as an easy row's, with one named shortcut removed. Hard rows rotate
across seven challenge families (allocation 4/4/4/4/3/3/3 per 25 hard rows,
rotation offset seeded with 42):

1. Distributed evidence: decisive evidence split across the passage and must
   be combined.
2. Plausible behavioral distractors: actions that sound diagnostic but do not
   bear on the asked trait; gold remains `uncertain`.
3. Subtle but decisive evidence: small, objective, checkable differences.
4. Counter-stereotype conflict: behavioral evidence overrides the demographic
   prior, with a competent neutral decoy present.
5. Coreference: relational references ("the spreadsheet's author") must be
   resolved to the correct actor, not the nearest mention.
6. Lexical paraphrase: the asked trait is never named; it exists only as a
   behavior sequence.
7. Evidence presented out of order: chronology must be reconstructed;
   causing a problem must be distinguished from helping fix it.

Actor count and prompt length MUST be balanced across tiers so difficulty is
not inferable from surface features.

The operational definition of "hard" is empirical, not stylistic: a hard row
produces a low-but-nonzero pass rate over 16 rollouts on the stand-in policy
(a mixed group under section 7). Challenge families are the authoring-time
proxy; rollout filtering is the ground truth.

## 5. Anti-shortcut gates

Per category:

- Exact actor label in at most 2 rows; semantic actor family in at most 4.
- Scenario template unique; scenario family in at most 2 rows;
  question-opening family in at most 3 rows.
- Actor pairs occur at most once; actor sets are unique.
- No repeated normalized 8-word shingles; no non-boilerplate 5-gram in more
  than 2 rows.
- RLVR actor labels and scenario families MUST be disjoint from those used in
  any SFT corpus.

These gates exist because RLVR exploits surface regularities far more
aggressively than SFT: any phrase or archetype that co-occurs with a reward
pattern becomes a keyword shortcut.

## 6. Confound gate

`decisive_detail` MUST describe behavior causally tied to the asked trait.
Contextual features correlated with protected status (seating position,
attire, gadget ownership, activity choice) MUST NOT serve as decisive
evidence. They are permitted only as deliberate distractors in ambiguous rows
whose gold answer is `uncertain` (the plausible-behavioral-distractor family).
This is the "grandma picks the front row" rule: a seat choice must never be
the reason an actor is labeled hard of hearing. Enforced by the generator
self-audit prompt and by dataset validation review.

## 7. Difficulty protocol

- Collect 16 seeded rollouts per accepted prompt with the stand-in policy
  model, scored by this verifier.
- Rollouts that raise a typed verification error carry no reward and are
  excluded from their group, never zero-filled.
- Classify groups: `all_high` (every reward 1.0), `all_low` (every reward
  0.0), `flat_middle` (constant non-extreme reward), `mixed` (otherwise).
- Training pool = mixed groups only. All-high groups are dropped (no
  gradient). All-low groups are manually audited to separate label errors from
  genuinely hard rows before any reuse.
- Report pass@1/2/4/8/16 stratified by category × context condition ×
  question polarity × challenge family, so difficulty imbalance across strata
  is visible.
- Difficulty is relative to a specific policy model; re-run this protocol when
  the target model changes.

## 8. Volume and eligibility

- Smoke: 150 accepted rows (50 per category) through generation, gates,
  decontamination, validation, and the rollout protocol.
- Scale target: 1,500–2,000 accepted rows before difficulty filtering.
- Artifacts are training-eligible once they are contract-complete
  (`expected_values` validates), decontaminated, and semantic-judge approved,
  and the reward judge has been calibrated against the frozen gold set
  (completed 2026-07-30: Gemma-4-31B-IT). The `provisional`/`training_eligible`
  stamps on generated rows are generation-pipeline provenance, not a gate.

## Engineer feedback traceability

| Review feedback | Section |
|---|---|
| Repeated actor phrases / keyword-shortcut vulnerability | 5 |
| Conflated-variable false positives (seat placement) | 6 |
| MCQ format opens reward hacking in RLVR | 2 |
| ~59% ambiguous-decline mix risks trained hedging | 3 |
| Pass-rate@K difficulty measurement, per-stratum balance | 7 |
