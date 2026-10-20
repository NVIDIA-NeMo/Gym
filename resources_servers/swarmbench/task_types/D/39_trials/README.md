# 39_trials: clinical-trial eligibility screening (family D, source A)

## Task
The research office pre-screens N synthetic patients, each assigned to one of 16 real recruiting trials from
ClinicalTrials.gov. For every patient the agent must say whether the patient is eligible and list exactly which
criterion ids fail. Inclusion lines (I1, I2, ...) fail when not met; exclusion lines (E1, ...) fail when they apply.

**Unit:** one patient screened against its assigned trial. Answer per patient: `{"eligible": bool, "failed": [ids]}`.
A unit is correct only if both fields match gold exactly. Score = the fraction of correct patients.

## Data
- **Criteria:** 16 trials (5 NSCLC, 4 breast cancer, 7 type 2 diabetes), 5-15 lines each, 122 lines in total.
  - Each line is copied verbatim from the registry's eligibility text. Only markdown escape backslashes and leading
    bullets are removed (`build_snapshot.py`).
  - Only lines that `trialspec.py` encodes exactly are shown. Vague lines are never shown.
  - Two lines come from structured registry fields instead: `Sex: Female` for two female-only trials and
    `Ages eligible: 18 Years to 65 Years` for NCT07613307.
- **Encoding:** `trialspec.py`, hand-written by Claude, maps each shown line to a predicate. Kinds: age, sex, lab range,
  ULN multiple with an alternative limit, ECOG, BMI, exposure window with exceptions, current or ever drug class,
  therapy for an indication, condition scope, procedure, pregnancy, lactation, stage, histology, biomarker, diagnosis
  duration, other malignancy with exceptions, and any/all/not combinations.
- **Patients:** synthetic and procedural. Each one is generated per (seed, i), so a smaller n is a prefix of a larger
  n.
  - The chart has 9 sections (summary, problems, oncology, medications, procedures, immunizations, labs, vitals,
    performance), plus dated notes.
  - The notes are built from K3 templates and carry some facts: ECOG, pregnancy tests, breastfeeding, family history,
    ruled-out diagnoses, last doses, outside vaccines and surgeries, and weights. The rest are filler notes.
- **Gold:** `type.ev` evaluates the predicates on the generated facts at the screening date. Arithmetic is Decimal on
  the displayed values, with unit conversion as listed in the prompt. Values shown in an alternate unit stay at least
  6% from any threshold, so rounding cannot flip a result. Exact-boundary values appear only in the canonical unit.

## Traps
At least one trap appears in 100% of units (test: at least 90%). Most units carry several. About 6% of patients are
eligible and 94% fail 1-4 lines. The trap kinds:
- **Superseded values:** an older lab, ECOG or weight on the other side of the threshold.
- **Units:** alternate units (g/dL, /uL, umol/L, mmol/mol, lb/in).
- **Exact boundaries:** values exactly at a limit, where the comparator decides (`>=` vs `>`).
- **ULN multiples:** a non-standard ULN printed per result; alternative limits for liver or bone metastases and for
  Gilbert's syndrome.
- **Age:** a birthday a few days before or after the screening date.
- **Time windows:** dates 2-6 days outside the window; a stop date found only in the notes; a palliative RT exception;
  stable long-term steroids; surgery related to the breast cancer, which is permitted.
- **Drug class:** Shingrix vs live vaccines; inhaled, topical or nasal steroids; GLP-1 injections vs insulin;
  bevacizumab vs checkpoint inhibitors; a biopsy that is not therapy.
- **Conditions:** family history and ruled-out diagnoses, which do not count (except the MTC line, where family history
  counts); near-name conditions (IBS vs IBD, autologous vs allogeneic, NYHA III vs IV, CKD 3b vs 4); resolved vs
  active; "concurrent" HBV and HCV.
- **Malignancy:** excepted skin and in-situ cancers, and cancers older than 5 years.
- **Stage and biomarkers:** a neighbouring stage (IIIB vs IIIC), and a KRAS decoy next to the EGFR/ALK results.

## Why one context cannot do the hard tier
- Each patient needs 6-9 tool calls (trial, 4-6 chart sections, notes) and about 120 lines of reading.
- Each patient also needs per-line arithmetic (units, windows, ULN multiples).
- The hard tier has 80 independent patients.

## Tiers
`TIERS = {"easy": (8, 1800), "medium": (24, 1200), "hard": (80, 1200)}`.
- **Unit cost:** in the 1-unit K3 loop a unit takes about 115 s and 12 tool calls.
- **Sizing assumption:** OpenCode solo is about 2.5x faster, roughly 45 s per unit.
- **Resulting load:** about 3600 s of solo work at hard, 3x the budget. Medium fits the budget, at about 1100 s.

These are first guesses, not yet calibrated.

## Tools
All tools are read-only, offline and bounded at 8k characters:
- `list_patients` (paged)
- `list_trials`
- `read_trial`
- `read_chart(patient_id, section)`
- `read_notes(patient_id, offset, limit)`
- `calculator`

No tool evaluates eligibility.

## Baselines (check 2)
Over 10 seeds per tier:

| baseline | easy n=8 | medium n=30 | hard n=80 |
|---|---|---|---|
| regex: an exclusion applies if two of its content words appear in the chart; inclusions assumed met | mean 0.075, max 0.125 | mean 0.047, max 0.133 | mean 0.039, max 0.100 |
| all eligible | mean 0.037, max 0.125 | mean 0.05, max 0.10 | mean 0.048, max 0.075 |
| careless reader: oldest values, no unit conversion, family history counts, any drug counts, notes skipped | mean 0.087 | mean 0.077 | mean 0.066, max 0.15 |

`checks.json` (seeds 0 and 1, n = 8 / 24 / 80):
- **Regex:** 0.00, 0.125 / 0.083, 0.083 / 0.10, 0.025.
- **Partial readers** (each ignores one trap kind):

| partial reader | score range |
|---|---|
| first_values | 0.25-0.38 |
| no_unit_conversion | 0.6-0.88 |
| history_any | 0.4-0.58 |
| skip_notes | 0.47-0.75 |
| any_drug | 0.5-0.79 |

At n=8 one unit is worth 0.125.

## Check 5 and verifier audit (self-hosted K3, single-unit tool loop, n=80 instances)

| run | seeds | correct | notes |
|---|---|---|---|
| k3 | 0, 1 | 18/20 | P0051 model error: insulin stopped 3 months earlier vs "who are taking". P0037 gold defect: HIV shown as "resolved". **Fixed:** only episodic conditions can be resolved. |
| k3v2 | 2, 3 | 18/20 | Two gold ambiguities. Allogeneic stem-cell transplant vs "allogeneic tissue/solid organ"; **fixed:** solid organs only. A stable long-term steroid decoy, where dose stability is not charted; **fixed:** decoy removed. |
| k3v3 | 4, 5 | 18/20 | Two model errors: a non-breast major surgery read as therapy for breast cancer. Realism fix: every surgery now carries a non-cancer indication. |
| k3v4 | 6, 7 | 17/20 | One model error (same insulin-taking reading). Two gold ambiguities. A DKA episode straddling the window edge; **fixed:** an episode counts if any part of it is in the window, and decoys sit at least 4 days clear of the window. "Hepatitis B, resolved past infection" shown as active; **fixed:** status is consistent with the name. |
| **k3v5 (final code)** | 8, 9 | **20/20** | |

- **Final code:** 20/20.
- **All 100 hand-audited units:**
  - 6 graded-wrong units came from gold or ambiguity defects. All were fixed before the final run.
  - 5 graded-wrong units are model errors.
  - 0 graded-right units were wrong.
- **Contested reading (resolved):** the stopped-insulin decoy behind "who are taking insulin" was dropped after review; only the NCT06884059 units that used it changed.

## Sources
- **ClinicalTrials.gov API v2** (US NLM). Per https://clinicaltrials.gov/about-site/terms-conditions the content is not
  under NLM copyright and may be reused with attribution. Retrieved 2026-10-14.
- **Raw files** are search pages for 6 conditions. Their sha256 hashes are in `data/SOURCES.json`, and the
  `data/raw/` files are kept on the build host only.
- **Shipped snapshot:** `data/trials.json`, sha256 in `data/SOURCES.json`.

## Provenance
- **K3 text:** all task prose comes from self-hosted Kimi K3 (pool replica, served as `policy_model`): intro, rules,
  answer line, unit prompt, tool descriptions and every note template. It was written by `k3gen.py` from
  `text/request.json` on 2026-10-14, in one attempt with no problems. The raw reply is in `text/k3_raw.json` and the
  frozen text in `text/k3.json`.
- **K3 prompt:** the "user" field of `text/request.json`, with k3gen's system prompt.
- **Claude-written:** code, the criterion encoding, and code labels such as condition, drug and vaccine names, section
  headers and the unit-conversion list.
- **Not shipped:** `text/dev_stub.json`, a Claude stub used only for offline tests.
- **Canary:** in every prompt and in `world/CANARY.txt`.
