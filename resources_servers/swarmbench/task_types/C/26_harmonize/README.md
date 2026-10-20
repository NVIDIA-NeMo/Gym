# 26 harmonize: unit and currency harmonization across facility reports (family C, P)

## Task
A fictional manufacturer has `n` facilities. Each facility publishes two fiscal-year energy reports (fiscal years end
in a facility-specific month, Feb-Oct), so calendar year 2025 spans both reports. Every report has a meter table,
monthly tables per meter (consumption in the meter's unit, read type A/E, cost in the local currency, plus decoy
budget-cost and prior-year columns), notes and a prior-period adjustments section. A conventions document gives the
exact factor from every unit to MWh and the rules; an FX file gives monthly USD rates per currency.

For every facility the agent reports calendar-2025 total energy in MWh and total cost in USD:
`{"FAC0001": {"energy_mwh": 1234.567, "cost_usd": 98765.43}, ...}`.

Unit = one facility. Units: kWh, MWh, GJ, therms, MMBtu, m3 (gas), litres and US gallons (diesel), tonnes and
short tons (coal); a meter's unit may change between the two reports. Currencies: EUR, GBP, JPY, CAD, CHF, AUD, SEK,
USD (USD rare). Each cost is converted at its own month's rate.

## Traps (2-4 per facility, all five kinds mixed; each verified to move energy or cost by >= 1%)
| kind | what the report says | careless result |
|---|---|---|
| scale | a table note: both consumption columns are in thousands of the unit | 1000x too small |
| mult | meter replaced in a month; from then on the figures are register readings x multiplier (2-20) | too small |
| corr | one month's consumption misreported; corrected figure given in the notes (x0.1 .. x10 error) | wrong month |
| sub | a meter is a sub-meter of the main meter; already included, must not be added | double count |
| est | a month marked E in the FY2025 report is replaced by an actual reading and restated cost in the FY2026 report's prior-period adjustments (cross-report) | estimate kept |
Distractors: other E rows that stand, budget and prior-year columns, a budget-revision note and a prior-year note per
report, benign notes, the FX annual-average row, the non-calendar months of each report.

## Why one context cannot do the hard tier
Each facility has at most 2 meters: two billed meters, or one billed meter plus its sub-meter (listed, excluded).
Reports average about 3.2k characters each, so the hard tier (n=160) is about 1.03M characters. The cost is
arithmetic, not reading: every facility needs 12 calendar months x 1-2 billed meters of unit conversions and
month-by-month FX products, applied after the notes. That is about 25-50 multiplications per facility, done through
the calculator tool (no shell). A careful unit takes about 6-10 tool calls. Units are independent, so the work
splits by facility.

## Tiers (n facilities / budget s)
easy 8 / 1800, medium 40 / 1200, hard 80 / 1200.

Hard is set to 80 by the coordinator: n=80 is the next sweep, which decides whether the type is promoted or dropped.
Calibration on seed 9100 at n=160 gave:
- solo: 0.19, 0.00 and 0.00 (the last two hit the deadline with no answer);
- task: 1.00, 0.00 and 0.00 (the last two were cut off while still assigning work to subagents).

A solo K3 probe on the earlier 3-4 meter design scored 0.77 at n=64 (about 28 s per facility). `generate` takes about
0.1 s at n=160. `list_facilities` pages are at most 60 rows (5.9k characters).

## Tools
`list_facilities(offset, limit)`, `read_report(report_id, offset)` (7.8k-char pages), `read_conventions()`,
`read_fx(currency, start_month, end_month)` (plus an annual-average row), `search_report(facility_id, text)`,
`calculator(expression)`. All read-only, offline, bounded, thread-safe (world loaded once under a lock).

## Scoring
A facility is correct if both values are within 0.1% relative. The graded answer is the last JSON object keyed by
facility ids. If none parses (for example a truncated answer), every complete `"FACnnnn": {...}` entry is graded on
its own (row salvage); malformed or missing facilities count as wrong, and format-only answers score 0. `score` = fraction of facilities correct;
`score_partial` = fraction of the 2n values correct; `score_exact` = all correct.

## Baselines (checks.json, seeds 0-1, n = 8/40/80, max over instances)
- Oracle: 1.00.
- Required plausible-wrong answers: naive (every figure read as kWh, raw local cost) 0.00, ignore all notes 0.00,
  fiscal year instead of calendar 0.00-0.01.
- Partial readers at n=80:
  - no FX conversion: 0.01-0.05;
  - annual-average FX rate: 0.09-0.19;
  - ignore one trap kind: 0.31-0.59.
- Partial readers at n=8-40 run higher (up to 0.50 for annual-average FX and 0.88 for one ignore-one-trap reader)
  because there are only a few facilities.
- Format-only answers score 0. Tool robustness: 0 errors.

## Change log
- **2026-10-14, after check 5: at most 2 meters per facility.** Previously facilities had 2-3 billed meters plus an
  optional sub-meter, about 70-100 products per facility. Self-hosted K3 then solved 16/20 single units, and all 4
  failures were slips in arithmetic or copying (see the audit below). Now there are two billed meters, or one billed
  meter plus its sub-meter. Every trap kind is still available, with 2-4 traps per facility, each moving energy or
  cost by >= 1%. The rule of converting each month's cost at its own month's rate is unchanged. The K3 text is
  unchanged. Tiers went from 5/16/64 to 6/24/96, then to 8/40/160 after the solo K3 probe, then to hard 80 by coordinator decision (see Tiers). Row salvage was added to `parse_answer`; regrading the rows-calA calibration answers changes no 26_harmonize score.
- **Same day:** `list_facilities` now caps a page at 60 rows. At n=96, a 100-row page passed the 8k output cap and cut
  off rows, which the new hard tier exposed (oracle 0.84 before the fix).

## Files
`type.py` (generator, `compute` = the rules over a facility's facts, tools, scorer), `solutions.py` (tool-only
oracle that parses tables and notes with regexes built from the K3 templates, baselines, single units),
`tests/test_type.py`, `checks.json`, `text/request.json`, `text/dev_stub.json`.
`generate` asserts that the facts parsed back from the rendered numbers reproduce the simulated truth exactly, and
resamples a facility until every trap moves a value by >= 1%. Facility i depends only on (seed, i), so smaller n is a
prefix of larger n.

## Sources
Procedural (P). Conversion factors are stated in the task's own conventions document (common engineering values,
rounded); no external data.

## Provenance
All task-facing prose (prompt, conventions intro and rules, headings, column labels, meter descriptions, every note
template, background sentences, FX heading, tool descriptions, unit prompt) is to be written by self-hosted Kimi K3
from `text/request.json` via `../k3gen.py` and frozen in `text/k3.json`. **Status: pending** (K3 replicas not up yet);
the code currently runs on `text/dev_stub.json`, Claude-written for local tests only, which never ships (check 9
requires `text_source == "k3"`). Claude wrote code, verifiers and tests only. Company, facility and city names,
units, currencies, ids and all numbers are code.

## Verifier audit of the single-unit check (check 4)
Source: `../_units/26_harmonize-k3/units.jsonl` (self-hosted K3, n=16, seeds 0-1, 20 units, 16 recorded correct).
All 20 verdicts were hand-checked and regraded (`audit/units_audit.json`). Result: **FP 0, FN 0**.
- 16 correct: both values match gold to rounding (relative error < 1e-6). Every trap moves a value by at least 1%, so
  an exact match implies every trap was applied.
- 4 failures, all model errors. None matches an ignore-one-trap baseline, and no rule, text, tolerance or unit-prompt
  ambiguity was found:
  - FAC0010 (s1): cost +0.30%. One month's cost was mis-transcribed (238,817.05 instead of 217,983.38 SEK).
  - FAC0012 (s1): cost -0.50%. The January cost of the second meter was dropped when the restated estimate was applied.
  - FAC0009 (s1): energy -17%. Its listed subtotals are right to within 0.03%, but the final addition is wrong.
  - FAC0015 (s1): energy -2.8%. The calculator expression used 101.824 instead of 101,824 kWh, against the model's
    own summary.
- Failure mode: long hand arithmetic and transcription (~70-100 products per facility under the earlier 3-4 meter design; reduced since, see Change log).
  Unit solvability is 80%, below the 90% bar. The cause is the model, not the verifier.

### Audit after the 2-meter redesign and calibration (check 4, continued)
- **Check-5 rerun:** 20 units at n=24, seeds 0-1; 19 recorded correct. All 20 verdicts were hand-checked and
  regraded: **FP 0, FN 0**.
  - The 19 correct units match gold to rounding.
  - The one failure (n24-s0 FAC0006) is a model error: it applied the x5 meter multiplier to the cost column too
    (cost +16.5%, energy exact).
  - Optional brief tweak: K3's `mult_note` says the multiplier gives consumption but, unlike `corr_note`, does not
    add "the cost shown is unchanged". Adding that sentence would make the two notes symmetric.
- **Calibration:** 18 runs from `rows-calA` (seed 9100); every regrade matches.
  - In the medium and hard runs, only 2 answered facilities are graded wrong. Both are model errors:
    - n40 solo, FAC0001: energy -22%, cost exact. Three other runs got this facility exactly right.
    - n160 solo, FAC0012: the Jun 2025 M1 energy was dropped; the gap is exactly 326.696 MWh.
  - Every other uncredited facility is simply missing from the final JSON, and the answers say so. That means no
    parser false negatives.
  - 5 graded-right facilities were checked (2-4 traps each): all exact. **FP 0, FN 0.**
- **No-answer runs:** 4, all at n=160 (2 solo, 2 task). All hit the deadline (exit 124) with only a progress line and
  no JSON, so score 0 is correct.
