# 23 schemamap: schema mapping over real permit exports (family C, R/A)

## Task
A city agency receives building-permit exports ("sources", `SRC001`...) from partner organisations. Each source is
a CSV table (30 real rows) with its own column names and a data dictionary. For every source and every one of the
16 target fields, the agent names the source column that holds the field (or null) and one transform from a fixed
menu. One unit = one source; a unit counts only if all 16 fields are right.

Target fields: permit_id, permit_type (code 1-8), status, filed_date, issued_date, completed_date, street_number,
street_name, zipcode, estimated_cost, revised_cost, existing_stories, proposed_stories, existing_use, proposed_use,
description. Transforms: identity, date_mdy, date_dmy, date_yyyymmdd, cents_to_dollars, thousands_to_dollars,
decode_codelist, type_name_to_code, address_number, address_street.

## How a source is built (`make_source`, unit i depends only on (seed, i))
- 9-16 target fields present (each 80%); the rest map to null.
- Column names: K3 alias bank per field (some terse, e.g. `dt_1`, `amt_2`); 25% opaque (`c07`).
- Presentation: one date style per source (ISO, m/d/Y, d/m/Y, YYYYMMDD); each cost column in dollars, cents or
  thousands; permit type as numeric code, partner letter code (code list in the dictionary) or type name; street
  number + name either separate or one combined site-address column (35%).
- 3-5 decoy columns from real non-target fields with target-like names: record-created date, status date (often
  equal to the completion date), plan-approval date (often equal to the issue date), dwelling units (vs storeys),
  plan sets, block, lot, internal record id, district, neighborhood, construction type, owner mailing address.
- Traps resolved only by notes, errata or values (every source has at least one errata entry or a date-pair note;
  about 85% have errata, 40% a note):
  - erratum "descriptions of columns a and b were exchanged" (mostly confusable pairs: estimated/revised cost,
    existing/proposed storeys or use, filed/issued, issued/completed, issued/approved, completed/status date,
    filed/created, storeys/units, permit id/record id);
  - erratum "since release r, column c is exported in <unit>" (the description shows a stale unit);
  - erratum "column old is now named new" (the dictionary lists the old name, the export header the new one);
  - note "columns a and b hold the <f1> and the <f2>, which is which not documented": settled by the row-wise
    order of the values (the snapshot keeps only filed < issued < completed);
  - m/d/Y vs d/m/Y: settled by a day > 12 in the values (the generator falls back to YYYYMMDD otherwise).

## Tools (read-only, bounded, offline)
`read_target_schema`, `list_sources(offset, limit<=100)`, `read_dictionary(source)`, `read_rows(source, offset,
limit<=15)` (CSV with the export header), `column_profile(source, column)` (blanks, distinct, min, max, examples).
No tool reveals a mapping.

## Answer and scoring
One JSON object: `{"SRC001": {"permit_id": ["application_no", "identity"], "zipcode": null, ...}, ...}`. If no complete answer object parses (one malformed key, truncated output), every complete
`"SRCnnn": {...}` entry is graded on its own and malformed or missing sources count wrong. A bare
string means identity; a missing field means null; column names compare case-insensitively. `score` = fraction of
sources with all 16 fields right; `score_partial` = fraction of fields right; `score_exact` = all sources right.

## Tiers
easy 12 / 1800 s, medium 40 / 1200 s, hard 80 / 1200 s (coordinator, after calibration A). History: an OpenCode solo
K3 probe at n=80 scored 0.975, so the tiers were first raised to 12 / 80 / 240; in calibration A (seed 9100) every
n=240 run and most n=160 runs ended without a final answer at the budget (the answer alone is ~40k tokens at 240), so
hard went back to 80 and medium to 40. Generate takes < 0.1 s at every tier. Rows: each source gets a disjoint
30-row slice of a per-seed permutation of the 7,200-row snapshot (no row repeats within an instance up to n=240);
the owner-mailing-address decoy reuses the addresses of slice i+120. `list_sources` pages at <= 100 rows (5.8k
chars) and the oracle pages by rows returned.

## Baselines (checks.json, n = 12/40/80, seeds 0-1; K3 text)
| strategy | max score |
|---|---|
| oracle (tool-only, template regexes + values) | 1.00 (min) |
| name similarity (difflib on header names, date style from values) | 0.00 |
| dictionary only (ignores errata, notes and value evidence) | 0.08 |
| partial: ignore swaps / unit errata / renames / pair order / assume m/d/Y | 0.67 / 0.68 / 0.83 / 1.00 / 1.00 |
No-op and format-only answers 0; tool robustness 0 errors; deterministic hashes. The partial readers reach 1.00 only
on n=12 seeds whose sources happen to lack that trap kind; at n=40-80 they score 0.6-0.9. Dictionary-only by
size: 0.083 / 0.05 / 0.075 (max over seeds at n = 12 / 40 / 80).

## Why one context cannot do the hard tier
80 sources x (dictionary + 30 rows + reasoning) is ~170k tokens of reading and 80 independent judgments with
3-6 tool calls each plus a ~13k-token answer; calibration A shows where one agent stops keeping up, while the units
are independent and need no cross-source information.

## Sources
- San Francisco Building Permits, DataSF dataset `i98e-djp9`, retrieved 2026-10-14 from
  `https://data.sf.gov/resource/i98e-djp9.csv`: permits filed in 2019 (6,000 rows) and the first 12,000 permits by
  number filed in each of 2017, 2018, 2020, 2021 (queries and per-file sha256 in `data/SOURCES.json`). License: Open
  Data Commons Public Domain Dedication and License (PDDL 1.0), per the dataset metadata.
- Snapshot (`build_snapshot.py`, deterministic): `data/permits.csv`, 7,200 rows (types 3 and 8; others fail the
  completeness filter), sha256 `cac76bc2...288a5c`.
- Permit type code table: the real SF DBI permit type definitions (1-8).

## Provenance
All task-facing prose (prompt, schema header, field definitions, transform descriptions, alias banks, dictionary
sentences, note and errata templates, partner names, tool descriptions, unit prompt) must come from self-hosted
Kimi K3 via `../k3gen.py` with `text/request.json` (sha256 `08acd3f6...`), frozen as `text/k3.json`.
`text/k3.json` is in place (tests and checks pass with it). `text/dev_stub.json` is Claude-written test text for local development only and must
not ship (check 9 requires `text_source == "k3"`). After k3gen: rerun tests and `python3 checks.py 23_schemamap`
(the tests check that every K3 sentence maps to exactly one meaning, which K3 might violate).

## Files
`type.py`, `solutions.py`, `build_snapshot.py`, `tests/test_type.py`, `checks.json`, `text/request.json`,
`text/dev_stub.json`, `data/permits.csv`, `data/SOURCES.json`, `data/raw/`.

## Verifier audit (check 4, single units)
Self-hosted K3 single-unit run (n=24, seeds 0-1, 20 units; `_units/23_schemamap-k3/`): 19/20 correct. All 20
verdicts were hand-checked field by field against gold (`audit/units_audit.json`): **FP 0, FN 0**. The one failure
(s0 SRC011) is an agent error: it reversed estimated/revised cost when applying an explicit "descriptions exchanged"
erratum, while applying the identical storeys erratum correctly in the same answer. No ambiguity found; no code
or text changed. (That audit used the earlier 1,200-row snapshot; the 7,200-row rebuild changes instance contents
but not the templates or the grader.)

Second run on the 7,200-row snapshot (n=24, seeds 2-3, `_units/23_schemamap-k3v2/`): 17/20, all 20 hand-checked,
**FP 0, FN 0** (totals over both runs: 36/40, FP 0, FN 0). The three failures are agent errors against explicit
text: a descriptions-exchanged erratum applied backwards (storeys in one unit, uses in another; 3 of the 4 failures
over both runs are this), and the record-created decoy taken as `filed_date` in two sources that have no filing-date
column (gold null; the dictionary says "created in the partner's system"). No gold or verifier change.

### Calibration audit (rows-calA, seed 9100)
Checked all 3 graded-wrong units that exist in parsed answers (n=12 and n=80 answers all scored 1.0) and 5
graded-right units (re-derived from dictionary, errata and values): **FP 0, FN 0** at unit level. The wrong units are
agent errors: two ignored descriptions-exchanged errata (ZIP vs neighborhood; estimated vs revised cost) and one
named a column that does not exist in that source. 8 runs scored 0 with no final answer (budget ran out on a
progress line). **Parse-policy issue:** two `task` runs that did answer scored 0 because
the single answer object did not parse: n=160 b44fa2 (one missing quote; per-source fallback would give 0.994) and
n=240 32fca8 (truncated after 230 sources; fallback 0.954). Fixed (coordinator-approved): `parse_answer` now salvages complete per-source entries;
`_build/regrade.py` changes exactly these two rows (b44fa2 0 -> 0.994, 32fca8 0 -> 0.954). No-op and format-only
answers still score 0.
