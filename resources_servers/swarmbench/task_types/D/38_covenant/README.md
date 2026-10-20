# 38 covenant: financial covenant compliance review (family D, source R)

## Task
A bank's portfolio monitoring team checks N borrowers. For each borrower the agent reads four files (covenant excerpt
of the credit agreement, amendment file, correspondence file, three-year statement excerpt) and reports, for the
requested test period, every tested covenant's ratio (2 decimals, within 0.01 of exact) and pass/breach, plus overall
compliance. **Unit = one borrower**; a unit is correct only if every ratio, every pass flag and `compliant` are right.
Score = mean unit correctness (`score_partial` = mean fraction of right items).

Figures are **real**: each borrower is a real SEC filer (fictional alias; the CIK is only in gold.json) whose FY2022-FY2024
reported values (USD thousands) come from the SEC XBRL frames API. The agreement terms are per-borrower parameters
over K3-written clauses; gold is computed by code (`type.compute`) from the same figures.

## Traps (>= 1 non-period trap on every borrower; 2/3 test a non-latest year)
- wrong period: the requested test period (correspondence) is often not the latest column
- finance leases in or out of Consolidated Total Debt (original definition, or an amendment flips it)
- cash netting cap binds; add-back cap (% of operating income + D&A) binds; only listed add-backs count
- step-down leverage schedule keyed on the test-period end date
- executed amendments apply only to periods ending on/after their date; latest governs; drafts have no effect
- equity cure: counts only with a cure right, within the window, for that period, and only for net leverage
  (cure-bearing borrowers get a leverage threshold between the cured and uncured ratio, so the decision flips)
- distractors: revenue, net income, operating lease liabilities, routine correspondence

## Tiers (`TIERS`)
easy 8 / 1800 s, medium 24 / 1200 s, hard 90 / 1200 s (pool 103 companies; a careful unit is 4-5 reads plus
calculator calls, 7.7 tool calls and ~100 s per unit in the single-unit K3 loop). Why one context cannot do hard: 90 borrowers x ~4 file reads (~1.5-3k chars each) is ~360 tool
results (~600k chars) plus arithmetic; sized so n x K3 solo seconds per unit is about 3x the budget.

## Tools
`list_borrowers(offset, limit)`, `read_agreement|read_amendments|read_correspondence|read_statements(borrower_id)`,
`calculator(expression)`. Read-only, bounded (8k), offline, no search tool that could compute anything.

## Answer
One JSON object keyed by borrower id: `{"BR-0001": {"net_leverage": {"ratio": 2.31, "pass": true},
"interest_coverage": {...}, "compliant": true}}`. Tolerant parsing (code fences, `BR0001`, string numbers,
pass/breach/yes/no).

## Baselines (check 2, max over n = 8/24/90, seeds 0,1; see checks.json)
- regex (latest column, headline lines, no definitions/amendments/cure): 0.011
- gross_leverage (no cash netting, otherwise perfect): 0.078; adjacent_year_column: 0.00; shifted ids: 0.011
- partial readers (diagnostic, expected > 0.1): naive_reader 0.14, headline_figures 0.17, latest_year 0.63,
  no_amendments 0.75, amendment_dates_ignored 0.75, other single-rule skips 0.83-0.92

## Unit check (check 5)
K3 (pool replica, single-unit tool loop, n=32 seeds 0,1): 19/20. The one miss (s0 BR-0010) is a model error: K3
applied an equity cure although that agreement has no cure right; gold re-derived by hand from the files matches.
- format-only answers: 0

## Sources
- SEC EDGAR XBRL frames API, `https://data.sec.gov/api/xbrl/frames/us-gaap/{tag}/USD/{CY2022..CY2024|CY2022Q4I..CY2024Q4I}.json`,
  21 tags (data/SOURCES.json lists them and the sha256 of all 63 raw files). U.S. government data, public domain;
  fetched 2026-10-14 with a generic User-Agent at < 5 req/s.
- Snapshot `data/companies.json` (103 calendar-year filers with all required lines; `build_snapshot.py`):
  sha256 `030c04c95d49205ce005a92885eba6ca54a869289855ada89515271a43564587`. Raw files stay in data/raw (not shipped).

## Provenance
All task-facing prose (intro, rules, answer format, every clause/amendment/correspondence template, statement labels,
tool descriptions, unit prompt) is from self-hosted Kimi K3 (`policy_model`, moonshotai/Kimi-K3, family-C pool replica
http://10.49.67.132:8000/v1 on aga), one call, 239.8 s, 2,291 prompt / 8,741 completion tokens, 2026-10-14 16:18 UTC.
The exact prompt is `text/request.json` (sha256 `08839d8a...`), raw reply in `text/k3_raw.json`, frozen output
`text/k3.json` (sha256 `4cafb3bd0ff19b25f56d71abe89daf9b0828ce4a3833f900fdf33004bc4dcfaf`). `text/dev_stub.json`
(Claude, tests only) never ships. Claude wrote code, verifier and tests; borrower aliases come from code word lists.
Canary: first line of every prompt and `world/CANARY.txt`.

## Audit fix (2026-10-14)
The check-4 audit found that the original cure_notice template ("Notice dated {date}: ...") did not say when the cure
money was received, while gold treats {date} as the receipt date (5 of ~720 audited units). One self-hosted K3 call
(pool replica http://10.49.64.27:8000/v1, 11.6 s, first attempt valid) rewrote that one template from
text/request_cure.json (sha256 21211566bfbe...); the frozen reply text/k3_cure.json (sha256 8dcefe30fa40...) overrides T["cure_notice"] in
type.py. Gold logic is unchanged; oracle 1.0, check 2 unchanged, new instance hashes in checks.json (calibration in
RESULTS.md predates the fix).
