# B11: API migration after a dependency bump (real repos, hidden tests)

## Task
The world holds two real projects: Python-Markdown 3.6 (`repos/markdown`, the dependency, with its bundled
extensions and `docs/changelog.md`) and PyMdown Extensions 10.8.1 (`repos/pymdownx`). `n` extension modules
(listed in `UNITS.txt`) are rewritten to the extension API that Markdown 3.4 removed (deprecated since 3.0). The
agents migrate them to the 3.6 API with identical behavior. The other extension modules are already migrated.

Removed API that appears in the legacy modules (the real 3.4 changelog table is in the world):
`Registry.add(name, item, location)` with `'<name'`/`'>name'` locations, `reg[name] = item`, `del reg[name]`,
`extendMarkdown(self, md, md_globals)`, `markdown.util.etree`, `util.isBlockLevel`, `Processor.markdown`.

The trap: `register(item, name, priority)` takes a number. Each location has to become a priority that gives the
same position, so the agent looks up the priorities of the neighbors (in Markdown core or in the module itself).
`reg[name] = item` on an existing name keeps that item's priority.

## Units, oracle and hidden tests (build_catalog.py)
- Oracle = the real module. "Before" = the module rewritten by AST edits. A location is computed from the registry
  order recorded at each `register` call while the real test suites run; up to two equivalent locations
  (`'<next'` or `'>prev'`) are offered and the seed picks one.
- Every legacy module is validated as a correct program for the old API: the real tests run against Markdown 3.6
  with the 3.3.7 implementations of the removed API re-added (`recorder.py compat`). Outcomes and the registry
  orders after every `convert()` must match the oracle on every test that exercises the module, for both location
  variants (a module whose second variant differs keeps the first only). 9 of 36 candidate modules were dropped
  here (their old-API form is not order-equivalent in some multi-extension tests).
- Hidden tests per unit = the real tests (Markdown 996 and PyMdown 521 passing tests, each run twice for
  flakiness) that fail when only that module is legacy, plus a generated registry-order test (load the extension
  alone, compare the order of all five registries with the oracle). Every registry site must be discriminated:
  moving its item to either end of its registry must fail a hidden test, otherwise that site stays in new form.
- Units that a fixed-priority regex migrator already passes under every location variant were dropped
  (`filter_guessable.py`: codehilite, nl2br, smarty, highlight, snippets; tasklist). 22 units remain.
- Visible tests: the registry-order test plus every hidden real test when a unit has at most 24 (otherwise a
  sample of 16). Showing all of them exposes multi-extension priority ties (found in calibration: priority 25 in
  `_bypassnorm` ties with superfences' `fenced_code_block` when both load).
- PyMdown's `tests/test_syntax.py` parametrizes over `os.walk()` order, which differs between filesystems; the
  hidden copy sorts the walk (`swb_b_engine.sort_pymdownx_walk`) so `compareN` ids name the same case everywhere.
- Static check per unit: no removed API may remain (AST: `md_globals`, `util.etree`, `util.isBlockLevel`,
  `self.markdown`, `<registry>.add(...)`, `<registry>[...] =` / `del`).

## Scoring
Each unit is scored in its own sandbox: the oracle tree plus the agent's version of that one file, then its hidden
tests (subprocess, vendored pytest 8.3.3, 300 s timeout). Edits to other files are discarded for unit scoring.
Controls = every oracle-passing test in files that hold no hidden test of a selected unit; they run once on the
agent's whole tree and each broken control file costs one unit.
`score = max(0, units_passed - control_files_broken) / n`; `score_exact` = all units and no broken control.

## Tiers (seed 0)
| tier | n | budget s | unit lines | unit chars (~tokens) | edit sites | hidden tests |
|---|---|---|---|---|---|---|
| easy | 5 | 1800 | 1,403 | 51k (15k) | 25 | 128 |
| medium | 11 | 1200 | 3,313 | 122k (37k) | 59 | 510 |
| hard | 22 | 600 | 5,681 | 206k (62k) | 98 | 814 |
Generation < 1 s, scoring 5-15 s. The hard budget was cut from 1200 s to 600 s after calibration (solo K3 solved 22
units in 1200 s); see ../RESULTS.md.

## Baselines (`baselines.py`)
| baseline | n=5 s0/s1 | n=11 s0/s1 | n=22 s0/s1 |
|---|---|---|---|
| no-op | 0 / 0 | 0 / 0 | 0 / 0 |
| oracle | 1 / 1 | 1 / 1 | 1 / 1 |
| regex (sed-style renames, priority 200/1/50 by location kind) | 0.20 / 0 | 0 / 0 | 0.05 / 0.05 |
| partial (oracle except one registry site per module) | 0 / 0 | 0 / 0 | 0 / 0 |
| delete (renames, drop registry calls) | 0 / 0 | 0 / 0 | 0 / 0 |

## Provenance
- Code: Python-Markdown 3.6 (BSD-3-Clause, commit e524b8fe), PyMdown Extensions 10.8.1 (MIT, commit 8ed2fcb8);
  Markdown 3.3.7 (db98a263) is used at build time only, as the legacy-API reference. URLs and sha256 in
  `../repos/SOURCES.json`. Test runtime: vendored pytest/pluggy/iniconfig/packaging/Pygments/PyYAML
  (`../_vendor/SHA256SUMS`).
- Task text: the prompt (`prompt.txt`) was written by Kimi K3 (hosted `nvidia/moonshotai/kimi-k3`) from the
  instruction in `k3/prompt_instruction.txt` (output and cost in `k3/prompt_k3.json`). The migration guide is the
  real `docs/changelog.md` of Markdown 3.6. No Claude-written text is task-facing.
- Canary: the SwarmBench GUID line opens every prompt and is in `CANARY.txt` of every world.

## Files
`type.py` (generate/tools/score), `build_catalog.py` + `recorder.py` + `filter_guessable.py` (build time),
`catalog.json` (units, edits, hidden test ids, registry orders), `baselines.py`, `tests/test_type.py`
(`python -m unittest 11_api_migration/tests/test_type.py` from `types/B`).
