# B13: Type-annotation pass (mypy --strict per module, real repos)

## Task
The world holds four real, strictly typed packages: click 8.1.7, itsdangerous 2.2.0, jinja2 3.1.4 and markupsafe
2.1.5 (`repos/<name>/src/<pkg>`). In `n` modules (`UNITS.txt`) the annotations of up to 12 functions/methods
each were stripped. The agents restore them so that `mypy --strict` (Python 3.12, vendored mypy 1.11.2) reports
no new errors, without changing behavior.

## Units, oracle and hidden checks (build_catalog.py)
- Oracle = the real module. "Before" = the module with the parameter/return annotations and the in-body annotated
  assignments of the selected functions removed by byte-exact AST edits (one group per top-level function or
  method, nested defs included). Class bodies and module-level annotations stay (dataclass/NamedTuple fields need
  them at run time). The seed picks which 12 groups are stripped in a larger module.
- A module is a candidate if it has >= 3 groups, the fully stripped module keeps the package's whole test suite
  passing (stripping has no run-time effect), and mypy reports new errors. 42 of 46 candidates qualify
  (click `_winconsole.py` is dropped: Windows-only, mypy skips it).
- Hidden tests per unit: (1) a generated test that runs `mypy --strict --no-warn-unused-ignores` on the whole
  package with every other file at the oracle and fails on any error not in the oracle's baseline (click 1,
  jinja2 4, markupsafe 2 baseline errors, compared as (file, code, message) multisets, line numbers ignored);
  (2) the package's real test suite (click 586, itsdangerous 297, jinja2 851, markupsafe 36 tests, all passing
  twice on the oracle).
- Static check per unit (anti-shortcut): no inline `# mypy:` config; the number of `Any`, `type: ignore` and
  `cast(` occurrences may not exceed the oracle module's.
- Visible tests: the same mypy test (it lists the new errors) plus 16 sampled real tests.

## Scoring
Same engine as all B types: each unit in its own sandbox (oracle tree + the agent's file), hidden tests in a
subprocess (300 s timeout). Controls = the suites of packages with no selected unit, run on the agent's whole
tree. `score = max(0, units_passed - control_files_broken) / n`.

## Tiers (seed 0)
| tier | n | budget s | unit lines | unit chars (~tokens) | stripped functions |
|---|---|---|---|---|---|
| easy | 6 | 1800 | 4,417 | 145k (43k) | 72 |
| medium | 16 | 1200 | 9,226 | 296k (89k) | 170 |
| hard | 42 | 1200 | 25,394 | 846k (256k) | 413 |
Generation 1-2 s; scoring 15 s (n=6) to 55 s (n=42) on 8 cores.

## Baselines (`baselines.py`, seed 0)
| baseline | n=6 | n=16 | n=42 |
|---|---|---|---|
| no-op | 0 | 0 | 0 |
| oracle | 1 | 1 | 1 |
| all `Any` | 0 | 0 | 0 |
| heuristic annotator (literal-default types, `-> None` if no return, `object` otherwise) | 0 | 0 | 0 |
| partial (oracle annotations for half of the stripped functions) | 0 | 0 | 0 |
| stub (oracle annotations, stripped bodies replaced by `raise NotImplementedError`) | 0 | 0 | 0 |

## Provenance
- Code: click 8.1.7 (874ca2bc), itsdangerous 2.2.0 (096c8d42), jinja 3.1.4 (dd4a8b54), markupsafe 2.1.5
  (fbba4acd), all BSD-3-Clause (Pallets); URLs and sha256 in `../repos/SOURCES.json`. mypy 1.11.2 (MIT),
  mypy_extensions, typing_extensions vendored in `../_vendor` (sha256 in `SHA256SUMS`).
- Task text: `prompt.txt` written by Kimi K3 (hosted) from `k3/prompt_instruction.txt`; output and cost in
  `k3/prompt_k3.json`. No other task-facing text.
- Canary line first in every prompt and in `CANARY.txt`.
