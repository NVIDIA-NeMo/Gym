# B18: Security fix rollout: unescaped text in generated HTML (real repos, hidden tests)

## Task
The world holds four real HTML-producing packages: mistune 3.0.2, Python-Markdown 3.6, PyMdown Extensions 10.8.1
and jinja2 3.1.4 (+markupsafe). In `n` functions (`UNITS.txt`, `path::qualified.name`) an escaping call was
removed, so document or template text reaches the HTML unescaped (an injection hole). The agents find the missing
escaping in each listed function and restore it with the package's own helper, without double-escaping text that is
already HTML.

## Units, oracle and hidden checks (build_catalog.py)
- A site is a call to one of the packages' escaping helpers (`escape`, `escape_text`, `escape_url`, `safe_entity`,
  `html_escape`, `_escape`, `code_escape`, `_escape_attrib`, `_escape_cdata`) inside a function; "before" replaces
  the call by its first argument. Each of the 56 sites was removed alone and the package's real suite run
  hermetically against the real suite plus the escape probes below: 35 sites are caught; only those are used.
  Units = the 27 functions with a caught site (all caught sites of a function are removed together).
- Each unit is scored by splicing only that function from the agent's file into the oracle file.
- Hidden tests per unit: the package's whole real suite (mistune 941, Markdown 996, PyMdown 527, jinja 851 tests,
  each passing twice on the oracle) plus escape probes (`probes.py`: 11/8/4/13 crafted inputs full of `<&"'`
  through every construct the units touch; jinja under autoescape on and off; expected = the oracle's output,
  recorded twice). The probes were added after the verifier audit found a false positive: escaping the items of
  jinja's `join` filter passed every real test but double-escapes under autoescape. Static rule: the function makes at least as many escaping calls as the
  original. Visible tests: the tests that catch the unit's sites plus 16 sampled tests.

## Scoring
Same engine as all B types. Controls = suites of packages with no selected unit, on the agent's whole tree.
`score = max(0, units_passed - control_files_broken) / n`.

## Tiers (seed 0)
| tier | n | budget s | files | function lines | removed escapes | units per repo (mistune/markdown/pymdownx/jinja) |
|---|---|---|---|---|---|---|
| easy | 5 | 1800 | 4 | 105 | 5 | 4/0/1/0 |
| medium | 12 | 1200 | 8 | 353 | 16 | 5/2/2/3 |
| hard | 27 | 600 | 16 | 656 | 31 | 14/5/2/5 |

## Baselines (`baselines.py`)
| baseline | n=5 s0/s1 | n=12 | n=27 |
|---|---|---|---|
| no-op | 0 / 0 | 0 | 0 |
| oracle | 1 / 1 | 1 | 1 |
| escape everything concatenated or formatted into strings (sed-style) | 0 / 0 | 0 | 0 |
| escape late (wrap every return value) | 0 / 0 | 0 | 0.04 |
| right places, generic `html.escape` instead of the package's helper | 0.60 / 0.40 | 0.25 | 0.26 |
The last row is not a wrong fix: where the tests accept `html.escape`, the output is equally safe. It shows that the
task is finding where escaping is missing and what must not be escaped, not naming the helper.

## Limits
Small pool (27 units, 1-3 removed escapes each); functions are short, so a single unit is easy for a strong model.
The swarm signal comes from n and from spread across four packages.

## Provenance
- Code: mistune 3.0.2 (BSD-3-Clause, commit f997c863), Python-Markdown 3.6 (BSD-3-Clause), PyMdown Extensions
  10.8.1 (MIT), jinja2 3.1.4 and markupsafe 2.1.5 (BSD-3-Clause). URLs and sha256 in `../repos/SOURCES.json`.
- Task text: `prompt.txt` by Kimi K3 (hosted) from `k3/prompt_instruction.txt` (`k3/prompt_k3.json`).
- Canary line first in every prompt and in `CANARY.txt`.
