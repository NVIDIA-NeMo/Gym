# B19: Internationalization pass (real gettext code, catalog-exact msgids, hidden tests)

## Task
The world holds click 8.1.7 and humanize 4.9.0, both of which translate their user-facing messages with gettext.
In `n` functions (`UNITS.txt`, one `path::qualified.name` per line) the translation calls were removed and the
messages hard-coded. The agents put every message back through the catalog with the catalog's exact msgids while
keeping the English behavior. Catalogs: `repos/click/messages.pot` (extracted from the real click code: msgids,
plural forms, file references, no line numbers) and humanize's real `.po` files.

What the "before" code looks like (i18n_rules.inverse_edits), and what the agent has to recover:
- `_("Missing argument")` became `"Missing argument"`;
- `_("No such option: {name}").format(name=option_name)` became `f"No such option: {option_name}"` (the catalog's
  placeholder names must be recovered and the message translated before formatting);
- `ngettext(s, p, n)[.format(...)]` became `(s' if n == 1 else p')`;
- humanize `P_("1 (male)", "st")` became `"st"` (the msgctxt must be recovered from the .po file);
- humanize `_ngettext("%d day", "%d days", days) % days` became `("%d day" if days == 1 else "%d days") % days`.
Sites at module or class level, and functions that contain a nested function with sites, are left translated.

## Units, oracle and hidden checks (build_catalog.py)
- Oracle = the real function. Each unit is one function; it is scored by splicing only that function from the
  agent's file into the oracle file (i18n_rules.splice), so other functions of the same file do not interfere.
- Static check: the multiset of translation calls with literal msgids in the function (kind + msgid [+ plural or
  context]) must equal the oracle's. The message lists missing and unexpected msgids.
- Runtime check: a pytest plugin wraps `gettext.gettext/ngettext/pgettext` and the `NullTranslations` /
  `GNUTranslations` methods and records, per test, every translation call with its calling module and
  `co_qualname`; the sequence must equal the oracle's (recorded twice, stable). Translating an already formatted
  string or calling with the wrong msgid changes the trace.
- humanize's real translations: every `.po` is compiled to `.mo` in the hidden test tree (msgfmt in
  build_catalog.py), so its locale tests (ru_RU, fr_FR, de_DE, ...) check that the msgids and contexts really hit
  the translators' entries.
- Hidden tests per unit = the package's whole real suite (click 586, humanize 550 tests), with the trace check.
- Validation: with any one function reverted, no test outside the locale tests changes (the inverse preserves the
  English behavior) for all 49 candidates.
- Guessable units dropped (`filter_guessable.py`): those that the no-op, a catalog-guided regex (wrap every exact
  catalog string in `_()`) or a translate-after-format rewrite already pass: 13 of 49. 36 units remain.

## Scoring
Same engine as all B types (per-unit sandbox, vendored pytest, no site-packages). Controls = suites of packages
with no selected unit, on the agent's whole tree. `score = max(0, units_passed - control_files_broken) / n`.

## Tiers (seed 0)
| tier | n | budget s | files | function lines | hard-coded sites |
|---|---|---|---|---|---|
| easy | 6 | 1800 | 6 | 238 | 29 |
| medium | 14 | 1200 | 6 | 478 | 37 |
| hard | 36 | 300 | 9 | 1,186 | 92 |
Generation < 1 s, scoring 20-35 s. The hard budget was cut from 1200 s to 300 s after calibration (solo K3 finished
all 36 functions in ~16 min); see ../RESULTS.md.

## Baselines (`baselines.py`)
| baseline | n=6 s0/s1 | n=14 | n=36 |
|---|---|---|---|
| no-op | 0 / 0 | 0 | 0 |
| oracle | 1 / 1 | 1 | 1 |
| catalog-guided regex (wrap exact catalog strings) | 0 / 0 | 0 | 0 |
| translate after formatting (`_(f"...")`, `ngettext` over f-strings) | 0 / 0 | 0.07 | 0 |
| partial (oracle except the first site of each function) | 0 / 0 | 0 | 0 |

## Provenance
- Code and catalogs: click 8.1.7 (BSD-3-Clause, commit 874ca2bc), humanize 4.9.0 (MIT, commit 35e2d21b; its
  `.po` files are the real community translations). URLs and sha256 in `../repos/SOURCES.json`. freezegun,
  python-dateutil and six are vendored for humanize's tests (`../_vendor/SHA256SUMS`).
- Task text: `prompt.txt` by Kimi K3 (hosted) from `k3/prompt_instruction.txt` (`k3/prompt_k3.json`).
  `messages.pot` is generated from the real click source (msgids are real strings). No Claude-written task text.
- Canary line first in every prompt and in `CANARY.txt`.
