# 37 a11y: accessibility audit of static pages (family D, source R)

**Task.** An accessibility team audits N archived web pages (real English Wikipedia articles and Python 3.13
documentation pages, sanitized and lightly edited) against a seven-rule rulebook and reports, per page, the exact
offending elements of every rule.

**Unit** = one page. The answer for a page has seven entries: `img_alt`, `control_name`, `link_name`,
`heading_skip`, `table_header` (lists of `data-ref` values), `duplicate_id` (id values) and `lang_ok` (bool). A page
counts only if all seven are exactly right. Score = fraction of pages right (`score_partial` = fraction of entries).

**Gold.** `a11ydom.check` (exact, deterministic) runs on the page text exactly as the tools serve it. The rulebook
(K3 prose) states the same definitions: hidden = `hidden` / `aria-hidden="true"` / inline `display:none` or
`visibility:hidden` on the element or an ancestor; accessible-name sources per element kind; generic link phrases
(K3's list: click here, read more, learn more, here, more, this, link, continue, details); heading levels may rise by at most one; duplicate ids case-sensitive incl. hidden; lang well-formedness
only; tables need their own th unless role=presentation/none. Contrast was left out: inline colors with inheritance
and real CSS cannot be made exact without a renderer.

**Real offenders + seeded traps.** Real pages already fail (Wikipedia figure images without alt inside links, Python
docs sidebars starting at h3, tables without th). Each page also gets 11-18 seeded kinds (header nav, aside, footer form
and tables built from K3 phrase banks, plus edits of real elements):
- images: decorative alt="", role=presentation/none, hidden by each hiding mechanism, aria-hidden="false" (hides
  nothing), aria-labelledby to an existing / missing id, file-name alt, title only, no alt;
- form controls: label for= far away on the page, wrapping label, placeholder only, aria-labelledby ok / dangling,
  hidden inputs, controls in hidden containers, submit/reset, input type=button with / without value, select without
  label, title, label for= pointing at the wrong id, empty label, wrapping label that has its own for=;
- links/buttons: generic text in varied case and trailing punctuation, generic text overridden by a descriptive
  aria-label (ok), descriptive text overridden by a generic aria-label (offends), image links with alt / alt="", anchors
  without href, hidden links, title-only links, "generic + more words" (ok), icon buttons with aria-hidden glyphs with /
  without aria-label, aria-labelledby;
- structure: alt stripped from a real image, a real offending image hidden, a real h2/h3 raised two levels, a real
  table's th turned into td, hidden heading "bridges" (h(p+1) hidden then h(p+2) visible: offends) and hidden skips
  (exempt), duplicate ids copied from real ids / inside hidden elements / differing only in case (not duplicates),
  layout tables, header-less data tables, nested tables, hidden tables, valid and malformed lang values.

`tests/test_type.py::test_kind_intent` checks every seeded kind's intended effect against the checker.

**Why one context cannot do the hard tier.** A page is ~11.8k characters (~180 lines, ~70 links); a careful audit
needs 2 reads plus a full pass over every link, control, heading and table. Single-unit K3 takes ~147 s per page
(3 tool calls); at ~60 s per page for OpenCode solo, 56 pages are ~3,400 s of work (2.8x the 1200 s budget) and
~660k characters (~180k tokens) of HTML, more than one context reads carefully.

**Tiers** (`TIERS`): easy 4 pages / 1800 s, medium 16 / 1200 s, hard 56 / 1200 s (pool: 92 pages; n <= 92; unit i
uses pool page perm_seed[i], so smaller n is a prefix of larger n).

**Tools** (read-only, bounded to 8k chars, offline): `list_pages(offset, limit)`, `read_page(page_id, start_line)`
(numbered lines, ~7.4k chars per call), `search_page(page_id, query)` (<= 40 lines, cut to 300 chars). No tool runs a
check.

**Answer.** One JSON object `{page_id: {seven keys}}` at the end (code fences, upper-case refs, `"#id"`,
`"true"` tolerated).

## Checks
Offline (`python3 checks.py 37_a11y`, n = 4/16/56, seeds 0-1, `checks.json`): oracle 1.000 on all 6 instances;
format-only answers 0; tool robustness 0 errors; deterministic (sha256 c7e03bc1... for n=56 s0); canary in prompt
and world; text_source k3. Plausible-wrong (max over instances): `regex` 0.00 (it mis-grades 100% of pages at n=56;
per-entry partial 0.42), `shifted_pages` 0.00, `ignore_hidden` 0.09. Careless readers missing one exception kind
(partial by design): placeholder-as-label 0.59-1.0, no wrapping labels 0.44-0.75, labelledby unchecked 0.25-0.75,
file-name alt ok 0.5-1.0, aria-label ignored for links 0.25-0.5.

Check 5 (K3 single units, `../unitcheck.py`, n=16, seeds 0-1, 20 units): first run on 6.5k-text pages 17/20; pages
were then cut to 5k text characters and the rerun gave **18/20** (`../_units/37_a11y-k3v2`). Audit of all 5 failed
units across both runs (40 rollouts): every one is a model error (image-only link judged by the wrong image, a
role="navigation" value reported as a duplicate id, h5 -> h2 -> h3 flagged as skips); 0 checker errors.
Mean 147 s and 3.0 tool calls per unit; ~14k prompt + 3k completion tokens per unit. Baselines in `solutions.wrong`:
`regex` (tag regexes: img without alt=, controls without aria-label/title/label for, links with empty or generic
inner text, raw heading sequence, tables without `<th`) and `shifted_pages`; careless readers that each miss one
exception kind: `ignore_hidden`, `placeholder_is_label`, `no_wrapping_labels`, `labelledby_unchecked`,
`filename_alt_ok`, `link_content_first`.

## Sources
`data/SOURCES.json` (urls, licenses, raw sha256 of all 92 downloads, snapshot `data/pages.json` sha256 9eeb9d68...). Wikipedia: CC BY-SA 4.0,
revision ids pinned in `data/pages.json`. Python docs: PSF License Agreement for the documentation. The snapshot is
derived by `_build/snapshot.py` (run with `python3 -I`); raw downloads stay in `data/raw` (not shipped).

## Provenance
Claude wrote code only (`type.py`, `a11ydom.py`, `solutions.py`, tests, build script). All task prose (intro, the
seven rule paragraphs, hidden/scope definitions, answer and unit instructions, tool descriptions), the generic-phrase
list and every phrase bank used in seeded markup (labels, placeholders, buttons, link texts, icon names, alts,
captions, headings, notices, table cells/heads) were written by self-hosted Kimi K3 (`policy_model`, revision
9f62e4e9, family-C pool on aga) on 2026-10-14 (one attempt, 390 s) from the brief in
`text/request.json` (`user` field, verbatim) through `../k3gen.py`;
the frozen reply is `text/k3.json`, the raw call log `text/k3_raw.json`. `text/dev_stub.json` is a code-test stub only.
Page text is real (see Sources). Canary: first line of every prompt and `world/CANARY.txt`.
