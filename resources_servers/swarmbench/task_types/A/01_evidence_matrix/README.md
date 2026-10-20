# A01 evidence matrix: effect directions for many clinical-trial reports (family A, source A/R)

## Task
A literature matrix lists rows of (article, INTERVENTION, COMPARATOR, OUTCOME) for many randomized controlled trial
reports (PubMed Central open-access full texts). For every row the agent fills in what the article reports for the
outcome in the intervention group relative to the comparator: `increased` (significantly higher), `decreased`
(significantly lower) or `no_difference`. One unit = one article with 3-5 rows; an article counts only if all of its
rows are right.

Traps are natural to the data: direction is about the value of the measure, not benefit (less pain = decreased);
the prompt's intervention/comparator order can be the reverse of the article's; many results sit only in the body or
in flattened tables (about 45% of the source annotations were made outside the abstract); significance must be read
from p-values or confidence intervals; and every article mixes at least two labels, so no constant answer gets an
article right.

## Tools (read-only, bounded to 8k chars, offline)
`list_rows(offset, limit<=100)`, `list_articles(offset, limit<=200)` (id, title, length),
`read_article(article, offset, limit<=7000)` (character window). No tool reveals a label.

v1 also had `search_article(article, query)` (best passages of one article). It made a unit one call: solo K3 did 80
articles in 1200 s at 0.84 with 1.7 tool calls per article, and Qwen3.8-27B / GLM-5.3-Flash matched it at medium.
v2 removed it (articles must be read, ~4 windows each) and keeps only articles with at least one row whose annotators
needed more than the abstract (231 of the 329 filtered articles).

## Answer and scoring
One JSON object `{"R0001": "decreased", ...}`; label synonyms such as "significantly increased" are accepted.
`score` = fraction of articles with all rows right; `score_partial` = fraction of rows right; `score_exact` = every
article right.

## Tiers
easy 5 / 1800 s, medium 60 / 1200 s, hard 120 / 1200 s (articles; ~4 rows each). A single article takes K3 ~50 s and
5 tool calls; inside a batch solo K3 reads ~60 articles in 1200 s (0.84 at 60), so hard = 120 (solo 0.65, `task`
0.95). A 200-article probe left both K3 arms mostly without an answer at the hard limit (see RESULTS.md).
Generate takes < 0.1 s.

## Why one context cannot do the hard tier
120 articles x ~26k characters is ~0.8M tokens of reading and ~480 row judgments, several times one context window;
solo K3 compacts and runs out of time, while articles are independent and split cleanly across subagents.

## Sources and gold
- Evidence Inference 2.0 (DeYoung et al., BioNLP 2020), `http://evidence-inference.ebm-nlp.com/v2.0.tar.gz`,
  sha256 `6abe0d4e...1586`, MIT license (code and annotations); articles are PubMed Central open-access texts as
  distributed in the release. Recorded in `data/SOURCES.json`.
- Candidate prompts (`_build/build_ei_pool.py`): >= 2 annotators gave the same label and the verifier accepted it;
  prompt ids on the release README's incorrect / questionable / malformed lists dropped; duplicate ICO triples
  dropped; articles with >= 3 such prompts and >= 2 distinct labels (1,233 articles; the first 343 by PMCID were K3-labelled).
- K3 filter (`_build/ei_prep.py`, `_build/k3batch.py`, `_build/finalize_ei_pool.py`): self-hosted K3 read each whole
  article (no tools) and labelled its prompts; a prompt is kept only if K3 agrees with the annotators, an article only
  if >= 3 prompts with >= 2 labels remain. 343 articles labelled, 1,629 / 1,690 prompts agree (96.4%); pool = 329
  articles, 1,602 prompts (`data/pool.jsonl.gz`; stats and a disagreement sample in `data/build/filter_stats.json`).
  Inspected disagreements are mostly gold noise or genuinely ambiguous prompts (e.g. "improvement in PSQI" labelled
  increased while the score fell; hazard ratios read in the wrong direction; baseline-as-comparator prompts).
- The oracle is this key (real annotations cannot be re-derived by code); it checks the parser/scorer round trip.

## Provenance
Task-facing prose (prompt, tool descriptions, unit prompt) was written by self-hosted Kimi K3 (`policy_model`,
moonshotai/Kimi-K3 rev 9f62e4e9, the shared pool in `../../K3_POOL.md`) via `../k3gen.py` from `text/request.json`;
frozen in `text/k3.json` (raw reply and timing in `text/k3_raw.json`). Article text, ICO strings and ids are real.
`text/dev_stub.json` is Claude-written development text and never ships (check 9 requires `text_source == "k3"`).
Every prompt starts with the SwarmBench canary line; every world holds `CANARY.txt`.

## Files
`type.py`, `solutions.py`, `tests/test_type.py`, `checks.json`, `text/`, `data/pool.jsonl.gz`, `data/SOURCES.json`,
`data/build/` (candidates, K3 requests/outputs, filter stats); build scripts in `../_build/`.
