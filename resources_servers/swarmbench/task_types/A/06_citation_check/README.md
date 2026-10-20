# A06 citation check: audit every citation of a long literature report (family A, source R/P)

## Task
The agent gets an NLP literature report (paragraphs `P001`..., sentences `S0001`...) whose every sentence cites one
paper of the report's bibliography (`[R0123]`; titles and abstracts of real arXiv cs.CL papers). For each sentence it
gives a verdict:
- `supported`: the cited paper's abstract states it (paraphrase is fine);
- `unsupported`: the cited paper is the right one but the sentence misstates it, and no paper in the bibliography
  states it;
- `wrong_source`: the sentence accurately states a DIFFERENT paper of the bibliography (a topically close one); the
  agent must also give that paper's id as `correct_ref`.
One unit = one paragraph of 5 sentences; it counts only if all 5 verdicts (and every correct_ref) are right. Mix per
sentence: 45% supported, 40% unsupported, 15% wrong_source (seeded per paper). The bibliography also holds 1 uncited
decoy paper per paragraph, so the true source of a misattributed sentence has to be found by reading.

The misstatements are built only from words, names and numbers of the same abstract (swapped winner, number attached
to the wrong metric, swapped datasets or languages, reversed effect, limitation turned into a strength), so they look
lexically like faithful summaries. A first version that let K3 substitute new words was dropped: a word-overlap judge
scored 0.20-0.35 on it.

## Tiers
easy 4 / 1800 s, medium 40 / 1200 s, hard 150 / 1200 s (paragraphs of 5 sentences; hard = 750 citations, 959
bibliography entries). v1 (700 papers, 4 / 25 / 75) was too small: K3 `task` scored 0.99 and GLM-5.3-Flash solo 0.91
at 75; a paragraph takes K3 ~75 s as a single unit but ~10-30 s inside a batch (abstracts are read 6 at a time).

## Tools (read-only, bounded, offline)
`read_report(offset, limit<=20)` (paragraphs), `list_refs(offset, limit<=120)`, `read_refs(ids<=6)` (title +
abstract), `search_refs(query, limit<=15)` (BM25 over the bibliography).

## Answer and scoring
`{"S0001": {"verdict": "supported"}, "S0002": {"verdict": "wrong_source", "correct_ref": "R0456"}, ...}` (a string
`"wrong_source:R0456"` is also accepted). `score` = fraction of paragraphs right; `score_partial` = fraction of
sentences right; `score_exact` = all right.

## Sources and gold
- arXiv metadata via OAI-PMH (`https://oaipmh.arxiv.org/oai`, set cs, records stamped 2024-03-01 to 2024-04-25),
  primary category cs.CL, ids 24xx, abstracts of 600-2,200 characters: 1,479 papers (`data/refs.jsonl`, with the 8
  nearest neighbours by tf-idf). arXiv metadata (titles, abstracts) is released CC0; per-file sha256 of the raw XML in
  `data/SOURCES.json`.
- Citing sentences: self-hosted K3 wrote, for each of the 1,479 papers, one faithful sentence and one misstatement
  (`_build/cite_prep.py gen`; misstatement prompt v2, temperature 0.6).
- Verification (`_build/cite_prep.py verify`, `_build/finalize_cite.py`): blind K3 judgements. A faithful sentence is
  kept only if judged supported by its own abstract; a misstatement only if judged unsupported; a paper can be the
  true source of a wrong_source citation only if its faithful sentence is judged unsupported by its nearest
  neighbour's abstract (which is then the cited paper).
- Kept: 1,479 faithful (all judged supported), 1,461 misstatements (of 1,477), 1,477 neighbour pairs. Pool = 1,479
  papers (`data/pool.jsonl`, stats in `data/build/filter_stats.json`). One uncited decoy per paragraph.
- The oracle is the construction key.

## Provenance
All citing sentences and the task text are by self-hosted Kimi K3; titles and abstracts are real arXiv metadata.
Canary in every prompt and world.
