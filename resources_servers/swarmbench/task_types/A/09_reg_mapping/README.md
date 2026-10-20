# A09 regulation mapping: trace plain-language procedures to federal regulation paragraphs (family A, source R)

## Task
A consumer-finance company's operating manual has many procedure clauses (`K0001`...). Each clause was written to
implement exactly one first-level paragraph of ten parts of the US Code of Federal Regulations (Regulations B, E, P,
V, X, DD; COPPA; the FTC Safeguards and Identity Theft rules; the Bank Secrecy Act rules for banks). For each clause
the agent cites the paragraph, e.g. `12 CFR 1024.35(e)`; a deeper citation inside it (`1024.35(e)(1)`) is accepted.
One unit = one clause.

The clauses are written as everyday staff procedures in plain business vocabulary (customer, home loan, mistake,
letter, keep, look into) with every number kept, and never name the regulation. So keyword search over the
regulations does not find the paragraph: the agent must work from the table of contents and section text and match
the clause's actors, triggers, deadlines and amounts to the right paragraph, often among parallel provisions (error
resolution vs information requests, initial vs annual vs revised privacy notices, furnisher vs reseller duties,
different account types). A first version written in the regulations' own vocabulary was dropped: BM25 with the
clause as query hit the source paragraph 95% of the time; the plain-language version 1 / 16 in a probe.

## Tiers
easy 6 / 1800 s, medium 60 / 1200 s, hard 180 / 1200 s. Single clauses take K3 ~90 s and ~5 tool calls; in a batch K3
reads ~46 sections for 60 clauses (several clauses share a section), so solo K3 covered 60 in 1200 s at 0.95 (v2
probe) and hard was raised to 180 (most of the pool).

## Tools (read-only, bounded, offline)
`list_clauses(offset, limit<=80)`, `list_sections(part, offset, limit<=250)` (table of contents with headings and the
part names), `read_section(section, start)` (paragraphs with designations, ~7k chars per page), `search(query,
limit<=15)` (BM25 over all paragraphs; citation + start of text).

## Answer and scoring
`{"K0001": "12 CFR 1024.35(e)", ...}`; the section and first-level paragraph are compared (title number optional).
`score` = fraction of clauses right; `score_partial` = right section; `score_exact` = all right.

## Sources and gold
- eCFR full text, versioner API at date 2026-10-01 (`https://www.ecfr.gov/api/versioner/v1/full/2026-10-01/
  title-N.xml?part=P`), public domain (US Government work); per-file sha256 in `data/SOURCES.json`.
- `data/regs.json` (`_build/ecfr_parse.py`): 206 regulation-body sections with paragraph designations (appendices and
  official interpretations excluded). Units: 419 first-level paragraphs of 250-3,500 characters outside definition,
  scope, authority, effective-date, model-form and similar sections.
- Clauses (`_build/reg_prep.py clause2`): one per unit, written by self-hosted K3 from the paragraph and its section
  heading (temperature 0.6). Gold = the source paragraph.
- Verification (`_build/reg_prep.py verify`, `_build/finalize_reg.py`): a blind K3 pass picked, for each clause, one
  of 8 candidate paragraphs (the source, up to 4 sibling paragraphs of the same section, and the paragraphs most
  similar to the source); 358 / 370 parsed clauses mapped back correctly.
- Adversarial lexical filter: clauses whose source paragraph is among the first 3 distinct paragraphs BM25 returns
  with the clause as the query (the agent's own search tool) are dropped (133). Dropping only top-1 hits left a
  top-2 baseline at 0.2-0.3.
- Twin filter (v3, after the verifier audit): units whose paragraph has a near-twin (tf-idf cosine >= 0.8, e.g.
  1016.7(d)/(f) differ only in which institutions they cover; 1022.136(f)/1022.137(d); 1022.31(a)/1022.32(a)) are
  dropped (16): a plain-language clause cannot tell them apart. Pool = 209 clauses over all 10 parts.
- The oracle is the generation key.

## Provenance
All task-facing prose (prompt, tool descriptions, unit prompt, and every clause) is by self-hosted Kimi K3; the
regulation text is the real eCFR text. Canary in every prompt and world; `SNAPSHOT.txt` records the regs.json hash.
