# FACTS Search V2 method ledger

The source of record is the public [FACTS Leaderboard](https://www.kaggle.com/benchmarks/google/facts), its Search V2 page, and the linked implementation notebook `aminmohamedmohami/facts-search-on-implementation` version 3.

| Public method | Status | Exact contract |
| --- | --- | --- |
| Search-On | Implemented | Exact published system prompt; Brave Web Search only; five results per query; multiple parallel searches per hop; seven search hops followed by the published forced-final instruction; one-sentence final answer; A/B/C judge; public F1, accuracy, attempted accuracy, hedging rate, average search hops, and 1,000-resample bootstrap CI. |

There are no other public FACTS Search methods on the current leaderboard. Search-Off is a dataset-construction/adversarial-filtering control described in the paper, not a scored V2 leaderboard method.

## Public artifact gaps

- The V2 page says 1,842 questions split evenly (921 public, 921 private), but its linked public dataset remains version 1 with exactly 890 rows. The 31 additional advertised public rows are not published through the linked dataset. Preparation pins and emits all 890 available rows and fails on drift.
- The entire 921-row private split is intentionally hidden by Kaggle.
- The linked CSV contains only `example_id`, `problem`, and `gold answer`; Google does not publish the subset label for each row, so per-subset metrics cannot be reconstructed without inventing metadata.
- The current page identifies Gemini 3.5 Flash as the grader. The linked notebook still names Gemini 2.0 Flash, which is retired for V2. As of 2026-10-28 the Gemini API serves `gemini-3.6-flash` for every `gemini-3.5-flash` request and offers no pinned 3.5 version, so 3.5 Flash can no longer be reproduced. This adapter defaults to `gemini-3.8-flash`, the newest Flash model, which is served under its own name, and keeps the notebook’s public grader prompt unchanged. Scores graded by 3.5 Flash and by 3.8 Flash are not directly comparable.
