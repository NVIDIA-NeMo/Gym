# LongFact SAFE resource server

The server evaluates one LongFact response. It extracts atomic claims, makes
each claim self-contained, filters claims irrelevant to the original prompt,
generates up to three verification queries, retrieves the top three results
per query, and labels each remaining claim as `SUPPORTED`, `CONTRADICTED`, or
`UNSURE`. Three-way mode is the default; set
`LONGFACT_CLAIM_VERDICT_MODE=binary` only for historical comparisons. The
default provider is local Tantivy retrieval backed by `TANTIVY_INDEX_DIR` and
`CORPUS_DB_PATH`; set `LONGFACT_SEARCH_BACKEND=tavily` for live Tavily search
or `LONGFACT_SEARCH_BACKEND=milvus` for retrieval from a compatible vector
collection. Backend endpoints and credentials must be supplied at runtime.

Its reward and aggregate metric use LongFact SAFE's F1@100 calculation. This
is not the official SAFE implementation: SAFE used Serper/Google Search and
free-form parsing, while this resource uses Tantivy, Tavily, or Milvus and
schema-validated NeMo Gym evaluator calls. Rollouts record the selected search
provider and per-claim evidence traces so they cannot be confused with
official scores.

Three-way mode retains the existing strict precision and F1@100 metrics by
collapsing contradicted and unsure claims into the legacy not-supported count.
It additionally reports contradicted and unsure counts, mean contradiction and
unsure rates, and decisive precision over only supported plus contradicted
claims. Retrieval/evaluator errors, including a judge that gives no parseable
verdict, remain `ERROR`, not `UNSURE`, and an episode with an `ERROR` claim
fails its rollout so that resume runs it again.

Tavily `read_write` mode uses a persistent, request-addressed cache. Entries
are atomically published with response checksums and cross-process file locks.
A corrupt entry is moved under `.corrupt/` and fetched again; replay mode
instead reports corruption without changing the cache. Legacy unchecksummed
entries remain readable in replay mode, but `read_write` mode quarantines and
refreshes them before use.
