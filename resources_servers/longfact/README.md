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

The primary benchmark pairs `simple_agent` with the standard one-turn
environment server and stateless Resources Server `/verify` endpoint.
Verification returns the saved policy response, reward and claim diagnostics
together with the grading observations.

This server and its held-out prompts are for evaluation only. Like upstream
SAFE, scoring does not deduplicate extracted claims; repeated claims can
inflate F1@100 when extracted more than once, making this reward unsuitable
for RL training. Prepared and vendored prompts retain their upstream canary
and must not enter training corpora.

For Milvus, add `'+config_paths=[benchmarks/longfact/milvus.yaml]'` to select
the backend and install its optional dependencies through
`component_extras: {resources_servers/longfact: [milvus]}`. The default Tantivy
and Tavily configurations do not install that extra. See the
[benchmark guide](../../benchmarks/longfact/README.md#milvus) for backend setup.

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
claims. Unusable judge replies, including incomplete or filtered responses and
exhausted malformed verdicts, fail verification through the judge-failure path.
Retrieval errors also fail verification rather than becoming `UNSURE`. These
failures preserve the saved answer and provenance for recovery.

Tavily `read_write` mode uses a persistent, request-addressed cache. Entries
are atomically published with response checksums and cross-process file locks.
A corrupt entry is moved under `.corrupt/` and fetched again; replay mode
instead reports corruption without changing the cache. Legacy unchecksummed
entries remain readable in replay mode, but `read_write` mode quarantines and
refreshes them before use.
