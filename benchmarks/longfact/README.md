# LongFact

This benchmark evaluates long-form factuality with the prompts from
[LongFact](https://arxiv.org/abs/2403.18802) and a SAFE-style verifier. It
extracts atomic claims, retrieves evidence, classifies each relevant claim as
`SUPPORTED`, `CONTRADICTED`, or `UNSURE`, and reports factual precision and
SAFE F1@100.

The implementation supports three interchangeable evidence backends:

- **Tantivy** for reproducible local lexical retrieval.
- **Milvus** for retrieval from a compatible vector collection.
- **Tavily** for live web search, with optional persistent caching.

These are not the original SAFE Serper/Google backend, so results should name
the selected backend and are not official SAFE leaderboard values.

## Datasets

The `longfact` dataset contains all 2,280 upstream prompts. It is downloaded
from a pinned upstream commit by the preparation script:

```bash
gym eval prepare --benchmark longfact
```

The `longfact_hard200` dataset is checked in at
`benchmarks/longfact/data/hard200.jsonl`. It contains 100 Objects and 100
Concepts prompts selected using contradiction counts from an earlier model
evaluation. The selection manifest and provenance are in the same directory.
Use this dataset when a stable 200-prompt stress test is desired.

## Configuration

Configure the policy and evaluator model servers in the usual NeMo Gym way,
then select a retrieval backend with `LONGFACT_SEARCH_BACKEND`.

The benchmark config uses a vLLM-served Qwen judge
(`benchmarks/longfact/evaluator_local_vllm.yaml`). For an OpenAI-compatible
endpoint instead, add `'+config_paths=[benchmarks/longfact/evaluator_openai.yaml]'`.
Both read `LONGFACT_EVALUATOR_BASE_URL`, `LONGFACT_EVALUATOR_API_KEY` and
`LONGFACT_EVALUATOR_MODEL`. With the OpenAI-compatible judge, set
`LONGFACT_EVALUATOR_RESPONSE_MODEL` to the model name the endpoint returns to
check the judge identity.

### Tantivy

```bash
export LONGFACT_SEARCH_BACKEND=tantivy
export TANTIVY_INDEX_DIR=/path/to/tantivy-index
export CORPUS_DB_PATH=/path/to/corpus.db
```

The index must store `doc_id` and `text` fields. `corpus.db` must provide the
document text expected by the provider.

### Milvus

```bash
export LONGFACT_SEARCH_BACKEND=milvus
export LONGFACT_MILVUS_URI=https://milvus.example.com
export LONGFACT_MILVUS_COLLECTION_NAME=documents
export LONGFACT_MILVUS_EMBEDDING_MODEL=google/embeddinggemma-300m
```

If the deployment requires authentication, provide
`LONGFACT_MILVUS_TOKEN` through the environment or your secret manager. The
evaluator does not contain or default to a private endpoint or credential. The
collection must contain the configured vector field (default `emb`) and the
`orig_id` and `text` scalar fields. The embedding model can be changed when the
collection was built with another compatible embedding model.

### Tavily

```bash
export LONGFACT_SEARCH_BACKEND=tavily
export TAVILY_API_KEY=your-api-key
export LONGFACT_TAVILY_CACHE_MODE=read_write
export LONGFACT_TAVILY_CACHE_DIR=/path/to/shared/cache
```

Use `live` to disable the persistent cache or `replay` to require an existing
cache without network access. In replay mode, `TAVILY_API_KEY` is not required.
API keys are read only from the environment and must never be committed.

## Running

Run the full suite from the benchmark configuration, or override its input
with the checked-in stress set. For example:

```bash
gym eval run --benchmark longfact \
  --input benchmarks/longfact/data/hard200.jsonl
```

Add the policy and evaluator model configuration required by your deployment.

The default claim extraction prompt is `legacy_v1` for compatibility with
historical runs. The prompt-aware extraction contract used for hard-200 can be
selected with:

```bash
export LONGFACT_CLAIM_EXTRACTION_PROMPT=claim_listing_v1
```

Three-way verdicts are the default. For historical binary comparisons only:

```bash
export LONGFACT_CLAIM_VERDICT_MODE=binary
```

Use a fresh output directory after changing the extraction prompt, verdict
mode, evaluator model, or search backend so completed rollouts are not mixed.
Set `LONGFACT_MAX_CLAIMS` to a small positive value for inexpensive smoke
tests; `0` evaluates every extracted claim.

Rollout metadata records the search provider, claim prompt version, verdict
mode, evidence, and per-claim traces. Prompts with no judged claims receive
precision `1.0` and F1@100 `0.0`.
