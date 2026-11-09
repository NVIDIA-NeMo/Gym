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

The prompts are Copyright 2024 DeepMind Technologies Limited and licensed
under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). The pinned
[upstream license statement](https://github.com/google-deepmind/long-form-factuality/blob/9d27158d198ced0a9d8271a80147cae580614601/README.md#license-and-disclaimer)
distinguishes the prompt materials from its Apache-2.0 software. Preparation
adds NeMo Gym fields; the hard-200 set also selects and reorders prompts.

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

The index must index `text` and `url` and store `docid` for each result.
`corpus.db` must contain a `corpus` table with `docid`, `url`, and `text`
columns; its `docid` values must match the stored index IDs. Stored `title`
and `url` fields are optional display overrides.

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

## GPT-5 system card adaptation

`benchmarks/longfact/gpt5_card.yaml` grades with an adaptation of the method in
the [GPT-5 system card](https://cdn.openai.com/gpt-5-system-card.pdf)
(section 3.7 and Appendix 2). It is not the card's harness, and no equivalence
with the card's numbers has been validated.

What it takes from the card:

- The claim-listing prompt (`claim_listing_v1`).
- Fact-checks in batches of 10 claims with the card's fact-checking prompt,
  which labels each claim true, false or unsure.
- The card's sentences for a policy without web access.
- Claim-level error rates, reported for Objects and Concepts separately.

How it differs:

- The card's grader browses the web itself and can search again after reading
  evidence. Here the grader plans up to three searches per claim once, the
  configured search backend runs them, and the grader labels the batch from
  those results in one pass.
- Each claim in a batch has a `claim_id`. A search plan or a verdict list must
  name every `claim_id` exactly once, or the call is repeated up to
  `evaluator_max_attempts` times; an exhausted batch fails the rollout.

The card does not say whether unsure claims count in the denominator or how
rates are averaged, so several are reported:

- `longfact_safe/micro_error_rate`: false / (true + false + unsure), pooled.
- `longfact_safe/micro_decisive_error_rate`: false / (true + false), pooled.
- `longfact_safe/micro_unsure_rate`: unsure / (true + false + unsure), pooled.
- `longfact_safe/mean_error_rate`: false / (true + false + unsure) per prompt,
  averaged over prompts.

Each is also reported per subset under `longfact_safe/objects/` and
`longfact_safe/concepts/`. The card names o3 as its grader.

## Running

Configure the policy variables below and the evaluator and search-backend
variables above. Run all 2,280 prompts from the benchmark configuration with
sampling disabled (the default):

```bash
gym eval run --benchmark longfact \
  --model-type vllm_model --model "$POLICY_MODEL" \
  --model-url "$POLICY_BASE_URL" --model-api-key "$POLICY_API_KEY" \
  --split benchmark --output results/longfact/rollouts.jsonl
```

For the checked-in 200-prompt stress set, start the servers and then collect
against the running LongFact agent in another terminal:

```bash
gym env start --benchmark longfact --model-type vllm_model \
  --model "$POLICY_MODEL" --model-url "$POLICY_BASE_URL" \
  --model-api-key "$POLICY_API_KEY"

# After the servers are ready, in another terminal:
gym eval run --no-serve --agent longfact_longfact_gymnasium_agent \
  --input benchmarks/longfact/data/hard200.jsonl \
  --output results/longfact-hard200/rollouts.jsonl
```

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
