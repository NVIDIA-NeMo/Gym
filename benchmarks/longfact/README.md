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
Prepared and vendored rows retain the upstream training-contamination canary.
These prompts are held out for evaluation and must not enter training corpora.

The `longfact` dataset contains all 2,280 upstream prompts. It is downloaded
from a pinned upstream commit by the preparation script:

```bash
gym eval prepare --benchmark longfact
```

The stress set is checked in at `benchmarks/longfact/data/hard200.jsonl`.
It contains 100 Objects and 100
Concepts prompts selected using contradiction counts from an earlier model
evaluation. The selection manifest and provenance are in the same directory.
Use this dataset when a stable 200-prompt stress test is desired.

## Measurement contract and onboarding

`manifest.yaml` declares the primary `simple_agent` and
`single_agent_turn_legacy` environment pairing, with prepared prompts. The
environment runs one policy turn and calls the Resources Server `/verify`
endpoint. The manifest's
`full-objects-and-concepts` split names the union of all 2,280 upstream Objects
and Concepts prompts. The default is one repeat
and one policy answer per prompt; the model sees the source prompt and has
no retrieval tools. Search belongs to the verifier. The hard-200 stress set
and any sampled run are distinct protocols and must be identified separately.

The per-prompt reward is SAFE-style F1@100 in `[0, 1]`, higher is better.
Precision counts supported claims against all relevant judged claims;
recall is the supported count divided by 100, capped at 1. The primary
aggregate averages per-prompt F1; pooled claim metrics are reported
separately. Contradicted and unsure claims both count as unsupported in
F1. Judge/search failures fail the rollout for recovery; a failed judge masks
the row and retains its original policy answer rather than assigning a model
zero. The default configuration uses `safe_v1` extraction,
three-way per-claim verdicts and Tantivy; the card overlay below changes
this protocol and is not interchangeable with that default.

This server is for evaluation only. Like upstream SAFE, scoring does not
deduplicate extracted claims. Repeated claims can inflate F1@100 if the judge
lists them more than once, so this reward is unsuitable for RL training.
The optional `claim_listing_v1` prompt asks the judge to list each claim once;
that instruction is not a scoring-level deduplication guarantee.

Policy sampling, context and token budgets come from the configured model
and run, so record them alongside the resolved configuration, judge identity,
retrieval settings and repeat count. LongFact imposes no episode deadline.
Its evaluator HTTP calls default to 180 seconds, and Tavily calls to 90
seconds, with their configured bounded retries. The agent's 60-second
environment-cleanup timeout is a separate bound.

Grading follows path A of the benchmark onboarding contract: the resources
server wraps the complete verification in `observe_llm_judge_grading`,
calls the judge through `call_judge`, and accepts a reply only after parsing
and, for card batches, exact claim-ID validation. The returned observation
binds the original policy response and F1 reward to the judge dependencies.
Rejected or malformed received replies remain unaccepted even when a later
retry succeeds, so that attempt has unknown grading coverage; it is not
reported as complete evidence. Resources Server verification uses only its request and
configuration (`ReverifyMode.STATELESS`), without legacy reset state. A
re-judge must preserve the saved answer, task provenance and scoring settings.
The offline fixture exercises this production scorer with scripted
external judge/search transports. It does not certify a live runtime run.

Complete the manifest-backed local checks after preparation:

```bash
gym eval prepare --benchmark longfact
gym env validate --sync longfact
gym env test longfact
gym env test --resources-server longfact
gym env publish longfact
```

Static validation resolves the selected judge configuration and reads the
prepared data. The workload fixture checks scoring without live services;
component tests cover the server implementation. These local checks do not
replace the representative real model/search rollout, reference comparison,
and reward profiling required before benchmark approval.

## Configuration

Configure the policy and evaluator model servers in the usual NeMo Gym way,
then select a retrieval backend with `LONGFACT_SEARCH_BACKEND`.

The benchmark config uses a vLLM-served Qwen judge
(`benchmarks/longfact/evaluator_local_vllm.yaml`). Its default temperature is
`0.1`, matching [pinned SAFE's fact-checking temperature](https://github.com/google-deepmind/long-form-factuality/blob/9d27158d198ced0a9d8271a80147cae580614601/eval/safe/config.py#L38),
and it applies no presence penalty. For an OpenAI-compatible
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

Add `'+config_paths=[benchmarks/longfact/milvus.yaml]'` when starting or
running the benchmark. This overlay selects Milvus and requests its optional
dependencies through the supported component mapping:

```yaml
component_extras:
  resources_servers/longfact: [milvus]
```

NeMo Gym forwards this to the component's locked environment as `--extra milvus`.
Setting only `LONGFACT_SEARCH_BACKEND=milvus` does not install the extra.
For a manually managed component environment, use
`uv sync --project resources_servers/longfact --locked --extra milvus`.
Tantivy and Tavily do not need this extra.

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
the [GPT-5 system card](https://arxiv.org/html/2601.03267v1)
(section 3.7 and Appendix 2). It is not the card's harness, and no equivalence
with the card's numbers has been validated.

Prompt text is attributed to OpenAI under CC-BY-4.0 at that paper revision.
The adaptation changes formatting, adds validated claim IDs, and replaces
iterative browsing with one search-planning pass.

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
gym eval run --no-serve --agent longfact_longfact_simple_agent \
  --input benchmarks/longfact/data/hard200.jsonl \
  --output results/longfact-hard200/rollouts.jsonl
```

The default claim extraction prompt is `safe_v1`. The prompt-aware extraction
contract used for hard-200 can be
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
