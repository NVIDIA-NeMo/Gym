# FACTS Search V2

This package ports the public Search-On component of Google DeepMind's [FACTS Leaderboard](https://arxiv.org/abs/2512.10791) into NeMo Gym. It evaluates a policy model in a seven-hop agent loop with Brave Web Search, then grades the final answer against the public gold answer with the current Gemini 3.5 Flash A/B/C grader.

```bash
gym eval prepare --benchmark facts_search
gym env start --benchmark facts_search --model-type vllm_model --config env.yaml
gym eval run --no-serve --benchmark facts_search --model-type vllm_model \
  --config env.yaml --agent facts_search_public \
  --input benchmarks/facts_search/data/facts_search_public.jsonl \
  --temperature 0.0 --output facts_search_rollouts.jsonl
```

Required runtime variables:

- `FACTS_SEARCH_BRAVE_API_KEY`: Brave Search API subscription token.
- `FACTS_SEARCH_JUDGE_BASE_URL`, `FACTS_SEARCH_JUDGE_API_KEY`, and optionally `FACTS_SEARCH_JUDGE_MODEL` (default `google/gemini-3.5-flash`): an OpenAI-compatible endpoint exposing the current canonical judge.

The public artifact mismatch and all inaccessible rows are recorded in [METHODS.md](METHODS.md). No substitute search engine or judge is accepted by the default configuration.

Preparation downloads `deepmind/facts-search-public` version 1, verifies the source CSV SHA-256, and writes all 890 downloadable public rows. The dataset and public protocol are Apache-2.0. The 921-row private split and 31 advertised but unavailable public rows are outside this adapter's scope.
