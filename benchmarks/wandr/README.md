# WANDR

[WANDR](https://arxiv.org/abs/2608.14747) (Wide ANd Deep Research, Perplexity, 2026) has 500 web data-collection
tasks for research agents. Each task asks for many entities (breadth), several facts and sources per entity (depth),
and one JSONL record per entity and source with verbatim excerpts. There is no gold answer set. WANDR's verifier
fetches every cited page again, and a task-specific GPT-5.4 judge checks each record against that page.

- Integration profile: `custom-gym-verifier`, agent `simple_agent`, scorer `resources_servers/wandr`.
- Reward: soft F1 under the full verdict (`soft_f1_full`), the paper's headline metric. The verify response also
  has soft and hard precision, recall and F1 for the full and retrieval-only verdicts.

## Provenance

| Source | Pin | Used for |
| --- | --- | --- |
| [perplexity-ai/wandr](https://huggingface.co/datasets/perplexity-ai/wandr) | `066ee3d71cd332cb2fb4f49146cad7f9c7249f86` | split `test` (500 scored tasks, canonical order, output files and item fields); split `smoke` (1 unscored task, first row of `resources_servers/wandr/data/example.jsonl`) |
| [perplexityai/wandr](https://github.com/perplexityai/wandr) | `ccb0baeb96f1c77a48e47f92122c57479ee99700` | `datasets/wandr/<task>`: instruction, task config and the task-local verifier; one shared copy of `tests/wandr_core` |

Licence: Apache-2.0 for Perplexity's material, with third-party notices; see the dataset's `LICENSE`, `NOTICE` and
`THIRD_PARTY.md`. Four instructions (`forbes_250_claims`, `forbes_250_cross`, `forbes_250_errors`,
`hbcu_proxy_directors`) embed third-party evidence and are withheld on Hugging Face. Prepare takes them from the
pinned GitHub package, as the upstream harness does. Six tasks are `self_contained=false`. All 500 tasks are kept,
matching the paper.

Nothing is committed: prepare downloads both sources (about 150 MB) and writes about 60 MB under `data/`.

## Prepare and run

```bash
gym eval prepare --benchmark wandr

gym eval run --benchmark wandr --split benchmark \
    --model <policy model> --output results/wandr/rollouts.jsonl \
    ++wandr_resources_server.resources_servers.wandr.inference_api_key=\${oc.env:INFERENCE_API_KEY} \
    ++wandr_resources_server.resources_servers.wandr.exa_api_key=\${oc.env:EXA_API_KEY}
```

`INFERENCE_API_KEY` is an NVIDIA inference API key for the verifier's judge (`openai/openai/gpt-5.4`).
`EXA_API_KEY` pays for the agent's web search and the verifier's page fetch, which both use Exa by default.
Two settings choose the providers, from config only: `search_backend` (`exa`, or `perplexity` for the NVIDIA
inference API's `perplexity-search` with `INFERENCE_API_KEY`) and `verifier_fetch_backend` (`exa`,
`perplexity` or `direct`).

## Task contract

The model sees a system prompt that explains the tools, and the upstream `instruction.md` as the user message. It has
three tools from the resources server:

- `web_search(query, max_results)`: Exa or Perplexity search results (title, URL, date, snippet).
- `fetch_page(url, start)`: readable text of an HTML page or PDF, 12,000 characters at a time.
- `write_records(file, records)`: appends records to one of the task's output files. It applies the upstream row
  checks and returns each rejected row with the reason, because one bad row makes WANDR reject the whole submission.

Verification rebuilds the output files from the accepted `write_records` calls in the trajectory, so it is
stateless and `gym eval reverify` works. It then runs the task's pinned `verify.py` (see the resources server README).

## Differences from the paper

- **Solver.** The paper scores six commercial research APIs, each with its own search stack. Here the policy model
  runs in Gym's `simple_agent` with the three tools above. Scores measure the model with this harness and are not
  comparable to the paper's systems. `simple_agent` keeps the whole trajectory in context, so large tasks can hit
  the model's context limit.
- **Search and page fetch.** WANDR's verifier fetches cited pages through the Perplexity Agent API. Our
  Perplexity org has zero data retention on, which that API's sandbox and background mode do not support, so by
  default the agent searches with Exa and the verifier fetches pages with Exa `/contents`. Exa returns the page
  text, title, author and published date, the fields the judge sees. WANDR's browser fallback stays off. Set
  `verifier_fetch_backend: perplexity` and `perplexity_api_key` (an org with zero data retention off) to use the
  official fetch, and report both backends with every score.
- **Judge.** The same GPT-5.4 model and effort settings as the paper, through the NVIDIA inference API.

## Cost and run time

Each record gets one judge call, plus triage, canonicalization and deduplication calls. Targets range from dozens
to thousands of records per task. The verifier runs after the rollout and can take minutes to hours for a large
task (`verifier_timeout_s`, default 4 hours).
