# Finance Agent v2 (FABv2) Resource Server

A NeMo Gym integration of the official [Vals Finance Agent Benchmark v2](https://github.com/vals-ai/finance-agent-v2)
that **reuses Vals's own tool code directly** (tools-only wrap) instead of
reimplementing it. The upstream `finance_agent.tools.*` classes are imported and
exposed as HTTP endpoints; the existing nemo-gym `finance_agent` agent loop drives
them. Scoring uses our own judge (path A) — see [Verification](#verification).

This is the v2 counterpart to `resources_servers/finance_sec_search` (v1). The key
difference: v1 reimplements the tools; v2 imports them from upstream so the tool
descriptions, parameters, and behavior track Vals automatically via a dependency bump.

## Tools (imported from `finance_agent.tools`)

| Tool | Description | Requires |
|------|-------------|----------|
| `web_search` | Tavily web search (`TavilyWebSearch`) | `tavily_api_key` |
| `edgar_search` | sec-api.io full-text EDGAR search (`EDGARSearch`) | `sec_api_key` |
| `price_history` | Tiingo daily OHLC for equity/etf/crypto/fx (`PriceHistory`) | `pricing_data_api_key` |
| `parse_html_page` | Fetch + parse a page to text, store under a key (`ParseHtmlPage`) | — |
| `retrieve_information` | LLM over stored docs via `{{key}}` prompts (`RetrieveInformation`) | `retrieval_model_server` |
| `calculator` | Safe arithmetic via simpleeval (`Calculator`) | — |
| `submit_final_result` | Submit the final answer; ends the loop (`SubmitFinalResult`) | — |

`parse_html_page` and `retrieve_information` share a per-session **data storage**
(`state`) dict, scoped by the HTTP session cookie. A tool whose required key/model
is not configured is registered as unavailable and its endpoint returns a clear
error (the agent can route around it). New in v2 vs v1: `calculator`, `price_history`;
removed: `sec_filing_search`. The upstream date clamp is `MAX_END_DATE = 2026-03-01`
(enforced inside the Vals tools).

## Dependencies

`requirements.txt` pins both upstream packages from git (`model-library` is **not**
on PyPI; `finance-agent` requires `model-library==0.1.25`, i.e. tag `v0.1.25`):

```
-e nemo-gym[dev] @ ../../
model-library @ git+https://github.com/vals-ai/model-library.git@v0.1.25
finance-agent @ git+https://github.com/vals-ai/finance-agent-v2.git@<pinned-sha>
```

Both nemo-gym and finance-agent require Python >=3.12.

## Setup (`env.yaml`)

This is a self-contained Gym environment: run it with **two configs** — this
environment config (`configs/finance_agent_v2.yaml`) plus a model config
(`responses_api_models/openai_model/configs/openai_model.yaml` for OpenAI, or
`responses_api_models/vllm_model/configs/vllm_model.yaml` for a self-hosted
vLLM endpoint).

Secrets live in `env.yaml` at the repo root (gitignored — never commit a populated
copy). Copy `env.yaml.example` and fill in. Only the **model endpoints** go here;
the **tool API keys are read directly from your shell** by the environment config
(`${oc.env:SEC_API_KEY}` etc.), so just export them:

```bash
export OPENAI_API_KEY=...        # policy + judge (OpenAI)
export SEC_API_KEY=...           # edgar_search (sec-api.io)
export TAVILY_API_KEY=...        # web_search (Tavily)
export TIINGO_API_KEY=...        # price_history (Tiingo)
```

```yaml
# env.yaml — model endpoints only
policy_base_url: https://api.openai.com/v1
policy_api_key: ${oc.env:OPENAI_API_KEY}
policy_model_name: gpt-5-mini

search_judge_model_base_url: https://api.openai.com/v1
search_judge_model_api_key: ${oc.env:OPENAI_API_KEY}
search_judge_model_name: gpt-5-mini
```

> Note: "airgap-friendly" applies to **grading** only. The v2 tools call external
> APIs (Tavily, sec-api.io, Tiingo) at rollout time and need network egress + keys.
> A tool whose key is unset simply registers as unavailable (no crash).

## Run

```bash
# Smoke test (mocks external services)
ng_test +entrypoint=resources_servers/finance_agent_v2

# Prepare the benchmark dataset (builds tool schemas from the imported Vals
# classes; ingests questions from benchmarks/finance_agent_v2/data/)
ng_prepare_benchmark "+config_paths=[benchmarks/finance_agent_v2/config.yaml]"

# End-to-end rollout collection
ng_e2e_collect_rollouts "+config_paths=[benchmarks/finance_agent_v2/config.yaml]"
```

### Quickstart: public 27-question smoke run (OpenAI gpt-5-mini)

Run the Vals public question set end-to-end on the OpenAI API (`gpt-5-mini` for
both policy and judge), limited to 3 rollouts to confirm the agent + tools +
binary grading path works. Two configs only: the environment config and the
OpenAI model config.

1. **Secrets** — populate `env.yaml` (model endpoints) and export the tool keys
   (`OPENAI_API_KEY`, `SEC_API_KEY`, `TAVILY_API_KEY`, `TIINGO_API_KEY`) as shown
   in [Setup](#setup-envyaml).

2. **Convert** the raw Vals CSV (downloaded from
   [finance-agent-v2](https://github.com/vals-ai/finance-agent-v2)) to benchmark
   JSONL. Prompts come from `finance_agent.prompt` and tool schemas from
   `finance_agent.tools`. The public CSV ships only rubric *criteria* (no single
   gold answer), so the script synthesizes a GOLD `expected_answer` from those
   criteria for our judge. The CSV's `rubric` is also copied through verbatim for
   reference only (it is **not** used for reward):

   ```bash
   python resources_servers/finance_agent_v2/scripts/convert_questions.py \
     -i resources_servers/finance_agent_v2/data/vals_v2_public_27q.csv \
     -o resources_servers/finance_agent_v2/data/vals_v2_public_27q.jsonl
   ```

3. **Start the servers** — environment config + OpenAI model config:

   ```bash
   ng_run "+config_paths=[\
     resources_servers/finance_agent_v2/configs/finance_agent_v2.yaml,\
     responses_api_models/openai_model/configs/openai_model.yaml]"
   ```

4. **Collect rollouts** against the running servers, limited to 3 questions:

   ```bash
   ng_collect_rollouts "+config_paths=[\
     resources_servers/finance_agent_v2/configs/finance_agent_v2.yaml,\
     responses_api_models/openai_model/configs/openai_model.yaml]" \
     +agent_name=finance_agent \
     +input_jsonl_fpath=resources_servers/finance_agent_v2/data/vals_v2_public_27q.jsonl \
     +output_jsonl_fpath=results/finance_agent_v2_smoke.jsonl \
     +limit=3 \
     +num_samples_in_parallel=3
   ```

   Rewards land in `results/finance_agent_v2_smoke.jsonl` (1.0 = judge rated
   `[[2]]`, i.e. the answer covers all required facts in the GOLD reference;
   else 0.0). Drop `+limit` to run the full set. To run on a
   self-hosted model, swap the model config for
   `responses_api_models/vllm_model/configs/vllm_model.yaml` and point
   `policy_*` in `env.yaml` at your vLLM endpoint.

## Dataset & labels (path-A scoring)

The public FABv2 release ships **only question strings** (no ground truth/grader).
`benchmarks/finance_agent_v2/prepare.py` loads input by this precedence:

1. `data/labeled.jsonl` — labeled rows (enables real scoring)
2. `data/public.jsonl` — rows with at least `{question}`
3. `data/public.txt` — one question per line (FABv2 public format)
4. fallback: the resource server's `example_questions.jsonl`

**Labeled JSONL schema** (one object per line):

```json
{"question": "...", "expected_answer": "...", "rubric": "[{\"operator\": \"...\", \"criteria\": \"...\"}]"}
```

- `expected_answer` is the GOLD reference used by our `[[N]]` judge — this is what
  drives reward.
- `rubric` is propagated from the public CSV verbatim for reference/completeness
  only. It is **not** consumed by scoring. (The public FABv2 release has no
  official grader; Vals's private per-criterion rubric grader is licensed and is
  deliberately not reproduced here.)
- To source labels at scale, publish a labeled set to the GitLab Model Registry
  (mirrors v1's `finance_sec_search_vals_200_eval`) and uncomment the
  `gitlab_identifier` block in `benchmarks/finance_agent_v2/config.yaml`.

**Interim dry-run:** with no labels, `/verify` returns `reward=0` so the agent +
tools path can be validated before ground truth is available.

## Verification

The public FABv2 release ships **no official grader**, so scoring uses **our own**
approximation: the legacy `[[0]]/[[1]]/[[2]]` judge from
`resources_servers/finance_sec_search`. The public CSV has no single gold answer,
so the GOLD `expected_answer` is synthesized from the rubric criteria (see
`scripts/convert_questions.py`); the judge awards `[[2]]` (reward 1.0) only when
the answer covers all required facts. The dataset's `rubric` field plays **no**
role in reward.

Set `reward_mode` in the resource server config:

| Mode | Mapping | Use |
|------|---------|-----|
| `binary` | `[[2]]` → 1.0, else 0.0 | **default (public)** — strict pass/fail |
| `scaled` | `[[0]]`/`[[1]]`/`[[2]]` → 0.0/0.5/1.0 | shaped reward (training) |

Judge prompts live in `prompt_templates/`.

## File structure

```
resources_servers/finance_agent_v2/
├── app.py                         # Resource server: tool endpoints + retrieval shim + verify
├── requirements.txt               # Pins nemo-gym + Vals model-library + finance-agent
├── env.yaml.example
├── configs/
│   └── finance_agent_v2.yaml      # The environment config: resources server + judge model + agent + dataset (binary scoring)
├── scripts/
│   └── convert_questions.py       # Vals public CSV -> benchmark JSONL (upstream prompts/tools; criteria -> GOLD expected_answer; rubric copied through for reference)
├── prompt_templates/              # judge / retrieval
├── data/                          # example.jsonl, example_rollouts.jsonl, example_questions.jsonl, vals_v2_public_27q.jsonl
└── tests/test_app.py

benchmarks/finance_agent_v2/
├── prepare.py                     # Builds tool schemas from upstream classes; ingests questions
└── config.yaml                    # Benchmark wiring (inherits base, overrides dataset)
```

## Licensing

**This environment's code** (everything under `resources_servers/finance_agent_v2/`)
is licensed under **Apache-2.0**, consistent with NeMo Gym and the SPDX headers in
each source file (`app.py`, `scripts/convert_questions.py`, `tests/test_app.py`,
`benchmarks/finance_agent_v2/*`).

**Upstream dependencies** (imported, not vendored — see `requirements.txt`):

| Package | Source | License |
|---------|--------|---------|
| `finance-agent` (`finance_agent.tools` / `finance_agent.prompt`) | [vals-ai/finance-agent-v2](https://github.com/vals-ai/finance-agent-v2) | MIT |
| `model-library` (`model_library.*`) | NVIDIA fork of [vals-ai/model-library](https://github.com/vals-ai/model-library)@`v0.1.25` (openai floor dropped, static version) | MIT |

We import these packages at install time and do not copy their source into this
repo, so their MIT terms apply to that code as distributed by the upstream/fork.

**Dataset.** `data/vals_v2_public_27q.*` and the `example.jsonl` questions derive
from the **public** Vals Finance Agent Benchmark v2 release
([vals-ai/finance-agent-v2](https://github.com/vals-ai/finance-agent-v2)); use is
subject to that project's terms. The public release ships **no official grader**.

**Grading is our own.** Reward is computed by our `[[0]]/[[1]]/[[2]]` judge (an
approximation reused from `resources_servers/finance_sec_search`) against a GOLD
`expected_answer` we synthesize from the public rubric criteria. The dataset's
`rubric` field is propagated **for reference only** and is not used for scoring.
Vals's private per-criterion rubric grader (prompts + reward logic) was obtained
under a separate license and is **deliberately not reproduced** in this public
code.
