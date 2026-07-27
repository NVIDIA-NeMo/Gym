# Finance SEC Search (Vals v1)

50-question financial information retrieval benchmark from the
[Vals AI finance-agent](https://github.com/vals-ai/finance-agent) public
dataset. Questions cover SEC EDGAR filings, financial metrics, and
company analysis.

The tool implementation, prompts, verification, and example fixtures live in
the [`finance_sec_search` resource server](../../resources_servers/finance_sec_search/README.md).

## Verification

Uses LLM-as-judge with a financial grading rubric (0/1/2 scale).
Only fully correct answers (`[[2]]`) receive reward 1.0. The judge
prompt and rubric are defined in the resource server's `prompt_templates/`.

## Tools

| Tool | Description |
|------|-------------|
| `sec_filing_search` | Search SEC EDGAR for filing metadata by stock ticker symbol |
| `parse_html_page` | Fetch and parse an HTML page, optionally cache SEC content, and store it under a key |
| `retrieve_information` | Query stored documents via LLM prompt with `{{key}}` placeholders |
| `submit_final_result` | Submit the final answer (required to receive a reward) |
| `web_search` | Internet search via Tavily API (optional — requires `tavily_api_key` in `env.yaml`) |

## Configure models

For example, to use GPT-5 mini for both the policy and judge, create `env.yaml`
in the Gym repository root:

```yaml
policy_base_url: https://api.openai.com/v1
policy_api_key: ${oc.env:OPENAI_API_KEY}
policy_model_name: gpt-5-mini

search_judge_model_base_url: https://api.openai.com/v1
search_judge_model_api_key: ${oc.env:OPENAI_API_KEY}
search_judge_model_name: gpt-5-mini

# Required only when running the web-search variant.
tavily_api_key: ${oc.env:TAVILY_API_KEY,null}
```

```bash
export OPENAI_API_KEY=...
# Export only for the web-search variant:
export TAVILY_API_KEY=...
```

## Data preparation

Without web search:

```bash
gym eval prepare --benchmark finance_sec_search/config_no_web_search
```

With the web-search tool included:

```bash
gym eval prepare --benchmark finance_sec_search/config_web_search
```

Downloads `public.csv` from the Vals AI GitHub repo and writes benchmark
JSONL to `data/`. A Tavily key is not needed during preparation; it is required
when running the web-search variant.

| Config | Output file |
|--------|-------------|
| `config_no_web_search.yaml` | `data/finance_sec_search_benchmark.jsonl` |
| `config_web_search.yaml` | `data/finance_sec_search_benchmark_web_search.jsonl` |

## Running servers

```bash
gym env start \
  --model-type openai_model \
  --benchmark finance_sec_search/config_no_web_search
```

## Collecting rollouts

```bash
gym eval run --no-serve \
  --agent finance_sec_search_benchmark_agent \
  --input benchmarks/finance_sec_search/data/finance_sec_search_benchmark.jsonl \
  --output results/finance_sec_search_rollouts.jsonl \
  --num-repeats 1
```

Use `--limit 1` for a quick end-to-end check. For the web-search variant, use
`finance_sec_search_web_search_benchmark_agent` and
`data/finance_sec_search_benchmark_web_search.jsonl`.
