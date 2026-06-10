# Finance SEC Search

50-question financial information retrieval benchmark from the
[Vals AI finance-agent](https://github.com/vals-ai/finance-agent) public
dataset. Questions cover SEC EDGAR filings, financial metrics, and
company analysis.

## Verification

Uses LLM-as-judge with a financial grading rubric (0/1/2 scale).
Only fully correct answers (`[[2]]`) receive reward 1.0. The judge
prompt and rubric are defined in the `finance_sec_search` resource
server's `/prompt_templates`.

## Tools

| Tool | Description |
|------|-------------|
| `sec_filing_search` | Search SEC EDGAR for filing metadata by stock ticker symbol |
| `parse_html_page` | Fetch and parse any HTML page (SEC URLs use disk cache), store under a key |
| `retrieve_information` | Query stored documents via LLM prompt with `{{key}}` placeholders |
| `submit_final_result` | Submit the final answer (required to receive a reward) |
| `web_search` | Internet search via Tavily API (optional — requires `tavily_api_key` in `env.yaml`) |

## Data preparation

Without web search:

```bash
ng_prepare_benchmark '+config_paths=[benchmarks/finance_sec_search/config_no_web_search.yaml]'
```

With web search (requires `tavily_api_key` in `env.yaml`):

```bash
ng_prepare_benchmark '+config_paths=[benchmarks/finance_sec_search/config_web_search.yaml]'
```

Downloads `public.csv` from the Vals AI GitHub repo and writes benchmark
JSONL to `data/`.

| Config | Output file |
|--------|-------------|
| `config_no_web_search.yaml` | `data/finance_sec_search_benchmark.jsonl` |
| `config_web_search.yaml` | `data/finance_sec_search_benchmark_web_search.jsonl` |

## Running servers

```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/finance_sec_search/config_no_web_search.yaml"
ng_run "+config_paths=[$config_paths]"
```

## Collecting rollouts

```bash
ng_collect_rollouts \
    +agent_name=finance_sec_search_benchmark_agent \
    +input_jsonl_fpath=benchmarks/finance_sec_search/data/finance_sec_search_benchmark.jsonl \
    +output_jsonl_fpath=results/finance_sec_search_rollouts.jsonl
```
