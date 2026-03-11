# Finance SEC Search Resource Server

Financial information retrieval using SEC EDGAR filings with optional web search via Tavily.

**Only companies listed in the [SEC company tickers file](https://www.sec.gov/files/company_tickers.json) are supported.** Questions about companies not in this list will fail at the ticker lookup step.

## Tools

| Tool | Description |
|------|-------------|
| `sec_filing_search` | Search SEC EDGAR for filing metadata by stock ticker symbol |
| `download_and_parse_filing` | Download and parse a filing (HTML → text), store under a key |
| `retrieve_information` | Query stored documents via LLM prompt with `{{key}}` placeholders |
| `submit_final_result` | Submit the final answer (keeps model in tool-calling mode until ready) |
| `web_search` | Internet search via Tavily API (optional — requires `tavily_api_key`) |

If `tavily_api_key` is not configured, `web_search` returns an error directing the model to use SEC tools instead.

## Setup

### env.yaml

Create `env.yaml` in the Gym root:

```yaml
policy_base_url: http://localhost:5000/v1
policy_api_key: empty
policy_model_name: /hf_models/Qwen3-30B-A3B

search_judge_model_base_url: http://localhost:5000/v1
search_judge_model_api_key: ""
search_judge_model_name: /hf_models/Qwen3-30B-A3B

finance_sec_search_resources_server:
  resources_servers:
    finance_sec_search:
      cache_dir: cache
      # tavily_api_key: <your-tavily-key>
```

## End-to-End Rollout

### 1. Prepare the dataset

The input is a JSONL file with question/answer pairs. An example is provided at
`resources_servers/finance_sec_search/data/example_questions.jsonl`:

```json
{"question": "What is the number of shares of common stock outstanding as of November 14, 2025 for Nvidia?", "expected_answer": "24.3 billion"}
{"question": "As of September 24, 2022 how many full-time equivalent employees did Apple have?", "expected_answer": "164,000"}
```

Convert the questions into the Gym input format (adds tool definitions, system prompt, etc.)

Use the `--include-web-search` flag to include the optional `web_search` tool:

```bash
python resources_servers/finance_sec_search/scripts/convert_questions.py \
  --input resources_servers/finance_sec_search/data/example_questions.jsonl \
  --output resources_servers/finance_sec_search/data/example.jsonl \
  --include-web-search
```

#### SecQue benchmark

To prepare the [SecQue](https://huggingface.co/datasets/nogabenyoash/SecQue) dataset (filters to questions mentioning known companies and converts to Gym format):

```bash
cd resources_servers/finance_sec_search
python scripts/prepare_secque_questions.py
```

This writes `data/secque_questions.jsonl`. Use it as the `input_jsonl_fpath` in step 4 below.

**Note that this dataset is not used for training anywhere and is only used for eval/benchmark purposes.**


### 2. Start the vLLM server

Launch a vLLM-compatible model server (e.g. Qwen3-30B-A3B) so the policy and judge endpoints are available.

### 3. Start the Gym servers

```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,resources_servers/finance_sec_search/configs/finance_sec_search.yaml"
ng_run "+config_paths=[$config_paths]"
```

### 4. Collect rollouts

```bash
ng_collect_rollouts \
  +agent_name=finance_agent \
  +input_jsonl_fpath=resources_servers/finance_sec_search/data/example.jsonl \
  +output_jsonl_fpath=results/finance_sec_search_rollouts.jsonl
```

Add `+limit=1` for a quick single-question test:

```bash
ng_collect_rollouts \
  +agent_name=finance_agent \
  +input_jsonl_fpath=resources_servers/finance_sec_search/data/example.jsonl \
  +output_jsonl_fpath=results/finance_sec_search_rollouts.jsonl \
  +limit=1
```

### Run tests

```bash
ng_test +entrypoint=resources_servers/finance_sec_search
```

## Verification

Uses LLM-as-judge with a financial grading rubric (0/1/2 scale). Only fully correct answers ([[2]]) receive reward 1.0. The judge prompt and rubric are defined in /prompt_templates.

