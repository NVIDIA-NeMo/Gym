# Finance Agent Resource Server

Financial information retrieval using SEC EDGAR filings with optional web search via Tavily. Architecture aligns with the [Vals Finance Agent benchmark](https://www.vals.ai/benchmarks/finance_agent).

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

finance_agent_resources_server:
  resources_servers:
    finance_agent:
      cache_dir: cache
      # Optional: uncomment to enable web_search
      # tavily_api_key: <your-tavily-key>
```

### Prepare dataset

Convert question/answer pairs to the Gym input format with tool definitions:

```bash
python resources_servers/finance_agent/scripts/convert_questions.py \
  -i resources_servers/finance_agent/data/questions_numeric.jsonl \
  -o resources_servers/finance_agent/data/example.jsonl
```

Input format:
```json
{"question": "What was Apple's revenue in 2024?", "expected_answer": "$391.0 billion"}
```

### Run

```bash
# 1. Start vLLM server
# 2. Start Gym servers
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,resources_servers/finance_agent/configs/finance_agent.yaml"
ng_run "+config_paths=[$config_paths]"

# 3. Collect rollouts
ng_collect_rollouts \
  +agent_name=finance_agent \
  +input_jsonl_fpath=resources_servers/finance_agent/data/example.jsonl \
  +output_jsonl_fpath=results/example_verified.jsonl
```

### Run tests

```bash
ng_test +entrypoint=resources_servers/finance_agent
```

## Verification

Uses LLM-as-judge with a financial grading rubric (0/1/2 scale). Only fully correct answers ([[2]]) receive reward 1.0. The judge prompt and rubric are defined in `app.py`.

