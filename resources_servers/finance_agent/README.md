# Finance Agent Resource Server

A resource server for financial information retrieval using SEC filings, with optional web search capabilities. Designed to align with the [Vals Finance Agent benchmark](https://www.vals.ai/benchmarks/finance_agent) architecture.

## Architecture

The agent follows a **RAG-style** architecture matching the Vals benchmark:

1. **`sec_filing_search`** — Search SEC EDGAR for filing metadata (replaces Vals' `edgar_search`, uses open-source SEC.gov API instead of paid SEC-API).
2. **`prepare_filing`** — Download, parse (HTML → plain text), and store a filing in an in-memory data storage keyed by a descriptive name (replaces Vals' `parse_html_page`).
3. **`retrieve_information`** — Query stored documents via a separate LLM call with a user-crafted prompt. The prompt uses `{{key_name}}` placeholders that get replaced with document text. Supports optional `input_character_ranges` for targeting specific portions.
4. **`submit_final_result`** — Explicitly submit the final answer via tool call. Keeps the model in tool-calling mode (searching, retrieving, verifying) until it deliberately submits.
5. **`web_search`** *(optional)* — Public internet search via Tavily API.

## Two Modes

| Mode | Config | Tavily API Key | Tools Available |
|------|--------|----------------|-----------------|
| **SEC-only** | `finance_agent.yaml` | Not required | `sec_filing_search`, `prepare_filing`, `retrieve_information`, `submit_final_result` |
| **SEC + Web** | `finance_agent_with_web.yaml` | Required | All SEC tools + `web_search` |

## Setup

### Environment Configuration

Create an `env.yaml` file in the root of the Gym repository.

#### SEC-only mode (no Tavily API key needed):

```yaml
policy_base_url: http://localhost:5000/v1
policy_api_key: empty
policy_model_name: /hf_models/gpt-oss-120b

search_judge_model_base_url: http://localhost:5000/v1
search_judge_model_api_key: ""
search_judge_model_name: /hf_models/gpt-oss-120b

finance_agent_resources_server:
  resources_servers:
    finance_agent:
      cache_dir: cache
```

#### SEC + Web Search mode (requires Tavily API key):

```yaml
policy_base_url: http://localhost:5000/v1
policy_api_key: empty
policy_model_name: /hf_models/gpt-oss-120b

search_judge_model_base_url: http://localhost:5000/v1
search_judge_model_api_key: ""
search_judge_model_name: /hf_models/gpt-oss-120b

tavily_search_resources_server:
  resources_servers:
    tavily_search:
      tavily_api_key: <tvly-dev-key>
      exclude_domains_file_path: /lustre/fsw/portfolios/llmservice/users/rgala/frozen/2025_12_15_nv_tdm_opt_out_registry.json

finance_agent_resources_server:
  resources_servers:
    finance_agent:
      cache_dir: cache
      tavily_api_key: <tvly-dev-key>
      exclude_domains_file_path: /lustre/fsw/portfolios/llmservice/users/rgala/frozen/2025_12_15_nv_tdm_opt_out_registry.json
```

Replace `<tvly-dev-key>` with your actual Tavily API key.

## Usage

### 1. Prepare Your Dataset

Create your dataset from simple question/ground truth pairs. This converts them to the correct prompt format with tool definitions for running the agent.

Input format (`questions.jsonl`):
```json
{"question": "What was Apple's revenue in 2024?", "expected_answer": "$391.0 billion"}
```

**For SEC-only mode** (no web search tool):

```bash
python tools/convert_questions.py \
  -i resources_servers/finance_agent/data/questions_numeric.jsonl \
  -o resources_servers/finance_agent/data/test_sec_numeric.jsonl \
  --no-web
```

**For SEC + Web Search mode** (includes `web_search` tool):

```bash
python tools/convert_questions.py \
  -i resources_servers/finance_agent/data/questions_numeric.jsonl \
  -o resources_servers/finance_agent/data/test_sec_numeric.jsonl
```

The converter adds the agent prompt (matching Vals benchmark), tool schemas (`sec_filing_search`, `prepare_filing`, `retrieve_information`, `submit_final_result`, and optionally `web_search`), and wraps everything in the Gym input format.

### 2. Start vLLM Server

Start the vLLM server (assuming nemo-flow setup or at least `my_cluster` config):

```bash
uv run ns start_server \
  --cluster=my_cluster --partition interactive \
  --model=/hf_models/gpt-oss-120b \
  --server_type=vllm \
  --server_gpus=8 \
  --server_args="--async-scheduling --tool-call-parser openai --enable-auto-tool-choice"
```

Or use your own vLLM command if you have a different setup.

Use `squeue` to find the job ID of the running server.

> **Note:** A single vLLM server handles both the agent (policy) model and the retrieval model since LLM API calls are stateless. The `retrieval_model_server` config points to the same `policy_model` by default, but can be changed to a different model if needed.

### 3. Start Gym Servers

On another terminal, connect to the same job and start the Gym servers:

```bash
# Connect to the running job
srun --jobid=<job-id> --overlap --pty bash

# Navigate to Gym directory
cd Gym

# Activate virtual environment
source .venv/bin/activate
```

**For SEC-only mode**:

```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,resources_servers/finance_agent/configs/finance_agent.yaml"
ng_run "+config_paths=[$config_paths]"
```

**For SEC + Web Search mode**:

```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,resources_servers/finance_agent/configs/finance_agent_with_web.yaml"
ng_run "+config_paths=[$config_paths]"
```

Wait for both the vLLM server and Gym servers to be ready.

### 4. Collect Rollouts

Once both servers are running, open a third terminal:

```bash
# Navigate and activate environment
cd Gym
source .venv/bin/activate
```

**For SEC-only mode**:

```bash
ng_collect_rollouts \
  +agent_name=finance_agent \
  +input_jsonl_fpath=resources_servers/finance_agent/data/test_sec_numeric.jsonl \
  +output_jsonl_fpath=results/test_sec_numeric_verified.jsonl
```

**For SEC + Web Search mode**:

```bash
ng_collect_rollouts \
  +agent_name=finance_agent_with_web \
  +input_jsonl_fpath=resources_servers/finance_agent/data/test_sec_numeric.jsonl \
  +output_jsonl_fpath=results/test_sec_numeric_verified.jsonl
```

### 5. Analyze Results

#### Model behavior analysis

Run the diagnostic tool to analyze model behavior (tool call patterns, errors, etc.):

```bash
python tools/analyze_model_behavior.py results/test_sec_numeric_verified.jsonl
```

#### LLM-as-a-judge evaluation

Use the judge script to evaluate answers against ground truth using an external LLM (default is GPT-5). This uses a stricter 0-1-2 grading rubric tailored for financial data:

```bash
OPENAI_API_KEY=<your-key> python tools/verify_labels.py \
  --input_file results/test_sec_numeric_verified.jsonl \
  --output_file results/test_sec_numeric_verified_judged.jsonl
```

The judge extracts the question and the model's final answer (from `submit_final_result` tool call or last assistant message) automatically from the Gym output format. Ratings:
- **2** = fully correct
- **1** = partially correct
- **0** = incorrect

Reports both strict accuracy (2 only) and lenient accuracy (1+2).

## Features

- **SEC Filings Search**: Query SEC EDGAR database for company filings (10-K, 10-Q, DEF 14A by default)
- **RAG-style Retrieval**: Separate LLM call for document querying with anti-hallucination safeguards
- **Data Storage**: In-memory key-value store for parsed filings, with descriptive keys generated by the agent
- **Submit Final Result**: Explicit tool-call submission matching Vals benchmark, promoting agent persistence
- **Web Search Integration**: Enhanced search capabilities via Tavily API (optional)
- **Result Caching**: Speeds up repeated queries with local caching
- **Automated Verification**: Built-in verification for rollout quality using equivalence LLM-as-a-judge, plus external judge script

## Configuration Options

See the config files in `configs/` for available options:
- `finance_agent.yaml`: SEC search only (no Tavily API key required)
- `finance_agent_with_web.yaml`: SEC search with web search integration (requires Tavily API key)

Key config fields:
- `cache_dir`: Directory for caching ticker mappings and filing metadata
- `retrieval_model_server`: Model server for `retrieve_information` LLM calls (defaults to `policy_model`)
- `judge_model_server`: Model server for built-in answer verification
- `judge_prompt_template`: Prompt template for the equivalence judge
