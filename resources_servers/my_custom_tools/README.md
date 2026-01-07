# Custom Tools Resources Server with Dynamic Verifier Delegation

This resources server demonstrates how to create custom tool implementations while delegating verification to existing verifiers based on per-sample configuration.

## Features

- **Custom Tools**: `search_database`, `execute_calculation`, `fetch_data`
- **Dynamic Verification**: Each sample specifies which verifier to use via `verifier_type`
- **Supported Verifiers**: `xlam_fc` (function call matching), `mcqa` (multiple choice), `equivalence_llm_judge` (LLM-based)

## Usage

### 1. Run Example Validation

```bash
config_paths="resources_servers/my_custom_tools/configs/my_custom_tools.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml"

ng_prepare_data "+config_paths=[${config_paths}]" \
    +output_dirpath=data/my_custom_tools \
    +mode=example_validation
```

### 2. Start Servers

```bash
ng_run "+config_paths=[${config_paths}]"
```

### 3. Collect Rollouts

```bash
ng_collect_rollouts \
    +agent_name=my_custom_agent \
    +input_jsonl_fpath=data/my_custom_tools/example.jsonl \
    +output_jsonl_fpath=results/my_custom_tools_rollouts.jsonl
```

## Sample Data Format

Each sample must include:
- `verifier_type`: Which verifier to use (`xlam_fc`, `mcqa`, `equivalence_llm_judge`)
- `responses_create_params`: The request parameters for the model
- Verifier-specific fields (e.g., `expected_answers` for xlam_fc, `expected_answer` and `options` for mcqa)

### Example: xlam_fc verifier

```json
{
  "verifier_type": "xlam_fc",
  "expected_answers": [{"name": "search_database", "arguments": {"query": "revenue"}}],
  "responses_create_params": {...}
}
```

### Example: mcqa verifier

```json
{
  "verifier_type": "mcqa",
  "expected_answer": "B",
  "options": [{"A": "option1"}, {"B": "option2"}],
  "responses_create_params": {...}
}
```

## Custom Tools

### search_database

Search a database with a query.

```json
{"query": "search terms", "database": "default"}
```

### execute_calculation

Execute a mathematical expression.

```json
{"expression": "100 * 1.15", "precision": 2}
```

### fetch_data

Fetch data from a source.

```json
{"source_id": "customers", "fields": ["name", "email"]}
```

## Architecture

```
Sample with verifier_type="xlam_fc"
           │
           ▼
   MyCustomToolsResourcesServer
           │
           ├── Tool calls: /search_database, /execute_calculation, /fetch_data
           │
           └── /verify → delegates to xlam_fc_verifier based on verifier_type
```

