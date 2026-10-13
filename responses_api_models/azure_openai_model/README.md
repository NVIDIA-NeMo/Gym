# Azure OpenAI Model

Use this model server to access Azure OpenAI endpoints for LLM-as-a-judge.
## Configuration

Set up your `env.yaml` file:
```yaml
policy_base_url: https://my.end.point.com/v1/azure
policy_api_key: <API_KEY>
policy_model_name: gpt-5-nano
```

## Usage

### Running the server
Set the API version. It usually looks something like "2024-10-21".
```bash
gym env start \
    --model-type azure_openai_model \
    --resources-server equivalence_llm_judge \
    +policy_model.responses_api_models.azure_openai_model.default_query.api-version=<api_version>
```

### Collecting Rollouts

```bash
gym eval run --no-serve \
  --agent equivalence_llm_judge_simple_agent \
  --input resources_servers/equivalence_llm_judge/data/example.jsonl \
  --output results/example_rollouts.jsonl \
  --limit 5
```

### Test cases

```bash
gym env test +entrypoint=responses_api_models/azure_openai_model
```

## Licensing information

- **Code**: Apache 2.0
- **Data**: N/A

## Dependencies

- `nemo_gym`: Apache 2.0
