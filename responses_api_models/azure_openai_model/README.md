# Azure OpenAI Model

Use this model server to access Azure OpenAI endpoints for LLM-as-a-judge.
## Configuration

Set up your `env.yaml` file:
```yaml
policy_base_url: https://my-resource.openai.azure.com
policy_api_key: <API_KEY>
policy_model_name: my-gpt-4-deployment
```

## Usage

### Running the server
Set the API version. It usually looks something like "2024-10-21".
```bash
config_paths="responses_api_models/azure_openai_model/configs/azure_openai_model.yaml, \
resources_servers/equivalence_llm_judge/configs/equivalence_llm_judge.yaml"

ng_run "+config_paths=[${config_paths}]" \
    +policy_model.responses_api_models.azure_openai_model.default_query.api-version=<api_version>
```

### Collecting Rollouts

```bash
ng_collect_rollouts \
  +agent_name=equivalence_llm_judge_simple_agent \
  +input_jsonl_fpath=resources_servers/equivalence_llm_judge/data/example.jsonl \
  +output_jsonl_fpath=results/example_rollouts.jsonl \
  +limit=5
```

### Test cases

```bash
ng_test +entrypoint=responses_api_models/azure_openai_model
```

## Licensing information

- **Code**: Apache 2.0
- **Data**: N/A

## Dependencies

- `nemo_gym`: Apache 2.0
