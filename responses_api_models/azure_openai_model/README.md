# Azure OpenAI Model

## Configuration

Set up your `env.yaml` file:
```yaml
policy_base_url: https://prod.api.nvidia.com/llm/v1/azure
policy_api_key: <API_KEY>
policy_model_name: gpt-5-nano
policy_api_version: 2024-10-21
```

## Usage

### Running the server
```bash
config_paths="responses_api_models/azure_openai_model/configs/azure_openai_model.yaml, \
resources_servers/equivalence_llm_judge/configs/equivalence_llm_judge.yaml"

ng_run "+config_paths=[${config_paths}]"
```

### Collecting Rollouts

```
ng_collect_rollouts +agent_name=equivalence_llm_judge_simple_agent   +input_jsonl_fpath=resources_servers/equivalence_llm_judge/data/example.jsonl   +output_jsonl_fpath=results/example_rollouts.jsonl   +limit=5
```

### Test cases:

```
ng_test +entrypoint=responses_api_models/azure_openai_model
```

# Licensing information
Code: Apache 2.0
Data: N/A

Dependencies
- nemo_gym: Apache 2.0
