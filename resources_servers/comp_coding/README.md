# Competitive Coding Resources Server

### Overview
Verifies competitive programming solutions by executing submitted code against unit tests. The server consumes agent trajectories and returns a reward based on whether the assistant's code produces the correct outputs for given test inputs.
Model registry link:
- Train: https://gitlab-master.nvidia.com/bxyu/nemo-gym/-/ml/models/80#/
- Validation: https://gitlab-master.nvidia.com/bxyu/nemo-gym/-/ml/models/76#/

### Input schema
- `responses_create_params`: OpenAI Responses create params
  - Use only a user message with the problem statement and instructions (e.g., "You are an expert competitive programmer...").
  - `verifier_metadata` (required):
    - `unit_tests` (required): dict with `inputs` and `outputs` arrays containing test cases.
      - `inputs`: list of strings representing stdin input for each test case
      - `outputs`: list of strings representing expected stdout output for each test case

**Notes**
- All test cases must pass for a solution to receive a reward of 1.0
- Failed test cases result in a reward of 0.0 with detailed error information

### Test execution (for now)
We use the LiveCodeBench execution code.

### Example of rollouts and usage

```bash
# Running the server
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/comp_coding/configs/comp_coding.yaml"
ng_run "+config_paths=[$config_paths]"

# Collect rollouts from example problems
ng_collect_rollouts +agent_name=comp_coding_simple_agent \
    +input_jsonl_fpath=resources_servers/comp_coding/data/example.jsonl \
    +output_jsonl_fpath=resources_servers/comp_coding/data/example_rollouts.jsonl \
    +limit=null
```

## Licensing information
Apache 2.0
