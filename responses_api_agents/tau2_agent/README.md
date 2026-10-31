Tau2 agent — how to run experiments
=================================

This document shows the minimal steps to run tau2 experiments locally.

*Steps*
1) Configure your API Key
```bash
echo "policy_base_url: https://api.openai.com/v1
policy_api_key: your-openai-api-key
policy_model_name: gpt-4.1-2025-04-14" > env.yaml
```

2) Setup Tau^2 data

- Download the `tau2` folder (https://github.com/sierra-research/tau2-bench/tree/main/data/tau2). 
- Save it to `resources_servers/tau2_bench/data/`.
- Configure data path (*don't forget* to modify the path accordingly):
```bash
export TAU2_DATA_DIR="/your_path/to/resources_servers/tau2_bench/data/"
```

3) Launch the NemoGym server
- In the *first terminal*, launch the server.

Example server for `openai_model`:
```bash
config_paths="responses_api_agents/tau2_agent/configs/tau2_agent.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/tau2_bench/configs/tau2_bench.yaml"

ng_run "+config_paths=[$config_paths]" \
+tau2_agent.responses_api_agents.tau2_agent.resources_server.name=tau2_bench_resources_server
```

Example server for `vllm_model`:
```bash
config_paths="responses_api_agents/tau2_agent/configs/tau2_agent.yaml,\
responses_api_models/vllm_model/configs/vllm_model.yaml,\
resources_servers/tau2_bench/configs/tau2_bench.yaml"

ng_run "+config_paths=[$config_paths]" \
    +tau2_agent.responses_api_agents.tau2_agent.resources_server.name=tau2_bench_resources_server \
+policy_model.responses_api_models.vllm_model.return_token_id_information=true
```

4) Prepare experiment input
- Prepare an input JSONL file describing which domain/task(s) to run. Set the path in the `input_jsonl_fpath`. An example is in `resources_servers/tau2_bench/data/example_retail_demo.jsonl`

5) Collect rollouts from Tau^2 Bench (separate terminal)
- In the *second (separate) terminal*, launch the rollout script to kick off the experiment:

```bash
ng_collect_rollouts +agent_name=tau2_agent \
    +input_jsonl_fpath=resources_servers/tau2_bench/data/example_retail_demo.jsonl \
    +output_jsonl_fpath=resources_servers/tau2_bench/data/example_retail_demo_rollouts.jsonl \
    +limit=1 \
    +num_repeats=1 \
    +num_samples_in_parallel=null
```
