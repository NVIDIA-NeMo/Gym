# Description

This environment is for training on τ²-bench style tasks with custom synthetic data via [tau2-synth](https://app.primeintellect.ai/dashboard/environments/prime/tau2-synth) on Prime Intellect Environments hub through verifiers integration. See tau2-synth for the full description and configuration details.

Domains: `library`, `fitness_gym`, `tech_support`, `telecom`, `cloud_incident_response`, `daily_planner`, `ev_charging_support`.

For additional details on the prime verifiers integration in NeMo Gym, see [responses_api_agents/verifiers_agent/README.md](../../responses_api_agents/verifiers_agent/README.md).

## Install Gym

```
git clone https://github.com/NVIDIA-NeMo/Gym
cd Gym
uv venv; source .venv/bin/activate; uv sync
```

## Test tau2-synth example

First set `env.yaml`, for example for a vLLM served model:
```
policy_base_url: "http://localhost:8000/v1"
policy_api_key: EMPTY
policy_model_name: "Qwen/Qwen3-4B-Instruct-2507"
```

```
# start nemo gym servers
ng_run "+config_paths=[environments/prime_tau2_synth/config.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"

# generate a rollout
ng_collect_rollouts \
    +agent_name=prime_tau2_synth_agent \
    +input_jsonl_fpath=environments/prime_tau2_synth/data/example.jsonl \
    +output_jsonl_fpath=environments/prime_tau2_synth/data/example-rollouts.jsonl \
    +limit=1

# view the rollout
tail -n 1 environments/prime_tau2_synth/data/example-rollouts.jsonl | jq | less
```

# Licensing information
Code: Apache 2.0
Data: N/A

Dependencies
- nemo_gym: Apache 2.0
- verifiers: Apache 2.0
