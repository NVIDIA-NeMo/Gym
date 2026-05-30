# Description

This environment runs τ²-bench with custom synthetic domains via Prime Intellect [verifiers](https://github.com/PrimeIntellect-ai/verifiers) environments. See [tau2-synth on Prime Intellect](https://app.primeintellect.ai/dashboard/environments/prime/tau2-synth) for the full description.

Domains: `library`, `fitness_gym`, `tech_support`, `telecom`, `cloud_incident_response`, `daily_planner`, `ev_charging_support`.

For installing additional prime environments from the Environments Hub and generating datasets, see [responses_api_agents/verifiers_agent/README.md](../../responses_api_agents/verifiers_agent/README.md).

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

## Integration notes

The patch to include prompt and generation token ids for preventing retokenization error when training with NeMo RL has been upstreamed into verifiers' `NeMoRLChatCompletionsClient`. We currently track verifiers `main` (`git+https://github.com/PrimeIntellect-ai/verifiers.git@main`) for this support; once verifiers `0.1.13` is released, the requirements pin will move to that tagged version.

For installing new prime environments and generating datasets, use a separate venv (outside of Gym) to avoid dependency conflicts with the `exclude-dependencies` section of Gym `pyproject.toml` and the server's pinned verifiers version. After generating your dataset, deactivate the separate venv and return to the Gym venv for running servers. Make sure to restart NeMo Gym servers with `ng_run` after any environment changes to ensure the pinned version of verifiers is used.

# Licensing information
Code: Apache 2.0
Data: N/A

Dependencies
- nemo_gym: Apache 2.0
- verifiers: Apache 2.0
