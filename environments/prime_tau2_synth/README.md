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

The user simulator also runs on the policy endpoint. To use a different user model, override `vf_env_args.user_model`, `user_base_url` and `user_api_key_var` in `config.yaml`.

```
# start nemo gym servers
gym env start --environment prime_tau2_synth --model-type vllm_model

# generate rollouts
gym eval run --no-serve \
    --agent prime_tau2_synth_agent \
    --input environments/prime_tau2_synth/data/example.jsonl \
    --output environments/prime_tau2_synth/data/example-rollouts.jsonl \
    --limit 3

# view the rollout
tail -n 1 environments/prime_tau2_synth/data/example-rollouts.jsonl | jq | less
```

## Prepare data

`vf_env_args.domain` in `config.yaml` picks the domain for every row, so prepare data with the same `--domain`. Run `prepare.py` in the agent venv, which `gym env start` creates:
```
responses_api_agents/prime_tau2_synth_agent/.venv/bin/python environments/prime_tau2_synth/prepare.py \
    --domain library \
    --output environments/prime_tau2_synth/data/library_train.jsonl
```

`data/example.jsonl` is the first 5 library tasks (`--limit 5`).

# Licensing information
Code: Apache 2.0
Data: Apache 2.0 (tau2-synth tasks from Prime Intellect's [prime-envs](https://github.com/PrimeIntellect-ai/prime-envs))

Dependencies
- nemo_gym: Apache 2.0
- verifiers: MIT
- tau2-synth: Apache 2.0
- tau2: MIT
