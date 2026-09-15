# Description

This agent enables running Prime Intellect [verifiers](https://github.com/PrimeIntellect-ai/verifiers) environments, including many in Prime Intellect's [Environments Hub](https://app.primeintellect.ai/dashboard/environments?ex_sort=by_sections) in NeMo Gym. 

For a tested, one-task setup without a Prime Hub account, start with the canonical
[legacy Verifiers onboarding recipe](https://docs.nvidia.com/nemo/gym/main/get-started/verifiers-onboarding).
It provisions the agent's independent server environment and uses the pinned
Verifiers **v0.1.14** API. Current Hub packages must be checked individually for
compatibility; this adapter does not implement the Verifiers v1 migration.

## Install Gym

```
git clone https://github.com/NVIDIA-NeMo/Gym
cd Gym
uv sync --frozen --extra dev --python 3.13.14
source .venv/bin/activate
```

## Test acereason-math example 

First set `env.yaml`, for example for a vLLM served model:
```
policy_base_url: "http://localhost:8000/v1"
policy_api_key: EMPTY
policy_model_name: "Qwen/Qwen3-4B-Instruct-2507"
```

```
# start nemo gym servers
gym env start \
    --config responses_api_agents/verifiers_agent/configs/verifiers_agent.yaml \
    --model-type vllm_model \
    +verifiers_agent.responses_api_agents.verifiers_agent.max_tokens=256

# generate a rollout
gym eval run --no-serve \
    --agent verifiers_agent \
    --input responses_api_agents/verifiers_agent/data/acereason-math-example.jsonl \
    --output responses_api_agents/verifiers_agent/data/acereason-math-example-rollouts.jsonl \
    --limit 1 --num-repeats 1 --concurrency 1 --max-output-tokens 256

# view the rollout
tail -n 1 responses_api_agents/verifiers_agent/data/acereason-math-example-rollouts.jsonl | jq | less
```


## Testing new prime environments from environments hub

Some examples: `primeintellect/acereason-math`, `primeintellect/ascii-tree` and `primeintellect/alphabet-sort`.

The historical Hub workflow below is not a compatibility guarantee for current
package releases. Choose a package version that supports Verifiers v0.1.14;
use the local onboarding recipe above as the first control. Apply an approved
request/token allowance before model runs: sample limits do not bound retries.

### Install an environment
```
# deactivate the main nemo gym virtual environment
deactivate

# create a separate venv for installing prime environments
# (avoids dependency conflicts with the Gym or server venvs)
# for example, use ~/prime_venv
mkdir -p /path/to/prime_venv && cd /path/to/prime_venv
uv venv && source .venv/bin/activate

# install prime CLI and the environment
uv pip install prime
prime env install primeintellect/ascii-tree
```

### Create dataset
```
# navigate to the verifiers_agent directory (with the prime venv still active)
cd <gym-root>/responses_api_agents/verifiers_agent
python3 scripts/create_dataset.py --env-id primeintellect/ascii-tree --size 5 --output data/ascii-tree-example.jsonl
```

### Update agent server requirements
```
-e nemo-gym[dev] @ ../../
verifiers @ git+https://github.com/PrimeIntellect-ai/verifiers.git@v0.1.14
--extra-index-url https://hub.primeintellect.ai/primeintellect/simple/
ascii-tree
```
### Update agent config
Create `configs/ascii-tree.yaml`, primarily updating env id, and any other env specific args: 
```
verifiers_agent:
  responses_api_agents:
    verifiers_agent:
      entrypoint: app.py
      model_server:
        type: responses_api_models
        name: policy_model
      model_name: ""
      vf_env_id: ascii-tree
      vf_env_args: {}
      max_tokens: 256
      temperature: 1.0
      top_p: 1.0

```

```
# return to Gym/ root and activate Gym/ virtual environment
cd ../../
deactivate
source .venv/bin/activate

# start nemo gym servers
gym env start \
    --config responses_api_agents/verifiers_agent/configs/ascii-tree.yaml \
    --model-type vllm_model

# generate a rollout
gym eval run --no-serve \
    --agent verifiers_agent \
    --input responses_api_agents/verifiers_agent/data/ascii-tree-example.jsonl \
    --output responses_api_agents/verifiers_agent/data/ascii-tree-example-rollouts.jsonl \
    --limit 1 --num-repeats 1 --concurrency 1 --max-output-tokens 256
```

## Integration notes

The support for prompt and generation token IDs used by NeMo RL is in verifiers' `NeMoRLChatCompletionsClient`. The adapter pins `verifiers @ git+https://github.com/PrimeIntellect-ai/verifiers.git@v0.1.14`; it does not track `main`. Use Python 3.13.14 for the shared Gym and legacy Verifiers dependency range. See the canonical recipe for the tested setup and current error/token preservation limitations.

For installing new prime environments and generating datasets, use a separate venv (outside of Gym) to avoid dependency conflicts with the `exclude-dependencies` section of Gym `pyproject.toml` and the server's pinned verifiers version. After generating your dataset, deactivate the separate venv and return to the Gym venv for running servers. Make sure to restart NeMo Gym servers with `gym env start` after any environment changes to ensure the pinned version of verifiers is used.

# Licensing information
Code: Apache 2.0
Data: N/A

Dependencies
- nemo_gym: Apache 2.0
- verifiers: Apache 2.0
