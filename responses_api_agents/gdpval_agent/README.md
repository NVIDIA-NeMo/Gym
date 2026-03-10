# Description

GDPVal responses-api agent for running tool-augmented task rollouts and delegating scoring to a configured resources server.

# Terminal 1: Starting the servers
```bash
# Start terminal 1
srun --time 4:00:00 -A llmservice_modelalignment_sft --gres=gpu:8 -p interactive --pty /bin/bash

# Init Env
cd /lustre/fsw/portfolios/llmservice/users/vadams/Gym
source .venv/bin/activate

uv pip install tavily-python

export NVIDIA_API_KEY=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
export TAVILY_API_KEY="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"

# bash_sandbox.yaml starts: bash_sandbox_resources_server + bash_sandbox_agent
# nano_v3_single_node.yaml starts: policy_model (local vLLM Nemotron)
RESOURCE_AND_AGENT_CONFIG=resources_servers/bash_sandbox/configs/bash_sandbox.yaml
MODEL_SERVER_CONFIG=responses_api_models/vllm_model/configs/vllm_model.yaml

# Start all servers
ng_run "+config_paths=[${RESOURCE_AND_AGENT_CONFIG},${MODEL_SERVER_CONFIG}]" \
      "++policy_base_url=https://integrate.api.nvidia.com/v1" \
      "++policy_api_key=${NVIDIA_API_KEY}" \
      "++policy_model_name=nvidia/nemotron-3-nano-30b-a3b"
```

# Terminal 2: Running the agent
```bash
# Start terminal 2
srun --jobid=9854513 --overlap --pty -c 8 --gres=none bash

# Init Env
cd /lustre/fsw/portfolios/llmservice/users/vadams/Gym
source .venv/bin/activate

# Test the agent
python responses_api_agents/gdpval_agent/client.py prepare \
    --output-jsonl tmp/gdpval_tasks.jsonl \
    --split train \
    --limit 220 \
    --output-dir tmp/gdpval_output

ng_collect_rollouts +agent_name=bash_sandbox_agent \
    +input_jsonl_fpath=tmp/gdpval_tasks.jsonl \
    +output_jsonl_fpath=tmp/gdpval_rollouts.jsonl \
    +limit=220 \
    +num_samples_in_parallel=16

ng_collect_rollouts +agent_name=bash_sandbox_agent \
    +input_jsonl_fpath=tmp/web_search_tst.jsonl \
    +output_jsonl_fpath=tmp/web_search_rollout.jsonl \
    +limit=1 \
    +num_samples_in_parallel=1
```

# Scratchpad
```bash
ng_init_resources_server +entrypoint=resources_servers/bash_sandbox
```

# Licensing information
Code: Apache 2.0
Data: N/A

Dependencies
- nemo_gym: Apache 2.0


