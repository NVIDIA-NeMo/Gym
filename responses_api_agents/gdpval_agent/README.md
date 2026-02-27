# Description

GDPVal responses-api agent for running tool-augmented task rollouts and delegating scoring to a configured resources server.

# Terminal 1: Starting the servers
```bash
cd /lustre/fsw/portfolios/llmservice/users/vadams/Gym
source .venv/bin/activate

# bash_sandbox.yaml starts: bash_sandbox_resources_server + bash_sandbox_agent
# nano_v3_single_node.yaml starts: policy_model (local vLLM Nemotron)
RESOURCE_AND_AGENT_CONFIG=resources_servers/bash_sandbox/configs/bash_sandbox.yaml
MODEL_SERVER_CONFIG=responses_api_models/local_vllm_model/configs/nano_v3_single_node.yaml

# Start all servers
ng_run "+config_paths=[${RESOURCE_AND_AGENT_CONFIG},${MODEL_SERVER_CONFIG}]"
```

# Terminal 2: Running the agent
```bash
cd /lustre/fsw/portfolios/llmservice/users/vadams/Gym
source .venv/bin/activate

# Test the agent
python responses_api_agents/gdpval_agent/client.py
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


