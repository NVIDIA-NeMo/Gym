# Quick Start: Running SWE Agents

This guide shows how to run the SWE agents that use OpenAI GPT-4 (or any other model) to solve real-world GitHub issues.

## Prerequisites

1. **Install Apptainer** (for container execution):
```bash
# Install Apptainer on Ubuntu/Debian
apt install -y wget && \
    cd /tmp && \
    wget https://github.com/apptainer/apptainer/releases/download/v1.4.1/apptainer_1.4.1_amd64.deb && \
    apt install -y ./apptainer_1.4.1_amd64.deb

# Verify installation
apptainer --version
```


## Step 1: Configure Your API Key

Create or update your `env.yaml` file in the NeMo-Gym root directory:

```yaml
# For OpenAI models
policy_base_url: https://api.openai.com/v1
policy_api_key: {your OpenAI API key}
policy_model_name: gpt-4.1-2025-04-14
```

You can also host a vLLM model.

Start VLLM server (in separate terminal):
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --max-model-len 131072 \
  --enable-expert-parallel \
  --tensor-parallel-size 4 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --port 8000 \
  --enforce-eager
```
Then set
```yaml
policy_base_url: http://localhost:8000/v1
policy_api_key: dummy
policy_model_name: Qwen/Qwen3-Coder-30B-A3B-Instruct
```


## Step 2: Run the SWE Agents

Start the servers with SWE-agent configuration:

```bash
# Define config paths
# OpenAI model
config_paths="responses_api_agents/swe_agents/configs/swebench_swe_agent.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml"

or
# vLLM model
config_paths="responses_api_agents/swe_agents/configs/swebench_swe_agent.yaml,\
responses_api_models/vllm_model/configs/vllm_model.yaml"

# Run the servers
# If you have pre-downloaded images, you can set the path with container_formatter, e.g.
ng_run "+config_paths=[$config_paths]" \
     +swe_agents.responses_api_agents.swe_agents.container_formatter=/lustre/xxx/images/swe-bench/swebench_sweb.eval.x86_64.\{instance_id\}.sif \
     +swe_agents.responses_api_agents.swe_agents.model_server.name=vllm_model 

```

To run OpenHands server, simply replace the SWE-agent config path to OpenHands config 
```bash
responses_api_agents/swe_agents/configs/swebench_openhands.yaml
```

For how to download images and convert to .sif, you can refer to https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/dataset/swe-bench/dump_images.py


You should see output like:
```
INFO:     Started server process [1815588]
INFO:     Uvicorn running on http://127.0.0.1:25347 (Press CTRL+C to quit)
INFO:     Started server process [1815587]
INFO:     Uvicorn running on http://127.0.0.1:56809 (Press CTRL+C to quit)
```

## Step 3: Query the Agent

In a new terminal, run the client script:

```bash
python responses_api_agents/swe_agents/client.py
```


## Advanced usage: Run Batch  Evaluation/Data Collection

For multiple problems, use rollout collection:

```
# Collect rollouts
ng_collect_rollouts +agent_name=swe_agents \
    +input_jsonl_fpath=swebench-verified-converted.jsonl \
    +output_jsonl_fpath=swebench-verified.openhands.qwen3-30b-coder.jsonl
```
By default, the concurrency of ng_collect_rollouts is 100. You may want to adjust it based on your hardware configuration accordingly. 

## Step 6: View Results

View the collected results:

```bash
ng_viewer +jsonl_fpath=swebench-verified.openhands.qwen3-30b-coder.jsonl
```


## Expected Output

A successful run will show:
```json
{
  "id": "swebench-astropy__astropy-12907",
  "output": [
    {
      "type": "text",
      "text": "{\n  \"instance_id\": \"astropy__astropy-12907\",\n  \"swe-bench-metrics\": {\n    \"resolved\": true,\n    \"patch_exists\": true,\n    \"patch_successfully_applied\": true\n  },\n  \"swe-bench-outputs\": {\n    \"model_patch\": \"diff --git ...\\n...\"\n  }\n}"
    }
  ],
  "metadata": {
    "swebench_result": {
      "instance_id": "astropy__astropy-12907",
      "swe-bench-metrics": {
        "resolved": true,
        "patch_exists": true,
        "patch_successfully_applied": true
      }
    },
    "agent_framework": "swe_agent"
  }
}
```
