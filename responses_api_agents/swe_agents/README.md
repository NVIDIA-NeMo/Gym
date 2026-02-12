# Quick Start: Running SWE Agents

This guide shows how to run the SWE agents that use OpenAI GPT-4.1 (or any other model) to solve real-world GitHub issues.

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
    +output_jsonl_fpath=swebench-verified.openhands.qwen3-30b-coder.jsonl \
    +model=Qwen/Qwen3-Coder-30B-A3B-Instruct \
    +temperature=0.7 \
    +top_p=0.8
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
  "responses_create_params": {
    "background": null,
    "include": null,
    "input": [
      {
        "content": "You are OpenHands agent, a helpful AI assistant...",
        "role": "system",
        "type": "message"
      },
      {
        "content": "I've uploaded a python code repository...",
        "role": "user", 
        "type": "message"
      }
    ],
    "instructions": null,
    "max_output_tokens": null,
    "max_tool_calls": null,
    "metadata": {
      "instance_id": "astropy__astropy-12907",
      "base_commit": "",
      "dataset_name": "princeton-nlp/SWE-bench_Verified",
      "split": "test",
      "problem_statement": "Modeling's `separability_matrix` does not compute separability correctly for nested CompoundModels\nConsider the following model:\n\n```python\nfrom astropy.modeling import models as m\nfrom astropy.modeling.separable import separability_matrix\n\ncm = m.Linear1D(10) & m.Linear1D(5)\n```\n\nIt's separability matrix as you might expect is a diagonal:\n\n```python\n>>> separability_matrix(cm)\narray([[ True, False],\n[False, True]])\n```\n\nIf I make the model more complex:\n```python\n>>> separability_matrix(m.Pix2Sky_TAN() & m.Linear1D(10) & m.Linear1D(5))\narray([[ True, True, False, False],\n[ True, True, False, False],\n[False, False, True, False],\n[False, False, False, True]])\n```\n\nThe output matrix is again, as expected, the outputs and inputs to the linear models are separable and independent of each other.\n\nIf however, I nest these compound models:\n```python\n>>> separability_matrix(m.Pix2Sky_TAN() & cm)\narray([[ True, True, False, False],\n[ True, True, False, False],\n[False, False, True, True],\n[False, False, True, True]])\n```\nSuddenly the inputs and outputs are no longer separable?\n\nThis feels like a bug to me, but I might be missing something?"
    },
    "model": "Qwen/Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "parallel_tool_calls": true,
    "previous_response_id": null,
    "prompt": null,
    "reasoning": null,
    "service_tier": null,
    "store": null,
    "temperature": 0.7,
    "text": null,
    "tool_choice": "auto",
    "tools": [...]
  },
  "response": {
    "id": "swebench-astropy__astropy-12907",
    "created_at": 1757366053,
    "error": null,
    "incomplete_details": null,
    "instructions": null,
    "metadata": null,
    "model": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "object": "response",
    "output": [
      {
        "id": "msg-2",
        "content": [
          {
            "annotations": [],
            "text": "I'll help you implement the necessary changes...",
            "type": "output_text",
            "logprobs": null
          }
        ],
        "role": "assistant",
        "status": "completed",
        "type": "message"
      }
    ],
    "parallel_tool_calls": true,
    "temperature": null,
    "tool_choice": "auto",
    "tools": [...],
    "top_p": null,
    "background": null,
    "max_output_tokens": null,
    "max_tool_calls": null,
    "previous_response_id": null,
    "prompt": null,
    "reasoning": null,
    "service_tier": null,
    "status": null,
    "text": null,
    "top_logprobs": null,
    "truncation": null,
    "usage": null,
    "user": null
  },
  "reward": 1.0,
  "swebench_metrics": {
    "patch_is_None": false,
    "patch_exists": true,
    "patch_successfully_applied": true,
    "resolved": true,
    },
  "resolved": 1,
  "patch_exists": 1,
  "patch_successfully_applied": 1,
  "metadata": {
    "instance_id": "astropy__astropy-12907",
    "agent_framework": "openhands",
    "patch_exists": true,
    "patch_successfully_applied": true,
    "resolved": true
  }
}
```
