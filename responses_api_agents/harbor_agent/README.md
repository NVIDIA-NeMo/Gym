# Harbor Agent for NeMo Gym

This agent integrates [Harbor](https://github.com/laude-institute/harbor) into NeMo Gym.
It runs Harbor agents (e.g., `terminus-2`) in Harbor-managed environments and returns NeMo Gym-compatible outputs.

## Quick Start

### 1) Prerequisites

- Install Apptainer/Singularity.

```bash
apt-get update && apt-get install -y git wget
cd /tmp
wget https://github.com/apptainer/apptainer/releases/download/v1.4.2/apptainer_1.4.2_amd64.deb
apt-get install -y ./apptainer_1.4.2_amd64.deb
apptainer --version
```

- Prepare Apptainer/Singularity images. For how to download images and convert to .sif, you can refer to https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/dataset/swe-bench/dump_images.py.

- (Optional) Set private registry credentials for Singularity pulls.

```bash
export APPTAINER_DOCKER_USERNAME=<registry-username>
export APPTAINER_DOCKER_PASSWORD=<registry-password-or-token>
```

### 2) Configure model endpoint in `env.yaml`

Harbor agent reads model routing from NeMo Gym global config:

```yaml
policy_base_url: http://localhost:8000/v1
policy_api_key: placeholder
policy_model_name: meta-llama/Llama-3.1-8B-Instruct
```

### 3) Configure Harbor agent

Modify `configs/harbor_agent.yaml`.

To use local custom wrappers (and keep Harbor installed from upstream), set import
paths under `responses_api_agents/harbor_agent/custom_agents` and
`responses_api_agents/harbor_agent/custom_envs`:

```yaml
harbor_agent_name: null
harbor_agent_import_path: "responses_api_agents.harbor_agent.custom_agents.terminus_2_nemo_gym:Terminus2NemoGym"
harbor_environment_type: null
harbor_environment_import_path: "responses_api_agents.harbor_agent.custom_envs.singularity.singularity:SingularityEnvironment"
harbor_agent_kwargs:
  collect_rollout_details: true
  nemo_model_server_api_key: placeholder
```

To route Harbor through a configured NeMo Gym model server (same pattern as
`swe_agents`/`mini_swe_agent`), add:

```yaml
model_server:
  type: responses_api_models
  name: policy_model
```

Harbor agent resolves the base URL from the configured `model_server` host/port.
`model_server` is required.

### 4) Start NeMo Gym servers

You only need the Harbor agent config path (no separate NeMo model-server config required).

```bash
config_paths="responses_api_agents/harbor_agent/data/harbor_agent_test.yaml"
ng_run "+config_paths=[${config_paths}]"
```

### 5) Test Harbor agent

```bash
python responses_api_agents/harbor_agent/client.py
```

### 6) Collect rollouts

```bash
ng_collect_rollouts +agent_name=harbor_agent \
  +input_jsonl_fpath=responses_api_agents/harbor_agent/data/example_input.jsonl \
  +output_jsonl_fpath=responses_api_agents/harbor_agent/data/example_output.jsonl
```

### 7) View trajectories

```bash
ng_viewer +jsonl_fpath=responses_api_agents/harbor_agent/data/example_output.jsonl
```

## Notes

- Harbor agent is self-contained: Harbor handles task environment + verifier internally.
- The NeMo output converter behavior:
  - If `trajectory.json` exists: keep rich output (`message`, `function_call`, `function_call_output`).
  - If `rollout_details` exist: overlay token IDs/logprobs onto assistant turns.
  - If neither exists: return empty `output`.

For Harbor related questions, check out the official Harbor docs: https://harborframework.com/docs. 
