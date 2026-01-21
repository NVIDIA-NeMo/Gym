(env-multi-node-docker)=
# Multi-Node Deployment with Containers

Scale NeMo Gym environments across multiple machines using container orchestration.

:::{card}

**Goal**: Deploy NeMo Gym across multiple nodes using Docker containers.

^^^

**In this tutorial, you will**:

1. Configure coordinator and worker nodes
2. Deploy containerized resources servers
3. Connect to NeMo RL for distributed training

:::

:::{button-ref} user-modeling
:color: secondary
:outline:
:ref-type: doc

← Previous: User Modeling
:::

---

## Overview

NeMo Gym supports two deployment approaches for scaling beyond a single machine:

| Approach | Use Case | Orchestration |
|----------|----------|---------------|
| **Standalone containers** | Running environments at scale without training | Manual or Docker Compose |
| **NeMo RL integration** | Production training with GRPO, DPO, etc. | Slurm + Ray (automatic) |

This guide covers standalone container deployment. For training workflows, see the {doc}`/tutorials/nemo-rl-grpo/index` tutorial.

:::{dropdown} Terminology
:icon: book

**Rollout**
: A complete interaction sequence between the model and environment—from initial prompt through tool calls to final reward.

**GRPO** (Group Relative Policy Optimization)
: A reinforcement learning algorithm for training language models.

**DPO** (Direct Preference Optimization)
: An alternative to RLHF that learns from preference data.
:::

---

## Architecture

NeMo Gym uses a head server for service discovery and configuration distribution:

```text
┌─────────────────────────────┐
│       Head Server           │
│    (port 11000 default)     │
│  /global_config_dict_yaml   │
│  /server_instances          │
└──────────────┬──────────────┘
               │
    ┌──────────┼──────────┐
    │          │          │
┌───┴───┐  ┌───┴───┐  ┌───┴───┐
│ Agent │  │ Model │  │ Rsrc  │
│Server │  │Server │  │Server │
└───────┘  └───────┘  └───────┘
```

| Component | Role |
|-----------|------|
| **Head Server** | Configuration distribution, service discovery |
| **Agent Server** | Orchestrates rollout collection |
| **Model Server** | Provides inference (vLLM, OpenAI API) |
| **Resources Server** | Environment-specific logic (verification, rewards) |

---

## Single-Node (Default)

The `ng_run` command starts all servers in a single process:

```bash
config_paths="resources_servers/example_single_tool_call/configs/example_single_tool_call.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml"

ng_run "+config_paths=[$config_paths]"
```

This starts:
- Head server on port 11000
- All configured servers as subprocesses
- Servers register with the head server automatically

---

## Multi-Node Deployment

For multi-node deployment, run `ng_run` on each node with configuration pointing to a shared head server.

### Step 1: Configure the Coordinator Node

Create a complete configuration file for the coordinator:

```yaml
# coordinator-config.yaml

# Head server configuration - accessible to all nodes
head_server:
  host: "192.168.1.100"  # Coordinator's IP
  port: 11000

# Resources server
example_resources:
  resources_servers:
    example_single_tool_call:
      entrypoint: app.py
      domain: agent
      host: "192.168.1.100"
      port: 8080

# Agent server
example_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      host: "192.168.1.100"
      port: 8081
      resources_server:
        type: resources_servers
        name: example_resources
      model_server:
        type: responses_api_models
        name: policy_model
      datasets:
      - name: train
        type: train
        jsonl_fpath: resources_servers/example_single_tool_call/data/example.jsonl
```

Start the coordinator:

```bash
ng_run "+config_paths=[coordinator-config.yaml]"
```

### Step 2: Configure Worker Nodes

On each worker node, create a configuration that points to the coordinator's head server:

```yaml
# worker-config.yaml

# Point to the coordinator's head server
head_server:
  host: "192.168.1.100"  # Same as coordinator
  port: 11000

# Model server (GPU-intensive, runs on worker)
policy_model:
  responses_api_models:
    openai_model:
      entrypoint: app.py
      host: "192.168.1.101"  # This worker's IP
      port: 8000
      openai_base_url: "http://localhost:8080/v1"  # Local vLLM
      openai_api_key: "not-needed"  # pragma: allowlist secret
      openai_model: "meta-llama/Llama-3.1-8B-Instruct"
```

Start the worker:

```bash
ng_run "+config_paths=[worker-config.yaml]"
```

### Step 3: Connect to Existing Ray Cluster (Optional)

For CPU-intensive workloads, connect all nodes to a shared Ray cluster:

```yaml
# Add to any config file
ray_head_node_address: "ray://192.168.1.100:10001"
```

When `ray_head_node_address` is set, NeMo Gym connects to that cluster instead of starting a new one.

---

## Containerized Deployment

### Dockerfile for Resources Server

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install NeMo Gym
RUN pip install --no-cache-dir nemo-gym

# Copy server code
COPY . .

EXPOSE 8080

# Server reads config from environment variable
CMD ["python", "app.py"]
```

### Building the Container

```bash
cd resources_servers/example_single_tool_call
docker build -t my-resources-server:latest .
```

### Running with Configuration

Pass configuration via the `NEMO_GYM_CONFIG_DICT` environment variable:

```bash
docker run -p 8080:8080 \
    -e NEMO_GYM_CONFIG_DICT='
head_server:
  host: "192.168.1.100"
  port: 11000
example_resources:
  resources_servers:
    example_single_tool_call:
      entrypoint: app.py
      host: "0.0.0.0"
      port: 8080
' \
    my-resources-server:latest
```

---

## Production Training with NeMo RL

```{admonition} Different from Docker deployment
:class: important

For production training, use NeMo RL's Slurm + Ray orchestration instead of manual Docker deployment. NeMo RL handles cluster management automatically.
```

**How it works**:

1. **Slurm** allocates physical nodes (GPUs, memory)
2. **Ray** connects nodes into a unified compute cluster
3. **NeMo Gym** runs as a Ray actor, receiving the cluster address automatically

### Launch Script

```bash
# From NeMo RL repository (not NeMo Gym)
cd $REPO_LOCATION

WANDB_API_KEY=$WANDB_API_KEY \
NUM_ACTOR_NODES=4 \
CONTAINER_IMAGE_PATH=$CONTAINER \
SLURM_ACCOUNT=$ACCOUNT \
SLURM_PARTITION=$PARTITION \
bash examples/nemo_gym/launch_nemo_gym_multinode_training.sh
```

**Source**: `nemo_rl/examples/nemo_gym/launch_nemo_gym_multinode_training.sh`

---

## Network Requirements

| Port | Purpose | Must Be Accessible To |
|------|---------|----------------------|
| 11000 | Head server | All worker nodes |
| 8080 | Resources server (default) | Agent server |
| 8000 | Model server (default) | Agent server |

### Firewall Configuration

```bash
# Allow head server access from worker subnet
sudo ufw allow from 192.168.1.0/24 to any port 11000
```

---

## Service Discovery

Servers discover each other via the head server using name-based routing:

```python
# Servers call each other by name, not IP
await self.server_client.post(
    server_name="example_resources",
    url_path="/verify",
    json=payload
)
```

---

## Verifying Your Deployment

### Check Head Server Status

```bash
curl http://192.168.1.100:11000/server_instances
```

**Expected output** (JSON array of registered servers):

```json
[
  {
    "process_name": "example_resources",
    "server_type": "resources_servers",
    "name": "example_single_tool_call",
    "host": "192.168.1.100",
    "port": 8080,
    "url": "http://192.168.1.100:8080",
    "pid": 12345,
    "entrypoint": "app.py",
    "config_path": "example_resources",
    "start_time": 1704067200.0
  }
]
```

```{tip}
The response may include additional fields like `dir_path`, `status`, and `uptime_seconds`.
```

### Check Configuration Distribution

```bash
curl http://192.168.1.100:11000/global_config_dict_yaml
```

**Expected output** (YAML configuration):

```yaml
head_server:
  host: 192.168.1.100
  port: 11000
example_resources:
  resources_servers:
    example_single_tool_call:
      entrypoint: app.py
      host: 192.168.1.100
      port: 8080
# ... rest of config
```

---

## Troubleshooting

| Symptom | Cause | Solution |
|---------|-------|----------|
| "Could not connect to head server" | Head server unreachable | Verify network connectivity, check firewall rules |
| Servers not finding each other | Different head server configs | Ensure all nodes use same `head_server.host` |
| Ray tasks failing | Ray cluster not shared | Set `ray_head_node_address` to same cluster |
| Empty `/server_instances` response | Servers not registered | Check server startup logs, verify config paths |

### Startup Order

1. **Start coordinator first** — Head server must be running before workers connect
2. **Wait for head server** — Verify with `curl http://<coordinator>:11000/server_instances`
3. **Start workers** — Workers will register with the head server on startup

---

## Production Checklist

Before deploying to production:

- [ ] **Network isolation**: Servers not exposed to public internet
- [ ] **TLS termination**: Use nginx/traefik reverse proxy for HTTPS
- [ ] **Firewall rules**: Restrict access to known IPs only
- [ ] **Log aggregation**: Ship logs to centralized logging system
- [ ] **Health monitoring**: Set up alerts for server availability
- [ ] **Resource limits**: Configure container memory/CPU limits

```{warning}
NeMo Gym servers have no built-in authentication. Use network-level security (VPC, firewalls) to restrict access.
```

---

## Limitations

```{note}
NeMo Gym's multi-node support is designed primarily for integration with NeMo RL training pipelines. Standalone multi-node deployment requires manual configuration of each node.

Future releases may include:
- CLI flags for head-only and worker modes
- Automatic worker registration
- Built-in container orchestration templates
```

---

## Next Steps

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Train with NeMo RL
:link: training-nemo-rl-grpo-index
:link-type: ref
Connect NeMo Gym environments to GRPO training.
:::

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Deployment Topology
:link: /infrastructure/deployment-topology
:link-type: doc
Understand deployment patterns for different scales.
:::

:::{grid-item-card} {octicon}`broadcast;1.5em;sd-mr-1` Distributed Computing with Ray
:link: /infrastructure/ray-distributed
:link-type: doc
Scale CPU-intensive tasks across nodes.
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Containerize Resources Servers
:link: /resources-server/containerize
:link-type: doc
Package custom servers for deployment.
:::

::::
