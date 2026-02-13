(infra-deployment-topology)=
# Deployment Topology

Understand how to deploy NeMo Gym components for different scales and use cases.

**Quick Start**: For most use cases, a single `ng_run` command starts all servers:

```bash
config_paths="resources_servers/example_single_tool_call/configs/example_single_tool_call.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml"

ng_run "+config_paths=[$config_paths]"
```

---

## Topology Patterns

NeMo Gym supports three deployment patterns, each suited to different stages of development and scale.

:::::{tab-set}

::::{tab-item} All-in-One (Development)

All servers run in a single process—ideal for local development and debugging:

```
┌─────────────────────────────────┐
│         Single Machine          │
│                                 │
│  ┌─────────────────────────┐   │
│  │      Head Server        │   │
│  │   ├── Model Server      │   │
│  │   ├── Resources Server  │   │
│  │   └── Agent Server      │   │
│  └─────────────────────────┘   │
└─────────────────────────────────┘
```

**Best for:** Local development, debugging, quick iteration

::::

::::{tab-item} Separated Services (Testing)

Services run as separate processes on a single machine:

```
┌─────────────────────────────────┐
│         Single Machine          │
│                                 │
│  ┌───────────┐ ┌────────────┐  │
│  │   Head    │ │   Model    │  │
│  │  Server   │ │   Server   │  │
│  └───────────┘ └────────────┘  │
│  ┌───────────┐ ┌────────────┐  │
│  │ Resources │ │   Agent    │  │
│  │  Server   │ │   Server   │  │
│  └───────────┘ └────────────┘  │
└─────────────────────────────────┘
```

**Best for:** Integration testing, validating service communication, Docker Compose setups

::::

::::{tab-item} Distributed (Production)

Services distributed across multiple nodes for horizontal scaling:

```
┌─────────────┐     ┌─────────────┐
│ Coordinator │     │  GPU Node   │
│             │     │             │
│ ┌─────────┐ │     │ ┌─────────┐ │
│ │  Head   │ │────▶│ │ Model   │ │
│ │ Server  │ │     │ │ Server  │ │
│ └─────────┘ │     │ └─────────┘ │
└─────────────┘     └─────────────┘
        │
        │           ┌─────────────┐
        │           │ CPU Node    │
        └──────────▶│             │
                    │ ┌─────────┐ │
                    │ │Resources│ │
                    │ │ Server  │ │
                    │ └─────────┘ │
                    └─────────────┘
```

**Best for:** Production workloads, large-scale rollout collection, multi-GPU training

::::

:::::

---

## Choosing a Pattern

Use this decision matrix to select the right deployment pattern:

| Factor | All-in-One | Separated | Distributed |
|--------|------------|-----------|-------------|
| **Team size** | Individual | Small team | Production team |
| **Data volume** | Small datasets | Medium datasets | Large datasets |
| **GPU requirements** | Single GPU or CPU | Single machine, multiple GPUs | Multi-node GPU cluster |
| **Fault tolerance** | None | Process isolation | Full isolation |
| **Setup complexity** | Minimal | Moderate | Higher |

### When to Scale

**Scale from All-in-One to Separated when:**
- You need to restart individual services without restarting everything
- Multiple team members are developing different components
- You want to simulate production networking locally

**Scale from Separated to Distributed when:**
- Single machine cannot handle rollout throughput
- GPU memory on one machine is insufficient
- You need fault isolation between coordinator and workers

---

## Configuration

:::::{tab-set}

::::{tab-item} Single-Node

```bash
# Define your server configurations
config_paths="resources_servers/example_single_tool_call/configs/example_single_tool_call.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml"

# Start all servers in one process
ng_run "+config_paths=[$config_paths]"
```

This starts:
- Head server on port 11000
- All configured servers as subprocesses
- Servers register with the head server automatically

::::

::::{tab-item} Multi-Node

For multi-node deployment, configure each node with YAML files that point to a shared head server.

**Coordinator node** (`coordinator-config.yaml`):

```yaml
head_server:
  host: "192.168.1.100"  # Coordinator's IP
  port: 11000

example_resources:
  resources_servers:
    example_single_tool_call:
      entrypoint: app.py
      host: "192.168.1.100"
      port: 8080
```

**Worker node** (`worker-config.yaml`):

```yaml
head_server:
  host: "192.168.1.100"  # Points to coordinator
  port: 11000

policy_model:
  responses_api_models:
    openai_model:
      entrypoint: app.py
      host: "192.168.1.101"  # This worker's IP
      port: 8000
```

Start each node with its configuration:

```bash
# On coordinator
ng_run "+config_paths=[coordinator-config.yaml]"

# On each worker
ng_run "+config_paths=[worker-config.yaml]"
```

::::

:::::

---

## Networking

### Service Discovery

The head server (port 11000) acts as a service registry. Servers register on startup, enabling name-based routing. You can query registered servers:

```bash
# List all registered server instances
curl http://localhost:11000/server_instances

# Get the resolved global configuration
curl http://localhost:11000/global_config_dict_yaml
```

Within server code, use `ServerClient` for inter-server communication:

```python
from nemo_gym.server_utils import ServerClient

# Servers call each other by name, not IP
await self.server_client.post(
    server_name="my_resources",
    url_path="/verify",
    json=payload
)
```

### Port Allocation

| Server | Default Port | Notes |
|--------|--------------|-------|
| Head server | 11000 | Fixed default (`DEFAULT_HEAD_SERVER_PORT` in `global_config.py`) |
| Model server | Dynamic | Assigned from available ports unless specified in config |
| Resources server | Dynamic | Assigned from available ports unless specified in config |

```{tip}
Specify explicit ports in your YAML config for predictable deployments:
```yaml
my_resources:
  resources_servers:
    my_server:
      host: "0.0.0.0"
      port: 8080  # Explicit port
```
```

### Network Requirements

| Requirement | Guideline |
|-------------|-----------|
| **Latency** | Low latency between head and workers improves throughput |
| **Bandwidth** | Higher bandwidth recommended for large model weights |
| **Firewall** | Allow TCP traffic between nodes on configured ports |

```{note}
Actual requirements depend on your model size and rollout complexity. Test your specific workload.
```

---

## Resource Sizing

### Per-Pattern Recommendations

| Component | All-in-One | Separated | Distributed |
|-----------|------------|-----------|-------------|
| **Coordinator CPU** | 4+ cores | 4+ cores | 8+ cores |
| **Coordinator RAM** | 16 GB | 16 GB | 32 GB |
| **Worker CPU** | — | — | 4+ cores per worker |
| **Worker GPU** | 1× 16GB+ | 1-8× 16GB+ | 1+ per worker |

### Capacity Planning

Throughput varies significantly based on model size, GPU type, environment complexity, and network latency. The following are rough estimates for planning purposes:

| Rollout Type | Characteristics | Scaling Factor |
|--------------|-----------------|----------------|
| Simple (1-step) | Single tool call, fast verification | Linear with workers |
| Multi-step (5 steps) | Sequential tool calls | ~5× slower per rollout |
| Multi-turn (10 turns) | Conversation with state | ~10× slower per rollout |

```{important}
Benchmark your specific workload before capacity planning. Use {doc}`/get-started/rollout-collection` to measure actual throughput.
```

---

## Migration Paths

::::{dropdown} Development → Testing
:icon: arrow-right

1. Extract configuration into separate YAML files per server
2. Start servers in separate terminals/processes
3. Verify inter-service communication via logs

::::

::::{dropdown} Testing → Production
:icon: arrow-right

1. Containerize each server type
2. Deploy head server on coordinator node
3. Deploy workers with GPU resources
4. Configure networking and service discovery

::::

::::{dropdown} Scaling Existing Deployments
:icon: arrow-right

To add workers to a running cluster:

1. Create a worker config pointing to the existing head server
2. Start the worker with `ng_run "+config_paths=[worker-config.yaml]"`
3. Verify registration via `curl http://coordinator:11000/server_instances`

Workers auto-register with the head server on startup.

::::

---

## Monitoring

### Server Status

Check server status using the CLI or API:

```bash
# CLI: Show all running servers
ng_status

# API: List registered server instances
curl http://localhost:11000/server_instances
```

Example response:

```json
[
  {
    "process_name": "example_resources",
    "server_type": "resources_servers",
    "name": "example_single_tool_call",
    "host": "127.0.0.1",
    "port": 62920,
    "status": "success"
  }
]
```

### Key Metrics to Monitor

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| Server registration | Number of servers in `/server_instances` | < expected count |
| Process health | Server process exit codes | Non-zero exit |
| GPU utilization | Per-worker GPU usage (via `nvidia-smi`) | < 50% (underutilized) |
| Request latency | Head → worker round-trip | > 100ms |

---

## Security Considerations

```{warning}
NeMo Gym servers have **no built-in authentication**. They are designed for trusted internal networks only.
```

### Network Security

- **Internal networks only**: Do not expose servers to the public internet
- **VPC isolation**: Deploy within a private VPC or subnet
- **Firewall rules**: Restrict access to known IP ranges
- **TLS termination**: Use a reverse proxy (nginx, Traefik) for HTTPS

### Production Checklist

- [ ] Servers deployed in private network (not internet-accessible)
- [ ] Firewall rules restrict access to known IPs
- [ ] API keys stored in environment variables, not config files
- [ ] Monitoring alerts configured for anomalies
- [ ] Logs shipped to centralized logging system
- [ ] Reverse proxy configured for TLS termination

---

## Training Framework Deployment

When NeMo Gym is used for RL training (not standalone rollout collection), it runs alongside a training framework. NeMo Gym's Model Server acts as an HTTP proxy for policy model inference — it translates between the Responses API and Chat Completions API formats, forwarding requests to the training framework's generation endpoint (e.g., vLLM). NeMo Gym can also run other models on GPU (e.g., reward models, judge models) through its own resource servers.

This section covers resource requirements, cluster strategies, and how to choose between them. For a detailed integration walkthrough from the training framework side, see how [NeMo RL integrated with NeMo Gym](https://github.com/NVIDIA-NeMo/RL/blob/main/docs/design-docs/nemo-gym-integration.md). For guidance on integrating a new training framework, see {doc}`/contribute/rl-framework-integration/index`.

### Resource Requirements

NeMo Gym and the training framework have different compute profiles:

| Component | Compute | Role |
|-----------|---------|------|
| **NeMo Gym** | CPU by default | Orchestrates rollouts, executes tools, computes rewards. Some resource servers may use GPUs (e.g., running local reward or judge models via vLLM). |
| **Training framework** (e.g., NeMo RL) | GPU | Holds model weights, runs policy training, serves inference via an OpenAI-compatible HTTP endpoint (e.g., vLLM) |

### Cluster Co-location Strategy

The deployment strategy depends on how the training framework manages its cluster.

#### Single Ray Cluster

If `ray_head_node_address` is specified in the config, NeMo Gym connects to that existing Ray cluster instead of starting its own. Training frameworks using Ray set this address so that NeMo Gym attaches to the same cluster.

```{mermaid}
%%{init: {'theme': 'default', 'themeVariables': { 'lineColor': '#5c6bc0', 'primaryTextColor': '#333'}}}%%
flowchart LR
    subgraph Cluster["Single Ray Cluster"]
        subgraph RL["Training Framework · GPU"]
            GRPO["GRPO Loop"]
            Policy["Policy Workers"]
            vLLM["vLLM + HTTP"]
        end
        subgraph Gym["NeMo Gym · CPU"]
            Agent["Agent Server"]
            Model["Model Server"]
            Resources["Resources Server"]
        end
    end

    vLLM <-->|HTTP| Model

    style Cluster fill:#f5f5f5,stroke:#999,stroke-width:2px
    style RL fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style Gym fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
```

**How it works:**
1. The training framework initializes the Ray cluster and creates vLLM workers with HTTP servers
2. The training framework creates a NemoGym Ray actor within the same cluster
3. The NemoGym actor spawns Gym servers (Head, Agent, Model, Resources) as subprocesses
4. NeMo Gym's Model Server proxies inference requests to the training framework's vLLM HTTP endpoints
5. Results flow back through the actor to the training loop

Both systems share a single Ray cluster, so Ray has visibility into all available resources.

**Version Requirements**

When NeMo Gym connects to an existing Ray cluster, the same Ray and Python versions must be used in both environments.

---

#### Gym's Own Ray Cluster

When the training framework **does not use Ray**, NeMo Gym spins up its own independent Ray cluster for coordination.

```{mermaid}
%%{init: {'theme': 'default', 'themeVariables': { 'lineColor': '#5c6bc0', 'primaryTextColor': '#333'}}}%%
flowchart LR
    subgraph TF["Training Framework · GPU"]
        Loop["Training Loop"]
        vLLM["vLLM + HTTP"]
        Workers["Policy Workers"]
    end
    subgraph Gym["NeMo Gym · CPU"]
        Agent["Agent Server"]
        Model["Model Server (proxy)"]
        Resources["Resources Server"]
    end

    vLLM <-->|HTTP| Model

    style TF fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style Gym fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
```

The training framework runs its own orchestration (non-Ray). NeMo Gym spins up a separate Ray cluster.

**When to use:**
- The training framework has its own orchestration (not Ray-based)
- You still want NeMo Gym's HTTP-based rollout collection
- The generation backend exposes OpenAI-compatible HTTP endpoints that NeMo Gym can reach

#### Separate Clusters

When the training framework and NeMo Gym are **not started together** (independently deployed), they run on fully separate clusters connected only by HTTP.

```{mermaid}
%%{init: {'theme': 'default', 'themeVariables': { 'lineColor': '#5c6bc0', 'primaryTextColor': '#333'}}}%%
flowchart LR
    subgraph A["Training Cluster · GPU"]
        Loop["Training Loop"]
        vLLM["vLLM + HTTP"]
        Workers["Policy Workers"]
    end
    subgraph B["Gym Cluster · CPU"]
        Agent["Agent Server"]
        Model["Model Server (proxy)"]
        Resources["Resources Server"]
    end

    vLLM <-->|HTTP| Model

    style A fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style B fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
```

**When to use:**
- Training framework and Gym are deployed independently by different teams
- Clusters have different lifecycle requirements (e.g., Gym always on, training runs are transient)
- Network security policies require isolation between training and environment infrastructure
- Hybrid cloud setups where training runs on GPU cloud and environments run on CPU cloud

**Requirements:**
- The training cluster must expose its generation backend (e.g., vLLM) as HTTP endpoints reachable from the Gym cluster
- Network connectivity and firewall rules between clusters must allow HTTP traffic on the configured ports

### Choosing a Deployment Strategy

| Factor | Single Ray Cluster | Gym's Own Ray Cluster | Separate Clusters |
|--------|-------------------|----------------------|-------------------|
| **Training framework** | Ray-based | Non-Ray | Any |
| **Startup** | Co-launched | Independent | Independent |
| **Resource visibility** | Unified | Separate | Separate |
| **Network requirements** | Intra-cluster | Intra-cluster | Cross-cluster HTTP |



## Related Guides

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Architecture Overview
:link: /about/architecture
:link-type: doc
Understand the server-based architecture.
:::

:::{grid-item-card} {octicon}`workflow;1.5em;sd-mr-1` Integrate RL Frameworks
:link: /contribute/rl-framework-integration/index
:link-type: doc
Implement NeMo Gym integration into a new training framework.
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` NeMo RL GRPO Training
:link: /tutorials/nemo-rl-grpo/index
:link-type: doc
End-to-end GRPO training tutorial with NeMo RL.
:::

:::{grid-item-card} {octicon}`link-external;1.5em;sd-mr-1` NeMo RL Integration (RL-side)
:link: https://github.com/NVIDIA-NeMo/RL/blob/main/docs/design-docs/nemo-gym-integration.md
:link-type: url
Detailed integration architecture from the NeMo RL perspective.
:::

::::
