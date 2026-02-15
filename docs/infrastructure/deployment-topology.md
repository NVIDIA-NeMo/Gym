(infra-deployment-topology)=
# Deployment Topology

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
:link: /about/concepts/architecture
:link-type: doc
Understand the server-based architecture.
:::

:::{grid-item-card} {octicon}`workflow;1.5em;sd-mr-1` Integrate RL Frameworks
:link: /contribute/rl-framework-integration/index
:link-type: doc
Implement NeMo Gym integration into a new training framework.
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` NeMo RL GRPO Training
:link: /training-tutorials/nemo-rl-grpo/index
:link-type: doc
End-to-end GRPO training tutorial with NeMo RL.
:::

:::{grid-item-card} {octicon}`link-external;1.5em;sd-mr-1` NeMo RL Integration (RL-side)
:link: https://github.com/NVIDIA-NeMo/RL/blob/main/docs/design-docs/nemo-gym-integration.md
:link-type: url
Detailed integration architecture from the NeMo RL perspective.
:::

::::
