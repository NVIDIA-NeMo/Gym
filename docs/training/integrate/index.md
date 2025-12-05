(training-integrate-overview)=

# Integrate Custom Training Frameworks

Connect NeMo Gym to custom RL training pipelines using OpenAI-compatible HTTP endpoints. These guides are for users who cannot use NeMo RL's native integration and need to implement their own.

:::{tip}
**Using NeMo RL?** Skip these guides and use {doc}`/tutorials/integrate-training-frameworks/train-with-nemo-rl` instead — integration is handled automatically.
:::

---

## How Integration Works

Gym communicates with training frameworks through OpenAI-compatible HTTP endpoints. This decoupled architecture allows Gym to run independently while your training loop controls policy updates.

```{mermaid}
flowchart LR
    subgraph Training["Training Framework"]
        P["Policy"]
        O["Optimizer"]
    end
    
    subgraph Gym["NeMo Gym"]
        A["Agent"]
        R["Resources"]
    end
    
    subgraph Gen["Generation Backend"]
        V["vLLM / SGLang"]
    end
    
    P -->|"weights"| V
    V -->|"HTTP"| Gym
    Gym -->|"rollouts"| Training
    O -->|"update"| P
    
    classDef training fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#000
    classDef gym fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#000
    classDef gen fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000
    
    class P,O training
    class A,R gym
    class V gen
```

Each rollout returns:
- **Token IDs** — For gradient computation
- **Log probabilities** — For policy gradient methods  
- **Rewards** — From verification logic

---

## How-To Guides

Complete these in order for a new integration, or jump to the specific guide you need.

::::{grid} 1 1 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Expose an OpenAI-Compatible Endpoint
:link: expose-openai-endpoint
:link-type: doc
Configure your generation backend to serve an HTTP endpoint that Gym can connect to.
+++
{bdg-secondary}`vllm`
{bdg-secondary}`http`
{bdg-secondary}`20 min`
:::

:::{grid-item-card} {octicon}`plug;1.5em;sd-mr-1` Connect Gym to Your Training Loop
:link: connect-gym-to-training
:link-type: doc
Integrate Gym's rollout collection into your custom training pipeline.
+++
{bdg-secondary}`RunHelper`
{bdg-secondary}`rollouts`
{bdg-secondary}`30 min`
:::

:::{grid-item-card} {octicon}`iterations;1.5em;sd-mr-1` Process Multi-Turn Rollouts
:link: process-multi-turn-rollouts
:link-type: doc
Handle token alignment across multi-turn interactions for training.
+++
{bdg-secondary}`tokens`
{bdg-secondary}`alignment`
{bdg-secondary}`25 min`
:::

:::{grid-item-card} {octicon}`check-circle;1.5em;sd-mr-1` Validate Your Integration
:link: validate-integration
:link-type: doc
Verify your integration works correctly end-to-end.
+++
{bdg-secondary}`testing`
{bdg-secondary}`validation`
{bdg-secondary}`15 min`
:::

::::

## Background Reading

Understand the architecture before diving into implementation.

- {doc}`/about/concepts/training-integration-architecture` — Architecture and design rationale
- {doc}`/about/ecosystem` — Existing framework integrations

---

```{toctree}
:hidden:
:maxdepth: 1

expose-openai-endpoint
connect-gym-to-training
process-multi-turn-rollouts
validate-integration
```
