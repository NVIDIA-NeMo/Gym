(tutorial-train-with-verl)=

# Train with VeRL

In [Collecting Rollouts](../get-started/rollout-collection.md), you generated scored interactions between your agent and environment. Now you'll use VeRL (Volcano Engine Reinforcement Learning) to run distributed RL training with flexible backend support.

:::{card}

**Goal**: Train your first model using Gym and VeRL's distributed training framework.

^^^

**In this tutorial, you will**:

1. Configure VeRL to use Gym as the environment
2. Set up the generation backend (vLLM, SGLang, or HF rollout)
3. Run distributed GRPO/PPO training
4. Evaluate the trained model

:::

:::{button-ref} /get-started/index
:color: secondary
:outline:
:ref-type: doc

← Get Started
:::

---

## Before You Begin

Make sure you have:

- ✅ Completed [Collecting Rollouts](../get-started/rollout-collection.md)
- ✅ VeRL installed (refer to [VeRL documentation](https://github.com/volcengine/verl))
- ✅ A working resource server with `verify()` function
- ✅ GPU(s) with sufficient memory for training

**What you'll build**: A distributed on-policy training loop using VeRL's Ray-based architecture with Gym providing the reward signal.

---

## 1. Understand VeRL's Architecture

<!-- SME: Diagram showing VeRL + Gym integration -->

VeRL uses Ray for distributed training with multiple backend options:

```{mermaid}
flowchart LR
    VeRL["VeRL<br/>(Ray Orchestration)"] --> GenBackend["Generation Backend"]
    GenBackend --> Gym["NeMo Gym<br/>(Agent + Tools)"]
    Gym --> Reward["Reward Signal"]
    Reward --> VeRL
    
    subgraph backends["Backend Options"]
        vLLM
        SGLang
        HF["HF Rollout"]
    end
    
    GenBackend --> backends
```

Key advantages of VeRL:

- **Multiple backends**: vLLM, SGLang, or Hugging Face native
- **Ray-based**: Scales across nodes easily
- **Flexible**: Supports PPO, GRPO, and other algorithms

---

## 2. Choose Your Backend

<!-- SME: Show configuration for each backend option -->

::::{tab-set}

:::{tab-item} vLLM (Recommended)

```yaml
# TODO: vLLM backend configuration for VeRL
```

Best for: High throughput, production deployments

:::

:::{tab-item} SGLang

```yaml
# TODO: SGLang backend configuration for VeRL
```

Best for: Complex multi-turn interactions, structured generation

:::

:::{tab-item} HF Rollout

```yaml
# TODO: HF rollout configuration for VeRL
```

Best for: Simple setups, debugging, smaller models

:::

::::

---

## 3. Configure Gym Integration

<!-- SME: Show how to connect VeRL to Gym servers -->

Tell VeRL how to reach your Gym servers:

```python
# TODO: VeRL + Gym integration configuration
```

```{list-table} Gym Integration Parameters
:header-rows: 1
:widths: 30 15 55

* - Parameter
  - Type
  - Description
* - `gym_agent_url`
  - `str`
  - URL of the Gym agent server
* - <!-- SME: Add other parameters -->
  - 
  - 
```

**✅ Success Check**: <!-- SME: Verification step -->

---

## 4. Configure Training

<!-- SME: Show VeRL training configuration -->

Create your training configuration:

```python
# TODO: VeRL training configuration
```

```{list-table} Key Training Parameters
:header-rows: 1
:widths: 30 15 55

* - Parameter
  - Default
  - Description
* - `algorithm`
  - `grpo`
  - Training algorithm (grpo, ppo)
* - <!-- SME: Add key parameters -->
  - 
  - 
```

---

## 5. Start the Servers

<!-- SME: Show server startup sequence -->

Start your Gym servers:

```bash
# TODO: Gym server startup
ng_run "+config_paths=[...]"
```

Start Ray cluster (for multi-node):

```bash
# TODO: Ray cluster startup
```

**✅ Success Check**: All servers should report healthy.

---

## 6. Run Training

<!-- SME: Show training launch command -->

Launch distributed training:

```bash
# TODO: VeRL training command
```

Monitor training:

```text
# TODO: Example training output
```

**✅ Success Check**: You should see <!-- SME: success criteria -->

---

## 7. Evaluate Results

<!-- SME: Show evaluation procedure -->

After training:

```bash
# TODO: Evaluation command
```

---

## Multi-Node Scaling

<!-- SME: Show how to scale across nodes -->

```bash
# TODO: Multi-node Ray + VeRL configuration
```

---

## Troubleshooting

<!-- SME: Add common issues and solutions -->

:::{dropdown} Ray cluster connection issues
<!-- SME: Solutions -->
:::

:::{dropdown} Backend-specific errors
<!-- SME: Solutions for vLLM, SGLang, HF -->
:::

:::{dropdown} OOM during training
<!-- SME: Solutions -->
:::

---

## What's Next?

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` VeRL Documentation
:link: https://github.com/volcengine/verl

Explore VeRL's full feature set and advanced configurations.
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Custom Environments
:link: /tutorials/creating-resource-server
:link-type: doc

Build a custom resource server for your domain.
:::

::::

