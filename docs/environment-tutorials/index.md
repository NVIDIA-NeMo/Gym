(environment-tutorials-index)=

# Environment Tutorials

Build custom training environments that define how models receive rewards.

---

## Environment Patterns

NeMo Gym environments use the `verify()` method to compute rewards from model responses. Different patterns handle different training scenarios:

| Pattern | Description | Key Characteristic |
|---|---|---|
| **Single-step** | One model response per task | `verify()` evaluates the final response |
| **Multi-step** | Sequential tool calls within a turn | `/step` endpoint routes tool calls; model drives the loop |
| **Multi-turn** | Conversation with accumulated history | User messages alternate with assistant responses |
| **User modeling** | LLM-simulated user interactions | Generates synthetic training data at scale |

**Implementation reference**: All patterns inherit from `SimpleResourcesServer` (`nemo_gym/base_resources_server.py`), which provides `verify()` and `seed_session()` endpoints.

---

## Verification Methods

| Method | When to Use | Example Server |
|---|---|---|
| **Exact match** | Answers have one correct form | `mcqa/app.py` — Choice grading |
| **Library verification** | Domain-specific parsing needed | `math_with_judge/app.py` — Uses `math_verify` library |
| **LLM-as-judge** | Semantic equivalence matters | `equivalence_llm_judge/app.py` — Configurable judge prompts |
| **Reward model** | Learned preferences | NeMo RL `RewardModelEnvironment` |

---

## Tutorials

### Foundational

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Creating a Training Environment
:link: creating-training-environment
:link-type: doc

Build `verify()`, prepare data, connect to NeMo RL.

+++
{bdg-primary}`start here` {bdg-secondary}`45-90 min`
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Multi-verifier rollouts
:link: multi-verifier-rollouts
:link-type: doc

Run multiple training environments (multi-verifier) rollout collection.

+++
:::

::::

---

## Learning Path

**New to NeMo Gym?** Follow this sequence:

```{mermaid}
flowchart LR
    A[1. Setup] --> B[2. Resources Server]
    B --> C[3. Training Environment]
    C --> D[4. Train]
```

1. {doc}`/get-started/detailed-setup` — Install NeMo Gym
2. {doc}`/tutorials/creating-resource-server` — Build a basic resources server
3. {doc}`creating-training-environment` — Add verification and connect to training
4. Start training with one of the {doc}`/training-tutorials/index`

---

## Reference Implementations

NeMo Gym includes working examples in `resources_servers/`:

| Server | Pattern | Verification |
|---|---|---|
| `mcqa/` | Single-step | Regex extraction, exact match |
| `example_multi_step/` | Multi-step | Function call validation |
| `calendar/` | Multi-turn | State comparison |
| `equivalence_llm_judge/` | Single-step | LLM judge with swap check |
| `math_with_judge/` | Single-step | Library + judge fallback |
| `aviary/` | Multi-step | Aviary environment integration |
| `workplace_assistant/` | Multi-step | Session state, tool routing |

:::{tip}
Use `ng_init_resources_server +entrypoint=resources_servers/my_env` to scaffold a new environment from a template.
:::
