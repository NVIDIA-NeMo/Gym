(environment-tutorials-index)=

# Environment Tutorials

Build custom training environments that define how models receive rewards.

---

## Environment Properties

Training environments can be broadly characterized along four dimensions:
1. **Rollout structure**: The interaction pattern between the model, environment, and user.
2. **Core capabilities**: The behaviors or skills that a model needs in order to succeed in a given use case.
3. **Knowledge domain**: What subject area, area of expertise, or field of study is involved.
4. **Task type**: The high-level use case that is represented in the training environment.

The tables below are only a small subset of of the rollout structures and core capabilities that exist. As new training environments are added to NeMo Gym and as we discover new use cases, these tables below will grow.

| Rollout structure | Description |
|---|---|
| Multi-step | Interleaved assistant and tool messages |
| Multi-turn | Interleaved user and assistant messages |
| Multi-modal | Interleaved text, image, video, and/or audio messages |
| Long context | Message content is very large or the number of messages is very large |

| Core capability | Developer/User need | Rollout Structures Required |
|---|---|---|
| Information dependency | The model receives environment responses that may require changes to subsequent actions. | Multi-step |
| Proactive asking | Developers put the model in a situation where user context is missing. The model needs to recognize user context is missing and ask the user for the missing context. | Multi-turn |
| Schema adherence | Users need more than one piece of information delivered by the model at one time in a specified delivery format. | |
| Meta data instruction following | User constrains the meta-properties of the model response e.g. “respond in 5 words”. | |
| Counterintuitive instruction following | User provides instructions that are against conventional wisdom, typically making sense in the specific context in which the model is being used | |
| Information relevance | Given a large volume of inputs, the model needs to ignore content irrelevant to the task at hand. | Long context |
| Multiple intent synthesis | Users provide multiple tasks for the model to accomplish. | Multi-step, Multi-turn |

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

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Multi-Environment Training
:link: multi-environment-training
:link-type: doc

Run multiple training environments simultaneously for rollout collection.

+++
:::

::::

---

## Learning Path

**New to NeMo Gym?** Follow this sequence:

```{mermaid}
flowchart LR
    A[1. Setup] --> B[2. Training Environment]
    B --> C[3. Train]
```

1. {doc}`/get-started/detailed-setup` — Install NeMo Gym
2. {doc}`creating-training-environment` — Build a training environment with verification
3. Start training with one of the {doc}`/training-tutorials/index`

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
