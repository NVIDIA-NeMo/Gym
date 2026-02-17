(environment-tutorials-index)=

# Environment Tutorials

Learn how to build custom environments for training or evaluation using NeMo Gym.

:::{tip}
Looking to use an existing environment rather than build your own? See the [Available Environments](https://github.com/NVIDIA-NeMo/Gym#-available-environments) in the README.
:::

---

## Environment Properties

Training environments can be broadly characterized along five dimensions:
1. **Rollout structure**: The interaction pattern between the model, environment, and user.
2. **Core capabilities**: The behaviors or skills that a model needs in order to succeed in a given use case.
3. **Knowledge domain**: What subject area, area of expertise, or field of study is involved.
4. **Task type**: The high-level use case that is represented in the training environment.
5. **Verification method**: How the environment computes rewards from model responses. See {doc}`/about/concepts/task-verification` for details.

Below are a subset of rollout structures and core capabilities found across NeMo Gym environments. We plan to add these as structured metadata to environments in the future. If you have ideas for additional properties, please let us know by [opening an issue](https://github.com/NVIDIA-NeMo/Gym/issues).

### Rollout Structure
| Rollout structure | Description |
|---|---|
| Multi-step | Interleaved assistant and tool messages |
| Multi-turn | Interleaved user and assistant messages |
| Multi-modal | Interleaved text, image, video, and/or audio messages |
| Long context | Message content is very large or the number of messages is very large |

### Core Capabilities
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

## Tutorials

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
