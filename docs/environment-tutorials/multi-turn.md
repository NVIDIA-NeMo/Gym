(env-multi-turn)=
# Multi-Turn Environments

```{warning}
This article was generated and has not been reviewed. Content may change.
```

Build conversational training environments with extended dialogue interactions.

::::{grid} 2
:gutter: 3

:::{grid-item-card} {octicon}`clock;1em;` **Time**
30-45 minutes
:::

:::{grid-item-card} {octicon}`bookmark;1em;` **Prerequisites**

- Completed {doc}`/get-started/detailed-setup`
- Completed {doc}`/tutorials/creating-resource-server`

:::

::::

---

## What is Multi-Turn?

Multi-turn environments train models on extended conversations where context accumulates across multiple exchanges. The model must:

- Maintain coherent dialogue across many turns
- Track conversation state and user intent
- Build on previous exchanges to complete tasks

### Multi-Turn vs Multi-Step

These patterns are often confused. Here's the key distinction:

| Aspect | Multi-Turn | Multi-Step |
|--------|------------|------------|
| **Interaction type** | User ↔ Assistant dialogue | Tool calling loops |
| **Context growth** | Conversation history accumulates | Tool results accumulate |
| **Control flow** | User drives the conversation | Model drives the workflow |
| **Example** | Customer support chat | Trip planning with APIs |
| **Termination** | User goal achieved or conversation ends | Task complete or max steps |

**Multi-turn**: The model responds to user messages, building a conversation.
**Multi-step**: The model calls tools repeatedly to complete a task autonomously.

---

## When to Use

Multi-turn environments are ideal for:

- **Customer support agents** — Resolve issues through dialogue
- **Information-seeking dialogue** — Help users find answers iteratively
- **Negotiation scenarios** — Back-and-forth persuasion or deal-making
- **Teaching and tutoring** — Adaptive instruction based on student responses
- **Task completion through conversation** — Scheduling, booking, planning via chat

:::{tip}
Start with multi-turn when the task naturally involves human interaction. Use multi-step when the model should work autonomously with tools.
:::

---

## Quick Start

The `calendar` resources server demonstrates a complete multi-turn environment for training scheduling assistants.

### 1. Start the Servers

```bash
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/calendar/configs/calendar.yaml"

ng_run "+config_paths=[$config_paths]"
```

### 2. Collect Rollouts

```bash
ng_collect_rollouts \
    +agent_name=calendar_agent \
    +input_jsonl_fpath=resources_servers/calendar/data/example.jsonl \
    +output_jsonl_fpath=results/calendar_rollouts.jsonl \
    +limit=5
```

---

## Configuration

Multi-turn conversations typically need limits on conversation length:

```yaml
my_multi_turn_agent:
  responses_api_agents:
    simple_agent:
      max_steps: 20  # Maximum conversation turns
```

---

## Conversation History

In multi-turn environments, messages accumulate in `responses_create_params.input`:

```json
{
  "responses_create_params": {
    "input": [
      {"role": "developer", "content": "You are a helpful scheduling assistant."},
      {"role": "user", "content": "I need to schedule a team meeting"},
      {"role": "assistant", "content": "I'd be happy to help. What time works for you?"},
      {"role": "user", "content": "How about 10am tomorrow for 1 hour?"},
      {"role": "assistant", "content": "Done! I've scheduled your team meeting for 10am tomorrow, lasting 1 hour."}
    ]
  }
}
```

Each assistant response becomes part of the context for subsequent turns.

---

## Implementation

### Conversation State Management

Track state across turns using session data or by parsing the conversation history:

```python
class MultiTurnResourcesServer(SimpleResourcesServer):
    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        conversation = body.responses_create_params.input
        
        # Extract state from conversation history
        current_state = self.extract_state_from_conversation(conversation)
        expected_state = body.expected_state
        
        reward = 1.0 if current_state == expected_state else 0.0
        return BaseVerifyResponse(**body.model_dump(), reward=reward)
    
    def extract_state_from_conversation(self, messages: list[dict]) -> dict:
        """Parse conversation to determine current state."""
        state = {}
        for msg in messages:
            if msg["role"] == "assistant":
                # Extract structured data from assistant responses
                state.update(self.parse_response(msg["content"]))
        return state
```

### Turn-by-Turn Verification

Evaluate each assistant turn individually:

```python
async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
    conversation = body.responses_create_params.input
    
    # Score each assistant turn
    turn_scores = []
    for i, msg in enumerate(conversation):
        if msg["role"] == "assistant":
            context = conversation[:i]  # Messages before this turn
            score = self.evaluate_turn(msg["content"], context)
            turn_scores.append(score)
    
    # Average across turns
    reward = sum(turn_scores) / len(turn_scores) if turn_scores else 0.0
    return BaseVerifyResponse(**body.model_dump(), reward=reward)
```

### End-of-Conversation Detection

Define when a conversation should terminate:

```python
def is_conversation_complete(self, conversation: list[dict]) -> bool:
    """Check if the conversation goal has been achieved."""
    last_assistant = None
    for msg in reversed(conversation):
        if msg["role"] == "assistant":
            last_assistant = msg["content"]
            break
    
    if last_assistant is None:
        return False
    
    # Check for completion signals
    completion_signals = [
        "task completed",
        "is there anything else",
        "glad I could help",
    ]
    return any(signal in last_assistant.lower() for signal in completion_signals)
```

---

## Verification Strategies

:::::{tab-set}

::::{tab-item} Final State Verification

Verify only the end result of the conversation:

```python
async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
    final_state = self.extract_final_state(body.response)
    expected_state = body.expected_state
    
    reward = 1.0 if final_state == expected_state else 0.0
    return BaseVerifyResponse(**body.model_dump(), reward=reward)
```

**Best for**: Task-oriented dialogues with clear success criteria (scheduling, booking).

::::

::::{tab-item} Cumulative Turn Scoring

Weight turns by importance or recency:

```python
async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
    conversation = body.responses_create_params.input
    assistant_turns = [m for m in conversation if m["role"] == "assistant"]
    
    total_score = 0.0
    total_weight = 0.0
    
    for i, turn in enumerate(assistant_turns):
        weight = i + 1  # Later turns weighted more heavily
        score = self.evaluate_turn(turn["content"])
        total_score += score * weight
        total_weight += weight
    
    reward = total_score / total_weight if total_weight > 0 else 0.0
    return BaseVerifyResponse(**body.model_dump(), reward=reward)
```

**Best for**: Conversations where later turns build on earlier ones.

::::

::::{tab-item} Constraint-Based Verification

Check multiple criteria for correctness:

```python
async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
    conversation = body.responses_create_params.input
    
    checks = [
        self.check_no_conflicts(conversation),      # Events don't overlap
        self.check_constraints_met(conversation),   # Time constraints satisfied
        self.check_all_events_scheduled(conversation),  # All requested events exist
    ]
    
    reward = 1.0 if all(checks) else 0.0
    return BaseVerifyResponse(**body.model_dump(), reward=reward)
```

**Best for**: Structured tasks with multiple success criteria.

::::

:::::

---

## Data Format

Multi-turn training data includes conversation history and expected outcomes:

```json
{
  "responses_create_params": {
    "input": [
      {"role": "developer", "content": "You are a scheduling assistant. Output calendar state as JSON."},
      {"role": "user", "content": "Schedule a team meeting at 10am for 1 hour"}
    ]
  },
  "expected_state": {
    "1": {
      "event_name": "Team Meeting",
      "start_time": "10:00",
      "duration": 60
    }
  }
}
```

For multi-turn training, data can include partial conversations where the model must continue appropriately.

---

## Example: Calendar Assistant

The `resources_servers/calendar/` environment demonstrates multi-turn training for scheduling assistants:

**Task**: Schedule events based on conversational requests with time constraints.

**Features**:
- Multi-turn conversation tracking
- Constraint validation (before/after/between times)
- Conflict detection
- Binary reward based on final calendar state

**Verification logic**:
- Reward = 1.0 if all events correctly scheduled with no conflicts
- Reward = 0.0 if any constraint violated or events missing

See the [Calendar README](https://github.com/NVIDIA/NeMo-Gym/tree/main/resources_servers/calendar) for the complete implementation.

---

## Next Steps

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`people;1.5em;sd-mr-1` User Modeling
:link: user-modeling
:link-type: doc
Simulate users for realistic dialogue training.
:::

:::{grid-item-card} {octicon}`iterations;1.5em;sd-mr-1` Multi-Step Environments
:link: multi-step
:link-type: doc
Build autonomous tool-calling workflows.
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Train with NeMo RL
:link: training-nemo-rl-grpo-index
:link-type: ref
Start training models on your environment.
:::

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Prepare Data
:link: /data/prepare-validate
:link-type: doc
Learn more about data formats and validation.
:::

::::
