(env-user-modeling)=
# Multi-Turn via User Modeling

```{note}
This page is a stub. Content is being developed.

**Blocker**: No reference implementation exists. See [GitHub Issue #551](https://github.com/NVIDIA-NeMo/Gym/issues/551) to track progress. The `calendar/` resources server contains user modeling patterns that may be extracted.
```

Create training environments that simulate user interactions for dialogue training.

---

## What is User Modeling?

User modeling uses a second LLM to simulate user responses, enabling:
- Scalable multi-turn data generation
- Diverse interaction patterns
- Realistic dialogue training without human annotation

## Architecture

```
Training Flow:
  Agent Model → Response → User Model → Next User Message → Agent Model → ...
```

## Implementation

### User Model Configuration

```yaml
user_model:
  responses_api_models:
    openai_model:
      entrypoint: app.py
      openai_model: gpt-4o-mini  # Simulated user
```

### User Prompt Design

```python
USER_SIMULATION_PROMPT = """
You are simulating a user with the following persona:
{persona}

Current conversation goal: {goal}

Respond naturally as this user would. When your goal is achieved, 
respond with [CONVERSATION_COMPLETE].
"""
```

### Conversation Loop

```python
async def run_user_modeled_conversation(self, initial_task):
    conversation = [initial_task]
    
    while not is_complete(conversation):
        # Get agent response
        agent_response = await self.call_agent_model(conversation)
        conversation.append(agent_response)
        
        # Get simulated user response
        user_response = await self.call_user_model(conversation)
        if is_conversation_complete(user_response):
            break
        conversation.append(user_response)
    
    return conversation
```

## User Personas

Define diverse user personas for training variety:

```python
PERSONAS = [
    {"style": "polite", "expertise": "novice", "patience": "high"},
    {"style": "direct", "expertise": "expert", "patience": "low"},
    {"style": "confused", "expertise": "novice", "patience": "medium"},
]
```

## Verification

<!-- TODO: Document verification strategies for user-modeled conversations -->

## Data Generation

<!-- TODO: Document large-scale data generation with user modeling -->

## Quality Control

<!-- TODO: Document quality control for synthetic conversations -->
