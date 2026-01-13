(agent-server-openai-agents-sdk)=
# OpenAI Agents SDK Integration

```{note}
This page is a stub. Content is being developed. See [GitHub Issue #520](https://github.com/NVIDIA-NeMo/Gym/issues/520) for details.
```

Integrate agents built with the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) into NeMo Gym for RL training.

---

## Overview

The OpenAI Agents SDK provides:
- Structured agent workflows
- Built-in tool handling
- Conversation state management

## Prerequisites

```bash
pip install openai-agents
```

## Integration Architecture

```
NeMo Gym Agent Server
    └── OpenAI Agents SDK Agent
            ├── Model calls → NeMo Gym Model Server
            └── Tool calls → NeMo Gym Resources Server
```

## Configuration

<!-- TODO: Add configuration example -->

## Implementation

### Wrapping an OpenAI SDK Agent

```python
from openai_agents import Agent
from nemo_gym.base_responses_api_agent import SimpleResponsesAPIAgent

class OpenAIAgentWrapper(SimpleResponsesAPIAgent):
    def model_post_init(self, context):
        self.sdk_agent = Agent(
            name="my_agent",
            instructions="...",
            tools=[...]
        )

    async def run(self, body: BaseRunRequest) -> BaseVerifyResponse:
        # Route SDK agent through NeMo Gym infrastructure
        pass
```

### Tool Routing

<!-- TODO: Document how to route SDK tool calls to NeMo Gym resources server -->

### State Management

<!-- TODO: Document conversation state handling -->

## Example

<!-- TODO: Add complete working example -->

## Limitations

<!-- TODO: Document any limitations or unsupported features -->

## Troubleshooting

<!-- TODO: Add common issues and solutions -->
