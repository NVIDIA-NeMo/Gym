(agent-server-nemo-agent-toolkit)=
# NeMo Agent Toolkit Integration

```{note}
This page is a stub. Content is being developed. See [GitHub Issue #270](https://github.com/NVIDIA-NeMo/Gym/issues/270) for details.
```

Integrate agents from the [NeMo Agent Toolkit](https://github.com/NVIDIA/NeMo-Agent-Toolkit) into NeMo Gym for RL training.

---

## Overview

NeMo Agent Toolkit provides:
- Enterprise-ready agent implementations
- NVIDIA-optimized inference
- Production deployment patterns

## Prerequisites

```bash
pip install nemo-agent-toolkit
```

## Integration Architecture

```
NeMo Gym Agent Server
    └── NeMo Agent Toolkit Agent
            ├── Model calls → NeMo Gym Model Server (or direct)
            └── Tool calls → NeMo Gym Resources Server
```

## Configuration

<!-- TODO: Add configuration example -->

## Implementation

### Wrapping a NeMo Agent

```python
from nemo_agent_toolkit import Agent
from nemo_gym.base_responses_api_agent import SimpleResponsesAPIAgent

class NeMoAgentWrapper(SimpleResponsesAPIAgent):
    def model_post_init(self, context):
        self.nemo_agent = Agent(...)

    async def run(self, body: BaseRunRequest) -> BaseVerifyResponse:
        # Route NeMo agent through NeMo Gym infrastructure
        pass
```

### Tool Registration

<!-- TODO: Document how to register NeMo Gym tools with NeMo Agent Toolkit -->

### Workflow Integration

<!-- TODO: Document workflow compatibility -->

## Example

<!-- TODO: Add complete working example -->

## Benefits

- Leverage NVIDIA-optimized agent patterns
- Use existing enterprise agent deployments
- Combine with NeMo Gym's RL training infrastructure

## Troubleshooting

<!-- TODO: Add common issues and solutions -->
