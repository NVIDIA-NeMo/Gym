(about-architecture)=
# Architecture

```{note}
This page is a stub. Content is being developed. See [GitHub Issue #292](https://github.com/NVIDIA-NeMo/Gym/issues/292) for details.
```

NeMo Gym uses a modular server-based architecture that decouples environment development from training, enabling scalable rollout collection and seamless integration with RL training frameworks.

## Architecture Diagram

<!-- TODO: Add architecture diagram showing server relationships and request flow -->

## Design Principles

### Decoupled Server Architecture

NeMo Gym separates concerns into three server types that communicate via REST APIs:

- **Model Servers**: Stateless LLM inference—scales horizontally without coordination
- **Resources Servers**: Stateful environment logic—manages sessions, tools, and verification
- **Agent Servers**: Orchestration layer—coordinates rollout execution across servers

This separation allows teams to develop and test environments independently from training frameworks.

### REST API Interoperability

All servers expose REST APIs, enabling:

- Language-agnostic integration (Python, Go, Rust, etc.)
- Standard HTTP tooling for debugging and monitoring
- Easy containerization and deployment

### Stateless Models, Stateful Resources

Models remain stateless for horizontal scaling, while resources servers maintain session state. This design enables efficient GPU utilization for inference while preserving environment context across multi-step rollouts.

## Request Flow

A typical rollout follows this sequence:

```
Training Framework
        │
        ▼
┌───────────────────┐
│   Agent Server    │ ◄── Orchestrates rollout lifecycle
└───────────────────┘
        │
        ├──────────────────────────────────────┐
        ▼                                      ▼
┌───────────────────┐                ┌───────────────────┐
│  Model Server     │                │ Resources Server  │
│                   │                │                   │
│ /v1/responses     │                │ /seed_session     │
│ /v1/chat/...      │                │ /verify           │
└───────────────────┘                └───────────────────┘
```

1. **Initialize**: Agent calls Resources Server `/seed_session` to set up task state
2. **Generate**: Agent requests text generation from Model Server `/v1/responses`
3. **Execute**: Agent sends tool calls to Resources Server for execution
4. **Iterate**: Steps 2-3 repeat for multi-step rollouts
5. **Verify**: Agent calls Resources Server `/verify` to compute reward
6. **Return**: Agent returns complete rollout to training framework

## Service Discovery

The head server (port 11000) acts as a service registry, enabling servers to discover and communicate with each other by name rather than hardcoded addresses.

For deployment patterns and scaling strategies, see {doc}`../infrastructure/index`.

## Learn More

- {doc}`concepts/core-components` — Detailed component descriptions and examples
- {doc}`concepts/configuration` — Configuration system and Hydra integration
- {doc}`../infrastructure/deployment-topology` — Production deployment patterns
- {doc}`../infrastructure/ray-distributed` — Distributed scaling with Ray
