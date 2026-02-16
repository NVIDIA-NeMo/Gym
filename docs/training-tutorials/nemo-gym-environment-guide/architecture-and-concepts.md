(building-environments-architecture)=

# Architecture and Core Concepts

This section covers NeMo Gym's architecture and the foundational concepts you need before building environments.

---

## Overview: What is NeMo Gym?

NeMo Gym is a framework for building RL (Reinforcement Learning) environments designed specifically for training Large Language Models (LLMs). It provides a microservices-based architecture where environments (called **Resource Servers**) interact with LLM agents through well-defined HTTP APIs.

**Key characteristics:**
- **Microservices architecture**: Each component runs as an independent HTTP server
- **OpenAI-compatible APIs**: Uses the Responses API format for tool calling
- **Scalable**: Components can be distributed across machines
- **Composable**: Mix and match environments, agents, and models

### A Note on Terminology: Environments vs Resource Servers

In NeMo Gym, **"Resource Server" and "Environment" are the same thing**. The codebase uses "Resource Server" to emphasize the microservices architecture, but functionally these serve the same role as environments in traditional RL frameworks.

| Traditional RL (OpenAI Gym) | NeMo Gym |
|----------------------------|----------|
| Environment (Python object) | Resource Server (HTTP microservice) |
| `env.step(action)` | `POST /{tool_endpoint}` |
| `env.reset()` | `POST /seed_session` |
| Reward from `step()` | Reward from `POST /verify` |

This naming is reflected in the codebase:
- The base class is `SimpleResourcesServer` (see `nemo_gym/base_resources_server.py:55-73`)
- All environments live in the `resources_servers/` directory
- Config types use `ResourcesServerTypeConfig` (see `nemo_gym/config_types.py:449`)

Throughout this guide, "environment" and "resource server" are used interchangeably.

---

## Architecture: The Three-Tier System

NeMo Gym uses a three-tier microservices architecture:

```
┌─────────────────────────────────────────────────────────────┐
│              HeadServer (Port 11000)                        │
│   - Global configuration distribution                       │
│   - Server discovery and registry                           │
└─────────────────────────────────────────────────────────────┘
              ↓                    ↓                    ↓
┌──────────────────────┐ ┌──────────────────────┐ ┌──────────────────────┐
│ ResponsesAPIModel    │ │ ResponsesAPIAgent    │ │ ResourcesServer      │
│ (LLM inference)      │ │ (Orchestrator)       │ │ (Your Environment)   │
│                      │ │                      │ │                      │
│ /v1/chat/completions │ │ /v1/responses        │ │ /verify              │
│ /v1/responses        │ │ /run                 │ │ /seed_session        │
│                      │ │                      │ │ /{custom_tools}      │
└──────────────────────┘ └──────────────────────┘ └──────────────────────┘
```

### Component Responsibilities

| Component | Purpose | Key Endpoints |
|-----------|---------|---------------|
| **HeadServer** | Central registry, config distribution | `/global_config_dict_yaml`, `/server_instances` |
| **ResponsesAPIModel** | LLM inference (OpenAI, vLLM, etc.) | `/v1/chat/completions`, `/v1/responses` |
| **ResponsesAPIAgent** | Orchestrates rollouts | `/v1/responses`, `/run` |
| **ResourcesServer** | Your RL environment | `/seed_session`, `/verify`, `/{tools}` |

**Key insight**: Unlike traditional RL environments (like OpenAI Gym), NeMo Gym environments are HTTP servers that can be distributed, scaled, and composed independently.

---

## Core Concepts

### 1. The Resource Server Lifecycle

Every RL episode in NeMo Gym follows this lifecycle:

```
1. seed_session() → Initialize environment state for a task. Called once at the start of each `run` episode (one dataset row / rollout invocation)
         ↓
2. [Tool Calls]   → Agent interacts with environment via custom endpoints. This happens repeatedly over the course of the rollout.
         ↓
3. verify()       → Evaluate final response, compute reward. Called once at the end of the episode.
```

### 2. The Base Classes

The foundation of every Resource Server is `SimpleResourcesServer` from `nemo_gym/base_resources_server.py`:

```python
# simplified
from abc import abstractmethod
from fastapi import FastAPI
from pydantic import BaseModel
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import BaseRunServerInstanceConfig, BaseServer, SimpleServer


class BaseRunRequest(BaseModel):
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming


class BaseVerifyRequest(BaseRunRequest):
    response: NeMoGymResponse  # The model's full response


class BaseVerifyResponse(BaseVerifyRequest):
    reward: float  # The RL reward signal (typically 0.0-1.0)


class BaseSeedSessionRequest(BaseModel):
    pass  # Extend for custom initialization data


class BaseSeedSessionResponse(BaseModel):
    pass  # Extend for custom initialization responses


class SimpleResourcesServer(BaseResourcesServer, SimpleServer):
    config: BaseResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = FastAPI()
        self.setup_session_middleware(app)
        app.post("/seed_session")(self.seed_session)
        app.post("/verify")(self.verify)
        return app

    async def seed_session(self, body: BaseSeedSessionRequest) -> BaseSeedSessionResponse:
        return BaseSeedSessionResponse()

    @abstractmethod
    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        pass  # YOU MUST IMPLEMENT THIS
```

### 3. Key Abstractions

| Class | Purpose |
|-------|---------|
| `SimpleResourcesServer` | Base class for all environments |
| `BaseVerifyRequest` | Contains the model's response for evaluation |
| `BaseVerifyResponse` | Must include `reward: float` |
| `BaseSeedSessionRequest/Response` | Initialize episode state |
| `SESSION_ID_KEY` | Unique identifier for stateful sessions |

---

> **Next**: {ref}`Single-Step Environment <building-environments-single-step>` — Build your first environment.
