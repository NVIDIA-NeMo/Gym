(agent-server-nemo-agent-toolkit)=

# NeMo Agent Toolkit Integration

This page documents the **NeMo Gym agent-server integration surface** that external agent frameworks must build against. It focuses on the interface and behavior defined in this repository so you can adapt an external agent to NeMo Gym.

**Purpose**: Understand the NeMo Gym agent-server integration surface.  
**Audience**: Developers integrating external agent frameworks into NeMo Gym.

**Key capabilities**:

- **Endpoints**: `SimpleResponsesAPIAgent` registers `/v1/responses` and `/run`.
- **Run types**: `BaseRunRequest`/`BaseVerifyResponse` define the run path payload and return shape.
- **Reference behavior**: `SimpleAgent` shows model calls, tool routing, and loop control.
- **Configuration**: `SimpleAgentConfig` defines the expected agent server fields.

---

## Integration surface in NeMo Gym

**Integration surface** here means the concrete base classes, endpoints, and request/response types your adapter must provide to run as a NeMo Gym agent server.

NeMo Gym implements agent servers as subclasses of `SimpleResponsesAPIAgent`. The base class wires up two HTTP endpoints:

```python
class SimpleResponsesAPIAgent(BaseResponsesAPIAgent, SimpleServer):
    def setup_webserver(self) -> FastAPI:
        app = FastAPI()
        self.setup_session_middleware(app)
        app.post("/v1/responses")(self.responses)
        app.post("/run")(self.run)
        return app
```

The request/response types for the run path come from `nemo_gym.base_resources_server`:

```python
class BaseRunRequest(BaseModel):
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming

class BaseVerifyRequest(BaseRunRequest):
    response: NeMoGymResponse

class BaseVerifyResponse(BaseVerifyRequest):
    reward: float
```

---

## What you provide

An external agent adapter should adhere to the abstract interface in `SimpleResponsesAPIAgent`:

- `responses()` — handles `/v1/responses` and returns a `NeMoGymResponse`.
- `run()` — handles `/run` and returns a `BaseVerifyResponse`.

`SimpleResponsesAPIAgent` defines both methods as abstract, so an adapter must supply concrete methods.

---

## Reference behavior: `SimpleAgent`

The built-in `responses_api_agents/simple_agent` shows the default orchestration behavior:

- **Model calls**: `SimpleAgent.responses()` calls the configured model server at `/v1/responses` and validates the response.
- **Tool routing**: for each `function_call` in the model output, it calls the resources server at `/{tool_name}` and appends a `function_call_output`.
- **Loop control**: the loop ends when the model returns an assistant message without tool calls, or when `max_steps` hits its limit.

The `/run` endpoint performs a complete run:

1. Call resources server `/seed_session`
2. Call the agent server `/v1/responses`
3. Call resources server `/verify` to compute the `reward`

---

## Configuration shape (`responses_api_agents`)

The reference implementation (`SimpleAgent`) defines these configuration fields:

```python
class SimpleAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    max_steps: int = None
```

Configure agent servers under a `responses_api_agents` block in YAML. This example is from `resources_servers/math_with_judge/configs/math_with_judge.yaml`:

```yaml
math_with_judge_simple_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: math_with_judge
      model_server:
        type: responses_api_models
        name: policy_model
```

---

## Where to extend

If you are integrating an external agent framework, build against the interface defined in `nemo_gym/base_responses_api_agent.py`. The reference implementation is `responses_api_agents/simple_agent/app.py`.

## Next steps

- **Agent server overview**: See `docs/agent-server/index.md` for the full agent lifecycle.
- **Resources servers**: See `docs/resources-server/index.md` for tool and verification servers.
