(agent-server-openai-agents-sdk)=
# OpenAI Agents SDK Integration

```{note}
This page documents the current, code-backed status of OpenAI Agents SDK integration in NeMo Gym.
```

---

**Purpose**: Describe the current OpenAI Agents SDK integration status and the agent server interface in NeMo Gym.

**Audience**: Contributors integrating external agents into NeMo Gym.

## Key facts (code-backed)

- The repository has no code references to `openai_agents` or the OpenAI Agents SDK; the references are in documentation files.
- Agent servers subclass `SimpleResponsesAPIAgent`. That base class registers the `/v1/responses` and `/run` endpoints.
- Agent servers in `responses_api_agents/` are `simple_agent`, `aviary_agent`, and `mini_swe_agent`.
- Gym depends on `openai<=2.6.1`. The dependency list does not include the OpenAI Agents SDK package.

## Integration surface in NeMo Gym

`SimpleResponsesAPIAgent` is the base interface an external adapter must provide. It registers the two agent endpoints. It also requires `responses()` and `run()` implementations.

```python
class SimpleResponsesAPIAgent(BaseResponsesAPIAgent, SimpleServer):
    config: BaseResponsesAPIAgentConfig

    def setup_webserver(self) -> FastAPI:
        app = FastAPI()
        self.setup_session_middleware(app)
        app.post("/v1/responses")(self.responses)
        app.post("/run")(self.run)
        return app

    @abstractmethod
    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        pass

    @abstractmethod
    async def run(self, body: BaseRunRequest = Body()) -> BaseVerifyResponse:
        pass
```

## Reference behavior from `simple_agent`

The `simple_agent` implementation shows how the agent server orchestrates model calls, tool calls, and termination.

```python
model_response = await self.server_client.post(
    server_name=self.config.model_server.name,
    url_path="/v1/responses",
    json=new_body,
    cookies=model_server_cookies,
)
...
api_response = await self.server_client.post(
    server_name=self.config.resources_server.name,
    url_path=f"/{output_function_call.name}",
    json=json.loads(output_function_call.arguments),
    cookies=resources_server_cookies,
)
```

Other verified behaviors in `simple_agent`:

- Stops when the model produces assistant messages without tool calls.
- Stops when `max_steps` exists and the step counter reaches that value.
- Stops when the model response reports `max_output_tokens` in `incomplete_details`.

## What this means for OpenAI Agents SDK users

The repository ships the agent servers under `responses_api_agents/`. To integrate an OpenAI Agents SDK agent, add a new agent server that conforms to `SimpleResponsesAPIAgent` and follow the request/response patterns in `responses_api_agents/simple_agent/app.py`.
