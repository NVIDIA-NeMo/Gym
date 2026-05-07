# NeMo Gym Agent Patterns

Production code patterns for custom agent servers. Each pattern is self-contained.

---

## Base agent structure

```python
from pydantic import BaseModel
from starlette.requests import Request

from nemo_gym.servers.responses_api_agent import SimpleResponsesAPIAgent


class MyAgentConfig(BaseModel):
    max_turns: int = 3
    resources_server: dict = {}
    model_server: dict = {}
    name: str = "my_agent"


class MyAgent(SimpleResponsesAPIAgent):
    config: MyAgentConfig

    async def run(self, request: Request, body) -> dict:
        # All agent logic goes here
        ...
```

Key requirements:
- Extend `SimpleResponsesAPIAgent`
- `run()` must be `async def`
- Accept `request: Request` (for cookies) and `body` (the run request)

Inherited attributes:
- `self.server_client` — `ServerClient` instance for async HTTP calls to model/resources servers. Wraps aiohttp with retry logic (3 tries, exponential backoff) and connection pooling.
- `self.config` — the agent's Hydra config (resources_server, model_server, datasets, etc.)
- Override `aggregate_metrics()` for custom metric computation after rollout collection.

---

## Multi-turn correction loop

Model gets multiple attempts to solve a problem. Error feedback is sent back on failure.

```python
async def run(self, request: Request, body) -> dict:
    cookies = request.cookies
    current_input = body.model_dump()

    all_prompt_token_ids = []
    all_generation_token_ids = []
    all_generation_log_probs = []

    for turn in range(self.config.max_turns):
        # Model call
        gen_resp = await self.server_client.post(
            server_name=self.config.name,
            url_path="/v1/responses",
            json=current_input,
            cookies=cookies,
        )
        await raise_for_status(gen_resp)
        cookies = gen_resp.cookies

        model_response = await gen_resp.json()

        # Accumulate token IDs
        all_prompt_token_ids.extend(model_response.get("prompt_token_ids", []))
        all_generation_token_ids.extend(model_response.get("generation_token_ids", []))
        all_generation_log_probs.extend(model_response.get("generation_log_probs", []))

        output_text = model_response.get("output_text", "")

        # Verify
        verify_resp = await self.server_client.post(
            server_name=self.config.resources_server.get("name", ""),
            url_path="/verify",
            json={"output_text": output_text, "verifier_metadata": body.get("verifier_metadata", {})},
            cookies=cookies,
        )
        await raise_for_status(verify_resp)
        cookies = verify_resp.cookies

        verify_data = await verify_resp.json()

        if verify_data.get("reward", 0.0) == 1.0:
            break

        # Build error feedback for next turn
        error_msg = verify_data.get("errors", verify_data.get("feedback", "Incorrect. Try again."))
        current_input = {
            "input": current_input.get("input", []) + [
                {"role": "assistant", "content": output_text},
                {"role": "user", "content": f"That was incorrect. {error_msg} Please try again."},
            ]
        }

    # Attach accumulated token IDs
    verify_data["prompt_token_ids"] = all_prompt_token_ids
    verify_data["generation_token_ids"] = all_generation_token_ids
    verify_data["generation_log_probs"] = all_generation_log_probs

    return verify_data
```

**Critical requirements:**
- `cookies=cookies` on every `server_client.post()` call
- `cookies = response.cookies` after every response
- Token IDs accumulated with `.extend()` across all turns
- Max turns guard to prevent infinite loops

---

## External library wrapper

When wrapping a 3rd-party library that uses httpx internally.

### aiohttp adapter (replaces httpx transport)

```python
from pydantic import BaseModel
from nemo_gym.server_utils import request


class AIOHTTPClientResponse(BaseModel):
    """Drop-in replacement for httpx.Response."""
    status_code: int
    data: dict

    def json(self):
        return self.data


class AIOHTTPClient(BaseModel):
    """Drop-in replacement for httpx.AsyncClient."""
    headers: dict
    base_url: str

    async def post(self, endpoint: str, content: str, timeout: float) -> AIOHTTPClientResponse:
        response = await request(
            method="POST",
            headers=self.headers,
            url=f"{self.base_url}{endpoint}",
            data=content,
        )
        return AIOHTTPClientResponse(
            status_code=response.status,
            data=await response.json(),
        )

    @classmethod
    def from_httpx_client(cls, client, **kwargs):
        return cls(
            headers=dict(client.headers),
            base_url=str(client.base_url),
            **kwargs,
        )
```

### Agent wrapper

```python
import asyncio

from nemo_gym.servers.responses_api_agent import SimpleResponsesAPIAgent


class ExternalBenchmarkAgent(SimpleResponsesAPIAgent):
    config: ExternalBenchmarkConfig

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.library = ExternalLibrary()
        # Replace httpx with aiohttp adapter
        self.library._client = AIOHTTPClient.from_httpx_client(self.library._client)
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent)

    async def run(self, request, body) -> dict:
        # Pre-process: Gym schema → library format
        library_input = {
            "prompt": body["responses_create_params"]["input"][-1]["content"],
            "task_id": body["verifier_metadata"]["task_id"],
        }

        # Execute with concurrency control
        async with self.semaphore:
            result = await self.library.evaluate(library_input)

        # Post-process: library output → Gym response
        return {
            "reward": 1.0 if result["passed"] else 0.0,
            "output_text": result.get("output", ""),
            "response": {"output_text": result.get("output", "")},
        }
```

---

## Tool-call loop

Model calls tools iteratively until it produces a final answer.

```python
async def run(self, request: Request, body) -> dict:
    cookies = request.cookies
    current_input = body.model_dump()
    max_iterations = self.config.max_tool_calls

    all_prompt_token_ids = []
    all_generation_token_ids = []
    all_generation_log_probs = []

    for iteration in range(max_iterations):
        # Model call
        gen_resp = await self.server_client.post(
            server_name=self.config.name,
            url_path="/v1/responses",
            json=current_input,
            cookies=cookies,
        )
        await raise_for_status(gen_resp)
        cookies = gen_resp.cookies

        model_response = await gen_resp.json()

        # Accumulate token IDs
        all_prompt_token_ids.extend(model_response.get("prompt_token_ids", []))
        all_generation_token_ids.extend(model_response.get("generation_token_ids", []))
        all_generation_log_probs.extend(model_response.get("generation_log_probs", []))

        # Check if model wants to call tools
        tool_calls = model_response.get("tool_calls", [])
        if not tool_calls:
            break  # No more tool calls — model is done

        # Execute each tool call
        tool_results = []
        for tool_call in tool_calls:
            tool_resp = await self.server_client.post(
                server_name=self.config.resources_server.get("name", ""),
                url_path=f"/tools/{tool_call['function']['name']}",
                json=tool_call["function"]["arguments"],
                cookies=cookies,
            )
            cookies = tool_resp.cookies
            tool_result = await tool_resp.json()
            tool_results.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": json.dumps(tool_result),
            })

        # Build next turn with tool results
        messages = current_input.get("input", [])
        messages.append({"role": "assistant", "content": "", "tool_calls": tool_calls})
        messages.extend(tool_results)
        current_input = {"input": messages}

    # Final verification
    output_text = model_response.get("output_text", "")
    verify_resp = await self.server_client.post(
        server_name=self.config.resources_server.get("name", ""),
        url_path="/verify",
        json={"output_text": output_text, "verifier_metadata": body.get("verifier_metadata", {})},
        cookies=cookies,
    )
    cookies = verify_resp.cookies
    verify_data = await verify_resp.json()

    verify_data["prompt_token_ids"] = all_prompt_token_ids
    verify_data["generation_token_ids"] = all_generation_token_ids
    verify_data["generation_log_probs"] = all_generation_log_probs

    return verify_data
```

---

## YAML config for custom agents

```yaml
my_custom_agent:
  responses_api_agents:
    my_agent:                       # Must match the directory name
      entrypoint: app.py
      max_turns: 3
      max_tool_calls: 10
      resources_server:
        type: resources_servers
        name: my_benchmark          # Must match instance name
      model_server:
        type: responses_api_models
        name: policy_model          # Must match instance name
      datasets:
      - name: my_example
        type: example
        jsonl_fpath: resources_servers/my_benchmark/data/example.jsonl
```
