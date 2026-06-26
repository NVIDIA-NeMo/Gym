# Example MCP Weather

A minimal **Gym-owned MCP Resources Server**: it mounts a Streamable-HTTP MCP endpoint at `/mcp` on the
same FastAPI app as `/seed_session` and `/verify`. The `get_weather` MCP tool records its calls against the
per-rollout Gym session (resolved from a hidden `X-NeMo-Gym-Session-Token`), and `/verify` rewards a rollout
only if the tool was used **in that same session** and the final answer repeats the returned sentence.

This is the runnable companion to the [MCP Resources Server tutorial](https://github.com/NVIDIA-NeMo/Gym/tree/main/fern/versions/latest/pages/environment-tutorials/mcp-resources-server.mdx).

## Run with an agent (Claude Code)

Put your key in a repo-root `env.yaml` (the config interpolates `${anthropic_api_key}`), then start the servers
— the `claude_code_agent` runs `claude` with that key injected:

```bash
# env.yaml:  anthropic_api_key: sk-ant-...
gym env start --config resources_servers/example_mcp_weather/configs/example_mcp_weather.yaml
```

Then collect rollouts against `data/example.jsonl` and reward-profile as in the
[quickstart](https://github.com/NVIDIA-NeMo/Gym/tree/main/fern/versions/latest/pages/get-started/quickstart.mdx).
A correct rollout shows Claude Code calling `mcp__example_mcp_weather__get_weather` and a `reward` of `1.0`.

## Inspect the MCP round-trip without an LLM

Start the server and drive the endpoint directly — a `requests.Session` preserves the session cookie so
`/verify` sees the same session as the tool call:

```python
import requests

s = requests.Session()
meta = s.post("http://127.0.0.1:<port>/seed_session", json={"verifier_metadata": {"expected_city": "Paris"}}).json()["mcp"]
token = meta["headers"]["X-NeMo-Gym-Session-Token"]

# call the MCP tool over the mounted /mcp route, carrying the per-rollout token
s.post(
    f"http://127.0.0.1:<port>{meta['url_path']}",
    headers={"Accept": "application/json, text/event-stream", "X-NeMo-Gym-Session-Token": token},
    json={"jsonrpc": "2.0", "id": 1, "method": "tools/call",
          "params": {"name": "get_weather", "arguments": {"city": "Paris"}}},
)

# verify in the same session -> reward 1.0
print(s.post("http://127.0.0.1:<port>/verify", json={
    "responses_create_params": {"input": [{"role": "user", "content": "use the weather tool"}]},
    "verifier_metadata": {"expected_city": "Paris"},
    "response": {"id": "r", "created_at": 0, "model": "t", "object": "response", "output": [
        {"id": "m", "type": "message", "role": "assistant", "status": "completed",
         "content": [{"type": "output_text", "text": "The weather in Paris is sunny and 72 F.", "annotations": []}]}],
        "parallel_tool_calls": False, "tool_choice": "none", "tools": []},
}).json()["reward"])
```

## Tests

```bash
gym env test --resources-server example_mcp_weather
```
