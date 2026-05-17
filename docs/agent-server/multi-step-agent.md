# Multi-Step Agent

The Multi-Step Agent [`SimpleAgent`](https://github.com/NVIDIA-NeMo/Gym/tree/main/responses_api_agents/simple_agent) handles **single-turn, multi-step** interactions: one user message triggers a model response that may loop through multiple tool calls before verification produces a reward.

## How It Works

The agent loop sends the conversation to the model, and if the model makes tool calls, executes them against the resources server and feeds the results back. This repeats until the model produces a text response with no tool calls, `max_steps` is reached, or the context length is exceeded.

```python
class SimpleAgent:
    async def run(self, task_data):
        # 1. Initialize episode
        resources_server.seed_session(task_data)

        # 2. Run the agent loop (multi-step tool-call loop)
        response = self.responses(task_data.prompt, task_data.tools)

        # 3. Grade the result
        reward = resources_server.verify(response, task_data.ground_truth)
        return response, reward

    async def responses(self, prompt, tools):
        conversation = prompt

        while step < max_steps:
            # Call the model with the full conversation
            model_output = model_server.responses(conversation, tools)
            conversation.append(model_output)

            # Model produced text — done, no more tool calls
            if model_output is text:
                break

            # Execute tool calls against the resources server
            for tool_call in model_output.function_calls:
                result = resources_server.post(f"/{tool_call.name}", tool_call.arguments)
                conversation.append(result)

        return conversation
```

## Configuration

```yaml
simple_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: ???
      model_server:
        type: responses_api_models
        name: policy_model
      max_steps: null
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `resources_server` | ref | required | Resources server for tools and verification |
| `model_server` | ref | required | Policy model being trained/evaluated |
| `max_steps` | int | null | Max tool-call loop iterations. Null = unbounded; loop stops when model produces text |

## Stop Conditions

- **Text with no tool calls:** model is done
- **`max_steps` reached:** safety limit
- **`max_output_tokens` hit:** context length exceeded

## Output Format

The final `output` list contains all outputs from the single turn in the order they were produced. Each item has a `type` field:

| Type | Description |
|------|-------------|
| `reasoning` | Model's chain-of-thought (if reasoning/thinking is enabled) |
| `message` | Assistant text response (`role: "assistant"`, contains `output_text`) |
| `function_call` | Assistant requesting a tool call (name, arguments) |
| `function_call_output` | Tool result returned from the Resources server |

A typical sequence with tool calls:

```
[reasoning, function_call, function_call_output, reasoning, message]
```
