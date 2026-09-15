# Image Tools Agent

The image tools agent adds iterative image inspection to an existing NeMo Gym agent workflow. It sends each request to the configured model, executes supported image tool calls, and returns the final assistant response to the resource server for verification.

Each request must include `image_tools_base_agent_ref`. The referenced agent name selects an entry from `resource_servers_by_agent`, which determines the resource server used for tool execution and final verification.

## Configuration

The agent configuration controls:

- the model server and resource-server mappings
- the maximum number of model and tool steps
- crop output format, quality, and pixel bounds
- rewards and penalties for valid, invalid, and duplicate tool calls

See `configs/image_tools_agent.yaml` for the available settings.

## Testing

```bash
gym env test +entrypoint=responses_api_agents/image_tools_agent
```
