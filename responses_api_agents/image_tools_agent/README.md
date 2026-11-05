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

## Agent sessions

The agent also serves the agent sessions API, so a `single_agent_turn` Environment Server can run it
(`configs/image_tools_agent_single_agent_turn.yaml`). A session takes no granted tools and no sandbox: the image tools
run in the agent. The session's one activation runs the same tool loop; a retried activation returns the rollout it
already ran. The response names the agent and lists the crop files the rollout wrote under `image_tools_rollout`.

Under the Environment Server, the `image_tools_reward` resources server verifies instead of `/run`: it routes each row
by `image_tools_base_agent_ref`, verifies the final answer and adds the tool reward, which it recomputes from the
rollout (see `resources_servers/image_tools/README.md`). The reward rule is the same: the verifier's reward plus the
tool reward, at most 1.0.

## Testing

```bash
gym env test +entrypoint=responses_api_agents/image_tools_agent
```
