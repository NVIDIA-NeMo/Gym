# Legacy agent environment server

This Environment Server provides backward compatibility support for Agent Servers that have not been migrated to Environment Server protocols.
It relays `/run` to the agent without reading either side's contract (body, cookies, status, and headers other than connection and framing headers) and forwards `/aggregate_metrics` to the agent, which aggregates its own rollouts.

Use this server so rollout collection reaches an unmigrated agent through an Environment Server. It applies none of the base episode limits; the agent keeps its own.

This is the legacy path for running an episode. Gym plans to deprecate `legacy_agent` and remove support for it over time, so new agents should implement agent sessions and run under an Environment Server that drives them, such as [`single_agent_turn`](../single_agent_turn/README.md).

Every agent instance a run can dispatch to needs an Environment Server that names it. `scripts/add_legacy_agent_environment_servers.py` adds one of these for each agent instance in a config:

```yaml
my_benchmark_environment_server:
  environment_servers:
    legacy_agent:
      entrypoint: app.py
      agent_server:
        type: responses_api_agents
        name: my_benchmark_simple_agent
```
