# Legacy agent environment server

This Environment Server provides backward compatibility support for Agent Servers that have not been migrated to Environment Server protocols.
It relays `/run` to the agent without reading either side's contract (body, cookies, status, and headers other than connection and framing headers) and forwards `/aggregate_metrics` to the agent, which aggregates its own rollouts.

Use this server so rollout collection reaches an unmigrated agent through an Environment Server. It applies none of the base episode limits except `default_episode_timeout_seconds`, which is 21600 (six hours) by default, like the other Environment Servers; the agent keeps its own limits too. Set it to `null` to leave only the agent's limits. This server ends an episode whose `/run` outlives the deadline with a retryable deadline failure (`_ng_failure_class: environment_server_failed`, `_ng_failure_message: Episode timed out`), and the relay cancels its call to the agent.

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

An episode deadline stops the relay from waiting; it does not prove that the agent stopped before generating an answer. Its recorded deadline failure therefore blocks automatic agent replay. Completed replies remain unchanged.
