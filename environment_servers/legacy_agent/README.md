# Legacy agent environment server

This Environment Server provides backward compatibility support for Agent Servers that have not been migrated to Environment Server protocols.
It relays `/run` to the agent without reading either side's contract (body, cookies, status, and headers other than connection and framing headers) and forwards `/aggregate_metrics` to the agent, which aggregates its own rollouts.

Use this server so rollout collection reaches an unmigrated agent through an Environment Server. It applies none of the base episode limits except `default_episode_timeout_seconds`, which is unset by default; the agent keeps its own. When set, an episode whose `/run` outlives it ends as the retryable failure `single_agent_turn_legacy` returns for its deadline (`_ng_failure_class: environment_server_failed`, `_ng_failure_message: Episode timed out`), and the relay cancels its call to the agent.

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
