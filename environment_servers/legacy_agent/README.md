# Legacy agent environment server

This Environment Server provides backward compatibility support for Agent Servers that have not been migrated to Environment Server protocols.
It relays `/run` to the agent without reading either side's contract (body, cookies, status, and headers other than connection and framing headers) and forwards `/aggregate_metrics` to the agent, which aggregates its own rollouts.

Use this server so rollout collection reaches an unmigrated agent through an Environment Server. It applies none of the base episode limits; the agent keeps its own.

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

## Tasksets

A dataset that declares `taskset:` can route to this server through `environment_server_routes`, so an unmigrated agent can serve a taskset before it moves to an Environment Server protocol.
Rollout collection does not send this server an episode request. It rebuilds the run request row the agent reads today: the task's fields, the collector keys such as `_ng_task_index` and `_ng_rollout_index`, and the agent's `agent_ref`.
The agent's `/run` result is stored as the rollout, with the task's ID.

```yaml
environment_server_routes:
  my_benchmark: my_benchmark_environment_server

my_benchmark_resources_server:
  resources_servers:
    my_benchmark:
      datasets:
        - name: my_benchmark
          type: benchmark
          taskset: my_benchmark
          jsonl_fpath: benchmarks/my_benchmark/data/my_benchmark_benchmark.jsonl
          prepare_script: benchmarks/my_benchmark/prepare.py
```

This server binds no resources server. A taskset declared on a resources server routes here when the server's agent references that resources server, and collation validates the rows against that resources server's `task_data.py`.
