# Misalignment-monitored agent environment server

Fronts one agent server like `legacy_agent`, and hands each finished rollout to the misalignment monitor
(`resources_servers/misalignment_monitor`). Use it in place of the agent's `legacy_agent` environment server.
It covers agents on the legacy `/run` path only. Agents migrated to session harnesses are not monitored yet.
A config looks like this:

```yaml
my_benchmark_environment_server:
  environment_servers:
    misalignment_monitored:
      entrypoint: app.py
      agent_server:
        type: responses_api_agents
        name: my_benchmark_simple_agent
      monitor_server:
        type: resources_servers
        name: misalignment_monitor
```

- `/run` is relayed to the agent and its reply comes back unchanged. The rollout's result goes to the monitor's
  `/monitor` in the background, with its task and rollout indices and the rollout id its model calls were
  captured under, so the reward and timing are what they would be without the monitor.
- Before relaying a rollout, the server waits while the monitor's `/status` asks for a pause, which it does while a
  P0 alert is unresolved past `pause_after_unresolved_p0_minutes`. In-flight rollouts finish. The first rollouts
  wait for the first status answer. A monitor that does not answer within 10 seconds does not hold the run.
- Posts that fail are counted, and the first and every hundredth failure are printed. Shutdown waits up to a minute
  for posts in flight and prints the counts.
- When `gym eval run` finishes, `/aggregate_metrics` first asks the monitor to finish its reviews (`/drain`), so the
  last rollouts are reviewed before the judge stops. The wait is capped at `monitor_drain_timeout_s` (660 seconds),
  after which metrics go on without the monitor. Raise it with the monitor's `drain_timeout_s`.

# Licensing information

Code: Apache 2.0

Dependencies:
- nemo_gym: Apache 2.0
