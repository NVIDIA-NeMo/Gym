# Description

Simple Agent supports an explicit outcome policy for confirmed policy-model response timeouts on its legacy `/run` route. The default is `retry`: an ungraded, nonterminal failure eligible for Collection’s existing bounded retry on resume. It does not start another immediate model call.

```yaml
outcome_policy:
  by_failure_kind:
    model_policy_response_timeout: count_as_zero
```

`fail` stops the rollout ungraded; `count_as_zero` stops it with an explicit zero. All three actions retain the same classified failure, IDs and causes. Classification requires every observed attempt of the final model operation to have timed out waiting for a response after a confirmed completed upload. Missing phase or attempt evidence, failed uploads, connection/pool timeouts and mixed failures do not qualify. Older reports without upload evidence therefore cannot activate this scoring rule. Model-request and episode budgets remain separate.

Status rules take precedence over kind rules, then `default_action`. The adapter selects the event’s own status; it does not search inner causes. `verify` is rejected when loading this boundary’s policy. Environment Server policies retain their existing verification behavior.

`final_on_model_timeout: true` remains a compatibility alias for the rule above; explicit false selects `retry`. The alias is resolved once into `outcome_policy` and excluded from the serialized resolved configuration. An explicitly conflicting timeout rule or default is rejected. Prefer `outcome_policy` in new configurations.


## Sandbox mode

When its seed lends a sandbox, Simple Agent runs that agent session's loop inside it, instead of in the agent server's process. The environment decides: an Environment Server that lends a sandbox, such as `single_agent_turn` with `sandbox_provider` and `sandbox_config` set, runs every attempt's loop in a sandbox, and a seed that lends none runs it in process, as before. No agent setting changes this.

| Step | What the agent server does |
|------|----------------------------|
| Seed | Borrows the lent sandbox with `borrow()`, installs a private Python runtime in it (built on this host from the pinned CPython and this server's package versions), and unpacks this server's own `nemo_gym` sources and Simple Agent module there. |
| Activation | Stages the loop's arguments and runs `sandbox_runner.py` under Gym's process supervisor. The runner builds the same `SimpleAgent` class and calls the same `_create_episode` loop as in process, so `max_steps`, retries and the outcome policy cannot drift. It calls only the model server, at the address the sandbox reaches, under the same rollout path, and returns the output items unchanged. |
| Close | Stops the loop if it still runs and releases the borrowed client. The sandbox's owner stops the sandbox. |

There is no host fallback. A seed that lends a sandbox is refused (HTTP 422) when the model server or a `direct_http` tool grant is at a loopback address the sandbox cannot reach, and a session lent a sandbox never runs the loop in the agent server's process. A subclass that changes the agent loop (`_create_episode` or `_create_session_episode`) refuses a lent sandbox, since the sandbox runs Simple Agent's own loop. A model server's HTTP error the loop raises comes back with its status, headers and body; any other error the loop raises becomes a `RuntimeError` naming it.

A session lent a sandbox is not checkpointed: after a crash its episode starts over. Retiring or closing it stops the loop and releases the borrowed client. `sandbox_install_timeout_seconds` and `sandbox_runner_timeout_seconds` bound the runtime install and one activation; `session_close_timeout_seconds` bounds stopping the loop. Tool calls go to the granted URL from inside the sandbox, so a pairing that grants `direct_http` tools needs an egress rule that allows the tool server too. Rollout stops (`POST /rollouts/{rollout_id}/stop`) are not supported in sandbox mode yet.

The runtime's packages are pinned to this server's versions and installed as wheels, except OmegaConf's `antlr4-python3-runtime`, which PyPI ships only as source and the host builds from its PyPI URL.

# Licensing information
Code: Apache 2.0
Data: N/A

Dependencies
- nemo_gym: Apache 2.0
