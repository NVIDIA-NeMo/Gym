# EQ-Bench 4 environment server

Runs one [EQ-Bench 4](../../benchmarks/eqbench4/README.md) conversation per episode: the policy roleplays with a
simulated persona for up to `num_turns` turns per side, and the [`eqbench4` resources
server](../../resources_servers/eqbench4/README.md) grades the conversation against published conversations of the
same scenario.

## Episode

1. Seed a resources server session, then an agent session for each participant: the policy (`agent_server`) and the
   persona simulator (`persona_agent`). Each session's close is registered before it is seeded.
2. Alternate the turns. The policy opens.
   - Each policy activation sends that turn's whole input: the turn's system prompt (`evaluated_system_prompts[k]`)
     and the dialogue so far; the first turn sends the row's `responses_create_params` as prepared. A reply without
     visible text is asked again, up to `policy_max_attempts` (7) in all; a context-window overflow is not.
   - Each persona activation sends the persona's view: the conversation with the roles swapped, its own latest reply
     in full (its private `<internal>` monologue included) and earlier ones as dialogue only, and `[Turn k of n]`
     before the policy's latest message, with `persona_responses_create_params`. A reply needs a non-empty
     `<internal>` and `<dialogue>` without leaked control tags; an unusable one is shown back with
     `persona_correction_message`, up to `persona_max_attempts` (3) in all. `<status>ended</status>` ends the
     conversation.
3. Close both agent sessions. Their observations go in the result's `ng_agent_observations`, each session's
   invocations named after its agent instance (`eqbench4_policy_agent:root`, `eqbench4_persona_agent:root`, ...).
   Both sessions name their activations alike, and rollout collection joins a captured model call to the invocation
   its session ID names, so unscoped, each session's calls would also join the other's invocations.
4. Send the resources server's own flat verify body to `/verify`: the row, the policy's last reply as `response`, and
   the conversation (`transcript`, `conversation_status`, `conversation_error`, `policy_stop_reason`). The response's
   usage covers every model call of both sessions, the invocation tree the observations publish, as rollout health
   requires; its metadata carries the conversation, so a re-judge, which rebuilds the verify request from the task and
   the saved response, grades the same conversation. The verify response is the episode's result.
5. Close the resources server session.

`aggregate_metrics` forwards to the resources server, which holds the published comparisons the rating needs, and adds
performance metrics.

## Participants

The persona simulator is part of the environment, not of the policy:

- Only the policy's activations carry a training framework's sampling values and, with token capture on, the
  training-token tag (`applies_model_sampling`). The persona's carry `null` sampling, so it keeps its own settings.
- A persona failure is the environment's, never a score against the policy.
- Each session owns its own model calls in the published observations, so the persona's calls are never the policy's
  turns. Gym's trajectory record has no participant kind besides agents and verifiers, so the persona's session is
  published as an agent beside the policy's, and the response's usage covers both.

Both participants run on `simple_agent` with `max_steps: 1` and `execute_tools: false`, which answers each activation
from the input it is given. The resources server's `allowed_agents` lists the harnesses that passed a smoke run as the
policy.

## Failures

| Cause | Episode |
| --- | --- |
| The policy stopped: no visible reply in its attempts, a context overflow, or a request the model refused as it stands | Graded: a loss to every opponent, with `policy_stop_reason` |
| The persona gave no usable reply in its attempts, or its endpoint or session failed | Failed, `eqbench4:persona_failed`; retryable unless the request itself was refused (a 4xx) |
| A policy or persona model call timed out (`model_policy_response_timeout`) | Failed; retryable (`agent_outcome_policy`'s default here, as in `single_agent_turn`) |
| The policy's call failed (an endpoint outage, or a failed reply that the agent outcome policy retries) | Failed; retryable unless the request itself was refused |
| A judge outage | Graded by the resources server as `judge_failed`, masked, and re-judged from the saved conversation |
| The verifier or a session seed or close failed | Failed at that stage; retryable when the dependency's failure is |

A failed episode keeps both sessions' observations as evidence. A conversation that broke off keeps no response, so
a resumed run runs it again; a failure after the conversation finished (closing a session, or `/verify`) keeps the
graded response, so recovery grades it again.

## Configuration

`resources_servers/eqbench4/configs/eqbench4.yaml` defines the deployment (`eqbench4_environment_server`) with its
participants, and `benchmarks/eqbench4/config.yaml` routes the benchmark's taskset to it. Settings:
`resources_server`, `agent_server`, `persona_agent`, `persona_responses_create_params`, `persona_max_attempts`,
`persona_correction_message`, `policy_max_attempts`, `agent_outcome_policy`, and the shared episode limits
(`default_episode_timeout_seconds`, `cleanup_timeout_seconds`). `agent_server` and `persona_agent` must name
different agent servers, since each participant's invocations are named after its server. Episodes restart from their
input after a crash (`checkpoint_mode: restart_only`).

## Tests

```bash
ng_test +entrypoint=environment_servers/eqbench4
```

## Licensing

Code: Apache-2.0. The conversation protocol follows [EQ-bench/eqbench4](https://github.com/EQ-bench/eqbench4)
(BSD-3-Clause with the Commons Clause condition) as an independent implementation; nothing from it is committed.

Dependencies
- nemo_gym: Apache 2.0
