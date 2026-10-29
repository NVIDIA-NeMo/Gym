# BIRD-Interact Environment Server

Runs [BIRD-Interact](https://bird-interact.github.io/) episodes on `BaseUserSimulationEnvironmentServer`
(`nemo_gym/base_user_simulation_environment_server.py`). The `bird_interact` Resources Server holds each episode's
database sandbox, user simulator and grader; see `resources_servers/bird_interact/README.md`.

| `mode` | Turns | Agent access | `next_user_turn` |
|---|---|---|---|
| `c_interact` | Up to `max_turns`. The agent must resume its session across turns. | None | A reply with a ```` ```sql ```` or ```` ```sqlite ```` block is a submission (`/c_interact/submit`); any other reply is a question (`/c_interact/ask`). The next user message is the simulator's answer, the debug prompt or the follow-up question. |
| `a_interact` | One (`max_turns: 1`). | The Resources Server's tools under `/a_interact/` | `Stop("agent_finished")` |

The opening request has the instructions and the user's ambiguous request. In c-Interact it also has the schema,
the column meanings and the visible knowledge base. In a-Interact it has the nine tool definitions instead. It is
built from the context in the Resources Server's seed reply.

`finish_episode` calls the Resources Server's `/verify`. The result has the base fields (`reward`, `turns`,
`stop_reason`, agent observations) and BIRD's (`phases`, `questions_asked`, `coins_spent`, `finished`).

Stop reasons: `solved`, `failed` (the second attempt failed), `no_sql_submitted` (a question after the question
budget, after a submission, or in phase 2), `agent_finished` (a-Interact), and the base class's `max_turns` and
`interaction_timeout`.

The configuration is in `resources_servers/bird_interact/configs/bird_interact.yaml`. It defines one Environment
Server for each mode and routes the tasksets `bird_interact:c_interact_example` and
`bird_interact:a_interact_example` to them.
