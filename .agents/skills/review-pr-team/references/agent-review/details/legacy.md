# Legacy agents (item 1)

Line numbers are at main `099832fa7e`. `G` = `fern/versions/latest/pages/agent-server/add-or-update-a-harness.mdx`, `I` = `integrate-existing-agents.mdx`, `P` = the harness consistency plan (summarized in `details/consistency.md`).

## Rule on main

**Deprecated:**

- The `/run` path of `opencode_sandboxed_agent`, `pi_sandboxed_agent` and `hermes_sandboxed_agent`.
- The `/run` path of `terminus_2_sandboxed_agent` and `miniswe_sandboxed_agent`.

Their Harbor episode sessions are **not** deprecated. Do not build new benchmarks, agents or configs on any deprecated path (I:31-35, I:215; G:311, G:348, G:391, G:589).

**What new work uses:** agent sessions, driven by an Environment Server. Today the servers are `single_agent_turn` with typed tasksets, `single_agent_turn_legacy` for flat-row configs, and `harbor`. More may come, such as a multi-agent-turn server where several agents take turns in one episode. `legacy_agent` only relays to an old `/run`.

**Adapters stay server-agnostic.** The contract is seed, activate, close, plus rollout stop. An adapter that depends on one server's call order, row format or cookies breaks under the next server. Examples: assuming exactly one activation per session, or reading `single_agent_turn`-only metadata.

**`resources_server` in the agent config:** set only for the compatibility `/run`; `null` for sessions. Each agent emits a deprecation warning at startup, for example `hermes_sandboxed_agent/app.py:37-48` and `miniswe_sandboxed_agent/app.py:71-74`.

**Session adapters must be registered.** Any `app.py` that defines `_seed_agent_session_state` must be listed in `AGENT_SESSION_AGENT_TYPES` (`nemo_gym/config/global_config.py:217-245`). A test enforces this.

## What reviewers asked for

- **Scope every claim to its path.** Say whether it covers the session path, legacy `/run`, or reverification. On legacy `/run`, Simple Agent forwarded `agent_context` to `/verify`, which a session-only reading missed (#5703).
- **File an issue to track removing legacy-only code** when a PR adds a legacy-only flag or branch (#3902).

## Proposed (plan step 7, not accepted)

Delete each older adapter in the PR that moves its last consumer, together with its presets, its `AGENT_SESSION_AGENT_TYPES` entry, its Harbor override and its health signal (P "Step 7").

Open migrations:

- #3966 removes `hermes_sandboxed_agent`.
- #5825 and #5826 move the Pi and OpenCode presets.
- #5762, #5815 and #5819 move OpenCode consumers.

Several Terminus and mini-SWE pairs remain.

## Review questions

- Does the PR add a preset, feature or consumer on a deprecated `/run`? Flag it.
- Does a fix to one path leave the other path unchanged, or does the PR say otherwise?
- Do user-facing messages that name a replacement agent hold for every preset that shows them? See "User-facing guidance" in `.agents/skills/review-pr-team/references/agent-harness-review.md`.
