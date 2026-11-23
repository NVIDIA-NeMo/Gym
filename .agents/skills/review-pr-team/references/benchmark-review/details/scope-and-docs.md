# Blast radius, claims and shared callers (item 16)

## Map the blast radius

For each changed Resources Server, dataset or Harbor entry, list every config that loads it and trace each one:

```
config: benchmarks/<name>/, environments/<name>/, resources_servers/*/configs/, nemotron_recipes/,
        environments/harbor/<entry>/ (generated run config)
  -> Resources Server instance: allowed_agents, verified, sandbox config and ttl_s,
     what its seed returns (sandbox_access | sandbox_handle | sandbox_descriptor | none)
  -> Environment Server and route: single_agent_turn (+ taskset route, resources_tool_transports, tool_servers)
     | harbor | single_agent_turn_legacy | legacy_agent
  -> each harness in allowed_agents: session adapter or /run relay
  -> judge and helper model servers
```

```bash
grep -rln "<server_name>" benchmarks environments resources_servers/*/configs nemotron_recipes \
  responses_api_agents/*/configs
grep -n "allowed_agents\|verified\|ttl_s" resources_servers/<server>/configs/*.yaml
grep -n "environment_server_routes\|taskset\|single_agent_turn\|legacy_agent\|resources_tool_transports" \
  <each config above>
grep -n "sandbox_access=\|sandbox_handle\|sandbox_descriptor\|LentSandboxes" resources_servers/<server>/app.py
```

For a shared change (a base class, an Environment Server, the Harbor loader, a grading helper):

```bash
grep -rn "<ClassName or helper>" nemo_gym resources_servers environment_servers responses_api_agents
ls environments/harbor/            # every Harbor entry the loader or the harbor server reaches
```

Check every finding against every row of the map, not only the pairing the PR tested. Put the map in the leader's
summary when it has more than three rows.

## Claims match the code

This is the most frequent ask on benchmark PRs (about 43 of 69 PRs with a kept comment).

**Check that the text says what the code does:**

- the server's and the benchmark's `README.md` (path, supported harnesses, limits, how to run it);
- docstrings;
- Fern pages under `fern/versions/latest/pages/` that describe the changed behaviour, including pages outside the
  diff;
- the manifest (`reward.range`, `determinism`, `session_model`, `state`, `sandbox`);
- the PR description: what changed, why, the exact validation, rollout evidence or `N/A`.

**Each quoted command runs as written.** A `gym eval run` target exists; a Harbor entry needs `gym dataset pull`
first; README anchors resolve.

**A README that teaches the old path is a finding** in a new benchmark, even when it copies a stale tutorial (see
`known-conflicts.md` items 8-10).

## Shared changes reach every caller

- Trace the other callers: other benchmarks, other Harbor entries, reverification, other Environment Servers.
- State the score impact, or say there is none. When published aggregates change, bump the benchmark's version or
  add a compatibility note (#5455, #5991).
- Keep existing defaults; make new behaviour opt-in, or call out the change (#5149, #5397).
- Keep benchmark-specific fields and names out of shared request models, shared graders and general Fern pages
  (#4658, #3966, #5635). Reuse the shared constant or helper instead of copying it (#5620).
- Split a large shared part, or a live bug fix on main, into its own PR (#5480).
- An unrelated change to `num_repeats`, a prompt or a sampling setting alters cost and results; ask for it to be
  split out or justified (#5386).

## Written nowhere, enforced in review

Reviewers asked for these on several PRs. Treat them as `RISK`, and say they are review conventions:

- Judges run at temperature near 0; vendored data keeps its contamination canary (#5285).
- Gym code and comments stay neutral about the training framework (#4209).
- Example rollouts and READMEs carry no internal hostnames or job IDs (#5135, #3997).
- A test that still passes with its fix deleted is not a test (the mutation probe; `AGENTS.md` says only "assert
  real behavior").
