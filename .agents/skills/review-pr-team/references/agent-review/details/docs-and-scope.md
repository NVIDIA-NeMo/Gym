# Docs, scope and shared callers (items 11, 12)

## Claims match the code (item 12)

This was the most frequent finding on these PRs: 14 PRs, from 5 reviewers.

**Check that the text says what the code does:**

- the adapter README;
- docstrings;
- the Fern guide (`fern/versions/latest/pages/agent-server/`);
- the paired benchmark's harness table;
- migration notes;
- the PR description.

That includes docs outside the diff that describe changed behaviour.

**Examples:**

- The README said foreground tools were stopped; in fact they escaped (#3902).
- The guide still offered the uv/python-in-sandbox paths the PR removed (#5094).
- The PR description said the harness runs in the Agent Server; the code uses the sandbox (#5540).

**Scope each claim** to the session path, legacy `/run`, or reverification (#5703).

**Docs and catalog** (from `.agents/skills/review-pr-team/references/agent-harness-review.md`):

- The README states the harness version, the capability level, the supported pairings and the limits.
- The tables in `integrate-existing-agents.mdx` and `add-or-update-a-harness.mdx` match.
- Each quoted command runs as written.

## Shared changes reach every caller (item 11)

**Rule:** when a PR changes a shared helper, version pin, aggregation, supervisor, or Environment Server path:

- trace the other callers: other benchmarks, sibling harnesses, reverification, other Environment Servers;
- state the score impact, or say there is none;
- split the shared part into its own PR when it is large.

**Examples:**

- A supervisor change reached SWE-bench Pro and Hermes; add their owners or split it out (#3902).
- Published aggregates changed for critpt and gdpval; add a compatibility note (#5455).
- Harbor scores might shift; say so in the description (#4653).
- A version bump touched other presets; split it, or compare before and after (#5538).

**How to find callers:**

```bash
grep -rn "<helper or ClassName>" nemo_gym responses_api_agents environment_servers resources_servers
grep -rln "<agent_name>" benchmarks resources_servers/*/configs environments nemotron_recipes \
  responses_api_agents/*/configs nemo_gym/data/tasks/harbor/prepare.py
```

## Naming and API (part of item 16)

- **One term per concept,** named for its job, in Gym's vocabulary. Four names for one concept were flagged (#3961). "Execution metrics" added a third name where "perf" exists (#5455).
- **Export no unused or half-exported API.** A class was exported without its producer, and its name clashed with `sandbox_platform()` (#5152).
- **One implementation per protocol.**
- **Avoid qualifiers that carry no information,** such as "native", "legacy", "new" or "old" (repo `AGENTS.md`).
