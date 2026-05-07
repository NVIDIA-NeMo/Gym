# skill_eval_agent

Agent that orchestrates [agentskills.io](https://agentskills.io/specification)
skill evaluations on real NeMo Gym infrastructure. Produces a `reward` per
rollout equal to the fraction of behavioral assertions satisfied, so
`ng_collect_rollouts` yields an attributable scoreboard when paired with
`scripts/diff_skill_scoreboards.py`.

## Stack

Three servers are wired together:

| Role | Server | What it does |
|---|---|---|
| Tool sandbox | `resources_servers/skill_workspace` | Seeds a scoped tmpdir with scenario fixtures and — gated by `with_references`/`with_scripts` — the skill's `references/` and `scripts/`. `SKILL.md` is never seeded. Exposes `/run_bash` and `/read_file`. |
| Grader | `resources_servers/skill_judge` | LLM-as-judge that returns per-assertion binary grades; aggregate reward = fraction satisfied. |
| Policy | any `responses_api_models/*` | The model under evaluation. |

## /run flow

1. `POST /seed_session` → `skill_workspace` with `{skill_path, scenario_id, files, with_references, with_scripts}` returning `env_id`. The two flags are forwarded from `verifier_metadata`.
2. If `verifier_metadata.with_skill`, prepend a system message containing the SKILL.md body. Inject `run_bash`/`read_file` tool schemas when the incoming request has no tools.
3. Model ↔ tool loop (bounded by `max_steps`). Every tool call is dispatched to `skill_workspace` with the `env_id` and captured into a `ToolCallLogEntry`.
4. `POST /verify` → `skill_judge`, forwarding the captured `tool_calls` + `assertions` + model response via `verifier_metadata`.
5. `POST /close` → `skill_workspace` runs in a `finally` block so the workspace is cleaned up even on error.

## The 2×2 cells

Two independent flags in `verifier_metadata` control the control/treatment structure:

| `with_references` | `with_skill`=False | `with_skill`=True |
|---|---|---|
| `False` | `blind` — priors only | `skill-only` — SKILL.md in prompt, no supporting artifacts |
| `True` | `docs-only` — realistic reader without skill pack | `skill+docs` — realistic reader with skill pack |

`scripts/build_skill_eval_jsonl.py` emits all four cells by default. Use
`--cells=blind,skill+docs` etc. to restrict.

## Input JSONL shape

```json
{
  "responses_create_params": {"input": [{"role": "user", "content": "<task prompt>"}]},
  "verifier_metadata": {
    "skill_path": "/abs/path/to/.claude/skills/gym-review",
    "skill_name": "gym-review",
    "skill_md_sha": "851931cb5698",
    "evals_sha": "e7173def9214",
    "fixtures_sha": "ad3033003411",
    "judge_prompt_sha": "665408560ff4",
    "harness_version": "e28ce8330e99",
    "scenario_id": 1,
    "files": ["evals/scenario_1/broken_agent.py"],
    "cell": "skill+docs",
    "with_skill": true,
    "with_references": true,
    "with_scripts": true,
    "skill_md": "<contents of SKILL.md>",
    "assertions": ["response identifies httpx usage", "response recommends aiohttp"]
  }
}
```

Five content hashes ride through the full pipeline so downstream tooling
can attribute deltas to specific input changes:

- `skill_md_sha` — `sha256(SKILL.md)[:12]`
- `evals_sha` — `sha256(evals/evals.json bytes)[:12]`
- `fixtures_sha` — sha over the scenario's fixtures
- `judge_prompt_sha` — sha of the judge prompt template
- `harness_version` — sha over the three server `app.py` files

All five are populated on every record. `scripts/diff_skill_scoreboards.py`
uses them to render a `same-all` attribution tag in two-file mode so you
know whether a delta movement is attributable to a known input change or
is noise/judge drift.

## Config

See `configs/skill_eval_agent.yaml`. Tunables:

- `max_steps` (default 8) — caps the model↔tool loop length.
- `inject_tools` (default true) — auto-injects `run_bash`/`read_file` schemas if the JSONL carries no tools.
