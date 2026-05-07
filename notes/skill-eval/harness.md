# Skill-eval harness — build and operate

Internal note. Companion to [`results.md`](results.md).

## What it is

Three NeMo Gym servers that grade `.claude/skills/*/SKILL.md` content via paired rollouts, plus the scripts on top.

- `resources_servers/skill_workspace` — per-session sandbox tmpdir. Tools: `run_bash`, `read_file`. Seeds the skill's `scripts/` and `references/` (gated by per-request flags) and scenario fixtures. **`SKILL.md` is never seeded** — see contamination note below.
- `resources_servers/skill_judge` — LLM-as-judge. Returns per-assertion binary grades with evidence. Reward = fraction of assertions satisfied.
- `responses_api_agents/skill_eval_agent` — orchestrator. Seeds the workspace, optionally prepends SKILL.md as a system message, runs the model↔tool loop, forwards the transcript to the judge, closes the workspace.

Scripts:
- `scripts/build_skill_eval_jsonl.py` — emits a 4-cell 2×2 per scenario with five content-hash provenance fields.
- `scripts/diff_skill_scoreboards.py` — auto-detects 2×2 vs legacy and renders multi-axis (Δreward, Δtools, Δtokens) per skill, with `same-all` / `partial(N/5)` / per-field provenance attribution.
- `scripts/build_shape_probe.py` — one-off shape A/B builder.
- `scripts/eval_skills.py` — standalone runner for ad-hoc evaluation.

## The 2×2 cells

Two independent flags control the rollout cell:

|  | `with_references=False` | `with_references=True` |
|---|---|---|
| `with_skill=False` | **blind** — model priors only | **docs-only** — realistic reader without the skill pack |
| `with_skill=True`  | **skill-only** — SKILL.md in prompt, nothing on disk | **skill+docs** — realistic reader with the skill pack |

The diff tool reports four named marginal effects:

- `skill | refs=T` (skill+docs − docs-only) — realistic-deployment value of the skill overlay.
- `skill | refs=F` (skill-only − blind) — skill as a standalone doc.
- `refs | skill=T` (skill+docs − skill-only) — do refs still matter when the skill is prompted?
- `refs | skill=F` (docs-only − blind) — marginal value of references alone.

Default is "all four." Use `--cells=blind,skill+docs` (etc.) to restrict.

## Provenance — five hashes per record

Every JSONL record carries content hashes in `verifier_metadata`:

| field | hashes |
|---|---|
| `skill_md_sha` | `SKILL.md` |
| `evals_sha` | `evals/evals.json` |
| `fixtures_sha` | listed fixture files for this scenario |
| `judge_prompt_sha` | judge prompt template |
| `harness_version` | concatenated bytes of the three server `app.py` files |

Hashes ride through to the output JSONL. The diff tool uses them to render `same-all` / `md` / `md+evals` / `harness` etc. attribution tags so a delta-of-delta can be attributed to a specific input change. Known limit: stamped at JSONL-build time, not rollout time. If server code is edited between build and `ng_collect_rollouts`, the hash lies.

## Bring it up

```bash
# Setup once
uv venv && uv sync --extra dev --group docs

# Configure model endpoint at repo root
cat > env.yaml <<EOF
policy_base_url: <your-/v1-endpoint>
policy_api_key: <your-key>
policy_model_name: <your-model>
EOF

# Bring up the four servers
ng_run "+config_paths=[
  resources_servers/skill_workspace/configs/skill_workspace.yaml,
  resources_servers/skill_judge/configs/skill_judge.yaml,
  responses_api_models/openai_model/configs/openai_model.yaml,
  responses_api_agents/skill_eval_agent/configs/skill_eval_agent.yaml
]" +skip_venv_if_present=true

ng_status   # expect "4 healthy"
```

## Run a scoreboard

```bash
# Build input JSONL (4 cells × 8 skills × 3 scenarios = 96 records)
python scripts/build_skill_eval_jsonl.py \
  --skills-dir .claude/skills \
  --output responses_api_agents/skill_eval_agent/data/example.jsonl

# Collect rollouts at n=5
ng_collect_rollouts \
  +agent_name=skill_eval_agent \
  +input_jsonl_fpath=responses_api_agents/skill_eval_agent/data/example.jsonl \
  +output_jsonl_fpath=results/skill_evals/v8/rollouts.jsonl \
  +num_repeats=5 \
  +num_samples_in_parallel=6 \
  "+responses_create_params={max_output_tokens: 8192}"

# Render multi-axis 2×2 scoreboard
python scripts/diff_skill_scoreboards.py results/skill_evals/v8/rollouts.jsonl

# Diff against a prior run
python scripts/diff_skill_scoreboards.py \
  results/skill_evals/v7/rollouts.jsonl \
  --v2 results/skill_evals/v8/rollouts.jsonl --v1-label v7 --v2-label v8
```

Pace: 96 records × n=5 = 480 rollouts, ~30 min wall-clock at 6-way parallel on the NVIDIA inference API.

## Writing scenarios

Each skill has `.claude/skills/<skill-name>/evals/evals.json` with 3–5 scenarios:

```json
{
  "evals": [
    {
      "id": 1,
      "prompt": "Audit the dataset at evals/files/sample.jsonl. Identify every entry with a schema violation.",
      "files": ["evals/files/sample.jsonl"],
      "assertions": [
        "Line 2's missing required field is flagged",
        "Line 4's parallel_tool_calls / expected_tool_calls mismatch is flagged"
      ],
      "expected_output": null
    }
  ]
}
```

Assertions matter:

- **Specific.** "Line 2 is flagged: missing required field" beats "the agent flags issues."
- **Testable from the transcript.** The judge sees the model's output and tool calls. Don't assert on filesystem state.
- **Backtick the load-bearing nouns.** When the assertion expects a specific field name (`extracted_model_code`, `pass_threshold`), wrap it in backticks. The harness uses backtick-quoted tokens as keyword extraction targets for the diagnostic classifier (planned).
- **Avoid over-literal phrasing.** "Mentions `ng_reward_profile` (or a handoff to the gym-profile skill)" beats "Handoff to gym-profile mentioned" — the second punishes the model for paraphrasing.

`expected_output` is optional — if present the judge sees it; for most skills leave it `null` and let the assertions carry the spec.

## Reading the scoreboard

The diff tool prints, per skill:

```
gym-profile
  cell          reward  tools  tokens  n
  blind          0.65    5.1    1094  15
  docs-only      0.95    5.9    1010  15
  skill-only     0.96    3.1    1116  15
  skill+docs     0.99    3.0    1123  15
  effect           Δreward   Δtools   Δtokens
  skill | refs=T    +0.04    -2.87     +114
  skill | refs=F    +0.31    -2.00      +22
  refs | skill=T    +0.02    -0.13       +7
  refs | skill=F    +0.29    +0.73      -84
```

Reading rules (one cell at a time):

- **The realistic-deployment column is `skill | refs=T`.** "Does the skill overlay help a reader who already has the repo?" Most product decisions live here.
- **Big `skill | refs=F`, small `skill | refs=T`** → skill is mostly a compressed restatement of references. Either shrink the skill or promote references.
- **Big positive on both `skill | refs=T` AND `skill | refs=F`** → skill carries unique value. Keep.
- **Negative `skill | refs=T`, positive `skill | refs=F`** → skill is useful in isolation but competes with its own references. Rewrite SKILL.md to cross-reference the references rather than duplicate them.
- **Everything ~0** → ceiling-clipped or true null. If `docs-only ≥ 0.95`, scenarios are too easy.

Tool-call deltas are a second axis. A skill that doesn't move accuracy but cuts 4 tool calls per rollout is real value at deployment scale (latency + cost). Always read both axes.

## Iteration loop

```
1. Pick the worst (or most interesting) skill.
2. Break it out by scenario to localize which cell moved.
3. Read the partial-credit rollouts' grades[].evidence to find the failure mode.
4. Edit ONE input — SKILL.md, evals.json, a fixture, or harness code.
5. Regenerate input JSONL (every relevant hash updates automatically).
6. Rerun ng_collect_rollouts into a fresh output path.
7. Diff against the prior run with diff_skill_scoreboards.py.
8. If the provenance diff points at the thing you changed AND the delta change is larger than the noise floor, ship it.
```

The "one input at a time" rule is load-bearing. Edit a SKILL.md and an evals.json in the same run and you can't tell which moved the delta. The provenance diff will say `md+evals` and you'll burn a third run to disambiguate.

## Noise floor

Empirical, from v4↔v5 zero-edit reruns:

- ~±0.05 skill-level (averaged across 3 scenarios × n=5)
- ~±0.20 per-cell at n=5

Dominant noise source: judge non-determinism at temperature=0 on the NVIDIA inference API. Provenance hashes can confirm inputs are bit-identical; they say nothing about whether outputs will be. `same-all` is necessary for attribution but not sufficient — it's a cache-hit check, not a p-value.

If you need to resolve smaller effects: bump `num_repeats`. Every doubling of n roughly halves the per-cell stderr.

## The contamination story (don't repeat our mistakes)

The first six iterations measured a confound. Two contamination bugs, fixed in this order:

1. **`SKILL.md` on disk in both arms.** `seed_session` copied `SKILL.md` into the workspace unconditionally. The `with_skill` flag only controlled the system-prompt prepend. So the "without skill" arm still had `SKILL.md` on disk, and the model's first tool call was `ls && cat SKILL.md` — 100% peek rate. Fixed by removing the `SKILL.md` copy in `seed_session`.
2. **`references/` and `scripts/` on disk in both arms.** Same shape, one directory over. We measured 100% of `gym-profile` "without skill" rollouts reading `references/metrics-guide.md`, which contains every noun the assertions test for. Fixed by gating `references/` and `scripts/` seeding on independent `with_references` / `with_scripts` flags.

After both fixes, the harness produces the 4-cell 2×2 with a true blind control (only `.sandbox_bin` and scenario fixtures on disk).

The general lesson: **anything seeded into a workspace is part of the skill payload unless you flag it otherwise.** New skills that ship references/scripts inherit the gating automatically; fixtures still ride along (they're the scenario's input, not the skill's payload).

## Sandbox details

The bash subprocess runs with a stripped env:

- `PATH` = `/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin` (host venv stripped)
- A workspace-local `.sandbox_bin/` with a `python → python3` symlink (otherwise macOS rollouts hit `python: command not found` and thrash)
- `HOME`, `USER`, `LANG`, `LC_ALL`, `SHELL`, `TERM`, `TMPDIR` pass through; everything else is dropped

Output is capped at `output_cap_bytes` (default 50 KB combined). Bash timeout clamped to `[1, bash_timeout_hard_cap_seconds]` (default 120 s). Concurrent bash bounded by `max_concurrent_bash` (default 16).

This is **not a container**. Skill authors are assumed trusted NVIDIA contributors; if that changes, add a container layer.

## Architecture diagram

```
JSONL → skill_eval_agent /run
         ├─ /seed_session → skill_workspace
         ├─ loop:
         │    /v1/responses → policy_model
         │    /run_bash, /read_file → skill_workspace
         ├─ /verify → skill_judge
         │    /v1/responses → policy_model (judge call)
         └─ /close → skill_workspace
        ↓
       output JSONL: reward + per-assertion grades + verifier_metadata
```
