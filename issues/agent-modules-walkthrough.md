# Agent modules walkthrough (local copy)

Design doc: `issues/agent-modules-first-class-citizens.md` — **Agent modules**, implemented as `AgentModule`.

Fern doc: `fern/versions/latest/pages/evaluation-tutorials/agent-modules-walkthrough.mdx`

## Quick commands

```bash
# 1. Writable skills copy (adaptation mutates SKILL.md on failure)
cp -r benchmarks/skills/variant_a /tmp/agent_modules_skills_variant_a

# 2. Start servers (terminal 1)
source .venv/bin/activate
gym env start \
  --config benchmarks/agent_modules_tutorial/config.yaml \
  --resources-server agent_modules_tutorial_mcqa \
  --model-type openai_model

# 3. Eval (terminal 2)
mkdir -p results
gym eval run --no-serve \
  +agent_name=agent_modules_tutorial_simple_agent \
  +input_jsonl_fpath=benchmarks/agent_modules_tutorial/data/tiny.jsonl \
  +output_jsonl_fpath=results/agent_modules_tutorial_rollouts.jsonl \
  +skills.path=/tmp/agent_modules_skills_variant_a \
  +limit=2 +num_repeats=1

# 4. Inspect
head -1 results/agent_modules_tutorial_rollouts.jsonl | python -m json.tool
```

## What to look for

- `agent_module_refs[].type` — e.g. `working_memory`, `skill_library`
- `agent_module_refs[].hash` — content hash for provenance
- `skills_ref` — rollout collection stamp from `+skills.path`
- `agent_update_events` — skill adaptation after failed rollouts

Config: `benchmarks/agent_modules_tutorial/config.yaml`
