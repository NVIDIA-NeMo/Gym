# OpenEvolve Contrib Wrapper

This directory contains a small OpenEvolve evaluator that runs candidate prompt
or skill text through NeMo Gym's OpenHands `swe_agents` environment.

The wrapper:

- renders a candidate Markdown file into `system_prompt.j2` and `user_prompt.j2`
- writes a Gym prompt overlay for `CodexAgent`
- starts Gym with `ng_run`
- collects rollouts with `ng_collect_rollouts`
- parses Gym rollout JSONL into OpenEvolve's score dictionary

## Dry Run

Dry-run mode is the default. It renders artifacts and returns the Gym commands
that would run, but it does not start Gym.

```sh
PYTHONPATH=contrib/open_evolve \
uv run python contrib/open_evolve/evaluator.py \
  contrib/open_evolve/seed_skill.md
```

## Live Run

Set `OPENEVOLVE_OPENHANDS_EXECUTE=1` when the Gym checkout, model endpoint, and
input JSONL are ready.

```sh
PYTHONPATH=contrib/open_evolve \
NEMO_GYM_REPO="$PWD" \
OPENEVOLVE_OPENHANDS_EXECUTE=1 \
POLICY_BASE_URL=http://localhost:8001/v1 \
POLICY_API_KEY="$POLICY_API_KEY" \
POLICY_MODEL_NAME="$POLICY_MODEL_NAME" \
uv run python contrib/open_evolve/evaluator.py \
  contrib/open_evolve/seed_skill.md
```

## OpenEvolve

Point OpenEvolve at the candidate file and this evaluator entrypoint:

```sh
PYTHONPATH=contrib/open_evolve \
uv run openevolve-run \
  contrib/open_evolve/seed_skill.md \
  contrib/open_evolve/evaluator.py \
  --iterations 10
```

The default input dataset is
`responses_api_agents/swe_agents/data/example.jsonl`. Override it from Python by
constructing `OpenHandsEvaluationConfig(input_jsonl_fpath=...)`.
