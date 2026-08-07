# TALES environments

Integrates [TALES](https://github.com/microsoft/tale-suite)

Specifically, this pins the `train_test_splits` branch across five text-adventure frameworks:
`textworld`, `textworld_express`, `alfworld`, `scienceworld`, and `jericho`. These frameworks use
different upstream split mechanisms; see [Evaluation quality status](#evaluation-quality-status)
before treating a row's `split` value as held-out evidence.

## Install

`tale-suite` is pinned in `requirements.txt` (installed automatically with the server's
venv). `textworld_express` and `scienceworld` need a JRE/JDK (they launch a Java gateway
via py4j); `textworld`, `alfworld`, and `jericho` run without it. The Java binary must be on
the server process's `PATH`.

```bash
# Linux
sudo apt-get update && sudo apt-get install -y default-jre default-jdk
# macOS
brew install openjdk
export JAVA_HOME="$(brew --prefix)/opt/openjdk/libexec/openjdk.jdk/Contents/Home"
export PATH="$(brew --prefix)/opt/openjdk/bin:$PATH"
```

## Quickstart

```bash
# Set inference endpoint in env.yaml as in other Gym environments, then

# Start environment 
ng_run "+config_paths=[resources_servers/tales/configs/tales.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"

# Collect example rollouts
ng_collect_rollouts \
  +agent_name=tales_gymnasium_agent \
  +input_jsonl_fpath=resources_servers/tales/data/example.jsonl \
  +output_jsonl_fpath=results/tales_rollouts.jsonl \
  +num_repeats=1
```

## Per-task selection

Each dataset row selects a task via top-level keys (they arrive as `metadata` in
`reset()`); anything omitted falls back to the server config in `configs/tales.yaml`:

| field | meaning |
|---|---|
| `framework` | one of `textworld`, `textworld_express`, `alfworld`, `scienceworld`, `jericho` |
| `task_no` | index into the framework's task list |
| `split` | requested `train` or `test` selection; support is framework-specific (see below) |
| `seed` | environment seed |
| `max_episode_steps` | turns before the episode is truncated |

Example row (`data/example.jsonl`):

```json
{"framework": "alfworld", "task_no": 0, "split": "train", "seed": 1234,
 "responses_create_params": {"input": [{"role": "system", "content": "You are playing a text-based game..."}]},
 "agent_ref": {"type": "responses_api_agents", "name": "tales_gymnasium_agent"}}
```

## Reward & walkthroughs

Reward comes from the underlying Gymnasium env. For `textworld` the env reports a
cumulative score, so per-step reward is the score delta, while the other frameworks should already
report per-step reward. Ground-truth walkthroughs exist but are not unique, some envs use
nearest-neighbour parsers (eg `take lantern` / `get lantern` / `pick up lantern` are
equivalent), so acceptance is determined by stepping through the env, not by string-matching a
walkthrough.

Set `expose_admissible_commands: true` in the config to surface each env's
`admissible_commands` in the step/reset `info`.

## Evaluation quality status

The checked-in five-row `data/example.jsonl` is a smoke test, not a representative evaluation. It covers all five frameworks, but every row uses `task_no: 0`, `split: train`, and one seed/repeat.

Do not compare models using aggregate raw reward across frameworks: TALES frameworks use different score ranges, and partial positive reward does not necessarily mean the episode succeeded. Report success/win rate first, then normalized score, truncation rate, invalid-action rate, steps, and tokens, with macro results by framework and task family.

Before publishing a held-out eval, the adapter's split behavior needs framework-specific validation. The current implementation falls back to `environments` when a framework has no `train_environments` export, and forwards `split` to `gym.make` only for ScienceWorld. A row labeled `test` therefore does not prove held-out selection for every framework.

`eval/eval_readiness.yaml` records the proposed Tales eval-v0 bar and intentionally points at the current smoke set for gap analysis:

```bash
gym dataset validate-eval --manifest resources_servers/tales/eval/eval_readiness.yaml
```

The command is expected to fail until Tales has a real held-out eval set. The initial target is 40 tasks across all frameworks, at least three repeated rollouts per task, stable upstream task names, task-family/difficulty metadata, parser and long-horizon edge cases, oracle replay, and manual review of reward/success disagreements and parser loops.
