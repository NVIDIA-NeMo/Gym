# TALES environments

Integrates [TALES](https://github.com/microsoft/tale-suite)

Specifically, this uses the `train_test_splits` branch, which provides both a set of tasks for
training and evaluation across five text-adventure frameworks: `textworld`,
`textworld_express`, `alfworld`, `scienceworld`, and `jericho`.

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
| `split` | `train` or `test` |
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

