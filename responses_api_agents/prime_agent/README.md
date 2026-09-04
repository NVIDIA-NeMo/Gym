# Prime Agent

Runs [Prime Agent](https://github.com/PrimeIntellect-ai/prime-agent) in headless JSON mode.

Prime Agent must be on `PATH`. The adapter installs `prime_agent_version` with the official
installer when the default command is missing.

Set `policy_base_url`, `policy_api_key`, and `policy_model_name` in `env.yaml`.

```bash
gym env start --environment prime_agent_math --model-type openai_model

gym eval run --no-serve --agent prime_agent_math_agent \
  --input environments/prime_agent_math/data/example.jsonl \
  --output environments/prime_agent_math/data/example_rollouts.jsonl \
  --limit 5
```

Math and reasoning gym example environments are in `environments/prime_agent_math` and
`environments/prime_agent_reasoning_gym`.

Each request uses an isolated `HOME` and one-shot Prime Agent worker. `kernel_venv` is shared so
IPython dependencies are reused.

`model` uses `<provider>/<model-id>`. Define the provider in `models_config` or set `model_server`
to use a Gym model server. See `configs/prime_agent.yaml` for all options.
