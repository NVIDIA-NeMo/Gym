# tau2-synth

Prime Intellect Environments Hub `tau2-synth` environment, run through
`verifiers_agent`. Tau2-bench style multi-turn tool use with an LLM user
simulator.

## Run

```bash
gym env start --environment tau2_synth --model-type vllm_model
gym eval run --agent tau2_synth \
    --input environments/tau2_synth/data/example.jsonl \
    --output results/tau2_synth.jsonl \
    --limit 5
```

No prepare step. Tasks ship inside the installed package.

## User simulator

A second model plays the user, separate from the policy model.

That call goes through litellm, which needs an explicit provider prefix.
`user_model: gpt-4.1` fails with `LLM Provider NOT provided` even though the
name is valid. Use `openai/gpt-4.1`, and point `user_base_url` at your gateway.

`user_api_key_var` is the name of an environment variable, not the key. That
variable must be set in the server process.

```bash
gym env start --environment tau2_synth --model-type vllm_model \
    ++tau2_synth.responses_api_agents.verifiers_agent.vf_env_args.user_model=openai/MODEL \
    ++tau2_synth.responses_api_agents.verifiers_agent.vf_env_args.user_base_url=URL
```

## Packaging

Published under the `prime` hub namespace, not `primeintellect`. It also
declares `tau2` as a URL dependency, which uv needs stated directly. Both are
handled in `responses_api_agents/verifiers_agent/requirements.txt`.

## Metrics

`evaluate_tau2_task`, `num_turns`, `num_steps`, `num_errors`,
`num_assistant_tool_calls`, `num_user_tool_calls`.
