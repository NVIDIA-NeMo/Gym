# Reasoning Gym — Claude Code Agent

Claude Code agent harness for reasoning gym tasks.

Source benchmark: https://github.com/open-thought/reasoning-gym

## Configuration

Set Anthropic credentials in `env.yaml`:

```yaml
anthropic_api_key: sk-ant-...
anthropic_model_name: claude-sonnet-4-6
anthropic_base_url: null
```

For a local vLLM or Ollama endpoint that serves the Anthropic Messages API:

```yaml
anthropic_api_key: EMPTY
anthropic_model_name: Qwen/Qwen3-4B-Instruct-2507
anthropic_base_url: http://localhost:8000
```

`anthropic_base_url` should not include `/v1`. Claude Code appends `/v1/messages` itself.

See [`responses_api_agents/claude_code_agent`](../../responses_api_agents/claude_code_agent/README.md) for the full set of agent options (`thinking`, `max_thinking_tokens`, `allowed_tools`, `disallowed_tools`, `max_turns`, `timeout`, etc.).

## Quick start

```bash
ng_run "+config_paths=[environments/reasoning_gym_claude_code/config.yaml]"
```

```bash
ng_collect_rollouts \
    +agent_name=reasoning_gym_claude_code_agent \
    +input_jsonl_fpath=environments/reasoning_gym_claude_code/data/example.jsonl \
    +output_jsonl_fpath=results/reasoning_gym_claude_code_rollouts.jsonl
```

## Prepare training data

```bash
python environments/reasoning_gym_claude_code/prepare.py --task knights_knaves --size 1000 --output environments/reasoning_gym_claude_code/data/train_knights_knaves.jsonl
```

See `prepare.py` for all available tasks, categories, and config options.

Alternatively, a pre-built dataset is hosted on HuggingFace at [nvidia/Nemotron-RL-ReasoningGym-v1](https://huggingface.co/datasets/nvidia/Nemotron-RL-ReasoningGym-v1).
