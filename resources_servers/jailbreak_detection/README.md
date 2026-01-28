# Jailbreak Detection Resources Server

This resources server trains models to resist jailbreak attempts using an LLM-as-judge for verification.

## Overview

- **Policy Model**: Receives adversarial prompts designed to elicit harmful content
- **Judge Model**: Evaluates whether the policy model's response is SAFE (resisted jailbreak) or UNSAFE (was jailbroken)
- **Reward**: 1.0 for safe responses, 0.0 for unsafe responses

## Attack Types Covered

The example dataset includes various jailbreak attack patterns:
- DAN (Do Anything Now) prompts
- Educational/research bypass attempts
- Roleplay/persona attacks
- Grandma exploit
- Security training bypass
- Developer mode claims
- Fiction/creative writing bypass
- Completion attacks
- Code generation requests
- Hypothetical/thought experiment bypass

## Usage

1. Start the servers:
```bash
ng_run "+config_paths=[resources_servers/jailbreak_detection/configs/jailbreak_detection.yaml,responses_api_models/openai_model/configs/openai_model.yaml]"
```

2. Collect rollouts:
```bash
ng_collect_rollouts \
    "+config_paths=[resources_servers/jailbreak_detection/configs/jailbreak_detection.yaml,responses_api_models/openai_model/configs/openai_model.yaml]" \
    +agent_name=jailbreak_detection_simple_agent \
    +input_jsonl_fpath=resources_servers/jailbreak_detection/data/example.jsonl \
    +output_jsonl_fpath=results/jailbreak_detection_rollouts.jsonl
```

## Configuration

Key configuration options in `configs/jailbreak_detection.yaml`:

- `judge_model_server`: The model server to use as the safety judge
- `judge_prompt_template`: Template for the judge evaluation prompt
- `safe_label`: Label the judge outputs for safe responses (default: `[[SAFE]]`)
- `unsafe_label`: Label the judge outputs for unsafe responses (default: `[[UNSAFE]]`)
- `reward_if_safe`: Reward for safe responses (default: 1.0)
- `reward_if_unsafe`: Reward for unsafe responses (default: 0.0)

## Licensing

- Code: Apache 2.0
- Data: Apache 2.0

## Dependencies

- nemo_gym: Apache 2.0


