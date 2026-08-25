# GSM8K (Aviary)

This environment runs [GSM8K](https://arxiv.org/abs/2110.14168) through [Aviary](https://github.com/Future-House/aviary). It is a math question-answering environment with a calculator tool.

The commands below assume that a model endpoint is configured with `policy_base_url`, `policy_model_name`, and `policy_api_key` in `env.yaml`. See the [local configuration documentation](https://docs.nvidia.com/nemo/gym/reference/configuration#local-configuration-envyaml).

Generate task-index datasets when needed:

```bash
python environments/aviary_gsm8k/prepare.py --size 1000 --output environments/aviary_gsm8k/data/gsm8k_train.jsonl
python environments/aviary_gsm8k/prepare.py --start 1000 --size 100 --output environments/aviary_gsm8k/data/gsm8k_validation.jsonl
```

Run the checked-in examples and write rollout and metric artifacts:

```bash
gym env start --environment aviary_gsm8k --model-type vllm_model
```

Keep that terminal running, then in another terminal run:

```bash
gym eval run --no-serve \
  --agent gsm8k_aviary_agent \
  --input environments/aviary_gsm8k/data/example.jsonl \
  --output environments/aviary_gsm8k/data/example_rollouts.jsonl
```

## Licensing

- Code: Apache 2.0
- GSM8K data: MIT
