# HotPotQA (Aviary)

This environment runs [HotPotQA](https://aclanthology.org/D18-1259/) through [Aviary](https://github.com/Future-House/aviary). It is a multi-hop question-answering environment with Wikipedia search.

The commands below assume that a model endpoint is configured with `policy_base_url`, `policy_model_name`, and `policy_api_key` in `env.yaml`. See the [local configuration documentation](https://docs.nvidia.com/nemo/gym/reference/configuration#local-configuration-envyaml).

Generate task-index datasets when needed:

```bash
python environments/aviary_hotpotqa/prepare.py --size 1000 --output environments/aviary_hotpotqa/data/hotpotqa_train.jsonl
python environments/aviary_hotpotqa/prepare.py --start 1000 --size 100 --output environments/aviary_hotpotqa/data/hotpotqa_validation.jsonl
```

Run the checked-in examples and write rollout and metric artifacts:

```bash
gym env start --environment aviary_hotpotqa --model-type vllm_model
```

Keep that terminal running, then in another terminal run:

```bash
gym eval run --no-serve \
  --agent hotpotqa_aviary_agent \
  --input environments/aviary_hotpotqa/data/example.jsonl \
  --output environments/aviary_hotpotqa/data/example_rollouts.jsonl
```

## Licensing

- Code: Apache 2.0
- HotPotQA data: Creative Commons Attribution-ShareAlike 4.0 International
