# BixBench (Aviary)

This environment runs [BixBench](https://arxiv.org/abs/2503.00096) through [Aviary](https://github.com/Future-House/aviary). It is a scientific question-answering environment with a Jupyter notebook. Docker must be available, and the first run downloads and extracts the BixBench capsules.

The commands below assume that a model endpoint is configured with `policy_base_url`, `policy_model_name`, and `policy_api_key` in `env.yaml`. See the [local configuration documentation](https://docs.nvidia.com/nemo/gym/reference/configuration#local-configuration-envyaml).

Run the checked-in examples and write rollout and metric artifacts:

```bash
gym env start --environment aviary_bixbench --model-type vllm_model
```

Keep that terminal running, then in another terminal run:

```bash
gym eval run --no-serve \
  --agent bixbench_aviary_agent \
  --input environments/aviary_bixbench/data/example.jsonl \
  --output environments/aviary_bixbench/data/example_rollouts.jsonl
```

Generate a larger task-index dataset with `prepare.py --size SIZE --output PATH`.

## Licensing

- Code and BixBench data: Apache 2.0
