# BixBench-Hypothesis (BBH)

[BixBench-Hypothesis](https://huggingface.co/datasets/nvidia/Nemotron-RL-bixbench_hypothesis) measures hypothesis testing over bioinformatics data. The remote config connects Gym to an external [Hypotest](https://github.com/EdisonScientific/hypotest) dataset server; the bundled config runs Hypotest and its Enroot sandbox alongside Gym.

The commands below assume that a model endpoint is configured with `policy_base_url`, `policy_model_name`, and `policy_api_key` in `env.yaml`. See the [local configuration documentation](https://docs.nvidia.com/nemo/gym/reference/configuration#local-configuration-envyaml).

## Remote server

Start the Hypotest dataset server, then set `server_url` and `api_key` in `config_remote.yaml`. Task indices in the Gym dataset must refer to the same problems served by Hypotest. The checked-in [example dataset](data/example.jsonl) shows the expected format.

```bash
gym env start --environment aviary_bbh/config_remote --model-type vllm_model
```

Keep that terminal running, then in another terminal run:

```bash
gym eval run --no-serve \
  --agent bbh_aviary_agent \
  --input environments/aviary_bbh/data/example.jsonl \
  --output environments/aviary_bbh/data/example_rollouts_remote.jsonl
```

## Bundled server

Set the dataset paths, rubric model, and `container_sqsh_path` in `config_bundled.yaml`. The bundled environment requires Enroot and a Hypotest container image.

```bash
gym env start --environment aviary_bbh/config_bundled --model-type vllm_model
```

Keep that terminal running, then in another terminal run:

```bash
gym eval run --no-serve \
  --agent bbh_aviary_agent \
  --input environments/aviary_bbh/data/example.jsonl \
  --output environments/aviary_bbh/data/example_rollouts_bundled.jsonl
```

Generate a task-index dataset with `prepare.py --size SIZE --output PATH`.

## Licensing

- Code and Hypotest dependency: Apache 2.0
- BixBench-Hypothesis data: CC BY 4.0
