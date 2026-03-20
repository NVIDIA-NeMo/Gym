(run-benchmark-suite)=

# Run a benchmark or benchmark suite

## Prepare benchmark data
1. Request access to various gated HuggingFace datasets

|Benchmark|Gated dataset to request access to|
|---|---|
|GPQA|[Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa)|

2. Set your HuggingFace token in your env.yaml. This is needed to authenticate to HuggingFace and authorize local download of the gated datasets above.

```bash
echo "hf_token: ?\n" >> env.yaml
```

:::{tip}
You can create a HF token following these instructions https://huggingface.co/docs/hub/en/security-tokens
:::

3. Prepare benchmark data using `ng_prepare_benchmark`. In the command below, we prepare the `aime24`, `aime25`, and `gpqa` benchmark datasets.

```bash
config_paths="environments/eval/aime24/config.yaml,\
environments/eval/aime25/config.yaml,\
environments/eval/gpqa/config.yaml"
ng_prepare_benchmark "+config_paths=[$config_paths]"
```

## Configure Weights & Biases benchmark result upload
```bash
echo "wandb_api_key: ?\n" >> env.yaml
```

## Run benchmarks using an OpenAI model

1. Configure the benchmark run. We set the W&B project and experiment name which is used to control where outputs are saved.

```bash
WANDB_PROJECT=bxyu-gym-dev
EXPERIMENT_NAME=benchmark-dev/gpt-5-nano-2025-08-07

config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
environments/eval/aime24/config.yaml,\
environments/eval/aime25/config.yaml,\
environments/eval/gpqa/config.yaml"
```

2. For using `openai_model`, configure your OpenAI API key and other policy model information.

```bash
echo 'openai_api_key: ?
policy_base_url: https://api.openai.com/v1
policy_api_key: ${openai_api_key}' >> env.yaml
```

3. Run the benchmarks using `gpt-5-nano-2025-08-07`
```bash
ng_e2e_collect_rollouts \
    "+config_paths=[${config_paths}]" \
    +wandb_project=$WANDB_PROJECT \
    +wandb_name=$EXPERIMENT_NAME \
    ++output_jsonl_fpath=results/$EXPERIMENT_NAME.jsonl \
    ++resume_from_cache=true \
    ++split=benchmark \
    ++policy_model_name=gpt-5-nano-2025-08-07
```

:::{tip}
You can resume stopped or crashed rollouts using:
```bash
++resume_from_cache=true
```
:::
