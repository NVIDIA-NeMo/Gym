(run-benchmark-suite)=

# Run a benchmark or benchmark suite

## Prepare benchmark data
1. Request access to various gated HuggingFace datasets

|Benchmark|Gated dataset to request access to|
|---|---|
|GPQA|[Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa)|

2. Set your HF token in your env.yaml. This is needed to authenticate to HuggingFace and authorize local download of the gated datasets above.

```bash
echo "hf_token: ?" > env.yaml
```

:::{tip}
You can create a HF token following these instructions https://huggingface.co/docs/hub/en/security-tokens
:::

3. Prepare benchmark data using `ng_prepare_benchmark`. In the command below

```bash
config_paths="benchmarks/aime24/config.yaml,\
benchmarks/aime25/config.yaml,\
benchmarks/gpqa/config.yaml"
ng_prepare_benchmark "+config_paths=[$config_paths]"
```
