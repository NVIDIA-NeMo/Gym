# Ether0 eval environment

[Benchmark](https://huggingface.co/datasets/futurehouse/ether0-benchmark) and [paper](https://arxiv.org/pdf/2506.17238)

325 questions across 14 task types: property-regression-adme, property-regression-ld50, property-regression-pka, property-cat-safety, property-cat-eve, property-cat-smell, molecule-formula, molecule-name, molecule-completion, reaction-prediction, simple-formula, functional-group, oracle-solubility, and retro-synthesis. Retro-synthesis and oracle-solubility require a separate `ether0-serve` remotes server, see the [ether0 repo](https://github.com/Future-House/ether0/).

```bash
vllm serve futurehouse/ether0
```

Create env.yaml:
```
policy_base_url: http://localhost:8000/v1
policy_api_key: EMPTY
policy_model_name: futurehouse/ether0
```

```bash
ng_run "+config_paths=[resources_servers/ether0/configs/ether0.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
ng_collect_rollouts \
    +agent_name=ether0_simple_agent \
    +input_jsonl_fpath=resources_servers/ether0/data/example.jsonl \
    +output_jsonl_fpath=resources_servers/ether0/data/ether0_rollouts.jsonl
```
