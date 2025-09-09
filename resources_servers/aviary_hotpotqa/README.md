# Aviary HotPotQA Resource Server

```bash
HF_HOME=.cache/ HOME=. vllm serve     Qwen/Qwen3-30B-A3B     --dtype auto     --tensor-parallel-size 4     --gpu-memory-utilization 0.9     --enable-auto-tool-choice --tool-call-parser hermes     --host 0.0.0.0     --port 10240

aviary_config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,resources_servers/aviary_hotpotqa/configs/aviary_hotpotqa.yaml"
ng_run "+config_paths=[$aviary_config_paths]"

ng_collect_rollouts +agent_name=aviary_hotpotqa_simple_agent \
    +input_jsonl_fpath=resources_servers/aviary_hotpotqa/data/hotpotqa_20_examples.jsonl \
    +output_jsonl_fpath=resources_servers/aviary_hotpotqa/data/hotpotqa_20_rollouts.jsonl \
    +limit=5 \
    +num_repeats=null \
    +num_samples_in_parallel=null
```

# Description

Data links: ?

# Licensing information
Code: ?
Data: ?

Dependencies
- nemo_gym: Apache 2.0
?