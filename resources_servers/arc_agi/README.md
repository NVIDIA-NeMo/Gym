# ARC-AGI resources server

launch local vllm server
```bash
vllm serve Qwen/Qwen3-30B-A3B \
    --dtype auto \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.9 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --host 0.0.0.0 \
    --port 10240
```

Start ARC-AGI environment:
```bash
ng_run "+config_paths=[resources_servers/arc_agi/configs/arc_agi.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
```

or ARC-AGI-2 environment:
```bash
ng_run "+config_paths=[resources_servers/arc_agi/configs/arc_agi_2.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
```


collect rollouts:

ARC-AGI-1 example rollouts
```bash
ng_collect_rollouts +agent_name=arc_agi_simple_agent +input_jsonl_fpath=resources_servers/arc_agi/data/example_1.jsonl +output_jsonl_fpath=resources_servers/arc_agi/data/example_1_rollouts.jsonl +limit=5 +num_repeats=null +num_samples_in_parallel=null
```

ARC-AGI-2 example rollouts:
```bash
ng_collect_rollouts +agent_name=arc_agi_2_simple_agent +input_jsonl_fpath=resources_servers/arc_agi/data/example_2.jsonl +output_jsonl_fpath=resources_servers/arc_agi/data/example_2_rollouts.jsonl +limit=5 +num_repeats=null +num_samples_in_parallel=null
```

ARC-AGI-1 train set rollouts (400 problems):
```bash
ng_collect_rollouts +agent_name=arc_agi_simple_agent +input_jsonl_fpath=resources_servers/arc_agi/data/arc_agi_1_training.jsonl +output_jsonl_fpath=resources_servers/arc_agi/data/arc_agi_1_training_rollouts.jsonl +limit=null +num_repeats=null +num_samples_in_parallel=null
```

ARC-AGI-1 eval set rollouts (400 problems):
```bash
ng_collect_rollouts +agent_name=arc_agi_simple_agent +input_jsonl_fpath=resources_servers/arc_agi/data/arc_agi_1_evaluation.jsonl +output_jsonl_fpath=resources_servers/arc_agi/data/arc_agi_1_evaluation_rollouts.jsonl +limit=null +num_repeats=null +num_samples_in_parallel=null
```

ARC-AGI-2 train set rollouts (1000 problems):
```bash
ng_collect_rollouts +agent_name=arc_agi_2_simple_agent +input_jsonl_fpath=resources_servers/arc_agi/data/arc_agi_2_training.jsonl +output_jsonl_fpath=resources_servers/arc_agi/data/arc_agi_2_training_rollouts.jsonl +limit=null +num_repeats=null +num_samples_in_parallel=null
```

ARC-AGI-2 eval set rollouts (120 problems):
```bash
ng_collect_rollouts +agent_name=arc_agi_2_simple_agent +input_jsonl_fpath=resources_servers/arc_agi/data/arc_agi_2_evaluation.jsonl +output_jsonl_fpath=resources_servers/arc_agi/data/arc_agi_2_evaluation_rollouts.jsonl +limit=null +num_repeats=null +num_samples_in_parallel=null
```

run tests:
```bash
ng_test +entrypoint=resources_servers/arc_agi
```
