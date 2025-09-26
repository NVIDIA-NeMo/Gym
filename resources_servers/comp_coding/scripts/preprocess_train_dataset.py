"""
Run as:
```bash
HF_HOME=.cache \
HF_TOKEN={your HF token} \
python resources_servers/comp_coding/scripts/preprocess_train_dataset.py
```

Upload:
```bash
ng_upload_dataset_to_gitlab \
    +dataset_name=opencodereasoning_filtered \
    +version=0.0.1 \
    +input_jsonl_fpath=resources_servers/comp_coding/data/opencodereasoning_filtered_25k_train.jsonl
```

Rollout collection. We match the LCB setting for reward profiling. For gpt-4o-2024-05-13 this should be around 33%.
```bash
ng_collect_rollouts +agent_name=comp_coding_simple_agent \
    +input_jsonl_fpath=resources_servers/comp_coding/data/opencodereasoning_filtered_25k_train.jsonl \
    +output_jsonl_fpath=resources_servers/comp_coding/data/opencodereasoning_filtered_25k_train_1k_gpt-4o-2024-05-13_rollouts.jsonl \
    +responses_create_params.temperature=0.2 \
    +responses_create_params.max_output_tokens=2000 \
    +responses_create_params.top_p=0.95 \
    +limit=1000
```
"""

import json

from datasets import load_dataset


ds = load_dataset("Nexusflow/comp_prog_filtered_no_function", split="train")

# Largely taken from https://github.com/NVIDIA/NeMo-Skills/blob/0af0b169ba62be9097f6362c4fb29202849ae036/nemo_skills/prompt/config/eval/livecodebench/python_codegen_reasoning.yaml
prompt_template = """You are a helpful and harmless assistant. You should think step-by-step before responding to the instruction below.

Please use python programming language only.

You must use ```python for just the final solution code block with the following format:
```python
# Your code here
```

{question}"""


with open("resources_servers/comp_coding/data/opencodereasoning_filtered_25k_train.jsonl", "w") as f:
    for d in ds:
        row = {
            "responses_create_params": {
                "input": [
                    {
                        "role": "user",
                        "content": prompt_template.format(question=d["question"]),
                    },
                ],
            },
            "verifier_metadata": {"unit_tests": json.loads(d["unit_tests"])},
            # Carry over original columns, even though they are unused for Gym
            "hash_id": d["hash_id"],
            "dataset": d["dataset"],
            "source": d["source"],
        }
        f.write(json.dumps(row) + "\n")
