# Description
> Keywords: Instruction Following, Structured Outputs, StructEval, Code Generation

This is a resources server for evaluating structured output generation using the [StructEval](https://github.com/TIGER-AI-Lab/StructEval) evaluation framework.

Currently supports **non-renderable** output formats: JSON, YAML, CSV, TOML, XML.

## Scoring

For non-renderable outputs, the reward is:

```
reward = 0.2 * render_score + 0.8 * key_validation_score
```

- **render_score** (0 or 1): Whether the generated code can be extracted and parsed as valid syntax
- **key_validation_score** (0-1): Fraction of expected structural paths found in the parsed output

## Metrics Breakdown

`compute_metrics()` produces breakdowns matching StructEval's `summarize_results.py`:
- By `output_type`: JSON, YAML, CSV, TOML, XML
- By `input_type`: Text, CSV, JSON, XML, YAML, TOML
- By `task_name`: e.g. "Text to JSON", "CSV to YAML"
- Per-bucket: reward, render_score, key_validation_score

## Example Usage

### Running servers
```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
resources_servers/structeval/configs/structeval_nonrenderable.yaml"
ng_run "+config_paths=[${config_paths}]"
```

### Collecting rollouts
```bash
ng_collect_rollouts \
    +agent_name=structeval_nonrenderable_simple_agent \
    +input_jsonl_fpath=resources_servers/structeval/data/structeval_nonrenderable_train.jsonl \
    +output_jsonl_fpath=results/structeval_nonrenderable.jsonl \
    +resume_from_cache=True \
    +num_samples_in_parallel=256
```

## Data Preparation

Convert StructEval dataset to Gym format (prerequisite: original structeval repository):
```bash
python resources_servers/structeval/misc/prepare_data.py \
    --input /path/to/StructEval/dataset/nonrenderable.json \
    --output resources_servers/structeval/data/structeval_nonrenderable_train.jsonl \
    --example-output resources_servers/structeval/data/structeval_nonrenderable_example.jsonl \
    --example-count 5
```

## Testing
```bash
ng_test +entrypoint=resources_servers/structeval
```

## Licensing

Server code: Apache 2.0 (NVIDIA)

Data: Apache 2.0 (derived from [TIGER-Lab/StructEval](https://huggingface.co/datasets/TIGER-Lab/StructEval), originally MIT License)

Evaluation logic adapted from [StructEval](https://github.com/TIGER-Lab/StructEval) (Apache 2.0, TIGER-Lab)

Dependencies:
- nemo_gym: Apache 2.0
- xmltodict: [MIT](https://github.com/martinblech/xmltodict/blob/master/LICENSE)
