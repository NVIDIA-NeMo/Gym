# WMT24++ Translation Benchmark

English to {de_DE, es_MX, fr_FR, it_IT, ja_JP} segment-level translation
from [`google/wmt24pp`](https://huggingface.co/datasets/google/wmt24pp).

Verification is deterministic corpus-level BLEU (sacrebleu) per language
pair, with cross-pair aggregations `en->xx`, `xx->xx`, and `xx->{tgt}`.
Optionally augments with xCOMET-XXL neural QE scores when
`compute_comet: true` is set on the wmt_translation server.

See `resources_servers/wmt_translation/README.md` for the verifier
details and the Ray GPU-scheduled COMET path.

## Prepare benchmark data

```bash
ng_prepare_benchmark "+config_paths=[benchmarks/wmt24pp/config.yaml]"
```

## Running servers

```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/wmt24pp/config.yaml"
ng_run "+config_paths=[$config_paths]"
```

## Collecting rollouts

```bash
ng_collect_rollouts \
    +agent_name=wmt24pp_wmt_translation_simple_agent \
    +input_jsonl_fpath=benchmarks/wmt24pp/data/wmt24pp_benchmark.jsonl \
    +output_jsonl_fpath=results/wmt24pp_rollouts.jsonl \
    +num_repeats=4
```
