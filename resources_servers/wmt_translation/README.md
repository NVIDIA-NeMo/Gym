# wmt_translation

Generic machine-translation verifier for WMT-style benchmarks. Computes
corpus-level BLEU per `(source_language, target_language)` pair via
sacrebleu, with language-specific tokenizers (`13a` default; `ja-mecab`,
`ko-mecab`, `zh` as appropriate). Optionally augments with xCOMET-XXL
neural QE scores, scheduled onto a Ray GPU actor so the heavy checkpoint
is loaded exactly once per aggregation.

Ported from NeMo-Skills' `TranslationMetrics`
(`nemo_skills/evaluation/metrics/translation_metrics.py`) plus the
xCOMET-XXL judge script at `nemo_skills/evaluation/evaluator/comet.py`.

## Metric outputs

`compute_metrics()` emits Skills-equivalent keys:

- Per-pair: `<src>-><tgt>/bleu`, `<src>-><tgt>/bleu_std_dev_across_runs`,
  `<src>-><tgt>/comet`, `<src>-><tgt>/comet_std_dev_across_runs`
- Aggregated: `xx->xx/bleu`, `<src>->xx/bleu`, `xx-><tgt>/bleu`
  (and matching `/comet` keys when `compute_comet: true`)

`get_key_metrics()` returns the headline aggregates
(`xx->xx/bleu`, `xx->xx/comet`, `en->xx/bleu`, `en->xx/comet`).

## Per-sample reward

`verify()` returns `sentence_bleu(generation, [reference]) / 100` as the
`reward` field. This is a useful dense RL signal but it is NOT the
parity target — corpus-level BLEU in `compute_metrics()` is.

## COMET via Ray GPU scheduling

When `compute_comet: true`, `compute_metrics()` dispatches a single
`@ray.remote(num_gpus=<comet_num_gpus>)` task that loads
`Unbabel/XCOMET-XXL` and batch-predicts scores for every (src, mt, ref)
triple across all tasks and rollouts. The checkpoint is resolved via
`comet.download_model()` (cached under HF_HOME). If Ray dispatch fails
for any reason, BLEU metrics still emit and only `/comet` keys are
skipped.

## Example usage

```bash
# Running servers
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
resources_servers/wmt_translation/configs/wmt_translation.yaml"
ng_run "+config_paths=[$config_paths]"

# Collecting rollouts (5-example smoke test)
ng_collect_rollouts \
    +agent_name=wmt_translation_simple_agent \
    +input_jsonl_fpath=resources_servers/wmt_translation/data/example.jsonl \
    +output_jsonl_fpath=results/wmt_translation_rollouts.jsonl \
    +num_repeats=1
```

## Config

| Key                 | Default               | Meaning                                               |
| ------------------- | --------------------- | ----------------------------------------------------- |
| `compute_comet`     | `true`                | Toggle xCOMET-XXL scoring                             |
| `comet_model`       | `Unbabel/XCOMET-XXL`  | HF repo passed to `comet.download_model`              |
| `comet_batch_size`  | `16`                  | Batch size for `model.predict`                        |
| `comet_num_gpus`    | `1`                   | GPU allocation for the Ray COMET actor                |

## Licensing

- Code: Apache 2.0
- `Unbabel/XCOMET-XXL`: check model card (CC-BY-NC 4.0 at time of writing)
- Dependencies: `sacrebleu` (Apache 2.0), `unbabel-comet` (Apache 2.0)
