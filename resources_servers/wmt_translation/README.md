# wmt_translation

Generic machine-translation verifier for WMT-style benchmarks. Computes
corpus-level BLEU per `(source_language, target_language)` pair via
sacrebleu, with language-specific tokenizers (`13a` default; `ja-mecab`,
`ko-mecab`, `zh` as appropriate). Optionally augments with xCOMET-XXL
neural QE scores, served by a persistent Ray actor pool that loads the
checkpoint once per actor and stays resident for the whole run.

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

> **Note:** `compute_metrics()` emits corpus-level BLEU/COMET keyed by
> language pair, not the `pass@k/{name}` pattern produced by
> `compute_pass_majority_metrics()`. This is intentional — translation
> quality is a corpus-level score, not a per-task correctness probability,
> so the Tier 1 pass@k template in `migrate-benchmark` doesn't apply.
> Parity with NeMo-Skills is on the corpus aggregates.

## Per-sample reward + per-row COMET

`verify()` returns `sentence_bleu(generation, [reference]) / 100` as the
`reward` field. This is a useful dense RL signal but it is NOT the
parity target — corpus-level BLEU in `compute_metrics()` is.

When `compute_comet: true`, `verify()` also dispatches a per-row score
request to the persistent xCOMET-XXL actor pool and awaits the result
before returning. Each rollout in `rollouts.jsonl` therefore carries
its own `comet_score` (or `None` when the model produced an empty
generation), and `compute_metrics()` aggregates those per-row scores
into per-pair / cross-pair means and std-dev keys.

## COMET actor pool

When `compute_comet: true`, `_ensure_comet_actors()` lazily spawns
`comet_num_shards` Ray actors (one per GPU on the extra_gpu node) on the
first `verify()` call. Each actor loads `Unbabel/XCOMET-XXL` once in
`__init__` and serves score requests from the resident model — no
per-call cold-load. `verify()` round-robins requests across the pool
under a small lock and awaits the future inline so per-row scoring is
interleaved with rollout collection.

The checkpoint and its xlm-roberta-xxl tokenizer are resolved via
`comet.download_model()` and `load_from_checkpoint()`, both of which hit
HF_HOME. The benchmark prepare step (`benchmarks/wmt24pp/prepare.py`)
pre-populates the cache so actors initialize fully offline; if the cache
is missing, the first actor falls back to fetching from HF Hub on
startup.

## Example usage

The xCOMET-XXL actor pool requires the `extra_gpu` Ray resource, which
is only advertised on multi-node SLURM deployments. Local / single-node
runs disable COMET via Hydra override and rely on corpus-BLEU only.
For an end-to-end SLURM run with COMET enabled, see the
[`ns nemo_gym_rollouts` block in benchmarks/wmt24pp/README.md](../../benchmarks/wmt24pp/README.md#end-to-end-reproduction-on-a-slurm-cluster-via-nemo-skills).

```bash
# Running servers (BLEU-only locally; flip compute_comet=true on cluster)
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
resources_servers/wmt_translation/configs/wmt_translation.yaml"
ng_run "+config_paths=[$config_paths]" \
    "++wmt_translation.resources_servers.wmt_translation.compute_comet=false"

# Collecting rollouts (5-example smoke test)
ng_collect_rollouts \
    +agent_name=wmt_translation_simple_agent \
    +input_jsonl_fpath=resources_servers/wmt_translation/data/example.jsonl \
    +output_jsonl_fpath=results/wmt_translation_rollouts.jsonl \
    +num_repeats=1
```

For a fully reproducible end-to-end SLURM run that brings up vLLM with
the right Ray topology (model node + a hidden `extra_gpu` node for the
COMET actor pool) and launches Gym in one shot, see the
[`ns nemo_gym_rollouts` block in benchmarks/wmt24pp/README.md](../../benchmarks/wmt24pp/README.md#end-to-end-reproduction-on-a-slurm-cluster-via-nemo-skills).

## Config

| Key                 | Default               | Meaning                                                         |
| ------------------- | --------------------- | --------------------------------------------------------------- |
| `compute_comet`     | `true`                | Toggle xCOMET-XXL scoring                                       |
| `comet_model`       | `Unbabel/XCOMET-XXL`  | HF repo passed to `comet.download_model`                        |
| `comet_batch_size`  | `16`                  | Batch size for `model.predict`                                  |
| `comet_num_shards`  | `8`                   | Number of CometActors in the pool; cap at the extra node's GPU count |
| `strip_reasoning`   | `true`                | Drop a `<think>...</think>` preamble before scoring             |

## Licensing

- Code: Apache 2.0
- `Unbabel/XCOMET-XXL`: check model card (CC-BY-NC 4.0 at time of writing)
- Dependencies: `sacrebleu` (Apache 2.0), `unbabel-comet` (Apache 2.0)
