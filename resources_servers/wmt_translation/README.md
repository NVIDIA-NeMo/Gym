# wmt_translation

Generic machine-translation verifier for WMT-style benchmarks. Computes
corpus-level spBLEU and chrF per `(source_language, target_language)` pair via
sacrebleu. spBLEU always uses SacreBLEU's `flores200` SentencePiece tokenizer,
regardless of the target language. Optionally augments them with xCOMET-XXL
neural QE scores, served by a persistent Ray actor pool that loads the
checkpoint once per actor and stays resident for the whole run, plus optional
language-consistency scoring through a configured backend.

> **Warning:** We switched from the default SacreBLEU tokenizer to the
> FLORES-200 SentencePiece tokenizer for computing BLEU. The new BLEU scores
> are reported as **spBLEU**—do **NOT** compare these spBLEU scores to prior
> BLEU scores reported by this benchmark in the past; they are not comparable.
> Also be very careful when comparing them to BLEU scores reported in papers or
> elsewhere: the tokenizer must be the same for the scores to be comparable.
> In general, we recommend using **chrF** instead of BLEU because it performs at
> least as well and requires no tokenization, so there is no chance of tokenizer
> mismatch.

## Metric outputs

Human-facing text and aggregate JSON metric keys use the canonical display
names **spBLEU** and **chrF**. Per-sentence Python and JSON fields remain
snake_case: `sentence_spbleu` and `sentence_chrf`.

Before sentence- or corpus-level chrF and spBLEU are computed, both the model
generation and reference translation are normalized to Unicode NFC. The
original generation text stored in rollout output is not modified.

`compute_metrics()` emits:

- Per-pair: `<src>-><tgt>/spBLEU`, `<src>-><tgt>/spBLEU_std_dev_across_runs`,
  `<src>-><tgt>/chrF`, `<src>-><tgt>/chrF_std_dev_across_runs`,
  `<src>-><tgt>/comet`, `<src>-><tgt>/comet_std_dev_across_runs`,
  `<src>-><tgt>/language_consistency`,
  `<src>-><tgt>/language_consistency_std_dev_across_runs`
- Aggregated: `xx->xx/{spBLEU,chrF}`, `<src>->xx/{spBLEU,chrF}`,
  `xx-><tgt>/{spBLEU,chrF}`
  (and matching `/comet` keys when `compute_comet: true`, and
  `/language_consistency` keys when a language-consistency backend is configured)

`get_key_metrics()` returns the headline aggregates
(`xx->xx/spBLEU`, `xx->xx/chrF`, `xx->xx/comet`,
`xx->xx/language_consistency`, `en->xx/spBLEU`, `en->xx/chrF`,
`en->xx/comet`, `en->xx/language_consistency`).

> **Note:** `compute_metrics()` emits corpus-level spBLEU/chrF/COMET keyed by
> language pair, not the `pass@k/{name}` pattern produced by
> `compute_pass_majority_metrics()`. This is intentional — translation
> quality is a corpus-level score, not a per-task correctness probability,
> so the Tier 1 pass@k template in `migrate-benchmark` doesn't apply.

## Per-sample reward + per-row COMET

`verify()` reports both `sentence_spbleu` and `sentence_chrf`, and returns
`sentence_chrf(generation, [reference]) / 100` as the `reward` field.
`compute_metrics()` reports both metrics at corpus level.

When `compute_comet: true`, `verify()` also dispatches a per-row score
request to the persistent xCOMET-XXL actor pool and awaits the result
before returning. Each rollout in `rollouts.jsonl` therefore carries
its own `comet_score` (or `None` when the model produced an empty
generation), and `compute_metrics()` aggregates those per-row scores
into per-pair / cross-pair means and std-dev keys.

When `language_consistency_backend` is configured, `verify()` also computes a
per-row `language_consistency_score` in-process. WMT24++ selects the
`wmt24pp_cld2` backend, which reports the fraction (0–1) of the generation that
CLD2 attributes to the target language (`0.0` for empty or wrong-language
output). This is a language-level check, not a regional-variety check:
`ar_EG`/`ar_SA`, `fr_CA`/`fr_FR`, `pt_BR`/`pt_PT`, and `sw_KE`/`sw_TZ`
each collapse to one CLD2 language code. Consequently, this metric cannot
detect a translation written in the wrong regional variety within one of
those pairs. FLORES selects `flores_glotlid`, which reports the summed GlotLID
probability for the expected full FLORES target code and any configured
equivalent codes. Both run on CPU without Ray/GPU dependencies, and
`compute_metrics()` aggregates their output into `/language_consistency` keys
alongside spBLEU, chrF, and COMET.

After aggregation, the server emits one warning listing every source-target
pair whose mean language-consistency score is below
`language_consistency_warning_threshold` (default: `50.0` on the 0–100 metric
scale). Each pair is printed with its score, ordered from lowest to highest.

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
runs disable COMET via Hydra override and rely on local spBLEU and chrF.
For an end-to-end SLURM run with COMET enabled, see the
[`ns nemo_gym_rollouts` block in benchmarks/wmt24pp/README.md](../../benchmarks/wmt24pp/README.md#end-to-end-reproduction-on-a-slurm-cluster-via-nemo-skills).

```bash
# Running servers (spBLEU + chrF locally; flip compute_comet=true on cluster)
gym env start \
    --model-type vllm_model \
    --resources-server wmt_translation \
    ++wmt_translation.resources_servers.wmt_translation.compute_comet=false

# Collecting rollouts (5-example smoke test)
gym eval run --no-serve \
    --agent wmt_translation_simple_agent \
    --input resources_servers/wmt_translation/data/example.jsonl \
    --output results/wmt_translation_rollouts.jsonl \
    --num-repeats 1
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
| `language_consistency_warning_threshold` | `50.0` | Warn for source-target pairs below this mean 0–100 language-consistency score |
| `strip_reasoning`   | `true`                | Drop a `<think>...</think>` preamble before scoring             |

## Licensing

- Code: Apache 2.0
- `Unbabel/XCOMET-XXL`: check model card (CC-BY-NC 4.0 at time of writing)
- Dependencies: `sacrebleu` (Apache 2.0), `sentencepiece` (Apache 2.0),
  `unbabel-comet` (Apache 2.0)
