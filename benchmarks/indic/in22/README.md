# IN22

This benchmark evaluates bidirectional translation between English and 13
Indic languages on the pinned `ai4bharat/IN22-Gen` and
`ai4bharat/IN22-Conv` test sets. The prompt, response
cleanup, language set, and corpus chrF, chrF++, and BLEU metrics follow the
supplied lm-evaluation-harness reference. Urdu preprocessing explicitly selects
its tokenization-only path; see the preprocessing policy below.

`config.yaml` is the primary IN22-Gen configuration. Use `config_conv.yaml`
for IN22-Conv. Both translate in English-to-Indic and Indic-to-English
directions with thinking disabled, a 4,096-token output limit, temperature 1,
top-p 0.95, no top-k truncation, and seed 42. These sampling settings deliberately
differ from the reference runner's greedy decoding and 50,000-token limit.
Defaults are stored on each IN22 row and survive `--model-type vllm_model`.
Override them with `--temperature`, `--top-p`, and `--max-output-tokens`, or
request metadata for `top_k`, `seed`, and `chat_template_kwargs`. Thinking is
requested off via `enable_thinking: false`; support depends on the served model's
chat template. Keep experiment configurations under the Git-ignored `models/` directory.

Prepare each configuration with:

```bash
gym eval prepare --config benchmarks/indic/in22/config.yaml

gym eval prepare --config benchmarks/indic/in22/config_conv.yaml
```

The datasets are gated on Hugging Face. Accept their access terms and log in
before preparation. Prepared JSONL files are ignored by Git.

Run an evaluation by adding the vLLM endpoint and model arguments used by your
deployment:

```bash
gym eval run --config benchmarks/indic/in22/config.yaml \
  --agent in22_gen_agent \
  --model-type vllm_model \
  --model YOUR_SERVED_MODEL \
  --model-url http://HOST:PORT/v1 \
  --model-api-key dummy \
  --split benchmark \
  --output results/in22_gen/rollouts.jsonl
```

For Conv, use `config_conv.yaml` and `--agent in22_conv_agent`. The configs have
separate agents and preparation entry points, so both can be selected together
without one subset replacing the other. They share the IN22 verifier instance.

For repeated sampling with distinct seeds, add both options to `gym eval run`:

```bash
  --num-repeats 3 +num_repeats_add_seed=true
```

This replaces each row's seed with the zero-based repeat index: `0`, `1`, `2`
for three repeats, independently for each task. It does not offset seed 42.
`--num-repeats 3` alone sends seed 42 for every repeat. Keep the seed policy
and sampling settings identical across models when comparing results.

Preparation pins Gen to `e042ab3d3063110b1a85efa0a59bdbf8553bb928` (1,024 rows)
and Conv to `18cd45870ff0a9e65df9b80dbbcc615eec0e4899` (1,503 rows). It validates
the source schema and nonempty translations before writing 26,624 Gen or 39,078
Conv tasks across 13 languages and both directions. Stable task IDs preserve
repeated source sentences in Conv. Re-prepare existing data after upgrading to
pick up the row defaults and IDs.

IN22 code is Apache 2.0. The two benchmark datasets are CC-BY-4.0.

The headline metrics are corpus chrF, chrF++, and BLEU on a 0–100 scale,
computed per translation direction and macro-averaged across directions.
`mean/reward` is mean sentence chrF divided by 100 for Gym rollout diagnostics;
it is not the benchmark corpus score. `en->xx/chrf++` and `xx->en/chrf++` are
the EN→IN and IN→EN macro-averages. Report Gen and Conv separately. The default
WMT profile still uses its existing spBLEU/chrF scoring; IN22 opts into its own
normalization, cleanup, and BLEU tokenization through `metric_profile: in22`.

## Reference and preprocessing

The supplied reference is `custom_configs/post_train_think/in22/utils.py` in
the local lm-evaluation-harness checkout, with SHA-256
`da5ee94c4e74d82dc6d0a689802932f5a6cae0ecd8df0a9fb8ba5b522fc56481`.
Those custom tasks are untracked in that checkout, so there is no upstream
commit permalink for the harness wrapper. The pinned public implementations
used to check its preprocessing and scoring are:

- [IndicNLP 0.92 normalization](https://github.com/anoopkunchukuttan/indic_nlp_library/blob/e478b0e4ca87ed3aac953c226541137d588cb1cd/indicnlp/normalize/indic_normalize.py)
  and [tokenization](https://github.com/anoopkunchukuttan/indic_nlp_library/blob/e478b0e4ca87ed3aac953c226541137d588cb1cd/indicnlp/tokenize/indic_tokenize.py).
- [SacreBLEU 2.6.0 chrF/chrF++](https://github.com/mjpost/sacrebleu/blob/2277caccfc7b956671a6a09f1646f62250034157/sacrebleu/metrics/chrf.py)
  and [BLEU](https://github.com/mjpost/sacrebleu/blob/2277caccfc7b956671a6a09f1646f62250034157/sacrebleu/metrics/bleu.py).

Both reference and hypothesis are stripped. English receives no Indic
preprocessing and uses SacreBLEU's `13a` BLEU tokenizer. Other targets use
IndicNLP 0.92 normalization with `remove_nuktas=False`, then trivial tokenization,
with one explicit exception: **Urdu always uses trivial tokenization only**.
UrduHack is never loaded, even when installed. For example, `یہ ایک مکمل جملہ ہے۔`
and `یہ  ایک مکمل جملہ ہے ۔` tokenize identically; Urdu character variants are
not normalized. Missing required IndicNLP dependencies raise an error rather
than changing the scoring pipeline. Indic targets use BLEU `tokenize="none"`
after this preprocessing. chrF uses character order 6, word order 0, and beta 2;
chrF++ adds word order 2. Scores are computed per language pair before averaging.

This policy preserves the supplied reference's behavior without optional
UrduHack. It can change EN→Urdu scores relative to older installations that
did load UrduHack; rerun those scores before comparing them with this version.

## Smoke validation

On 2026-10-14, batch job `20200242` completed a native `gym eval run` using
both configs and `--model-type vllm_model` on one H100. The model was
`gnani/gnani-evon-v3.3-30B-A3B` at revision
`fb9ddb6b58f2d235684388a9260671fca5ebccaf`, served with vLLM 0.21.0 and the
checkpoint's own chat template. The inputs were source indices `0` and `1`
for Hindi and Urdu, both translation directions, from each pinned dataset.

All **16/16 generated rollouts** completed with `finish_reason=stop`, nonempty
translations, and no structured reasoning or inline `<think>` tags. All 16
captured requests used `enable_thinking=false`, temperature 1, top-p 0.95,
top-k -1, seed 42, and a 4,096-token output limit. Input/reference routing,
sentence rewards, and separate Gen/Conv corpus aggregation were checked.
The 24 corpus scores (eight language/subset pairs × three metrics) matched
the supplied reference scorer's no-UrduHack path within `1e-8`.

| Subset | Completed | EN→HI chrF++ | EN→UR chrF++ | HI→EN chrF++ | UR→EN chrF++ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gen | 8/8 | 48.4564 | 59.3787 | 55.1939 | 65.8776 |
| Conv | 8/8 | 73.9737 | 50.0788 | 50.6030 | 59.2609 |

These two-example-per-pair scores validate integration; they are not a full
benchmark quality baseline. For example, the Conv Hindi→English source
`माँ, चलो कल एक फिल्म देखने चलते हैं।` generated
`Let's go watch a movie tomorrow, Mom.` against the reference
`Mom, let's go for a movie tomorrow.` (sentence chrF++ 57.7007).
Generated datasets, transport logs, and rollouts are retained outside Git.
