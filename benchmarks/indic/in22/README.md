# IN22

This benchmark evaluates bidirectional translation between English and 13
Indic languages on the pinned `ai4bharat/IN22-Gen` and
`ai4bharat/IN22-Conv` test sets. The prompt, response
cleanup, language set, IndicNLP preprocessing, and corpus chrF, chrF++, and
BLEU metrics match the supplied lm-evaluation-harness reference.

`config.yaml` is the primary IN22-Gen configuration. Use `config_conv.yaml`
for IN22-Conv. Both translate in English-to-Indic and Indic-to-English
directions with thinking disabled, a 4,096-token output limit, temperature 1,
top-p 0.95, no top-k truncation, and seed 42. These sampling settings deliberately
differ from the reference runner's greedy decoding and 50,000-token limit.
Model-specific sampling configurations can be supplied as additional local
config files. Keep experiment configurations under the Git-ignored `models/`
directory.

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
  --model YOUR_SERVED_MODEL \
  --model-url http://HOST:PORT/v1 \
  --model-api-key dummy \
  --split benchmark \
  --output results/in22_gen/rollouts.jsonl
```

The benchmark already includes the vLLM adapter. Do not append
`--model-type vllm_model`: Gym loads that adapter's default config after the
benchmark and clears its thinking and sampling overrides.

IN22 code is Apache 2.0. The two benchmark datasets are CC-BY-4.0.

The headline metrics are corpus chrF, chrF++, and BLEU on a 0–100 scale,
computed per translation direction and macro-averaged across directions.
`mean/reward` is mean sentence chrF divided by 100 for Gym rollout diagnostics;
it is not the benchmark corpus score. Gen and Conv share their inference
and scoring defaults through `config_base.yaml`.
