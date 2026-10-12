# MolDeTox (gym-native)

[MolDeTox](https://huggingface.co/datasets/MolDeTox/MolDeTox) is a
**toxicity-aware molecular editing** benchmark built from toxicity cliffs: pairs
of structurally near-identical molecules with opposite measured toxicity. Given
a toxic molecule and its endpoint context, the model must produce the paired
non-toxic molecule.

Paper: [arXiv:2605.12181](https://arxiv.org/abs/2605.12181).

This entry runs MolDeTox through the **gym-native** eval path. Scoring,
aggregation and the prompts all live in the `moldetox` resources server
(`resources_servers/moldetox/`); this benchmark only supplies data and wiring,
chaining to `resources_servers/moldetox/configs/moldetox.yaml`.

## Which config this registers

MolDeTox ships **eight** QA configs and publishes a separate figure for each, so
no single one is "the" benchmark. This entry registers
**`task3_smiles_gen_single`**, 943 test rows: whole-molecule detoxification
written as SMILES, single-fragment edit. It is the configuration upstream
headlines, and the one whose measured accuracy agrees with the published figure.

The other seven are prepared directly through the resources server's script and
scored by the same verifier — see
[the server README](../../resources_servers/moldetox/README.md).

## Metrics

- **Accuracy** — exact match after canonical-SMILES normalisation, which is the
  reward and the only headline-safe number here.
- `validity`, `tanimoto_rdk`, `tanimoto_maccs`, `tanimoto_morgan` and
  `levenshtein` ride along for diagnosis and are deliberately **not** promoted
  to `key_metrics`.

That exclusion is load-bearing. Cliff pairs are similar by construction, so
echoing the toxic input back unchanged scores 0.0 accuracy at mean Morgan
Tanimoto **0.711**. A do-nothing baseline looks strong on every similarity
metric.

## Prepare data

```bash
gym eval prepare --benchmark moldetox
```

Fetches the pinned dataset revision and writes all 943 rows to
`data/moldetox_benchmark.jsonl`. Each row carries upstream's system prompt and
question as its two input turns, both verbatim.

## Running servers

Scoring is deterministic — RDKit canonical SMILES, no judge model and no
external service in the scoring path — so only the policy model is needed.

```bash
gym env start \
    --model-type openai_model \
    --benchmark moldetox \
    --model "<model id>" \
    --model-url "<openai-compatible base url>" \
    --model-api-key "$API_KEY"
```

## Collecting rollouts and scoring

```bash
gym eval run --no-serve \
    --agent moldetox_benchmark_simple_agent \
    --input benchmarks/moldetox/data/moldetox_benchmark.jsonl \
    --output results/moldetox_rollouts.jsonl \
    --num-repeats 1
```

Upstream evaluates at **temperature 0.7 over three runs**, reporting mean and
standard deviation. A run that leaves the sampler unset is not comparable to the
published figure regardless of how faithful the prompts are.
