# RULER v2 (`ruler2`) resources server

Verifier for the RULER v2 long-context benchmark suite. Ports the
verification logic from
[`nemo_skills.evaluation.evaluator.ruler.eval_ruler2`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/evaluation/evaluator/ruler.py)
and the multichoice extractor from
[`nemo_skills.evaluation.evaluator.mcq.eval_mcq`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/evaluation/evaluator/mcq.py).

## Routing

A single server handles two evaluator types, dispatched by the per-row
`eval_type` field on the verify request:

- `eval_type=ruler2`: continuous reward in `[0, 1]` from
  `max(substring_match, 1 - WER)`, aggregated across the reference list
  according to the per-row `match_type`:
  - `match_type=all` — average soft-match across all references.
  - `match_type=part` — max soft-match across references after stripping
    `Document N:` document-prefix headers.
  - `match_type=2steps` — same aggregation as `all`, but only the last
    paragraph (`preds.split("\n\n")[-1]`) is matched.
- `eval_type=multichoice`: exact-match against a single uppercase letter
  extracted from `\boxed{}` (with relaxed regex fallback). Reward is 1.0
  or 0.0.

Both routes also normalize the prediction by replacing ASCII control
characters (`[\x00-\x1f]`) with newlines and stripping. This mirrors
`eval_ruler2.default_parse`.

## Why a separate server (not an extension of `ruler`)

The existing `ruler` server implements RULER **v1**, whose scoring is
substring-only (no WER fallback), supports only two match types
(`all`, `part`), and has no document-header stripping or per-row
multichoice route. RULER v2's verification is incompatible at every one
of those points, so v1 and v2 live as separate servers.

## Reasoning models

For reasoning models, start the model server with a reasoning parser
(e.g. `--reasoning-parser deepseek_r1`) so `<think>...</think>` blocks
are stripped before verification. The verifier itself does not strip
reasoning — if the prediction begins with a `<think>` preamble, soft
matching will still mostly work for `match_type=all`/`2steps` but
multichoice extraction may fail if the boxed answer falls inside a
truncated reasoning trace.

## Running servers

```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
resources_servers/ruler2/configs/ruler2.yaml"
ng_run "+config_paths=[$config_paths]"
```

## Collecting rollouts

```bash
ng_collect_rollouts \
    +agent_name=ruler2_simple_agent \
    +input_jsonl_fpath=resources_servers/ruler2/data/example.jsonl \
    +output_jsonl_fpath=results/ruler2_rollouts.jsonl \
    +num_repeats=1
```

For the full benchmark run see
[`benchmarks/ruler2/README.md`](../../benchmarks/ruler2/README.md).

## Licensing

- Code: Apache 2.0
- `editdistance`: MIT
