# PolyMath

[Qwen/PolyMath](https://huggingface.co/datasets/Qwen/PolyMath) — a
multilingual math benchmark covering 18 languages × 4 difficulty
tiers (low / medium / high / top, ~125 problems per (lang, tier)).

## Verification

Reuses the [`polymath`](../../resources_servers/polymath) resources
server (which itself subclasses `math_with_judge` in symbolic-only
mode) so the verification path is identical to NeMo Skills' default
`eval_type=math` for this benchmark. The PolyMath-specific bits are
metric aggregation:

- Difficulty-weighted `pass@k`, `pass@1[avg-of-k]`, `majority@k`
  (weights: low=1, medium=2, high=4, top=8) — emitted with the
  `_weighted` suffix.
- Per-language fan-out (`<lang>/pass@k/<score>`).

See `resources_servers/polymath/README.md` for details.

## Prompt

User-only prompt, character-for-character match with NeMo Skills'
`generic/default.yaml`:

```
{question}
```

The `{question}` field is the upstream PolyMath problem text wrapped
in PolyMath's own `QUESTION_TEMPLATE` (problem + per-language
instruction "Please answer …, place your final answer in \boxed{}",
optionally a language-control suffix). The instruction text is
downloaded at prepare time from upstream PolyMath's `instruction.py`,
matching Skills.

## Data preparation

```
ng_prepare_benchmark "+config_paths=[benchmarks/polymath/config.yaml]"
```

Writes `data/polymath_benchmark.jsonl` with one row per
(language × difficulty × problem).

## Quickstart

Start the servers (inherits the `polymath` resources server in
symbolic-only mode plus a vLLM model server):

```
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/polymath/config.yaml"
ng_run "+config_paths=[$config_paths]"
```

Collect rollouts (use `+num_repeats=4` for a quick parity pass,
`+num_repeats=16` for parity-grade evaluation):

```
ng_collect_rollouts \
    +agent_name=polymath_benchmark_simple_agent \
    +input_jsonl_fpath=benchmarks/polymath/data/polymath_benchmark.jsonl \
    +output_jsonl_fpath=results/polymath_rollouts.jsonl \
    +num_repeats=4
```

Start the model server with `--reasoning-parser <name>` (e.g.
`deepseek_r1` for Nemotron-3) so `<think>…</think>` is stripped from
model output before `\boxed{…}` extraction.

## Licensing

- Code: Apache 2.0
- Data: PolyMath (`Qwen/PolyMath`) — see the upstream HF repo.
