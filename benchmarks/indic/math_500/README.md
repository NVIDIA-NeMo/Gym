# Indic MATH-500

Evaluate [ai4bharat/indic-math-500](https://huggingface.co/datasets/ai4bharat/indic-math-500)
using Gym's English [MATH-500](../../math-500) pipeline.

## Details

- Data: 500 problems per language from `ai4bharat/indic-math-500` (Apache-2.0).
- Default languages: `bn`, `gu`, `hi`, `kn`, `ml`, `mr`, `ne`, `or`, `pa`, `ta`,
  `te`, `ur`; these 12 languages produce 6,000 rows. Assamese (`as`) and Sanskrit
  (`sa`) remain available through explicit `--languages` selection. Select `en`
  to use the original English problems for a paired baseline.
- Prompt: shared `benchmarks/prompts/generic/math.yaml`, including the English
  instruction to put the final answer inside `\boxed{}`. Only the problem text
  is translated; solutions are excluded from the prepared rows.
- Evaluation: English `simple_agent` and `math_with_judge`, with symbolic
  verification and `should_use_judge: false`. The original answers are unchanged.
- Metric: accuracy (pass@1), with one response per problem by default. Report
  results by language; pooled accuracy averages across the selected languages.
  For combined runs, join rollouts to prepared rows using `_ng_task_index` to
  recover language metadata. The shared verifier does not retain extra row fields.
- Generation: inherits the English configuration. Use the same model, sampling,
  token budget, and thinking settings for both languages in a comparison.

## Example usage

```bash
# Prepare the 12 default Indic languages.
gym eval prepare --benchmark indic/math_500

# Start servers with your configured vLLM endpoint.
gym env start --benchmark indic/math_500 --model-type vllm_model

# In another terminal, collect and score responses.
gym eval run --no-serve \
    --agent math_500_math_with_judge_simple_agent \
    --input benchmarks/indic/math_500/data/math_500_benchmark.jsonl \
    --output results/indic-math-500/rollouts.jsonl \
    --prompt-config benchmarks/prompts/generic/math.yaml
```

To prepare a language separately, use a distinct output path and pass it to
`gym eval run --input`:

```bash
python -m benchmarks.indic.math_500.prepare \
    --languages hi \
    --output-fpath benchmarks/indic/math_500/data/hi.jsonl

python -m benchmarks.indic.math_500.prepare \
    --languages en \
    --output-fpath benchmarks/indic/math_500/data/en.jsonl
```
