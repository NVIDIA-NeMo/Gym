# Reproducing the published evaluation results

This tutorial demonstrates how to reproduce the evaluation results for the NVIDIA
Nemotron 3.5 Lightning 30B A3B model using NeMo Gym.

The recipes are split by model:

- **[`instruct/`](./instruct/README.md)** — the instruct model. Most benchmarks are Gym
  recipes in [`instruct/gym/`](./instruct/gym/); Terminal-Bench and the two SWE-bench
  suites are NeMo Evaluator configs in
  [`instruct/nemo-evaluator/`](./instruct/nemo-evaluator/).
- **[`base/`](./base/README.md)** — the base (pretraining) model:
  `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Base-BF16`, 20 short-context benchmarks
  plus RULER. These are `nemo-evaluator-launcher` configs rather than Gym recipes, and
  their prerequisites differ.

## What you can reproduce

### Instruct model

| Benchmark | What it measures | Recipe |
|---|---|---|
| GPQA Diamond | Graduate-level science questions | [`gpqa.sh`](./instruct/gym/gpqa.sh) |
| HLE | Humanity's Last Exam — hard, broad knowledge | [`hle.sh`](./instruct/gym/hle.sh) |
| AA-LCR | Long-context reasoning | [`aa-lcr.sh`](./instruct/gym/aa-lcr.sh) |
| AA-Omniscience | Knowledge and hallucination | [`omniscience.sh`](./instruct/gym/omniscience.sh) |
| SciCode | Scientific coding, graded against numeric ground truth | [`scicode.sh`](./instruct/gym/scicode.sh) † |
| BrowseComp | Web browsing agent | [`browsecomp.sh`](./instruct/gym/browsecomp/browsecomp.sh) |
| Tau3 | Tool use against a simulated customer | [`tau3.sh`](./instruct/gym/tau3.sh) |
| CritPt | Research-level physics | [`critpt.sh`](./instruct/gym/critpt.sh) † |
| GDPval | Real-world work products, judged | [`gdpval.sh`](./instruct/gym/gdpval/gdpval.sh) † |
| PinchBench | The model as the brain of an OpenClaw agent | [`pinchbench.sh`](./instruct/gym/pinchbench.sh) † |
| Terminal-Bench 2.1 | Agentic terminal use | [`terminal-bench-2.1.yaml`](./instruct/nemo-evaluator/terminal-bench-2.1.yaml) † |
| SWE-bench Verified | Agentic coding against a repo's tests | [`swebench-verified.yaml`](./instruct/nemo-evaluator/swebench-verified.yaml) † |
| SWE-bench Multilingual | The same, across nine languages | [`swebench-multilingual.yaml`](./instruct/nemo-evaluator/swebench-multilingual.yaml) † |

† needs setup beyond installing Gym — see [`instruct/README.md`](./instruct/README.md).
The rest run as they are.

### Base model

Run as two configs rather than one recipe per benchmark: the suite below, plus RULER for
long context. Prerequisites differ from the instruct recipes — see
[`base/README.md`](./base/README.md).

| Group | Benchmarks |
|---|---|
| General knowledge | MMLU, MMLU-Pro, AGIEval, GPQA Diamond |
| Math | GSM8K, MATH-500 |
| Code | HumanEval (greedy + sampled), MBPP-sanitized (greedy + sampled) |
| Commonsense | CommonsenseQA, ARC-Challenge, HellaSwag, OpenBookQA, PIQA, Social IQa, WinoGrande |
| Reading comprehension | RACE |
| Multilingual | Global-MMLU-Lite, MGSM |
| Long context | RULER † |

† RULER needs a pre-generated dataset — see [`base/README.md`](./base/README.md).

## What to expect from your numbers

**Run the whole benchmark.** Every recipe can be limited for a smoke test — `LIMIT` for
the Gym recipes, `-O benchmarks.0.max_problems=<n>` for the NeMo Evaluator configs,
`limit_samples` for the base suite. A limited run tells you the setup works. It is not a
score, and should not be reported as one.

**Expect run-to-run variation.** These benchmarks sample at `temperature > 0` and average
over repeats, so two runs of the same recipe against the same endpoint will not agree to
the last digit. Judge a difference against that spread, not against the exact published
value.

**Serving configuration moves scores as much as the eval configuration does.** vLLM
version, tensor parallelism, cache dtype, prefix caching, and the reasoning and tool-call
parsers all change results. To compare against the published numbers, serve the model as
the recipes describe and point them at that. Running against an endpoint someone else
operates works fine, but then a gap is ambiguous: you cannot tell whether it came from
their serving stack or from the model. That is a sound way to measure a deployment you
already have — it is not a basis for confirming published numbers.

## License

Apache 2.0 — see the repository `LICENSE`.
