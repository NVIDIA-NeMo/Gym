# HumanEval+ benchmark

164 Python function-completion tasks with EvalPlus base + plus tests.
Mirrors NeMo-Skills' `human-eval` dataset — same data source, same
transformation (4-space → `\t` in `prompt`), same per-task verification
(EvalPlus base + plus subprocess execution).

Two named scores per task:
- `passing_base_tests` — passes the base HumanEval tests
- `passing_plus_tests` — passes base + EvalPlus extra tests (strict)

Reward (`= 1.0` iff plus pass) is the strict verdict. Both scores produce
their own pass@k / majority@k via the `evalplus` resource server's
`compute_metrics()`.

Verification runs in the `evalplus` resource server; this directory only
holds the dataset definition + prompt + prepare script.

### Example usage

```bash
# Prepare benchmark data
gym eval prepare --benchmark human_eval

# Running servers
gym env start \
    --model-type vllm_model \
    --benchmark human_eval

# Collecting rollouts. The +prompt_config= override is required because
# the prepared JSONL stores raw `question` rows (no `responses_create_params.input`);
# gym eval run --no-serve does not pick up the `prompt_config:` field on the dataset
# entry in config.yaml the way gym env start does.
gym eval run --no-serve \
    --agent human_eval_evalplus_simple_agent \
    --input benchmarks/human_eval/data/human_eval_benchmark.jsonl \
    --prompt-config benchmarks/prompts/generic/codegen.yaml \
    --output results/human_eval_rollouts.jsonl \
    --num-repeats 4
```

Start vLLM with `--reasoning-parser <name>` (e.g. `deepseek_r1` for
Nemotron-3) so `<think>…</think>` is stripped before the verifier sees
the model output. Without this, the last-fence extractor will skip
trailing reasoning blocks but `no_answer` rates will diverge from
Skills' `eval_type=evalplus` baseline (which strips reasoning at the
evaluator layer).

## Licensing information
MIT (HumanEval+ is MIT-licensed; see https://github.com/evalplus/evalplus).
