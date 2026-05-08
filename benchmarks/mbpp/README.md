# MBPP+ benchmark

378 Python function-completion tasks from the EvalPlus-curated subset of
MBPP, verified with EvalPlus base + plus tests. Mirrors NeMo-Skills'
`mbpp` dataset — same data source, same transformation (4-space → `\t`
in `prompt`), same per-task verification (EvalPlus base + plus
subprocess execution).

Two named scores per task:
- `passing_base_tests` — passes the base MBPP tests
- `passing_plus_tests` — passes base + EvalPlus extra tests (strict)

Reward (`= 1.0` iff plus pass) is the strict verdict. Both scores produce
their own pass@k / majority@k via the `evalplus` resource server's
`compute_metrics()`.

Verification runs in the `evalplus` resource server (shared with
`human_eval`); this directory only holds the dataset definition + prompt
+ prepare script.

### Example usage

```bash
# Prepare benchmark data
ng_prepare_benchmark "+config_paths=[benchmarks/mbpp/config.yaml]"

# Running servers
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/mbpp/config.yaml"
ng_run "+config_paths=[$config_paths]"

# Collecting rollouts. The +prompt_config= override is required because
# the prepared JSONL stores raw `question` rows (no `responses_create_params.input`);
# ng_collect_rollouts does not pick up the `prompt_config:` field on the dataset
# entry in config.yaml the way ng_run does.
ng_collect_rollouts \
    +agent_name=mbpp_evalplus_simple_agent \
    +input_jsonl_fpath=benchmarks/mbpp/data/mbpp_benchmark.jsonl \
    +prompt_config=benchmarks/mbpp/prompts/default.yaml \
    +output_jsonl_fpath=results/mbpp_rollouts.jsonl \
    +num_repeats=4
```

Start vLLM with `--reasoning-parser <name>` (e.g. `deepseek_r1` for
Nemotron-3) so `<think>…</think>` is stripped before the verifier sees
the model output. Without this, the last-fence extractor will skip
trailing reasoning blocks but `no_answer` rates will diverge from
Skills' `eval_type=evalplus` baseline (which strips reasoning at the
evaluator layer).

## Licensing information
MIT (MBPP+ is MIT-licensed; see https://github.com/evalplus/evalplus).
