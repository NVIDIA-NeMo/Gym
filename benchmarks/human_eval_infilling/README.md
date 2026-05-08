# HumanEval-Infilling (FIM) benchmark

Code Fill-in-the-Middle (FIM) tasks derived from HumanEval, introduced
in [Bavarian et al., 2022](https://arxiv.org/abs/2207.14255).
Each task gives the model a prefix and a suffix and asks it to emit the
missing span; the spliced program is then run against the same hidden
test that grades HumanEval's function-completion variant.

Three difficulty levels, fetched from the upstream
[`openai/human-eval-infilling`](https://github.com/openai/human-eval-infilling)
release (`HumanEval-{SingleLine,MultiLine,RandomSpan}Infilling.jsonl.gz`).
The HF mirror at `loubnabnl/humaneval_infilling` is a script-based
dataset and is no longer compatible with modern `datasets` versions;
the upstream raw files are content-equivalent and are what
`human_eval_infilling.data.read_problems(...)` reads (Skills uses the
same library at evaluation time):

| Split          | Tasks | What's masked                                  |
|----------------|-------|------------------------------------------------|
| `single_line`  |  1033 | a single statement (line) inside the function |
| `multi_line`   |  5815 | a contiguous span of >1 lines                  |
| `random_span`  |  1640 | a randomly chosen contiguous character span    |

`random_span` is the default (mirrors NeMo-Skills' `EVAL_SPLIT`).
Mirrors NeMo-Skills' `human-eval-infilling` dataset — same data source,
same per-row transformation (`prompt` → `prefix`; drop `entry_point`
and `test`; add `language="python"`, `split`, `comment_delimiter="#"`),
same per-task verification (`human_eval_infilling.execution.check_correctness`
splices `prefix + completion + suffix + test` and runs in a sandboxed
subprocess).

Verification runs in the `code_fim` resource server; this directory
holds only the dataset definition + prompt + prepare script.

### Example usage

```bash
# Prepare benchmark data (downloads all three default splits;
# default benchmark variant is random_span)
ng_prepare_benchmark "+config_paths=[benchmarks/human_eval_infilling/config.yaml]"

# Running servers
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/human_eval_infilling/config.yaml"
ng_run "+config_paths=[$config_paths]"

# Collecting rollouts. The +prompt_config= override is required because
# the prepared JSONL stores raw rows (no `responses_create_params.input`);
# ng_collect_rollouts does not pick up the `prompt_config:` field on the
# dataset entry in config.yaml the way ng_run does.
ng_collect_rollouts \
    +agent_name=human_eval_infilling_simple_agent \
    +input_jsonl_fpath=benchmarks/human_eval_infilling/data/random_span.jsonl \
    +prompt_config=benchmarks/human_eval_infilling/prompts/default.yaml \
    +output_jsonl_fpath=results/human_eval_infilling_rollouts.jsonl \
    +num_repeats=1
```

### Other splits

To benchmark a different split, point the resource server at it via the
config and feed the matching prepared JSONL:

```bash
ng_collect_rollouts \
    +agent_name=human_eval_infilling_simple_agent \
    '++policy_model.resources_servers.code_fim.split=single_line' \
    +input_jsonl_fpath=benchmarks/human_eval_infilling/data/single_line.jsonl \
    +prompt_config=benchmarks/human_eval_infilling/prompts/default.yaml \
    +output_jsonl_fpath=results/human_eval_infilling_single_line_rollouts.jsonl \
    +num_repeats=1
```

Start vLLM with `--reasoning-parser <name>` (e.g. `deepseek_r1` for
Nemotron-3) so `<think>…</think>` is stripped before the verifier sees
the model output. The extractor also strips `<think>...</think>` as a
fallback, but the parser-driven path is preferred for parity with
Skills' `eval_type=human_eval_infilling` baseline.

## Licensing information
MIT (the upstream HumanEval-Infilling library and the
`loubnabnl/humaneval_infilling` HF dataset are MIT-licensed; see
https://github.com/openai/human-eval-infilling).
