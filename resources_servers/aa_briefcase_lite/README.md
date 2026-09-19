# AA-Briefcase-Lite resources server

The server supplies the standard stateless session endpoint required by NeMo
Gym and grades artifacts independently from policy rollout. Grader-only checks,
prompts, source graphs, and reference artifacts are loaded server-side and do
not enter the agent request.

Binary mode implements all 55 released A/C checks with AA's released system and
user judge prompts. Each check is judged independently and malformed judge
output fails closed. Pairwise mode evaluates the eight released AQ/P criteria
against configured public reference submissions with GDPval's artifact
handling, compatible judge routing, and position-debiased trials. The default
reference is the public `gpt-5-5` example.

AA did not publish its production pairwise prompt, private comparison graph,
or full judge/aggregation details. Pairwise mode is therefore a local
diagnostic. `all` mode reports binary and pairwise metrics separately and also
returns a non-official convenience average. No output from this server is
AA-Briefcase leaderboard comparable.

## Public-reference example validation

The five example rows cover the four released Lite tasks and repeat the video
task once to meet Gym's five-example contract. These are judge-only binary
verification runs of AA's published `gpt-5-5` submission files, not policy
rollouts or leaderboard results. Pairwise comparison is disabled because the
candidate artifacts are themselves the public reference.

Use the pinned [public dataset](https://huggingface.co/datasets/ArtificialAnalysis/AA-Briefcase-Lite/tree/4dec557b47d43867a1648c0974db1d8208c8b677), including hydrated LFS files, at
`/opt/aa-briefcase-lite`. Stage each `submissions/gpt-5-5/<task>/submission/`
under `/opt/aa-briefcase-lite-demo/task_<task>/repeat_0/`; also stage the video
submission under `task_w1_t4/repeat_1/`. The example configuration inherits the
current benchmark judge settings. The stored responses were generated with
Claude at maximum effort and a 32,768-token limit; they do not validate the
current medium-effort, 49,152-token configuration.
Set `GDPVAL_JUDGE_REQUEST_TIMEOUT_SECONDS=600` for the example run.

The stored responses come from independent executions of this public example;
run-local collection indices are omitted. Report and presentation responses were
regenerated after preserving PDF filenames in judge inputs. The example checks
transport and grading behavior; it is not a full-run reliability result.

Configure judge credentials privately. Load the benchmark, policy adapter, and
example overlay together:

```bash
configs="benchmarks/aa_briefcase_lite/config.yaml,responses_api_models/openai_model/configs/openai_model.yaml,benchmarks/aa_briefcase_lite/example.yaml"
gym dataset collate "+config_paths=[$configs]" +mode=example_validation +output_dirpath=data/aa_briefcase_lite_example
gym env start "+config_paths=[$configs]"
# In another terminal, with the same configuration and server environment:
gym eval run --no-serve "+config_paths=[$configs]" +agent_name=aa_briefcase_lite_stirrup_agent \
  +input_jsonl_fpath=data/aa_briefcase_lite_example/example.jsonl \
  +output_jsonl_fpath=resources_servers/aa_briefcase_lite/data/example_rollouts.jsonl
```

The policy adapter is unused in judge-only mode. Dataset collation generates
`example_metrics.json`; it contains dataset statistics, not evaluation scores.

### Judge recovery and usage

Binary and pairwise judge requests use two SDK transport retries (three attempts
in total), separately from invalid-answer retries. Pairwise calls disable the
shared helper's outer transport retries to avoid multiplying attempts. An
exhausted transport failure leaves verification unscored.

The resource-server log records usage for every returned judge response,
including answers that later fail parsing. `Judge usage` entries contain the
model, judging mode, input/output/total tokens, optional reasoning and cached
input tokens, and finish reason. Missing provider usage is logged as `None`, not
zero. These entries contain no prompts, answer text, or credentials. Transport
failures without a response have no token-usage measurement.
