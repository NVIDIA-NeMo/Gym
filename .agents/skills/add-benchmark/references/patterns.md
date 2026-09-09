# Benchmark Integration Patterns

Use this reference to choose an implementation shape. Repository files are examples,
not universal schemas: always confirm the target checkout's config and request models.

## Runtime Sources of Truth

Trace the benchmark through these boundaries:

| Boundary | Source of truth | What to verify |
| --- | --- | --- |
| discovery/config | `nemo_gym/benchmarks.py`, `BenchmarkDatasetConfig` | one benchmark dataset, resolvable agent, paths, repeats |
| preparation | `nemo_gym/cli/eval.py` | importable synchronous `prepare()`, exact returned path |
| prompt materialization | `nemo_gym/prompt.py` | raw rows and `prompt_config`, or materialized input—not both |
| rollout input | `BaseRunRequest` and selected agent request model | required fields and routing |
| verification | selected resources-server request model and `verify()` | field meanings, reward, failures |
| aggregation | selected server's `compute_metrics()` / `get_key_metrics()` | grouping, caps, pass@k, official score |

The data producer and all consumers must agree on semantics, not merely field names.

## Pattern 1: Reuse a Shared Verifier

Start here when the benchmark changes the dataset but not the grading algorithm.

Examples:

- `benchmarks/aime26/` reuses `math_with_judge` and emits `question` plus
  `expected_answer` for a generic math prompt.
- `benchmarks/gpqa/` and `benchmarks/mmlu_pro/` reuse `mcqa` with benchmark-specific
  prompt templates.
- `benchmarks/wmt24pp/` and `benchmarks/flores200/` reuse `wmt_translation` while
  overriding evaluator settings needed by the dataset.
- `benchmarks/livecodebench/v5_2408_2502/` reuses `code_gen` with a nested benchmark
  identity (`livecodebench/v5_2408_2502/config`).

Typical config:

```yaml
config_paths:
  - resources_servers/math_with_judge/configs/math_with_judge.yaml

my_benchmark_resources_server:
  _inherit_from: math_with_judge

my_benchmark_simple_agent:
  _inherit_from: math_with_judge_simple_agent
  responses_api_agents:
    simple_agent:
      resources_server:
        name: my_benchmark_resources_server
      datasets:
        - name: my_benchmark
          type: benchmark
          jsonl_fpath: benchmarks/my_benchmark/data/my_benchmark.jsonl
          prompt_config: benchmarks/my_benchmark/prompts/default.yaml
          prepare_script: benchmarks/my_benchmark/prepare.py
          num_repeats: 1
```

Create an isolated inherited resources-server instance when the benchmark overrides
grader assets or settings. Point the agent at that isolated instance; otherwise the
override may affect or bypass a different composition.

Do not add a new resources server just to rename input fields. Prefer converting the
source rows to an existing, well-tested request contract when the grading semantics are
actually the same.

## Pattern 2: Reuse a Structured or Grouped Verifier

Coding, rubric, and grouped-test evaluators usually consume more than a question and
answer. Treat their request model and aggregation logic as a single contract.

Examples:

- `benchmarks/ioi/` emits competition, problem, subtask, score, and question fields,
  plus a separate metadata artifact consumed by
  `resources_servers/competitive_coding_challenges/`.
- `benchmarks/livecodebench/` converts official runner cases into the `code_gen`
  request shape.
- `benchmarks/finance_agent_v2/` preserves upstream rubrics, modifiers, prompts, and
  tools because each can change the score or agent trajectory.

For each selector or group field, answer all of the following:

1. Is the value present in the verifier's loaded metadata?
2. Does it select tests, label results, cap a score, or only annotate output?
3. If tests overlap across groups, does the evaluator award a group only after every
   required test ran and passed?
4. Does per-rollout reward use the same unit as aggregate metrics?
5. Can one row accidentally contribute credit to a different group or contribute the
   same test more than once?

Never introduce a synthetic selector in `prepare.py` unless the verifier explicitly
maps it to canonical tests and a score cap. A human-readable grouping label is not a
valid execution selector by itself.

Additional grader assets must be declared, generated, and tested as a set. If
`prepare()` writes the primary JSONL plus metadata, scripts, databases, or archives,
the config must point to the exact generated paths and a test must assert their mutual
consistency.

## Pattern 3: Choose Raw or Materialized Prompts

### Raw semantic rows

Use top-level semantic fields and a non-null `prompt_config` when the prompt can be
rendered from text values at rollout time:

```json
{"question": "...", "expected_answer": "...", "uuid": "stable-id"}
```

```yaml
prompt_config: benchmarks/my_benchmark/prompts/default.yaml
```

Every template placeholder must exist in every row. Prompt templates use Python-style
`str.format_map`; literal braces must be doubled. This mode supports prompt sweeps
without re-preparing the dataset.

### Materialized request rows

Use a pre-populated `responses_create_params.input` and `prompt_config: null` when the
request contains images, upstream-owned tool schemas, or agent-specific structure:

```json
{
  "responses_create_params": {
    "input": [{"role": "user", "content": "..."}],
    "tools": []
  },
  "instance_id": "stable-id"
}
```

Examples:

- `benchmarks/hle/config_vision.yaml` uses materialized multimodal inputs, while the
  text-only `benchmarks/hle/config.yaml` uses raw fields plus a prompt template.
- `benchmarks/finance_agent_v2/` materializes the upstream prompt and tool schemas.
- `benchmarks/legal_agent_bench/`, `benchmarks/scicode/`, and `benchmarks/osworld/`
  use custom-agent-compatible rows and `prompt_config: null`.

These modes are mutually exclusive. `nemo_gym/prompt.py` rejects rows with a non-empty
`responses_create_params.input` when a `prompt_config` is also configured.

## Pattern 4: Add a Custom Verifier or Agent Loop

Use a custom verifier when reward or aggregate metrics are new. Use a custom agent
when the trajectory—not only the final answer—is part of the benchmark.

Current scaffold profiles correspond to different extension points:

| Profile | Use when |
| --- | --- |
| `custom-gym-verifier` | Gym owns the verification logic; an existing agent can run it |
| `custom-gym-agent-loop` | Gym owns a custom multi-step agent loop |
| `external-agent-loop` | an upstream harness owns the interaction loop |
| `external-rollout-driver` | an external driver owns rollout orchestration |

Examples to inspect:

- `resources_servers/scicode/` plus `responses_api_agents/scicode_agent/` for a
  benchmark-specific, multi-step scientific-coding loop.
- `benchmarks/legal_agent_bench/` plus its Harbor agent/resources server for an
  upstream task bundle adapted into Gym with shared cache paths.
- `benchmarks/osworld/` plus `responses_api_agents/osworld_agent/` for an external
  environment with execution-backend profiles and operational tooling.
- `resources_servers/evalplus/` with `benchmarks/human_eval/` and `benchmarks/mbpp/`
  for multiple datasets sharing an official code-evaluation backend.

Agent loops must propagate session cookies through downstream calls. Training-capable
multi-turn loops must preserve monotonic trajectories and token metadata required by
the training stack. Follow `AGENTS.md` for async HTTP and concurrency rules.

When wrapping an upstream library:

1. run its official evaluator first and save versioned per-example outputs;
2. pin the upstream version and data revision;
3. map Gym inputs and outputs at a narrow adapter boundary;
4. compare Gym and upstream verdicts example by example;
5. separately compare aggregate metrics and unsupported/error cases.

## Pattern 5: Compose an Eval Suite

An eval suite is a composition, not a discoverable single benchmark. Keep each
benchmark independently preparable and runnable, then compose config paths:

```yaml
config_paths:
  - benchmarks/gpqa/config.yaml
  - benchmarks/scicode/config.yaml
  - benchmarks/hle/config.yaml
  - responses_api_models/vllm_model/configs/vllm_model.yaml
```

See `benchmarks/nemotron_3.5_super/eval_container_config.yaml` for a heterogeneous
suite and `nemotron_recipes/lightning-3.5/base/base-suite.yaml` for an external
evaluator task suite.

Suite checks should cover:

- config composition conflicts and intentional overrides;
- unique output/task identities across benchmarks;
- model capabilities required by each member (vision, tools, completions, logprobs);
- per-benchmark completion/error rates before computing a suite summary;
- repeat/sampling policy, caching, and aggregation per benchmark rather than one
  accidental global default.

Do not place several locally declared `type: benchmark` datasets in a config expected
to work with `gym eval prepare --benchmark <name>`; benchmark discovery intentionally
rejects that shape as a suite.

## Manifest-Backed Scaffolding

For a new complete benchmark that fits the manifest contract, use:

```bash
gym env init --benchmark my_benchmark
```

The scaffold creates the catalog artifact, config, manifest, and component/fixture
extension points appropriate to its profile. Reuse is available only when the existing
resources server exports `VERIFIER_FIXTURE`:

```bash
gym env init --benchmark my_benchmark \
  --reuse-verifier shared_verifier \
  --reward-range 0 1 \
  --higher-is-better
```

The manifest declares the public contract (kind, profile, domain, reward range,
determinism, components, and datasets) and mirrors resolved config fields. The config
remains authoritative for runtime wiring. Validate both static structure and verifier
fixtures with the manifest-aware commands before publication.

A benchmark manifest currently requires a standard prompt config. For a self-contained
materialized/custom-agent dataset that must keep `prompt_config: null`, follow the
existing config-only patterns rather than adding a dummy prompt. Likewise, do not add a
fixture or migrate a legacy shared verifier merely to make an otherwise focused
benchmark PR use `--reuse-verifier`.

Use the standalone `gym env init --resources-server my_server` only when adding one
component outside a complete catalog entry.

## Preparation Pattern

The CLI imports the module and calls `prepare(**prepare_script_args)`. The returned
path must exactly match `jsonl_fpath`. Keep default preparation callable with no args;
optional keyword arguments are useful for focused local preparation.

Robust preparation has these properties:

- expensive or optional preparation dependencies are imported lazily when practical;
- dependencies needed by `gym eval prepare` are available in the repository-root
  environment, not only a resources server's isolated requirements;
- network calls check failures and source versions/revisions are recorded;
- ordering and serialization are deterministic;
- IDs and evaluation-critical fields are validated before publication;
- a temporary file is flushed and atomically replaces the destination only after all
  validation succeeds;
- failure leaves an existing valid output untouched.

`benchmarks/legal_agent_bench/prepare.py` demonstrates deterministic rendering and an
atomic replace. `benchmarks/wmt24pp/tests/` demonstrates source-drift and failed-write
tests. `benchmarks/finance_agent_v2/tests/` demonstrates pinning rubric, modifier,
prompt, and tool semantics rather than only row counts.

Keep large generated JSONL and grader assets ignored unless they are intentional small
fixtures. Commit small examples only when they are needed for offline smoke tests.

## Verifier and Metric Pattern

A resources server normally defines a typed request and response around `verify()`.
The exact row shape varies by server, so validate against that type rather than a
generic imagined schema.

Verification tests should distinguish:

- valid full-credit output;
- valid but wrong/partial output;
- empty, malformed, or unparsable model output;
- grader exception, timeout, and missing external dependency;
- state isolation and cookie behavior for stateful environments.

Metric tests should be separate. Exercise every grouping key, cap, weight, tie rule,
deduplication rule, repeat policy, and missing-result behavior. For a benchmark with
subtasks, include at least one fixture where test membership overlaps; for a benchmark
with rubrics, include unequal weights and a missing criterion.

`resources_servers/competitive_coding_challenges/tests/test_app.py` is a useful example
of testing request forwarding, partial subtask reward, full-problem reward, score caps,
and cross-rollout aggregation. It does not remove the need for benchmark-specific tests
that prove prepared selectors match the metadata loaded by that server.

## Test Pattern by Boundary

| Boundary | Good repository examples | Important assertions |
| --- | --- | --- |
| prepare/import | `benchmarks/scicode/tests/test_prepare.py` | correct split, exact IDs and output path |
| deterministic/atomic output | `benchmarks/legal_agent_bench/tests/test_prepare.py` | stable bytes, count, failure preserves output |
| source drift | `benchmarks/finance_agent_v2/tests/test_prepare.py` | rubric/modifier/tool mapping fails loudly |
| prompt modes | `benchmarks/hle/prepare.py`, `nemo_gym/prompt.py` tests | raw vs materialized exclusivity |
| config resolution | `benchmarks/legal_agent_bench/tests/test_prepare.py` | isolated instances, agent routing, paths |
| reward and metrics | resources-server `tests/test_app.py` files | verdicts plus aggregation semantics |
| external operation | `benchmarks/osworld/tests/` | backend selection, sharding, scripts, cleanup |

Unit tests must not depend on downloading the full benchmark. Preserve a tiny fixture
with the same evaluation-critical structure, then run separate real-data and rollout
checks before declaring the integration complete.
