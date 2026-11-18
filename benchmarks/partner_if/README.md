# partner_if

Internal tasks built from instruction-following failures seen in production agent deployments. Each
task checks whether the model follows its instructions, graded against versioned rubrics by the
[partner_if](../../resources_servers/partner_if/README.md) resources server.

## Data

- **Source:** NVIDIA-internal rows in Gym's GitLab dataset registry: dataset `partner_if`, version
  `2.1.7`, file `partner_if.jsonl`. `prepare.py` pins the file's sha256 and refuses any other content. The
  rows are not in this repository.
- **Public examples:** the rows above are internal, but the partner_if resources server ships synthetic rows in
  the same format: five text tasks in
  [`data/example.jsonl`](../../resources_servers/partner_if/data/example.jsonl) and two mock-tool tasks in
  [`data/mock_tools_example.jsonl`](../../resources_servers/partner_if/data/mock_tools_example.jsonl), with
  recorded rollouts of both, plus a sandboxed version of the mock-tool tasks. Its
  [Usage](../../resources_servers/partner_if/README.md#usage) section shows how to run them.
- **License:** internal.
- **Size:** 28 tasks with 89 IF rubric items, plus 4 completion items: 18 text tasks, 2 search-tool tasks,
  3 build-tool tasks, 3 SWE tasks, 1 native Prime task and 1 recorded-context ordering task. There is one split, `benchmark`.
- **Provenance:** Reported instruction-following failures and reconstructions. The
  tool tasks' tools are simulations written for this benchmark, not real services. The SWE tasks' repositories
  and images come from Scale-SWE.
- **Rows:** each row holds the prompt (`responses_create_params`) and the grading contract
  (`verifier_metadata.partner_if`). A tool task's row also holds its tool backend (`tool_backend`): the
  simulated tools' code, fixture and pinned requirements. A SWE task's row also holds its Scale-SWE instance:
  the image, working directory, setup commands and tests.

The config routes all 28 published rows. The Prime route uses the experimental SDK child
entrypoint of `prime_agent`; its runtime packaging and harness merge checks are documented in
[the adapter README](../../responses_api_agents/prime_agent/README.md#sdk-child-sessions).
The full-release route-coverage test stays enabled as a readiness gate.

## Task contract

- **Harness:** Gym's `simple_agent` runs the text and simulated-tool tasks; OpenCode (`opencode_agent`)
  runs the SWE tasks; Prime Agent 0.7.0's depth-one child session runs the repository/report task; and
  Gym's stock `gymnasium_agent` runs the recorded-context ordering task. The SWE scores depend on
  OpenCode 1.17.11: the harness and its version are part of what they measure.
- **Routing:** `partner_if_dispatcher` routes each task to its environment, so supported tasks share one
  dataset and one score. Gym aggregates metrics per environment server, so without it the six task
  families would have separate scores.
- **Text tasks:** one reply from the policy model. Some rows include recorded tool calls and results, or ask
  for the next tool call. No tools run, and the grader sees any tool call only as part of the reply.
- **Search-tool tasks:** up to 30 steps with the row's simulated search and document tools.
- **Build-tool tasks:** up to 15 steps with the row's simulated project tools. These tasks don't require
  a final text answer.
- **SWE tasks:** OpenCode 1.17.11 works in the repository of the row's Scale-SWE image.
  - The row's setup commands write OpenCode's managed configuration (`/etc/opencode`, so they need root in the
    image). It replaces OpenCode's system prompt with the task's and adds text to two tool descriptions. The
    contract carries the judge's copies of both.
  - OpenCode keeps its default step limit and conversation compaction. It runs with `opencode_agent`'s command
    deny-list and without the web, subagent and skill tools.
  - `opencode_config.snapshot: false` disables native Git snapshots and undo history to avoid a post-generation
    stall. Gym still captures model calls and trajectories. This differs from the historical reference setting.
  - Set `context_window` on `partner_if_swe_agent` (262144 by default) to the policy model's: OpenCode compacts
    the conversation as it nears it.
  - The judge grades the visible trajectory, never reasoning: every agent message and tool call in order, with
    the arguments in full, and the tool outputs. An output longer than 500 characters keeps its first and last 250,
    with a `[… N characters omitted …]` marker between them. The SWE rubrics grade the agent's messages and tool
    calls, and with whole outputs a long agent run would not fit the judge's input.
- **Prime task:** a pinned source checkout, native IPython, and a recording parent controller.
  The actor has 48 complete assistant turns and a 900-second safety timeout. Its output
  reservation is 65,536 tokens per call, shrinking as context fills; temperature is 0.
  Its Environment Server has no outer episode deadline (`null`); the 1,800-second
  setup-command and agent-install limits, 900-second actor limit, 60-second agent-close
  limit and 360-second environment-cleanup limit remain explicit. Environment admission
  and agent concurrency are both three, matching the three dataset repeats. The
  600-second queue timeout applies to requests beyond those three active slots.
  Compaction and retries are disabled. Only corroborated parent deliveries reach the shared
  rubric judge; unsent drafts and final chat text do not count. These limits are part of
  the task contract. The dedicated `partner_if_prime_model` preserves the runtime's cap.
- **Sandbox:** the tool, SWE, and Prime tasks need OpenSandbox as configured: a provider with service endpoints and
  network policies.
  - Each tool rollout gets a fresh sandbox that holds only the row's tool backend: `python:3.13-slim`, one CPU
    and 1 GiB. By default the sandbox can reach only the package index, to install the backend's pinned
    requirements. With a prebuilt image, set `network: none`.
  - Each SWE rollout gets a sandbox of the row's Scale-SWE image: limits of 4 CPUs and 16 GiB, requests of 1 CPU
    and 4 GiB, and 20 GiB of disk. It needs egress to GitHub, to install OpenCode, and to the policy model server.
  - OpenCode calls the policy model server from inside the sandbox, so run with `++use_absolute_ip=true`, as other
    sandboxed-agent benchmarks do: the servers then listen on the host's address instead of loopback. Where the
    host's name does not resolve to an address the sandboxes can reach, also set `+default_host=<address>`.
  - Scale-SWE's own tests then run as a diagnostic, in up to two more sandboxes of the same image: an
    inconclusive result is retried once. Their verdict is kept in `task_outcome` and never changes the reward.
- **Sampling:** each run sets its policy's sampling, such as temperature and top_p, on the policy model server
  (`policy_model.responses_api_models.vllm_model.sampling_overrides`). The text, simulated-tool and SWE
  routes call `partner_if_policy_model`, a copy of the run's `vllm_model` policy server that keeps the run's
  sampling and sets null output caps. Prime and recorded-context routes use separate model-server copies
  that preserve their task-specific per-call limits and sampling described above and below.
  - No output cap means a null cap, which also replaces the per-call cap OpenCode sends. On a vLLM endpoint each
    reply may then use the rest of the model's context window.
  - An endpoint that applies a small default when the cap is null (some hosted OpenAI-compatible APIs default to
    4096) would cut replies short. Give it an explicit cap the size of the context window:
    `++partner_if_policy_model.responses_api_models.vllm_model.sampling_overrides.max_tokens=<window>`.
  - An endpoint that ends long requests needs a longer per-request timeout, since a reply at maximum reasoning
    effort can take more than 6 minutes. Some hosted gateways stop a request at 360 s unless its body sets
    `timeout`. The reference runs set `extra_body.timeout: 3600` on the policy and judge model servers.
  - Runs set no request parameters, such as `--temperature`, `--top-p`, `--max-output-tokens` or a reasoning
    effort: `opencode_agent` refuses them, so every SWE task would fail. Set sampling and model settings such as
    reasoning effort on the policy model server instead.
- **Prompts:** the rows are prepared, so no template applies. Some rows carry task-specific
  `temperature`, `top_p`, `max_output_tokens` or `tool_choice`; preserve those settings for targeted
  reproduction runs. The SWE tasks retain the model-server sampling policy described above.

## Scoring

- **Reward:** a rollout scores 1 when every IF rubric item passes, and 0 otherwise. Code checks
  (partner_if's built-in and fixed checks) and an LLM judge grade the items. Completion items are
  reported but never change the reward.
- **Provisional rubrics:** 25 of the 28 contracts in 2.1.7 are unsigned (`audit_signoff: false`); the 3 SWE
  contracts are signed. The config sets `allow_provisional_rubrics: true`, so the unsigned rows are scored and
  marked `provisional`.
- **Metrics:** reported on a 0–100 scale.
  - Headline: `pass@1[avg-of-3]/all_if_pass`, each task's pass rate over 3 repeats, averaged over tasks
    with equal weight.
  - Also `if_partial`, the passed fraction of IF items.
  - `coverage/*` counts masked rollouts.
- **Historical reference score (not a baseline for the expanded 2.1.7 cohort):** 57.33 on `pass@1[avg-of-3]/all_if_pass`, tolerance ±9.5 points (95%), on dataset 2.1.4.
  - Policy: GLM 5.3 (`nvidia/zai-org/glm-5.3`) at its standard sampling, set on the model server: temperature
    1.0 and top_p 0.95, maximum reasoning effort, 262,144 output tokens, 16 rollouts in parallel.
  - Judge: the default GPT-6 Sol (`azure/openai/gpt-6-sol`).
  - The tolerance is t(0.975, 6) × the pooled `std_dev_across_runs` (4.74) × √(2/3). Two more runs with the same
    judge scored 53.33 and 56.00.
- **Cross-check:** 50.67, with 0 masked rows, from regrading a different saved GLM 5.3 run (dataset 2.1.0, whose
  actor inputs match 2.1.4) with the 2.1.4 rubrics and GPT-6 Sol through a different gateway route.
- **History:** dataset 2.1.0 scored 61.33 with GLM 5.3 as both the policy and the judge.
- **Judge:** GPT-6 Sol (`azure/openai/gpt-6-sol`) by default, at temperature 0, with 32768 output tokens.
  Each of the five Resources instances (ordinary, simulated-tool, SWE, Prime and recorded-context) allows
  2 concurrent judge calls, so their combined limit is 10. Set `judge_base_url` and `judge_api_key`.
  A different judge makes the scores incomparable.
- **Failures:**
  - A missing or truncated answer scores 0.
  - Infrastructure, capture and judge failures are masked, with `failure_kind` and `failure_reason`, and
    leave the score. So is an OpenCode run that `opencode_agent` reports as stopped by its deadline or failed.
  - A judge reply that fails validation is asked for again once. If the second reply fails too, the row is masked
    as `judge_unparseable`.
  - A SWE task whose setup fails, its row's own setup commands included, fails its seed: its sandbox stops and
    the rollout goes to the failures file.
  - Run with `route_failures_to_sidecar: true`, so a failed `/run` lands in `<output>_failures.jsonl` instead
    of stopping the run.
  - A run with masked rows or failures is incomplete.
- **Timeouts:**
  - The text and tool tasks have no episode deadline: their step caps end an episode.
  - A SWE episode has no step cap, and OpenCode's deadline is 10800 s. Its sandbox has 600 s to start and lives
    at most 14400 s, and each of Scale-SWE's test runs gets 1200 s.
  - Text, simulated-tool and SWE environment cleanup times out after 180 s; Prime cleanup gets 360 s.
  - A tool sandbox has 600 s to start and 300 s to install its requirements, and its runner has 60 s to load.
  - Each tool call times out after 60 s.
  - A tool sandbox lives at most 3600 s.
- **Cost:** a full run is 84 rollouts (28 tasks, each repeated 3 times). Each of the 9 SWE rollouts uses
  an OpenCode sandbox and up to two test sandboxes of 4 CPUs and 16 GiB; each of the 15 simulated-tool
  rollouts uses one small sandbox; and each of the 3 Prime rollouts uses a repository sandbox of
  4 CPUs and 16 GiB. The 3 recorded-context rollouts use the in-process frozen simulator, with up to
  8 policy calls per rollout including Explore.
- **Seeds:** none are set, so runs are stochastic.

## Running

1. Set `mlflow_tracking_uri` and `mlflow_tracking_token` for Gym's GitLab dataset registry, and install
   Gym's `mlflow` extra.
2. Prepare the rows:

   ```bash
   gym eval prepare --benchmark partner_if
   ```
3. Add the judge endpoint and a sandbox provider config, and run the benchmark. The policy server must be
   `vllm_model`: its `sampling_overrides` carry the run's sampling, and the benchmark's copy adds the null caps:

   ```bash
   gym eval run --benchmark partner_if --model-type vllm_model \
     --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
     --output <run dir>/rollouts.jsonl ++use_absolute_ip=true
   ```

For the exit-check evidence, turn on model-call capture with `++observability_enabled=true
++model_call_capture_dir=<absolute dir>`, then run `gym eval health-check <run dir>`. With OpenTelemetry on,
add `sandbox` to the span groups to trace the tool and SWE sandboxes. The judge grades a rollout but is not part
of it, so its calls stay out of the model-call capture: each row keeps the judge's request and reply in
`judge_request` and `judge_response`, and any reply it asked for again in `judge_rejected_responses`. Checks on the
judge's calls read them there.


## Recorded-context ordering profile: pif_029

`config.yaml` includes `config_recorded_context.yaml` and routes `pif_029` through
Gym's stock `legacy_agent` relay and `gymnasium_agent`. The existing PartnerIF
Resources component implements Gymnasium `/reset`, `/step` and optional `/close`;
it owns the frozen tool simulator, Explore, shared semantic grading and the total
policy-call budget. There is no dataset-specific agent or Environment Server.

The run stops at the first decisive semantic ordering result. Failed ordering
scores zero. Passing requires a returned, identity-validated sibling probe;
unknown coverage, faults and undecided cutoffs are masked. All calls in one
assistant response precede their results, so a probe and prescription in the same
response cannot establish correct ordering.

Limits remain six parent turns and eight total policy calls including Explore,
with temperature 0.7, top_p 0.95 and 8,192 tokens per call. The other task routes
and their sampling are unchanged. The dispatcher lifts Resources' terminal report
into the collector row, retaining terminal tool outputs, all fresh usage and grades.

`prepare.py` still verifies the published private 2.1.7 source rows and data-only
assets. A local release-candidate command marks the changed harness in pif_029,
keeps every other row and all asset bytes, and records candidate digests:

```bash
.venv/bin/python -m benchmarks.partner_if.stage_recorded_context_release \
  --output-dir benchmarks/partner_if/data/release-candidate-2.1.8
```

The candidate is not published. Registry publication under a new version and
updating `prepare.py` to its verified pin remain release steps before merging;
2.1.7 must not be overwritten. Local development can use its pinned inputs with
the new configuration. See [recorded-context.md](recorded-context.md) for the
transport normalization and limitations, and
[recorded-context-provenance.md](recorded-context-provenance.md) for provenance.
The historical 25-task reference score does not apply to this 28-task review release.
