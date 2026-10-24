# partner_if

Internal tasks built from instruction-following failures seen in production agent deployments. Each
task checks whether the model follows its instructions, graded against versioned rubrics by the
[partner_if](../../resources_servers/partner_if/README.md) resources server.

## Data

- **Source:** NVIDIA-internal rows in Gym's GitLab dataset registry: dataset `partner_if`, version
  `2.0.0`, file `partner_if.jsonl`. `prepare.py` pins the file's sha256 and refuses any other content. The
  rows are not in this repository.
- **License:** internal.
- **Size:** 22 tasks with 83 IF rubric items, plus 4 completion items: 17 text tasks, 2 search-tool tasks and
  3 build-tool tasks. There is one split, `benchmark`.
- **Provenance:** 21 tasks reproduce a reported instruction-following failure and 1 reconstructs one. The
  tool tasks' tools are simulations written for this benchmark, not real services.
- **Rows:** each row holds the prompt (`responses_create_params`) and the grading contract
  (`verifier_metadata.partner_if`). A tool task's row also holds its tool backend (`tool_backend`): the
  simulated tools' code, fixture and pinned requirements.

## Task contract

- **Harness:** the policy model runs in Gym's `simple_agent`.
- **Routing:** `partner_if_dispatcher` routes each task to its environment, so all 22 tasks share one
  dataset and one score. Gym aggregates metrics per environment server, so without it the three task
  families would be three benchmarks with separate scores.
- **Text tasks:** one reply from the policy model. Some rows include recorded tool calls and results, or ask
  for the next tool call. No tools run, and the grader sees any tool call only as part of the reply.
- **Search-tool tasks:** up to 30 steps with the row's simulated search and document tools.
- **Build-tool tasks:** up to 15 steps with the row's simulated project tools. These tasks don't require
  a final text answer.
- **Sandbox:** the tool tasks need OpenSandbox as configured: a provider with service endpoints and network
  policies.
  - Each rollout gets a fresh sandbox that holds only the row's tool backend: `python:3.13-slim`, one CPU
    and 1 GiB.
  - By default the sandbox can reach only the package index, to install the backend's pinned requirements.
    With a prebuilt image, set `network: none`.
- **Prompts:** the rows are prepared, so no template applies. Rows set no `temperature`, `top_p` or
  `max_output_tokens`; some set `tool_choice`. Sampling and the output budget come from the run's policy
  model settings, which every run must state.

## Scoring

- **Reward:** a rollout scores 1 when every IF rubric item passes, and 0 otherwise. Code checks
  (partner_if's built-in and fixed checks) and an LLM judge grade the items. Completion items are
  reported but never change the reward.
- **Provisional rubrics:** every contract in 2.0.0 is unsigned (`audit_signoff: false`). The config sets
  `allow_provisional_rubrics: true`, so the rows are scored and marked `provisional`.
- **Metrics:** reported on a 0–100 scale.
  - Headline: `pass@1[avg-of-3]/all_if_pass`, each task's pass rate over 3 repeats, averaged over tasks
    with equal weight.
  - Also `if_partial`, the passed fraction of IF items.
  - `coverage/*` counts masked rollouts.
- **Reference score:** 62.12 on `pass@1[avg-of-3]/all_if_pass`, tolerance ±7.7 points (95%).
  - Policy and judge: GLM 5.3 (`nvidia/zai-org/glm-5.3`).
  - Policy settings: temperature 1.0, top_p 1.0, 16384 output tokens, no reasoning-effort setting, 16 rollouts
    in parallel.
  - The tolerance is t(0.975, 6) × the pooled `std_dev_across_runs` (3.83) × √(2/3). Two more runs with the same
    settings scored 65.15 and 63.64.
- **Judge:** `nvidia/zai-org/glm-5.3` by default, at temperature 0, with 32768 output tokens. Each of the
  two resources servers allows 2 concurrent judge calls, so a run makes at most 4. Set `judge_base_url` and
  `judge_api_key`. A different judge makes the scores incomparable.
- **Failures:**
  - A missing or truncated answer scores 0.
  - Infrastructure, capture and judge failures are masked, with `failure_kind` and `failure_reason`, and
    leave the score.
  - Run with `route_failures_to_sidecar: true`, so a failed `/run` lands in `<output>_failures.jsonl` instead
    of stopping the run.
  - A run with masked rows or failures is incomplete.
- **Timeouts:**
  - No episode deadline: the step caps end an episode.
  - Each environment's cleanup times out after 180 s.
  - A tool sandbox has 600 s to start and 300 s to install its requirements, and its runner has 60 s to load.
  - Each tool call times out after 60 s.
  - A sandbox lives at most 3600 s.
- **Seeds:** none are set, so runs are stochastic.

## Running

1. Set `mlflow_tracking_uri` and `mlflow_tracking_token` for Gym's GitLab dataset registry, and install
   Gym's `mlflow` extra.
2. Prepare the rows:

   ```bash
   gym eval prepare --benchmark partner_if
   ```
3. Add the judge endpoint and a sandbox provider config, and run the benchmark:

   ```bash
   gym eval run --benchmark partner_if --model-type vllm_model \
     --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
     --output <run dir>/rollouts.jsonl
   ```

For the exit-check evidence, turn on model-call capture with `++observability_enabled=true
++model_call_capture_dir=<absolute dir>`, then run `gym eval health-check <run dir>`. With OpenTelemetry on,
add `sandbox` to the span groups to trace the tool sandboxes.
