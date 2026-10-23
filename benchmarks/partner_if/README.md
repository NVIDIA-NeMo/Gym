# partner_if

Internal tasks built from instruction-following failures seen in production agent deployments. Each
task checks whether the model follows its instructions, graded against versioned rubrics by the
[partner_if](../../resources_servers/partner_if/README.md) resources server.

## Data

- **Source:** NVIDIA-internal rows in Gym's GitLab dataset registry: dataset `partner_if`, version
  `2.1.4`, file `partner_if.jsonl`. `prepare.py` pins the file's sha256 and refuses any other content. The
  rows are not in this repository.
- **License:** internal.
- **Size:** 25 tasks with 86 IF rubric items, plus 4 completion items: 17 text tasks, 2 search-tool tasks,
  3 build-tool tasks and 3 SWE tasks. There is one split, `benchmark`.
- **Provenance:** 21 tasks reproduce a reported instruction-following failure and 4 reconstruct one. The
  tool tasks' tools are simulations written for this benchmark, not real services. The SWE tasks' repositories
  and images come from Scale-SWE.
- **Rows:** each row holds the prompt (`responses_create_params`) and the grading contract
  (`verifier_metadata.partner_if`). A tool task's row also holds its tool backend (`tool_backend`): the
  simulated tools' code, fixture and pinned requirements. A SWE task's row also holds its Scale-SWE instance:
  the image, working directory, setup commands and tests.

## Task contract

- **Harness:** the policy model runs in Gym's `simple_agent`, and in OpenCode (`opencode_agent`) on the SWE
  tasks.
- **Routing:** `partner_if_dispatcher` routes each task to its environment, so all 25 tasks share one
  dataset and one score. Gym aggregates metrics per environment server, so without it the four task
  families would be four benchmarks with separate scores.
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
  - Set `context_window` on `partner_if_swe_agent` (262144 by default) to the policy model's: OpenCode compacts
    the conversation as it nears it.
  - The judge grades the visible trajectory, never reasoning: every agent message and tool call in order, with
    the arguments in full, and the tool outputs. An output longer than 500 characters keeps its first and last 250,
    with a `[… N characters omitted …]` marker between them. The SWE rubrics grade the agent's messages and tool
    calls, and with whole outputs a long agent run would not fit the judge's input.
- **Sandbox:** the tool and SWE tasks need OpenSandbox as configured: a provider with service endpoints and
  network policies.
  - Each tool rollout gets a fresh sandbox that holds only the row's tool backend: `python:3.13-slim`, one CPU
    and 1 GiB. By default the sandbox can reach only the package index, to install the backend's pinned
    requirements. With a prebuilt image, set `network: none`.
  - Each SWE rollout gets a sandbox of the row's Scale-SWE image: limits of 4 CPUs and 16 GiB, requests of 1 CPU
    and 4 GiB, and 20 GiB of disk. It needs egress to GitHub, to install OpenCode, and to the policy model
    server. Its label `nemo.nvidia.com/s3-hostpath: "false"` keeps the cluster's S3 mount out of the agent's
    shell.
  - OpenCode calls the policy model server from inside the sandbox, so run with `++use_absolute_ip=true`, as other
    sandboxed-agent benchmarks do: the servers then listen on the host's address instead of loopback. Where the
    host's name does not resolve to an address the sandboxes can reach, also set `+default_host=<address>`.
  - Scale-SWE's own tests then run as a diagnostic, in up to two more sandboxes of the same image: an
    inconclusive result is retried once. Their verdict is kept in `task_outcome` and never changes the reward.
- **Sampling:** the benchmark fixes the policy's sampling for every task: temperature 1.0, top_p 1.0 and no
  output cap. `config.yaml` pins it on `partner_if_policy_model`, a copy of the run's `vllm_model` policy
  server that every agent calls.
  - No output cap means the pin sends a null cap, which also replaces the per-call cap OpenCode sends. On a vLLM
    endpoint each reply may then use the rest of the model's context window.
  - An endpoint that applies a small default when the cap is null (some hosted OpenAI-compatible APIs default to
    4096) would cut replies short. Give it an explicit cap the size of the context window:
    `++partner_if_policy_model.responses_api_models.vllm_model.sampling_overrides.max_tokens=<window>`.
  - An endpoint that ends long requests needs a longer per-request timeout, since a reply at maximum reasoning
    effort can take more than 6 minutes. Some hosted gateways stop a request at 360 s unless its body sets
    `timeout`. The reference runs set `extra_body.timeout: 3600` on the policy and judge model servers.
  - Runs set no request parameters, such as `--temperature`, `--top-p`, `--max-output-tokens` or a reasoning
    effort: `opencode_agent` refuses them, so every SWE task would fail. Set model settings such as reasoning
    effort on the policy model server instead.
- **Prompts:** the rows are prepared, so no template applies. Rows set no `temperature`, `top_p` or
  `max_output_tokens`; some set `tool_choice`.

## Scoring

- **Reward:** a rollout scores 1 when every IF rubric item passes, and 0 otherwise. Code checks
  (partner_if's built-in and fixed checks) and an LLM judge grade the items. Completion items are
  reported but never change the reward.
- **Provisional rubrics:** 22 of the 25 contracts in 2.1.4 are unsigned (`audit_signoff: false`); the 3 SWE
  contracts are signed. The config sets `allow_provisional_rubrics: true`, so the unsigned rows are scored and
  marked `provisional`.
- **Metrics:** reported on a 0–100 scale.
  - Headline: `pass@1[avg-of-3]/all_if_pass`, each task's pass rate over 3 repeats, averaged over tasks
    with equal weight.
  - Also `if_partial`, the passed fraction of IF items.
  - `coverage/*` counts masked rollouts.
- **Reference score:** not yet measured on dataset 2.1.4. Dataset 2.1.0 scored 61.33 on
  `pass@1[avg-of-3]/all_if_pass`, tolerance ±8.5 points (95%).
  - Policy and judge: GLM 5.3 (`nvidia/zai-org/glm-5.3`).
  - Policy settings, all on the model server: temperature 1.0 and top_p 1.0 (the benchmark's pin), maximum
    reasoning effort, 262,144 output tokens, 16 rollouts in parallel.
  - The tolerance is t(0.975, 6) × the pooled `std_dev_across_runs` (4.24) × √(2/3). Two more runs with the same
    settings scored 58.67 and 57.33.
- **Judge:** `nvidia/zai-org/glm-5.3` by default, at temperature 0, with 32768 output tokens. Each of the
  three resources servers allows 2 concurrent judge calls, so a run makes at most 6. Set `judge_base_url` and
  `judge_api_key`. A different judge makes the scores incomparable.
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
  - Each environment's cleanup times out after 180 s.
  - A tool sandbox has 600 s to start and 300 s to install its requirements, and its runner has 60 s to load.
  - Each tool call times out after 60 s.
  - A tool sandbox lives at most 3600 s.
- **Cost:** a run is 75 rollouts. Each of the 9 SWE rollouts uses an OpenCode sandbox and up to two test
  sandboxes of 4 CPUs and 16 GiB, and each of the 15 tool rollouts uses one small sandbox.
- **Seeds:** none are set, so runs are stochastic.

## Running

1. Set `mlflow_tracking_uri` and `mlflow_tracking_token` for Gym's GitLab dataset registry, and install
   Gym's `mlflow` extra.
2. Prepare the rows:

   ```bash
   gym eval prepare --benchmark partner_if
   ```
3. Add the judge endpoint and a sandbox provider config, and run the benchmark. The policy server must be
   `vllm_model`, whose `sampling_overrides` carry the benchmark's sampling:

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
