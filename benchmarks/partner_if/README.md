# partner_if

Internal tasks built from instruction-following failures seen in production agent deployments. Each
task checks whether the model follows its instructions, graded against versioned rubrics by the
[partner_if](../../resources_servers/partner_if/README.md) resources server.

## Data

- **Source:** NVIDIA-internal rows in Gym's GitLab dataset registry: dataset `partner_if`, version
  `2.0.0`, file `partner_if.jsonl`. `prepare.py` pins the file's sha256 and refuses any other content. The
  rows are not in this repository.
- **License:** internal.
- **Size:** 22 tasks with 83 IF rubric items: 17 text tasks, 2 search-tool tasks and 3 build-tool tasks.
  There is one split, `benchmark`.
- **Rows:** each row holds the prompt (`responses_create_params`) and the grading contract
  (`verifier_metadata.partner_if`). A tool task's row also holds its tool backend (`tool_backend`), the
  simulated tools' code and fixture.

## Task contract

- **Text tasks:** one reply from the policy model. No tools run, and the grader sees any tool calls only
  as part of the reply.
- **Search-tool tasks:** up to 30 steps with the row's simulated search and document tools.
- **Build-tool tasks:** up to 15 steps with the row's simulated project tools. These tasks don't require
  a final text answer.
- **Sandbox:** the tool tasks need a sandbox provider with service endpoints, such as OpenSandbox. Each
  rollout gets a fresh sandbox that holds only the row's tool backend, with one CPU and 1 GiB. By default
  the sandbox can reach only the package index, to install the backend's pinned requirements. With a
  prebuilt image, set `network: none`.
- **Prompts:** the rows are prepared, so no template applies. Sampling settings come from each row's
  `responses_create_params` (some rows set `tool_choice` or `temperature`) and otherwise from the policy
  model's defaults.

## Scoring

- **Reward:** a rollout scores 1 when every IF rubric item passes, and 0 otherwise. Code checks
  (partner_if's built-in and fixed checks) and an LLM judge grade the items. Completion items are
  reported but never change the reward.
- **Headline metric:** `pass@1[avg-of-3]/all_if_pass`. Each task's pass rate over 3 repeats, averaged
  over tasks with equal weight.
- **Judge:** `nvidia/zai-org/glm-5.3` by default, at temperature 0, with 32768 output tokens and
  concurrency 2. Set `judge_base_url` and `judge_api_key`. A different judge makes the scores
  incomparable.
- **Failures:**
  - A missing or truncated answer scores 0.
  - Infrastructure, capture and judge failures are masked, with `failure_kind` and `failure_reason`, and
    leave the score. A run with masked rows is incomplete.
- **Timeouts:** each environment's cleanup times out after 180 s, and a tool sandbox lives at most
  3600 s.
- **Seeds:** none are set, so runs are stochastic.

## Running

Set `mlflow_tracking_uri` and `mlflow_tracking_token` for Gym's GitLab dataset registry and install Gym's
`mlflow` extra. Then add the judge endpoint and a sandbox provider config, and run the benchmark:

```bash
gym eval run --benchmark partner_if --model-type vllm_model \
  --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
  --output <run dir>/rollouts.jsonl
```

For the exit-check evidence, turn on model-call capture with `++observability_enabled=true
++model_call_capture_dir=<absolute dir>`, then run `gym eval health-check <run dir>`. With OpenTelemetry on,
add `sandbox` to the span groups to trace the tool sandboxes.
