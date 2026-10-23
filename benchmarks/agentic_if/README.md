# Agentic IF

Measures whether a coding agent follows instructions while it completes a real repository or terminal task. OpenCode
works each task in the task's own sandbox. The task's verifier scores the outcome, and an LLM judge grades the
recorded session against each instruction. OpenCode is part of what is measured: a score describes a model running
in OpenCode 1.17.11 with this configuration, not the model alone.

## Measurement contract

### Provenance

- 586 tasks. Each pairs a real repository task (Scale-SWE) or terminal task (Terminal-Bench format) with synthetic
  instructions about how to do the work. Tasks that proved problematic or too easy were filtered out.
- The task rows are internal. They live in Gym's dataset registry as dataset `agentic_if`, version `2.0.0`, with a
  second artifact that holds the `tests/` files of the terminal tasks. `prepare.py` pins both by SHA-256. Data
  license: internal. Code: Apache-2.0. The offline OpenCode bundle (see Settings) is OpenCode (MIT), ripgrep (MIT or
  Unlicense) and the npm packages that `opencode/package-lock.json` pins (MIT, ISC, Apache-2.0, BSD-3-Clause).
- The `example` split of `resources_servers/agentic_if` holds 5 public tasks whose synthetic instructions exercise the
  same features.

### Task contract

- Model-visible input: each row's literal system prompt and one user message, as prepared rows
  (`prompt_source: prepared`), plus the tools the row selects. Reference solutions and task tests are never visible.
- Tools: a per-row subset of OpenCode's built-in tools, with the descriptions and parameter descriptions the row sets.
  Web fetch, web search and skills are denied.
- Sandbox: the task's own image. The `scale_swe` or `terminal_bench_2_1` resources server starts it; the `agentic_if`
  resources server borrows it for the episode and lends it to the agent.
- Harness: `opencode_agent` runs OpenCode in that sandbox through `single_agent_turn`, with two plugins the resources
  server stages. A profile plugin applies the row's tool argument maps, read-before-edit guards and planted tool
  failures, and writes receipts. A boundary plugin applies the system prompt, tool selection and tool descriptions,
  and records every model call. All model calls go through Gym's model server.
- What the verifier reads: OpenCode's session exports, the plugins' receipts and call records, and the sandbox.
  Terminal tasks run their tests in it; repository tasks extract the patch and run the held-out tests.

### Scoring and run conditions

- `reward`: the task verifier's result, 0 or 1; higher is better. Instruction following does not change it.
- Instruction following, over rollouts the judge graded:
  - `mean/if_all`: the share that pass every applicable check;
  - `mean/if_frac`: the share of applicable checks passed, averaged;
  - `mean/task_and_if`: the share that pass both the task and every check.
- `if/graded`, `if/not_applicable` and `if/error` count rollouts; only graded rollouts enter the rates.
- Judge: `resources_servers/agentic_if/if_grading/data/judge.yaml` (`v2-rule-judge-12`), which asks
  `nvidia/zai-org/glm-5.3` at `https://inference-api.nvidia.com/v1` with temperature 0.6 and low reasoning effort,
  up to 4 attempts per check. Its calls are not among the rollout's captured model calls, which belong to the agent.
  Each rollout reports them in `judge_calls` (each call's reply ID and token usage) and `judge_usage` (totals), and
  each vote is saved with the episode under the resources server's `artifacts_dir`.
- Sampling: each row's temperature 1.0 and top_p 0.95. Receipts bind every request setting, so leave `--temperature`,
  `--top-p` and `--max-output-tokens` unset.
- Budgets: a 131,072-token context window and output limit, and no step limit.
- Timeouts: 3 hours for the OpenCode run and 5 hours for the whole episode (setup, run, verification and cleanup);
  terminal tests get 30 minutes.
- Concurrency: 8 episodes at a time, set by `opencode_agent`'s `concurrency` and the environment server's
  `max_concurrent_episodes`; raise both together to run faster. At 8, a full run can take days.
- Repeats: 1. Seeds are not supported; rollouts are stochastic.
- Failures:
  - An infrastructure failure is masked (`mask_sample: true` with a `failure_kind`), not scored.
  - An OpenCode run that times out or fills its context is scored.
  - A check whose judge call fails on every attempt is recorded as `judge_failed`;
    `gym eval reverify --judge-failed-only` grades those again.
- Re-verification (`gym eval reverify`) grades instruction following again from the episode saved under the resources
  server's `artifacts_dir` and keeps that episode's task verdict, so it needs that directory but not the sandbox.

## Settings

- `policy_base_url`, `policy_api_key` and `policy_model_name`, in `env.yaml` or as overrides. Our measurements set
  `propagate_context_overflow_errors: true` on a `vllm_model` policy server, so OpenCode sees context overflows as
  errors rather than empty replies.
- `judge_api_key`, and `judge_base_url` for another endpoint.
- `mlflow_tracking_uri` (`<your-gitlab-mlflow-tracking-uri>`, Gym's GitLab dataset registry) and
  `mlflow_tracking_token` for preparation, with the `mlflow` extra installed.
- The sandbox provider's settings, for example `OPENSANDBOX_DOMAIN` and `OPENSANDBOX_API_KEY`. With a remote provider,
  set `use_absolute_ip: true` so OpenCode in the sandbox can reach Gym's model server.
- Sandboxes without internet access need OpenCode's offline bundle (x86_64), which our measurements always pin;
  without it, `opencode_agent` installs its standard OpenCode 1.17.11 build from GitHub. Set these `opencode_agent`
  fields on `agentic_if_opencode_agent`, each with its SHA-256 (see `responses_api_agents/opencode_agent/README.md`):
  - `sandbox_executable_path`: `package/bin/opencode` of
    https://registry.npmjs.org/opencode-linux-x64/-/opencode-linux-x64-1.17.11.tgz, SHA-256
    `0254a429cd0e6cf0ba53fc01672cf98e4a8dc728f7fa94be88f1e4b3645e6ded`.
  - `sandbox_ripgrep_path`: `ripgrep-15.1.0-x86_64-unknown-linux-musl/rg` of
    https://github.com/BurntSushi/ripgrep/releases/download/15.1.0/ripgrep-15.1.0-x86_64-unknown-linux-musl.tar.gz,
    SHA-256 `ebeaf56f8a25e102e9419933423738b3a2a613a444fd749d695e15eba53f71f2`. OpenCode otherwise downloads it on
    first use.
  - `sandbox_plugin_dependencies_path`: the `@opencode-ai/plugin@1.17.11` install that OpenCode otherwise runs at
    startup, built from `opencode/` (its lockfile pins every package), SHA-256
    `77b03eba45c8d7fe3d0a58bfc215f6fb99a1db4dd18a7e3d91fd29981fddafb1` with Node 24.15.0, npm 11.12.1, GNU tar 1.34
    and GNU gzip 1.10 (another gzip build changes only the compressed bytes):

    ```bash
    cp -r benchmarks/agentic_if/opencode opencode  # npm records the folder name, so keep it
    (cd opencode && npm ci --ignore-scripts --no-audit --no-fund)
    tar -C opencode --sort=name --mtime=@1792627200 --owner=0 --group=0 --numeric-owner --mode=u+rwX,go+rX,go-w \
        --hard-dereference --format=gnu -cf - package.json package-lock.json node_modules \
        | gzip -n -9 > opencode-1.17.11-plugin-dependencies.tar.gz
    ```

## Prepare and run

```bash
gym eval prepare --benchmark agentic_if
gym eval run --benchmark agentic_if --split benchmark --model-type vllm_model \
    --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
    --output results/agentic_if.jsonl
```

`resources_servers/agentic_if/README.md` shows how to run the public `example` split.

## Limitations

- The task rows are internal: preparation needs read access to the registry project.
- Scores depend on OpenCode, the judge model and the task images, which are pinned by version or tag, not by content.
  A judge model update can move instruction-following scores; `judge.yaml` records the version these scores used.
