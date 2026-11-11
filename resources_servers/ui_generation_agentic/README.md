# ui_generation_agentic

An agentic UI-generation environment: a coding harness writes **one front-end UI file from scratch** inside a
sandbox that holds a ready-made project for one of three stacks, and this resources server grades that file. Each
harness has its own composition in `environments/ui_generation_agentic/` on top of the harness-neutral `base.yaml`:

| Composition | Harness | Runtime in the sandbox |
|---|---|---|
| `opencode.yaml` | `opencode_agent` (OpenCode 1.17.11) | baked into the agent images |
| `claude_code.yaml` | `claude_code_agent` (Claude Code 2.1.108) | uploaded at seed by the agent server (downloaded once on the Gym host) |
| `hermes.yaml` | `hermes_agent` (Hermes) | uploaded at seed (pinned CPython + Hermes, built once on the Gym host) |
| `codex.yaml` | `codex_agent` (Codex 0.144.4) | baked into the agent images when they are built with `WITH_CODEX=1` |

All of them call the policy Gym model server from inside the sandbox (OpenCode and Hermes through chat completions,
Codex through Responses, Claude Code through the Anthropic Messages endpoint `/v1/messages`), so any model Gym serves
can drive them.
Harnesses whose runtime must be installed from the internet inside the sandbox, or that need root in it, don't fit
these images as they are.

| Stack | Project in the sandbox | The agent writes |
|---|---|---|
| `ui-next` | Next.js 15 (app router), React 19, TypeScript, Tailwind 3, at `/app` | `/app/app/sample.tsx` (default-exported client component) |
| `ui-vite` | Vite 7, React 19, TypeScript, Tailwind 3, at `/app` | `/app/src/App.tsx` (default-exported `App`) |
| `ui-static` | nginx web root `/usr/share/nginx/html` | `index.html` (one self-contained page) |

There is no reference answer: the judge grades the file against the prompt alone.

## Episode

`single_agent_turn` (taskset routing) runs each episode:

1. **Seed.** The server starts the task's stack image (`LentSandboxes`, one per stack) and lends it to the agent. With
   `agent_network_mode: restricted` the sandbox can reach only the policy model server and Google Fonts
   (OpenSandbox egress policy, built with `restricted_network_policy`).
2. **Agent.** The harness borrows the sandbox and creates the entry file. No harness needs internet in the sandbox:
   OpenCode's runtime (and Codex's, with `WITH_CODEX=1`) is baked into the image, and Claude Code's and Hermes's
   runtimes are uploaded at seed.
3. **Verify** (in the same sandbox, after the agent session closed):
   - **G1** stops every process the agent left (dev servers, watchers), keeping the sandbox runtime;
   - **G2** snapshots the entry file once (regular file, size cap); only the snapshot is built and judged;
   - **G3** checks the image's root-owned read-only template and toolchain against `/opt/ui-gen/MANIFEST.sha256`
     (and an optional pinned SHA-256 of that manifest);
   - **G4** builds the snapshot in a fresh copy of the template with `scripts/ui_gen_verify.sh`, uploaded at every
     verify, so nothing the agent changed in its workspace reaches the build;
   - **G5** (`guard_mode: full` only) runs the build as the unprivileged `ui-verify` user;
   - **G6** takes the verdict from the build's exit status and the judge only.
4. **Judge.** An LLM judge (rubric `prompts/judge.yaml`, adapted from `resources_servers/ui_generation`) scores
   `prompt_adherence` and `quality` from 0 to 4.
5. **Close.** The sandbox is stopped.

`reward = build × quality / 4` (`reward_mode: gated`; `build` and `judge` are the alternatives). A missing,
symlinked or oversized entry file, a failed or timed-out build, or a changed template scores 0. A sandbox or judge
failure masks the sample.

`guard_mode: detection_only` is for sandboxes whose commands cannot switch user (for example a self-hosted
OpenSandbox server, where commands run as the image's user): G1 then spares the runtime and the verifier's own
processes, and G5 is skipped. Every rollout records `guard_mode`, `agent_network_mode` and `egress_enforcement`.

## Images: build and publish your own

The three agent images are defined by `docker/<stack>.Dockerfile`. Build them, push them to a registry your sandbox
provider can pull from, and point the config at the pushed digests:

```bash
IMAGE_PREFIX=<registry>/<namespace>/ui-gen-agent resources_servers/ui_generation_agentic/docker/build_images.sh --push
export UI_GEN_NEXT_IMAGE=<registry>/<namespace>/ui-gen-agent/ui-next@sha256:...
export UI_GEN_VITE_IMAGE=<registry>/<namespace>/ui-gen-agent/ui-vite@sha256:...
export UI_GEN_STATIC_IMAGE=<registry>/<namespace>/ui-gen-agent/ui-static@sha256:...
# optional manifest pins (guard G3): docker run --rm --entrypoint "" <image> sha256sum /opt/ui-gen/MANIFEST.sha256
export UI_GEN_NEXT_MANIFEST_SHA256=...   # and _VITE_, _STATIC_
```

The images contain only the stack projects (public npm packages), the `run-sample` script, OpenCode 1.17.11,
ripgrep 15.1.0 and OpenCode's plugin dependencies; `WITH_CODEX=1` also bakes in Codex's runtime (Node 22.19.0 and
`@openai/codex` 0.144.4, Apache-2.0) at `/opt/ui-gen/runtime/codex`, which `codex.yaml` needs. Check any image you
build (no data, no credentials) before you push it anywhere others can pull.

## Run

Supply the policy and judge models in `env.yaml` (`policy_base_url`, `policy_api_key`, `policy_model_name`,
`judge_base_url`, `judge_api_key`, `judge_model_name`); use a judge from a different model family than the policy.

The harness calls the policy model server from inside the sandbox, so that server must listen on an address the
sandboxes can reach. With a self-hosted OpenSandbox server on the same machine (Docker runtime), bind only the model
server to the Docker bridge:

```bash
++policy_model.responses_api_models.vllm_model.host=172.17.0.1
```

Otherwise use `++use_absolute_ip=true '++absolute_ip_server_types=[responses_api_models]'`.

Prepare tasks outside the repository and point the `train` dataset at them with a config overlay (the dataset list
cannot be overridden element-wise on the command line):

```bash
python environments/ui_generation_agentic/prepare.py --input <source.jsonl> --output <tasks.jsonl>
cat > overlay.yaml <<EOF
ui_generation_agentic_resources_server:
  resources_servers:
    ui_generation_agentic:
      datasets:
        - {name: train, type: train, jsonl_fpath: <tasks.jsonl>, taskset: "ui_generation_agentic:train", num_repeats: 1, license: <license>}
EOF
gym eval run \
  --config environments/ui_generation_agentic/opencode.yaml \
  --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
  --model-type vllm_model --config overlay.yaml \
  --split train --output <out>/rollouts.jsonl --num-repeats 1 --concurrency 4 \
  ++observability_enabled=true ++model_call_capture_dir=<out>/model-calls hydra.run.dir=<out>/hydra
```

`hydra.run.dir` keeps Hydra's run log, which holds the resolved config (endpoints and model names), with the run's
other outputs instead of in `outputs/` under the working directory. For a quick smoke test without your own tasks,
leave out the overlay and use `--split example` (the three synthetic tasks).

For another harness, use its composition instead of `opencode.yaml`:

- `claude_code.yaml`: the first seed downloads the pinned Claude Code binary to `~/.cache/nemo_gym/claude` on the Gym
  host, so that host needs HTTPS access once (or set `sandbox_local_executable_path` and `sandbox_executable_sha256`
  to a binary you supply).
- `codex.yaml`: build the images with `WITH_CODEX=1`. Codex has no per-call output cap of its own and refuses a
  request `max_output_tokens`, so cap it on the model server, for example
  `'++policy_model.responses_api_models.vllm_model.extra_body={max_tokens: 32768}'`.
- `hermes.yaml`: the agent server builds the Hermes runtime once on the Gym host (`uv` plus GitHub and PyPI access,
  or a filled cache). In sandbox mode the run's result file is writable by commands the model runs, so the Hermes
  README advises against using these rollouts as RL training data.

Set `UI_GEN_EGRESS_ENFORCEMENT` to what your OpenSandbox deployment enforces (`dns+nft` or `dns`; check a literal-IP
request from a restricted sandbox) so rollouts record it.

### Output cap and building in steps

`ui_generation_agentic_max_output_tokens` (default 32768; override per run with
`++ui_generation_agentic_max_output_tokens=<n>`) is the output cap of OpenCode and Hermes for each model reply:
reasoning plus one tool call. Claude Code keeps its own CLI default (32000), which a `settings` file can change through
`CLAUDE_CODE_MAX_OUTPUT_TOKENS`; Codex takes its cap from the model server (above). A reply cut off at the cap is lost
and the harness stops the run, which counts as a model failure: the file as it stands is graded, so an episode cut off
before its first write scores 0. Policies that write the whole page in one tool call can hit the cap, so the task
prompt (`prompts/task_prompt.txt`, applied by `prepare.py`) asks every harness, by default, to build the file in steps:
a short working version first, then edits of at most a few hundred lines each. It is part of the user message, so no
harness setting is needed. To prepare tasks without it, pass `prepare.py --prompt-template` a copy of the template
without that bullet.

## Rollout fields

`build_passed`, `build_exit_code`, `build_seconds`, `build_log_tail`, `final_code` (the graded file),
`final_code_sha256`, `tamper_detected`, `tamper_details`, `judge_quality`, `judge_prompt_adherence`, `judge_verdict`,
`judge_reasoning`, `reward_components`, `failure_reason`, `guard_mode`, `agent_network_mode`, `egress_enforcement`,
`agent_stop_reason`, plus the harness trajectory (`response`, `ng_trajectory`) and, with observability on, the
captured model calls.

`agent_stop_reason` is the harness's run status when the run did not complete (`incomplete`, `failed`), with the cause
when the harness reports one. OpenCode reports `incomplete:max_output_tokens` only for continuation sessions; for a
single-activation run, `incomplete` covers both an output-cap cut-off and the agent timeout. The per-stack metric
`agent_incomplete` is the share of unmasked rollouts that stopped incomplete.

## Data

No task data ships with this server: `data/example.jsonl` holds three hand-written tasks. `prepare.py` reads only the
first user prompt and the stack of each source row (any assistant answer is ignored) and writes to a path the runner
chooses.

## Licensing

Code: Apache 2.0. Data: supplied by the runner.
