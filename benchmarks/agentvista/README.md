# AgentVista evaluation through NeMo Gym

Two evaluation pipelines share the original AgentVista tasks, tools and answer judge:

| Gym benchmark | Execution harness | Factory benchmark |
|---|---|---|
| `agentvista_opencode` | OpenCode CLI 1.17.11 with the original tools over MCP | `agentvista-opencode` |
| `agentvista_default` | Original `engine.api.api_processors.process_single_sample` greedy loop | `agentvista-default` |

Gym owns all custom integration code: `responses_api_agents/agentvista_agent/runtime/`
contains both workers, the MCP transport, source/image bindings, original judge
adapter and transcript handling. `prepare_dataset.py` materializes the public
Parquet release; `prepare.py` creates Gym rows without actor or judge requests.
Factory only composes the driver, serving service and collection configuration.
There is no dependency on `agentvista_harness/`, AgentVista's `integrations/`, or
SaaS Bench Python modules.

The original benchmark implementation is an external dependency pinned to public
[AgentVista commit a3385d51](https://github.com/hkust-nlp/AgentVista/tree/a3385d51d0c3dd0c8501350a08d52065e3e94ab4).
Its code is not vendored or relicensed here. `source-pin.json` records SHA256 for
every execution, tool, prompt and judge source file; `setup.py` fetches that revision
or verifies an existing checkout without changing it. The evaluation image embeds
those verified source files. This does not require the separate AgentVista integration PR.

## Provision a runtime

From the Gym PR checkout, use workspace paths for all artifacts and caches:

```bash
python benchmarks/agentvista/setup.py --source "$WORKSPACE/agentvista-source"
python benchmarks/agentvista/prepare_dataset.py --output "$WORKSPACE/agentvista-data" \
  --revision 61de767ad8c72e258ac71a0aef952b89ca920a16
```

The prepared release contains 209 tasks and 307 embedded images. Original records
and image bytes are retained; stable IDs and `solution`/`category` aliases adapt the
public Parquet schema. Source/image hashes bind preparation to execution. The
upstream split name `train` is not a held-out evaluation protocol.

For a host worker, install `responses_api_agents/agentvista_agent/requirements-worker.txt`
in a separate environment. The default harness requires `qwen-vl-utils` and Torch
for its original image loader. OpenCode additionally requires CLI 1.17.11. Supply
`AGENTVISTA_ROOT`, `AGENTVISTA_DATASET`, `AGENTVISTA_IMAGES`,
`AGENTVISTA_OUTPUT_ROOT`, `AGENTVISTA_WORKER_PYTHON`, and, for OpenCode,
`AGENTVISTA_OPENCODE`. `PYTHONPATH` must contain the Gym PR checkout; the dispatcher
passes the Gym and upstream source paths to each isolated worker.

`runtime-image/Dockerfile` under the agent directory builds the shared worker
image from an approved Gym base. Supply a verified CLI binary as `opencode` in
the build context. `runtime-image/build.py` builds a SQSH directly with Enroot on
SVG, using explicit base image, standalone Python distribution, CLI, uv and source
inputs. It verifies the source and CLI, imports the worker dependencies, and writes
an image digest receipt. An import check does not establish live benchmark success.

The shared SVG image has been built and import-checked at
`/lustre/fsw/portfolios/nemotron/users/pritamb/gameplay_home/vlm_eval/saas_bench_assets/serving_containers/agentvista-eval-default-opencode-1.17.11-svg-amd64.sqsh`.
Its SHA256 is `d97355314b764b47fd29c2acd1ba7bef55af24ed03ac0bcfdbd3c27739d4beb5`;
the adjacent `.sqsh.json` records source, CLI and dependency provenance. Project
storage access is required. Factory recipes accept `AGENTVISTA_SVG_ROOT` for a
separately staged copy of the same assets.

[SVG smoke evidence](smoke-results/svg-20261111.json) records one bounded task for
each harness with local Super GA and hosted Kimi K3. All four completed execution
and original answer judging with unmasked zero rewards. These are pipeline checks;
they do not establish benchmark accuracy or published-score parity.

SVG rebuild example (uses a new output filename):

```bash
base=/lustre/fsw/portfolios/nemotron/users/pritamb/gameplay_home/vlm_eval
python responses_api_agents/agentvista_agent/runtime-image/build.py \
  --base-image /lustre/fsw/portfolios/nemotron/projects/nemotron_n3_post/eval/containers/gym/nemo-gym_dev-2026-07-05T17-22-34b47edf_amd64.sqsh \
  --python-distribution "$base/saas_bench_assets/runtime/python_references/20261102T012704Z/cpython313" \
  --opencode "$base/.runtime/opencode-1.17.11/opencode" \
  --uv "$base/.runtime/saas_eval_dual_harness/tools/uv-x86_64-unknown-linux-gnu/uv" \
  --source "$base/agentvista" --work-dir "$base/.runtime/agentvista-image-build" \
  --output "$base/saas_bench_assets/serving_containers/agentvista-eval-default-opencode-1.17.11-svg-amd64-rebuild.sqsh"
```

The builder does not submit a Slurm job. Run it in a suitable existing allocation.
It refuses an existing output and uses workspace caches without mounting user homes.

## Configure providers

Keep credentials in a private environment file, outside Git:

- `SERPAPI_KEY`: original Serper search and Google Lens tools.
- `JINA_API_KEY`, `IMGBB_API_KEY`: optional reader/upload backends.
- `AGENTVISTA_VISIT_API_KEY`, `AGENTVISTA_VISIT_API_ENDPOINT`,
  `AGENTVISTA_VISIT_MODEL_NAME`: fixed webpage summarizer; endpoint is a full
  `/chat/completions` URL.
- `VERIFIER_API_KEY`, `VERIFIER_END_POINT`, `VERIFIER_MODEL_NAME`: original answer
  judge; endpoint is the OpenAI-compatible base URL ending in `/v1`.
- `INFERENCE_API_KEY`: hosted Kimi policy credential in Factory's API recipes.

The actor always calls Gym's rollout-correlated model proxy. The default worker
sets the original runner's `REASONING_*` endpoint/model to that proxy and removes
fallback policy endpoints. The fixed visit summarizer is configured in memory.
The judge receives reference answers only after execution; the OpenCode actor
process receives neither judge credentials nor reference answers.

Rewards use the original binary answer-judge prompt and extraction. They are
**LLM-judged**, not deterministic verification. A completed wrong answer remains
an unmasked zero. Clean output-token exhaustion is reported as
`execution_health.termination=output_budget_exhausted` and sent to the original
judge, including an empty prediction when no answer was produced. Timeouts,
process/native errors and capture/judge failures retain distinct failure kinds
and remain masked and unscored;
the adapter makes the original judge's otherwise swallowed errors visible.

## Run evaluation

Choose one benchmark and keep its prepared input and output directories isolated:

```bash
benchmark=agentvista_default  # or agentvista_opencode
common=("+agentvista_dataset_path=$OUTPUT/input.jsonl" \
        "+agentvista_output_root=$OUTPUT/episodes" \
        "+policy_base_url=$POLICY_BASE_URL" "+policy_model_name=$POLICY_MODEL" \
        '+policy_api_key=${oc.env:AGENTVISTA_POLICY_API_KEY}')
gym eval prepare --benchmark "$benchmark" "${common[@]}" \
  "+prepare_script_args.output_fpath=$OUTPUT/input.jsonl"
gym eval run --benchmark "$benchmark" --model-type vllm_model \
  --agent agentvista_agent --split benchmark --output "$OUTPUT/rollouts.jsonl" \
  "${common[@]}" +observability_enabled=true
```

For OpenCode, add
`++policy_model.responses_api_models.vllm_model.entrypoint=agentvista_opencode.py`
to the run command; the Factory OpenCode binding supplies it automatically.
The old `agentvista` Gym benchmark keeps its OpenCode behavior.

Default-harness limits match `infer.py`: 16 turns, 16 images, 65,536 estimated
context tokens and 8,192 completion tokens. OpenCode defaults to 150 steps and
a 5,400-second actor timeout; OpenCode owns its history and compaction. These
protocols are different evaluation identities; do not combine their scores or
claim original published-score parity. Both retain original trajectories and Gym
model-call captures under the run directory. `harness` identifies the selected path.

## Validation and training compatibility

```bash
AGENTVISTA_ROOT=/path/to/verified/source \
  pytest responses_api_agents/agentvista_agent/tests
```

Worker/tool tests require the worker dependencies; the CLI fixture additionally
requires `AGENTVISTA_TEST_OPENCODE` and local TCP/Jupyter socket access. The
original-loop fixture tests image input, original tool feedback and valid zero,
valid one and judge-failure behavior without hosted requests. A real model smoke
must separately exercise both harnesses and inspect actor and judge artifacts.
`benchmarks/agentvista/smoke.sh` runs one selected task per Super/Kimi × harness
combination using an already served Super endpoint. It submits no allocations,
uses at most eight turns/steps per episode and does not retry failed episodes.

The existing NeMo RL adapters retain OpenCode-only sampling/token-capture
compatibility. The default harness is evaluation-only. No training, GPU optimizer
qualification or certified score is claimed by this PR. Live integration evidence
is recorded separately from accuracy qualification.

Episode outputs preserve the launcher-mounted path instead of resolving filesystem
aliases. Slurm exports `NEMO_GYM_BENCH_DIR` as the actual bound benchmark directory
for native and custom-command runs, so inputs, episode logs and result files survive
container exit.

## Completed 32K integration evaluations

[Completed evaluation evidence](smoke-results/svg-20261111-32k.json) records the
corrected runtime at `10f5459ed8c33f527c61f577488e1747e9e1cb7c` with Factory
`794bd75abf5496c3a72801a81f35b41d050c705d`. The full local Super/OpenCode run
collected all 209 tasks: 184 scored, 47 passed, and 25 actor timeouts remained
masked. Its 25.54% mean is over the scored subset, not a qualified full score.
All episode artifacts survived container exit.

The matching first 25 tasks were also evaluated with 32,768 output tokens:

| Model and harness | Collected | Scored | Passed | Mean over scored |
|---|---:|---:|---:|---:|
| Super GA, default | 25 | 25 | 7 | 28% |
| Kimi K3 API, OpenCode | 25 | 24 | 11 | 45.83% |
| Kimi K3 API, default | 25 | 25 | 12 | 48% |

The Kimi/OpenCode masked case reached the 5,400-second actor timeout. The default
harness retains its 16-turn limit; OpenCode retains its 150-step limit. These
results demonstrate integration and judging, not equivalent evaluation protocols.
OpenCode sent an effective `max_tokens=32000` with the configured 32,768-token
budget; startup admission must inspect the actual request instead of assuming
exact equality to the configured limit. Clean length endings and timeouts retain
the distinct scoring behavior described above.

For SVG GPU launches, the site wrapper must acknowledge real per-GPU monitoring
on every allocated node before model startup, fail closed on missing telemetry,
and release the allocation after 25 idle minutes, measured from allocation start
before first activity. Use a 20-minute model-readiness deadline and admit at most
two pending/warming replicas on the submission side. The tested matrix used one
TP8 replica on each of two nodes; it did not leave allocated GPUs waiting on a
startup lock. Generic Gym recipes do not install this site guard automatically.
CPU-short currently admits four submitted/running jobs per user; check actual
Slurm admission and keep credentials private when launching hosted evaluations.
