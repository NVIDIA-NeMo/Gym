# AgentVista evaluation through NeMo Gym

Two evaluation protocols share the pinned public AgentVista tasks, tools and
answer judge:

| Benchmark | Harness |
|---|---|
| `agentvista_default` | Original greedy loop, with the upstream published run-command limits |
| `agentvista_opencode` | OpenCode CLI 1.17.11 with the original tools exposed over MCP |

Gym owns preparation, worker dispatch, tool transport, scoring and runtime build
code. The [agent README](../../responses_api_agents/agentvista_agent/README.md)
lists the complete configuration and test commands. This is an experimental,
evaluation-only external-agent-loop integration. Swapping in another Gym harness
requires new prepared inputs and a resources-server/session integration; the
current rows alone are not a harness-independent visual benchmark.

## Public source and data

The original implementation is a runtime dependency pinned to
[AgentVista a3385d51](https://github.com/hkust-nlp/AgentVista/tree/a3385d51d0c3dd0c8501350a08d52065e3e94ab4).
It is not vendored or relicensed. The agent's
[`runtime/source-pin.json`](../../responses_api_agents/agentvista_agent/runtime/source-pin.json)
records 26 execution, prompt, tool and judge file hashes. Setup and both workers
verify those hashes before any model call; the episode records the public source
revision. The worker also checks the task/image binding from preparation.

```bash
python benchmarks/agentvista/setup.py --source "$WORKSPACE/agentvista-source"
python benchmarks/agentvista/prepare_dataset.py --output "$WORKSPACE/agentvista-data" \
  --revision 61de767ad8c72e258ac71a0aef952b89ca920a16
```

The CC-BY-4.0 release contains 209 tasks and 307 embedded images. Preparation
preserves original records and image bytes. The upstream split name `train` does
not establish a held-out evaluation protocol. Generated datasets stay outside
Git. The manifest uses `external-agent-loop` with canonical split `benchmark`;
validate it after local data preparation with `gym env validate --sync agentvista`.

## Provision a worker

Install the server using its managed `pyproject.toml` and `uv.lock`. The separate
CPU worker needs the original tools, Torch/image loader, MCP and Jupyter. Its
`requirements-worker.in` holds upgrade constraints; `requirements-worker.txt`
fully pins the resolved dependency set and artifact hashes for Linux x86-64 and
Python 3.13. Both image builders install that same file with hash verification,
and record the installed package set. Gym's server resolver exclusions must not
be applied to this separate worker environment.

To regenerate the worker lock from the Gym checkout:

```bash
uv --no-config pip compile responses_api_agents/agentvista_agent/requirements-worker.in \
  --python-version 3.13 --python-platform x86_64-unknown-linux-gnu \
  --index https://download.pytorch.org/whl/cpu --default-index https://pypi.org/simple \
  --index-strategy unsafe-best-match --emit-index-url --generate-hashes --no-header \
  -o responses_api_agents/agentvista_agent/requirements-worker.txt
```

Build a new image using explicit, trusted inputs. Obtain `OPENCODE_SHA256` from
trusted release/build provenance; the builder verifies it rather than accepting
only the executable's version string.

```bash
python responses_api_agents/agentvista_agent/runtime-image/build.py \
  --base-image "$WORKSPACE/images/gym-base.sqsh" \
  --python-distribution "$WORKSPACE/python313" \
  --opencode "$WORKSPACE/bin/opencode" --opencode-sha256 "$OPENCODE_SHA256" \
  --uv "$WORKSPACE/bin/uv" --source "$WORKSPACE/agentvista-source" \
  --work-dir "$WORKSPACE/image-build" --output "$WORKSPACE/images/agentvista-eval.sqsh"
```

The builder uses Enroot inside an appropriate existing allocation; it submits no
jobs and refuses an existing output. The Dockerfile is an alternative build path
using `BASE_IMAGE` and `OPENCODE_SHA256` build arguments. It supplies the same
pinned worker dependencies and verifies the same source pin and executable.
Import checks establish dependency availability, not evaluation accuracy.

## Configure providers and execution

Use a private environment file outside Git for policy/search/visit/judge settings:

- `AGENTVISTA_ROOT`, `AGENTVISTA_DATASET`, `AGENTVISTA_IMAGES`: verified source,
  original task records and image directory.
- `AGENTVISTA_WORKER_PYTHON`, `AGENTVISTA_OPENCODE`: worker interpreter and pinned CLI.
- `SERPAPI_KEY`: original search/Lens tool credential.
- `JINA_API_KEY`, `IMGBB_API_KEY`: optional reader/upload backends.
- `AGENTVISTA_VISIT_API_KEY`, `AGENTVISTA_VISIT_API_ENDPOINT`,
  `AGENTVISTA_VISIT_MODEL_NAME`: fixed webpage summarizer; endpoint is a full
  chat-completions URL.
- `VERIFIER_API_KEY`, `VERIFIER_END_POINT`, `VERIFIER_MODEL_NAME`: fixed answer
  judge; endpoint is a base URL ending in `/v1`. Both workers check these before
  any policy calls.

Both harnesses now run `code_interpreter` in a mandatory Bubblewrap namespace
sandbox supervised through NeMo Gym's sandbox lifecycle API. The boundary
excludes task datasets, reference answers, host process environments and judge
credentials. Only the Python runtime and verified tool code are mounted read-only;
images and results cross a bounded byte channel. Generated code has no network
access and cannot select host paths for returned images. Stateful Python and the
original tool schemas remain available.

Workers require `bwrap` and permitted nested user namespaces. Missing support or
sandbox transport failure is unscorable and never falls back to host execution.
Both runtime-image builders install Bubblewrap; rebuild older images before using
harness version **4.0.0**. Do not combine historical unsandboxed scores with this
version. See the [agent README](../../responses_api_agents/agentvista_agent/README.md)
for the namespace tests, runtime requirements and shared-kernel limitations.
Training remains unsupported. The resources-server/session migration with stock
`opencode_agent` remains a separate architectural follow-up.

## Run evaluation

Keep each protocol's inputs and outputs separate, and use workspace storage:

```bash
benchmark=agentvista_default  # or agentvista_opencode
export AGENTVISTA_OUTPUT_ROOT="$OUTPUT/episodes"
common=("+agentvista_dataset_path=$OUTPUT/input.jsonl" \
        "+agentvista_output_root=$OUTPUT/episodes" \
        "+policy_base_url=$POLICY_BASE_URL" "+policy_model_name=$POLICY_MODEL" \
        '+policy_api_key=${oc.env:AGENTVISTA_POLICY_API_KEY}')
gym eval prepare --benchmark "$benchmark" "${common[@]}" \
  "+prepare_script_args.output_fpath=$OUTPUT/input.jsonl"
gym eval run --benchmark "$benchmark" --model-type vllm_model \
  --agent agentvista_agent --split benchmark --concurrency 1 \
  --output "$OUTPUT/rollouts.jsonl" "${common[@]}" +observability_enabled=true
```

For OpenCode, also pass
`++policy_model.responses_api_models.vllm_model.entrypoint=agentvista_opencode.py`.
The default harness uses the normal model entrypoint and removes Gym's leading
reasoning envelope before the original parser sees an answer. Standard row/CLI
`temperature`, `top_p` and `max_output_tokens` override agent defaults on both
tracks. OpenCode advertises sampling support and receives its exact configured
output-token ceiling, including budgets above its built-in 32,000-token default.

The default loop now uses the published run command's 30 turns and 100 images,
65,536 context tokens and 8,192 output tokens. OpenCode defaults to 150 steps and
5,400 seconds of actor time, with its own history/compaction. The legacy relay
has no queue-wide deadline; the worker's episode timeout starts after acquiring
an active slot. OpenCode must list all four original MCP tools before a result
can be scored. Input image attachments are not silently resized by OpenCode.

## Scoring, evidence and compatibility

Both paths use the upstream judge prompt and first-word extraction rule: a
nonempty extracted word containing `1` is one, other nonempty words are zero.
Punctuation is retained; an empty word after judge retries is unscored. An empty
model prediction is still sent to the judge. For OpenCode, a clean reasoning-only stop and an
actor deadline with retained model text/reasoning/tool output are model outcomes
and are judged. A timeout with no retained event, failed MCP startup, provider or
capture failure remains masked. Tool errors and recovery remain in the evidence.
Rollout rows include `verifier_diagnostics`, `judge_response`, extracted prediction,
judge model, native events and source/task bindings. Counted-zero collector rows
remain in aggregate denominators even without verifier-generated fields.

Harness version 3.0.0 changes scoring, reasoning, sampling and baseline limits.
Prepare new rows and use new output directories. Older prepared rows are rejected;
do not combine their scores with this version. The historical 2.1.1 smoke records
under `smoke-results/` are retained for audit only: their default-harness reasoning
handling, reduced 16/16 limits, OpenCode token cap and masked timeout policy do not
qualify the corrected benchmark. They are not parity or accuracy claims.

Tests without external assets exercise Gym routing, row sampling, metrics, source
validation and model-versus-infrastructure outcomes. With a verified upstream
checkout and worker dependencies, the additional fixtures exercise real original
tools, Jupyter and the pinned CLI against local fake policy/judge endpoints.
`smoke.sh` provides bounded real-provider checks with explicitly supplied endpoint
and model settings. No training adapter or training qualification is included.
