# SpatialClaw evaluation harness

**Experimental integration.** Supply an authorized SpatialClaw checkout that
provides the dataset configurations and runtime described below. Its use is
subject to its own license. Gym includes no SpatialClaw source and does not
qualify the public SpatialClaw release with these presets.

SpatialClaw is a reusable evaluation harness for spatial reasoning, not a
Video-MME variant. The NeMo Gym integration runs the pinned SpatialClaw agent
and perception-tool loop while Gym owns serving, row scheduling, retry/resume,
verification, and metric aggregation.

The suite entry point is `benchmarks/spatialclaw/config.yaml`. It composes all
20 benchmark presets represented by the pinned SpatialClaw source revision.
Each preset can also be prepared and run independently.

Current Gym compatibility changes add environment-server routing, worker
entrypoint discovery, Python 3.13 dependencies, and a persistent HTTP loop for
native Jupyter visual queries. The Video-MME preset retains 256 key frames at
up to 2 FPS and one repeat.

## Architecture

A preset binds three benchmark-specific pieces to one shared agent:

1. A SpatialClaw dataset configuration selects the loader, prompt, frame
   sampling policy, and tools.
2. A preparation script converts the native samples to portable Gym JSONL
   rows. Images and videos remain paths; preparation does not pre-decode video.
3. A resource server scores predictions with the native SpatialClaw evaluator
   and reports its aggregate protocol.

The shared agent lives in
`responses_api_agents/spatialclaw_agent`. It supports single-image,
multi-image, single-video, multi-video, grouped-frame, and reference-image
samples. Relative media paths are resolved from `SPATIALCLAW_DATA_ROOT`.

All 20 configurations use `resources_servers/spatialclaw`, which calls the
evaluator selected by the pinned SpatialClaw dataset configuration.
Video-MME-v2 therefore retains SpatialClaw's native group-aware aggregation
without introducing a separate Gym execution or verification path. CV-Bench
here is SpatialClaw's multi-video CVBench dataset; it is distinct from Gym's
existing single-image `benchmarks/cvbench` integration.

## Included benchmarks

| Category | Benchmark | Preset | SpatialClaw dataset config | Scorer |
|---|---|---|---|---|
| Single-image spatial | ERQA | `erqa.yaml` | `erqa_spatialclaw` | Native |
| Single-image spatial | Omni3D | `omni3d.yaml` | `omni3d_spatialclaw` | Native |
| Single-image spatial | OmniSpatial | `omnispatial.yaml` | `omnispatial_spatialclaw_256f` | Native |
| Single-image spatial | SPBench | `spbench.yaml` | `spbench_spatialclaw_256f` | Native |
| Multi-view spatial | MindCube | `mindcube.yaml` | `mindcube_spatialclaw_256f` | Native |
| Multi-view spatial | MMSI | `mmsi.yaml` | `mmsi_spatialclaw_256f` | Native |
| Multi-view spatial | SPAR-Bench | `sparbench.yaml` | `sparbench_spatialclaw_256f` | Native |
| General spatial | BLINK | `blink.yaml` | `blink_spatialclaw_256f` | Native |
| General spatial | SpatialTree | `spatialtree.yaml` | `spatialtree_spatialclaw_256f` | Native |
| General spatial | ViewSpatial | `viewspatial.yaml` | `viewspatial_spatialclaw_256f` | Native |
| Video spatial / 4D | MMSI-Video | `mmsivideo.yaml` | `mmsivideo_spatialclaw_256f` | Native |
| Video spatial / 4D | OSI-Bench | `osibench.yaml` | `osibench_spatialclaw_256f` | Native |
| Video spatial / 4D | PAI-Bench | `paibench.yaml` | `paibench_spatialclaw_256f` | Native |
| Video spatial / 4D | VSI-Bench-U | `vsibench_unbiased.yaml` | `vsibench_unbiased_spatialclaw_256f` | Native |
| Video spatial / 4D | VSTI-Bench | `vstibench.yaml` | `vstibench_spatialclaw_256f` | Native |
| Video spatial / 4D | DSI-Bench | `dsibench.yaml` | `dsibench_spatialclaw_256f` | Native |
| General video | CV-Bench | `cvbench.yaml` | `cvbench_spatialclaw_256f` | Native |
| General video | PerceptComp | `perceptioncomp.yaml` | `perceptioncomp_spatialclaw_256f` | Native |
| General video | Video-MME | `videomme.yaml` | `videomme_spatialclaw` | Native |
| General video | Video-MME-v2 | `videomme2.yaml` | `videommev2_spatialclaw_256f` | Native |

The `_256f` configurations use the frame and FPS limits encoded in the
SpatialClaw source configuration. The preset does not independently redefine
those sampling rules.

## Source and data setup

The agent and native scorer require the same pinned SpatialClaw checkout. Set
these variables before preparing or running a preset:

```bash
export SPATIALCLAW_ROOT=/path/to/spatial_claw
export SPATIALCLAW_COMMIT="$(git -C "$SPATIALCLAW_ROOT" rev-parse HEAD)"
export SPATIALCLAW_DATA_ROOT=/path/to/spatial_claw/data
```

`SPATIALCLAW_ROOT` must be a Git worktree at `SPATIALCLAW_COMMIT`. To let
Gym prepare the checkout instead, set `SPATIALCLAW_SOURCE_URL` to an authorized
Git repository and leave `SPATIALCLAW_ROOT` unset. Preparation, verification,
and the reference agent require the same revision. Set `SPATIALCLAW_MODEL_CONFIG`
if the checkout requires a specific model-role configuration. Prepared rows
store media paths relative to `SPATIALCLAW_DATA_ROOT`.

The expected directory names are defined by
`spatial_agent/evals/factory.py` in the pinned checkout, including
`ERQA/`, `Omni3D-Bench/`, `MMSI-Video-Bench/`, `Video-MME/`, and
`Video-MME-v2/`. Dataset download and license requirements remain those of
the upstream benchmark and SpatialClaw repository.

Reconstruct and SAM3 are external SpatialClaw GPU services. Start them from the
same checkout or set `SPATIALCLAW_GPU_SERVER_REGISTRY` to a live
`spatial_agent/logs/gpu_server.json`. See
`responses_api_agents/spatialclaw_agent/README.md` for the runtime contract.

## Prepare a benchmark

For example, prepare ERQA with the native loader:

```bash
gym eval prepare --config benchmarks/spatialclaw/erqa.yaml
```

This writes
`benchmarks/spatialclaw/data/erqa_benchmark.jsonl`. All other configurations
follow the same naming convention and native preparation path. The first
preparation creates the SpatialClaw resource-server environment; later
configurations reuse it.

Preparing `benchmarks/spatialclaw/config.yaml` prepares the entire suite and
can be expensive. Prefer an individual preset unless all datasets are already
staged.

## Run a benchmark

Compose the benchmark preset with a policy-model server. For example:

```bash
gym eval run \
  --config benchmarks/spatialclaw/erqa.yaml \
  --config responses_api_models/vllm_model/configs/vllm_model.yaml
```

Replace `erqa.yaml` with any preset from the table. Model endpoint,
checkpoint, GPU allocation, and model-specific vLLM overrides belong in the
policy-model configuration, not in the benchmark preset.

To make all benchmark definitions available in one resolved configuration, use:

```bash
gym eval run \
  --config benchmarks/spatialclaw/config.yaml \
  --config responses_api_models/vllm_model/configs/vllm_model.yaml
```

That command schedules every composed benchmark. Use an individual YAML for
normal development and smoke tests.

## Metrics

Each native verifier returns the row's SpatialClaw `evaluate_single` score
when that benchmark supports per-sample scoring. At aggregation time the
resource server calls the benchmark's full `evaluate` method on the evaluated
rows, preserving benchmark-specific macro averages, mean-relative-accuracy
calculations, category breakdowns, and other native headline metrics. A native
evaluator may mark a row unscored; such a row remains visible and is not
silently converted into a successful result.

## Interchangeable harnesses through agent sessions

The session migration is a follow-up to Gym PR #3996. All 20 presets now use
`single_agent_turn`, with typed taskset routing. Resources owns the dataset,
canonical question, native media sampling, persistent Python kernel, tool state,
answer submission and native scoring. The Environment Server seeds Resources,
seeds the selected agent, activates it once, closes the agent, verifies, and
finally closes Resources. It does not invoke the agent's compatibility `/run`.

| Harness | Preset | Tool transport | Runtime |
| --- | --- | --- | --- |
| Native SpatialClaw | `videomme.yaml` (or another original preset) | Direct HTTP connection to the resource-owned kernel | Operator-supplied SpatialClaw checkout |
| OpenCode | `opencode/videomme.yaml` (or the matching `opencode/` preset) | Required authenticated MCP `get_context` and `execute` | Stock OpenCode 1.17.11 |

Both presets load exactly the same Resources configuration, preparation script,
tasks, media policy, and scorer. The OpenCode preset changes agent configuration
and tool-transport wiring. It uses the existing generic `opencode_agent`, not a
benchmark-specific OpenCode adapter. `opencode/config.yaml` composes the whole
suite for that harness.

The OpenCode preset uses the `spatialclaw_opencode_agent` configuration name,
so other benchmarks can use their own OpenCode settings in the same deployment.
Each SpatialClaw taskset selects one harness through its environment server route.
To compare two harnesses for the same benchmark in a single deployment, give the
second preset and its taskset distinct names while reusing the same prepared
payloads, sample IDs and scoring configuration. A new harness must support
required authenticated MCP tools, image content from tool replies and explicit
answer submission before it is added to `allowed_agents`.

Prepared rows now have `task_id` and `task_input` envelopes. Media references are
in verifier-owned `task_input.task_data.media`, and Responses input contains the
unchanged question text. **Re-run preparation for existing datasets**; older
flat rows with video/image metadata in Responses are incompatible with the
OpenCode session request contract.

The profile uses stock OpenCode 1.17.11, qualified by Gym's generic adapter.
Gym installs the runtime; no fork binary or `SPATIALCLAW_OPENCODE_BIN` is
required to resolve or run a preset. The external Python `OpenCodeWorkflow` is
not copied or launched. SpatialClaw source retains its separate license terms.

For example, after staging the SpatialClaw source and data:

```bash
gym eval prepare --config benchmarks/spatialclaw/videomme.yaml

gym eval run \
  --config benchmarks/spatialclaw/opencode/videomme.yaml \
  --config responses_api_models/openai_model/configs/openai_model.yaml
```

The supplied OpenCode profile uses a resource-owned lent Local sandbox for the
CLI and the native task kernel executes on the Resources host. These are trusted
development paths with shared filesystem access, **without filesystem or network
isolation**. The profile denies OpenCode's built-in host tools and grants only the
benchmark's MCP tools; that permission map does not turn Local execution into a
container. Production isolation and an isolated native kernel deployment are not
claimed. Both agent and Resources session workers use `num_workers: 1`. Resources stages
only selected media and an answer-free task manifest in the lent CLI workspace,
then destroys that workspace on close. The native reference harness uses the
resource kernel directly and does not request a CLI sandbox.

Native SpatialClaw keeps its planner, reflection, formatting and fallback policy.
OpenCode drives its own loop and submits through the benchmark kernel's
`ReturnAnswer`; plain assistant prose does not replace an explicit submission.
These harness behaviors, OpenCode's generic prompt/compaction behavior and its
model sampling defaults can differ from the historical standalone integration.
No score parity is claimed. Auxiliary VLM calls from the task kernel still use
the rollout-prefixed Gym model route; when they occur, a partial OpenCode-only
usage total is replaced with unknown usage and marked in response metadata.
Full invocation/turn ownership reconciliation and P0 qualification remain pending.

The native reference is an Evaluation-level pairing. Native SpatialClaw does not advertise a
function-calling shell tool, so the shell conformance probes do not apply to it.
The OpenCode pairing deliberately disables its shell tools and grants benchmark
Python execution instead; the shell probe suite is not a behavioral certificate
for this configuration. Real model smokes for both harnesses, health/evidence
inspection and full benchmark parity are required before certification.


## Multi-harness training configuration

This migration includes the pending core dependencies named in the November 9
multi-harness runbook: Gym [#5719](https://github.com/NVIDIA-NeMo/Gym/pull/5719)
(images in tool replies, lent sandboxes and OpenCode training controls) and
[#5833](https://github.com/NVIDIA-NeMo/Gym/pull/5833) (capture chains across the
step-limit note and tool-name repair). These prerequisites are now included in main. The shared
`tool_output_images` wire contract is used directly; this environment does not
carry a separate MCP image conversion implementation.

The local OpenCode profile is for evaluating trusted models. RL requires moving
the Python kernel to an environment that cannot read label files or prepared
rows containing gold answers. The supplied local profile does not provide that
isolation. The OpenCode profile disables automatic compaction, sets
an outer 30-step budget, returns its session trajectory, and caps each model call
at `SPATIALCLAW_MAX_OUTPUT_TOKENS` (default 16,384). Set the RL request budget at
least that high, or lower the configured limit explicitly for your experiment.
Request temperature/top-p support remains version-qualified by the generic
adapter on stock OpenCode 1.17.11. Answers at the outer step limit score
zero by default (`score_step_limit_answer: false`). The shared resource Python
step budget remains independently enforced for both harnesses.

MCP calls are allowed up to 1,800 seconds at the adapter level so GPU tools are
not cut off by the short connection default. OpenCode/Bun can still impose a
roughly 300-second fetch timeout; the configured MCP bound does not prove that a
long-running transport stays alive. Long-tool liveness must be checked against
the actual deployment.

This setup enables the Environment Server/session and capture interfaces used by
RL; it is not evidence of a completed RL run. Fresh real-model inference,
training smoke, zero-poisoned capture-chain checks and health/P0 inspection are
still pending cluster availability. Native SpatialClaw's own per-turn training
adapter and multimodal token-history contract remain outside this change.
