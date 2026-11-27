# CAD workspaces with native agents

`configs/multiharness.yaml` composes five native Agent Servers, each with its own
`single_agent_turn_legacy` Environment Server, sharing one CAD Resources Server
and policy Model Server. It uses the session interfaces from
[Gym #5950](https://github.com/NVIDIA-NeMo/Gym/pull/5950) alongside the Agentic-MME example
in [Gym #5720](https://github.com/NVIDIA-NeMo/Gym/pull/5720).
`configs/opencode.yaml` remains the standalone OpenCode recipe. There is no CAD
agent loop, custom MCP server, or `legacy_agent` relay; `legacy` here names the
flat-row input format.

The Resources Server lends a workspace containing target PNGs and view/color
settings. The agent writes `/workspace/model.py`, runs the supplied
`/workspace/.cad/cad_render.py`, and inspects `/workspace/previews/preview.png`
with its own image capabilities. Preview cameras and color match the targets;
axis views retain front/side/top order. Reference STEP files and reference
programs never enter the workspace. Operation hints remain task instructions.

On close, the resource collects the final regular UTF-8 `model.py`, retains it
for verification retries, stops the workspace, and calls the existing isolated
verifier. Prose, code fences, preview geometry, and caller-provided `final_code`
are not graded. Missing/invalid files, failed runs, and ordinary incomplete runs
score zero. Workspace collection failures are masked. Cleanup failure retains ownership for retry.
The stateful Resources Server requires one worker.

## Prepare and route inputs

Input JSONL must contain valid `verifier_metadata`, task text, and inline target
images from `dataset/prepare_gym.py`. Materialize lazy corpus rows before using
`--file-input`. The standalone inline OpenCode path still accepts validated
`deepcad_lazy` rows without rendering and defers workspace prompting to the CAD
dataset hook. Lazy file-input conversion fails with a materialization instruction.

```bash
python -m resources_servers.deepcad.prepare_opencode INPUT.jsonl opencode.jsonl \
  --file-input --harness opencode
python -m resources_servers.deepcad.prepare_opencode INPUT.jsonl pi.jsonl \
  --file-input --harness pi
```

Use `--file-input` for the shared five-agent recipe; non-OpenCode selections
require it. Preparation retains the
original images in explicit `target_images` for resource seeding, while the
outer activation contains task instructions and target-file
paths as user text. The resource decodes targets to RGB PNG files in input order
(`/workspace/target-0.png`, etc.). This is file-based image access, not a claim of
byte-identical inline delivery to every native model client. Without the flag,
activation retains inline images; not every native adapter accepts that format.

`--harness` selects the task-scoped `deepcad_<harness>_agent` and
`deepcad_<harness>_environment_server` routing fields.
`task_source` remains `deepcad_opencode`, the shared resource. Prepare separate
outputs for each harness and concatenate those rows to collect a mixed run; do
not replace their routing with a single global `--agent` override.

Load the model config first, then the CAD recipe. CAD inherits stock agents
under task-scoped names, so a profile that edits a stock agent block, such as
`benchmarks/hle/pi.yaml`, also changes the CAD agent that inherits that block.
CAD also sets the top-level `sandbox` block to Docker. Do not blend CAD with a
profile that edits a stock agent block, or with a recipe that sets another
`sandbox` provider.

`deepcad_policy_model` copies the selected `policy_model` backend while keeping
CAD's tool-image handling and token-capture options off the
shared model instance. Training integrations must route CAD calls to this alias
and include it in any explicitly required model-sampling list.

Preparation moves existing `temperature`, `top_p`, and `max_output_tokens` values
to `_ng_model_sampling`, rejecting conflicts with an existing marker. Lazy rows
retain their raw fields and receive the same validated marker, which survives
materialization. The training framework supplies model sampling requirements;
the recipe does not mandate them for ordinary evaluation. Preparation does
not invent sampling values. For a 32,768-token per-call comparison, supply that
explicit cap in the original request or marker before preparation. A smaller
source cap stays smaller. Do not reintroduce unsupported activation parameters
with collection-time sampling flags.

## Runtime prerequisites

Every workspace image must retain CadQuery, the CAD renderer, and its graphics
libraries. The supplied image builds only the OpenCode runtime.

```bash
docker build --platform linux/amd64 -t deepcad-cadquery:2.8.0 resources_servers/deepcad
docker build --platform linux/amd64 -t deepcad-opencode:1.17.11 \
  -f resources_servers/deepcad/sandbox_image/Dockerfile .
```

| `--harness` | Native runtime setup in the borrowed workspace |
|---|---|
| `opencode` | Pinned 1.17.11 binary and plugin archive are preinstalled by the supplied Dockerfile and SHA-256 checked. |
| `pi` | Pinned 1.5.2 plus private Node 22.19.0; existing installer or a verified `sandbox_runtime_dir` containing both. |
| `codex` | Pinned 0.144.4 plus private Node 22.19.0; existing installer or verified `sandbox_runtime_dir`. |
| `claude_code` | Pinned 2.1.108 Linux binary cached on the controller and uploaded; supplied local/sandbox binaries require `sandbox_executable_sha256`. |
| `openclaw` | Pinned 2026.6.11 plus private Node 22.19.0; existing installer or verified `sandbox_runtime_dir`. |

Pi, Codex, and OpenClaw installers need package-download access unless their
runtime directories are preinstalled; missing OS prerequisites may require root.
Claude obtains its artifact on the controller.
Use these native setup mechanisms rather than assuming the OpenCode image already
contains every CLI. Runtime setup and the task's actual Python/CadQuery imports
must be checked before a model run, including native shell environment behavior.

## Start and collect

```bash
gym env resolve --config deployment.yaml \
  --config resources_servers/deepcad/configs/multiharness.yaml
gym env validate --config deployment.yaml \
  --config resources_servers/deepcad/configs/multiharness.yaml \
  --server-types resources_servers,responses_api_agents,responses_api_models,environment_servers
gym env start --config deployment.yaml \
  --config resources_servers/deepcad/configs/multiharness.yaml
# In another terminal, with the same configuration:
gym eval run --no-serve \
  --config deployment.yaml --config resources_servers/deepcad/configs/multiharness.yaml \
  --input mixed.jsonl --output rollouts.jsonl
```

Load the deployment model configuration first, followed by the CAD recipe, so
CAD token-capture settings override model-adapter defaults.

`deployment.yaml` supplies the actual model endpoint/credentials, private
`reference_root`, and provider/image bindings. The defaults use Docker bridge for
workspaces and network-disabled Docker for verification. Docker cannot enforce
`model_only`; use a provider with network-policy support for that mode. Inspect
resolved addresses: allowing a model host permits other ports on that host, and
host-network sandboxes are not isolated by loopback pins.

For native Linux ARM64, build both images with `--platform linux/arm64` and
set these two fields under
`deepcad_opencode_agent.responses_api_agents.opencode_agent` in the deployment
configuration (the plugin archive includes an architecture-specific dependency):

```yaml
sandbox_executable_sha256: e51b4a9c0e8dfa90c3387f0aa41e7c362d56b1939eec102ae76c1aac0d911186
sandbox_plugin_dependencies_sha256: 12ce5f79b27aa8886b64082c1f5d23ab4bc7410e193b7d98146f5df1a61f243a
```

## Limits and validation boundary

- There is no Gym-level model-call cap. The original CAD limit of 10 is OpenCode's own
  `max_steps`; a harness's own turn limit and the episode's `agent_timeout_sec` bound each run.
  Codex, Pi, and OpenClaw have no native turn limit, so only the timeout bounds them.
- Image delivery uses each harness's existing support on main; this recipe adds none.
  OpenCode, Pi, and OpenClaw take `image_input: true`, and Codex sends `input_image`.
  Claude Code has no such option and reads the PNG files itself.
  Hermes and NOOA are not part of this recipe: Hermes' vision tool is not qualified in
  Gym and NOOA passes only text, so each needs its own image path and a real-model
  smoke run before it can be added.
- The row's output cap is an upper bound: the model middleware uses the smaller
  of that cap and the native request's cap. It does not raise a client's smaller
  output limit. Missing source caps leave native/backend defaults in effect. This
  multi-turn per-call limit is not an original single-turn total-token budget.
- Explicit context/output settings are applied where the public adapter supports
  them. Claude has no equivalent typed context/output controls; Codex
  relies on the row cap for the output ceiling. Inspect actual model requests before
  claiming equal effective context or output limits across all five.
- Native turn, summary, delegation, and compaction behavior remains relevant.
  Supported recipe settings disable some automatic behavior; the task prompt's
  prohibition is not a universal runtime switch. OpenCode's `build.steps` is
  advisory and can add its own final-turn instruction independently of the shared
  budget. Agent timeouts are 1,800 seconds; the episode limit is 2,100 seconds.
- Token capture is enabled. The serving backend must support the required-prefix
  extension. Saved tool text alone does not prove preview images reached the
  model; validate actual media, prefix continuity, terminal capture, and cleanup.

Focused checks are in `tests/test_opencode_sessions.py` and
`tests/test_opencode_manifest.py`. They cover preparation, routing, resource
sessions, file authority, and cleanup; they do not establish a fresh image build
or real-model rollouts. Earlier internal CAD quality/training experiments
used different runtime/protocol configurations and do not validate this public
file-input recipe. No new cross-harness quality or training parity is claimed.

## Real-model smoke

One real policy checkpoint, served by NeMo RL's vLLM worker, ran the five original examples
(`data/example.jsonl`) through each of the five harnesses on OpenSandbox, 25 episodes in all,
at commit `a7706529e`. Rewards come from this server's verifier.

| Harness | Episodes | Mean reward | Episodes where 3 or more distinct images reached the model |
|---|---|---|---|
| `claude_code` | 5 | 1.000 | 5 of 5 |
| `openclaw` | 5 | 1.000 | 5 of 5 |
| `codex` | 5 | 0.913 (one episode scored 0.562) | 5 of 5 |
| `opencode` | 5 | 0.893 (one episode scored 0.466) | 4 of 5 |
| `pi` | 5 | 1.000 | 3 of 5 |

All 25 episodes ended with the final file collected and a verifier result. Over all 25,
the mean reward is 0.961 (standard error 0.027).

How to read this:

- Rewards are close to 1.0 on these examples, so they do not separate the harnesses. A harness
  that never saw an image can still score 1.0: two Pi episodes and one OpenCode episode never
  showed an image to the model, and the two Pi episodes still scored 1.0.
- File-input rows carry no image in the first model request. The images arrive when the harness
  reads the workspace files, so "reached the model" is counted from every captured request of
  the episode, not the first one.
- No comparison with the original single-harness recipe was run, so no relative reward change is
  claimed.
- This is not training evidence. Of the 25 episodes, only 2 passed NeMo RL's token-capture
  selection. The rest were rejected because an output item could not be matched to a captured
  generation, or because a capture was attributed to another dispatched attempt. The verifier
  results above were read from the results that NeMo RL retains when selection fails.
- The cluster run differed from the defaults in this recipe in two ways. The verifier ran from a
  copy of the CadQuery image whose default user is root, because OpenSandbox's executor cannot
  switch users and the sealing step runs as root. The sandbox create timeout was raised to 600
  seconds, because creating a sandbox took about three minutes when the image was not yet on a
  node. The first pass left 5 episodes without a verifier result because their sandboxes stayed
  Pending; those 5 were run again in a second job and all finished.

## Validation

The following results validate the upstream standalone OpenCode recipe from #5950,
not this five-harness file-input recipe.

Automated tests cover resource sessions, final-file authority, retained source,
cleanup/retries, prompt/image preservation, and stock-harness input/config compatibility.

A Linux ARM64 smoke on 2026-11-16 used stock OpenCode 1.17.11, the Mixed76 step-76
vision checkpoint, a 131,072-token context setting, Apptainer workspaces, and Enroot
grading. All five synthetic tasks ran twice, with explicit instructions to render
a wrong draft, inspect the preview, edit, and render again; the smoke allowed 20 steps.

- All ten episodes collected `model.py` and graded without masking or failure.
  Mean B-rep IoU was 1.0 to floating-point tolerance.
- All ten saved rollouts retained their prompts and three decodable target PNGs.
  Each episode rendered twice and read at least one preview (14 preview reads total).
  Recorded model requests contained the actual additional preview PNGs, not only
  text acknowledging a read. These are controlled smoke results, not evidence of
  spontaneous feedback use or benchmark quality.
- Raw OpenCode `/run` responses did not carry token arrays. Gym's normal observation
  enrichment and token finalizer produced 72 token-bearing model calls across all
  ten episodes, with no token masks, unresolved parents, or dropped generated tokens.
  All 62 captured parent/child prefixes matched exactly, and token/log-probability
  arrays had equal lengths and finite log probabilities.
- Cleanup left no harness processes, listening service ports, Apptainer instances,
  or verifier sandboxes. Single-turn and reference/known-bad baseline results are
  recorded in the [component README](README.md#inputs-and-tests).

The initial standalone configuration masked six of ten token reconstructions and
scored the sphere at 0.666667. Preserving prior reasoning in the chat template fixed
all ten reconstructions in the repeat. The diagnostic configuration used:

```yaml
observability_enabled: true
model_call_capture_dir: /tmp/cad-model-calls
token_id_capture: {enabled: true, all_agents: true, dir: /tmp/cad-tokens}
policy_model:
  responses_api_models:
    vllm_model:
      return_token_id_information: true
      request_prompt_and_generation_token_ids: true
      supply_prefix_token_ids: false  # Standalone diagnostic only; native training keeps prefix supply.
      extra_body:
        chat_template_kwargs: {enable_thinking: true, truncate_history_thinking: false}
```

The standalone backend lacks `required_prefix_token_ids`, so this smoke disabled
prefix supply and checked the actual generation-time prefixes instead. Native
training retains exact-prefix supply. This does not validate distributed tensor/media
handoff, a full 128k-window workload, or the default Docker image and its isolation.
