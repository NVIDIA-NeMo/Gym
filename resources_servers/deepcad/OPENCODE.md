# CAD workspaces with native agents

`configs/multiharness.yaml` composes seven native Agent Servers, each with its own
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
score zero. An incomplete response carrying the native `ng_rollout_stop` JSON
marker with reason `model_call_budget_exhausted` is the sole exception: the
resource collects and grades the saved file, without changing response status.
This enables grading the artifact when the shared ten-call budget stops the
agent; it changes the earlier completed-only eligibility rule. Malformed markers
and other stop reasons remain ineligible. Workspace collection failures are masked. Cleanup failure retains ownership for retry.
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

Use `--file-input` for the shared seven-agent recipe; non-OpenCode selections
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

Load the model config first, then the CAD recipe, then benchmark profiles such as
`benchmarks/hle/pi.yaml`. CAD inherits stock agents under task-scoped names, but
reloading stock configs after a profile can overwrite that profile's settings.
In particular, placing CAD after HLE resets HLE's Pi version, model and timeout;
that ordering is not supported.

`deepcad_policy_model` copies the selected `policy_model` backend while keeping
CAD's generation budget, tool-image handling and token-capture options off the
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
libraries. The supplied image builds only the OpenCode runtime. It also makes
`/opt/nemo-gym-nooa` writable by the `cad` user so NOOA's
native installer can provision its private runtime without root privileges.

```bash
docker build --platform linux/amd64 -t deepcad-cadquery:2.8.0 resources_servers/deepcad
docker build --platform linux/amd64 -t deepcad-opencode:1.17.11 \
  -f resources_servers/deepcad/sandbox_image/Dockerfile .
```

| `--harness` | Native runtime setup in the borrowed workspace |
|---|---|
| `opencode` | Pinned 1.17.11 binary and plugin archive are preinstalled by the supplied Dockerfile and SHA-256 checked. |
| `pi` | Pinned 0.80.2 plus private Node 22.19.0; existing installer or a verified `sandbox_runtime_dir` containing both. |
| `hermes` | Controller builds and uploads the server-locked Hermes Python runtime through `SandboxPythonRuntime`; install the server component from its lock. |
| `codex` | Pinned 0.144.4 plus private Node 22.19.0; existing installer or verified `sandbox_runtime_dir`. |
| `claude_code` | Pinned 2.1.108 Linux binary cached on the controller and uploaded; supplied local/sandbox binaries require `sandbox_executable_sha256`. |
| `openclaw` | Pinned 2026.6.11 plus private Node 22.19.0; existing installer or verified `sandbox_runtime_dir`. |
| `nooa` | Sandboxed, component-locked NOOA runtime packages the launching tracked Gym source into private Python; use its existing offline `runtime_wheelhouse` for offline setup. |

Pi, Codex, and OpenClaw installers need package-download access unless their
runtime directories are preinstalled; missing OS prerequisites may require root.
Hermes and Claude obtain artifacts on the controller. NOOA requires a Git checkout
and Git on the controller; its offline bundle must match the packaged source.
Use these native setup mechanisms rather than assuming the OpenCode image already
contains every CLI. Runtime setup and the task's actual Python/CadQuery imports
must be checked before a model run, including native shell environment behavior.

## Start and collect

Choose an existing shared run directory and create a **new** budget directory for
each independent run. All policy-model workers in that run must see the same path.
Do not reuse another run's counters or give each worker an independent directory.

```bash
export CAD_BUDGET_DIR="$(mktemp -d "${CAD_RUN_ROOT:?set a shared run directory}/cad-budget.XXXXXXXX")"
gym env resolve --config deployment.yaml \
  --config resources_servers/deepcad/configs/multiharness.yaml
gym env validate --config deployment.yaml \
  --config resources_servers/deepcad/configs/multiharness.yaml \
  --server-types resources_servers,responses_api_agents,responses_api_models,environment_servers
gym env start --config deployment.yaml \
  --config resources_servers/deepcad/configs/multiharness.yaml
# In another terminal, with the same CAD_BUDGET_DIR and configuration:
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

- The shared model-server budget permits at most **ten generation admissions per
  rollout**, including auxiliary calls using that route. The next call is rejected.
  This is not ten tool actions or a guarantee of ten successful model responses.
  No shared-budget final prompt or forced final tool choice is configured.
- The row's output cap is an upper bound: the model middleware uses the smaller
  of that cap and the native request's cap. It does not raise a client's smaller
  output limit. Missing source caps leave native/backend defaults in effect. This
  multi-turn per-call limit is not an original single-turn total-token budget.
- Explicit context/output settings are applied where the public adapter supports
  them. Claude has no equivalent typed context/output controls; Codex and NOOA
  rely on the row cap for the output ceiling. Inspect actual model requests before
  claiming equal effective context or output limits across all seven.
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
or seven real-model rollouts. Earlier internal CAD quality/training experiments
used different runtime/protocol configurations and do not validate this public
file-input recipe. No new cross-harness quality or training parity is claimed.

## Validation

The following results validate the upstream standalone OpenCode recipe from #5950,
not this seven-harness file-input recipe or its ten-admission stopping rule.

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
