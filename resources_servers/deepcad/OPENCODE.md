# CAD workspaces with stock OpenCode

`configs/opencode.yaml` pairs this resources server with the existing
`responses_api_agents/opencode_agent` through `single_agent_turn_legacy`.
There is no CAD agent adapter, custom MCP tool server, or `legacy_agent` relay.
The word `legacy` in the Environment Server name refers to its flat JSONL input
format; it still drives agent sessions through seed, activation, and close.

The resources server lends a workspace containing only target PNGs and view/color
settings. The sandbox image supplies CadQuery and a renderer. OpenCode writes
`/workspace/model.py`, uses `bash` to run the renderer, and uses `read` to inspect
`/workspace/previews/preview.png`. The preview uses the target cameras and color;
axis views are combined in front/side/top order. No reference STEP or source enters
the workspace. Prompts, including operation hints, belong to prepared task rows.

After OpenCode closes, the resources server collects the final regular UTF-8 file,
retains its source for verification retries, stops the workspace, and invokes the
existing isolated verifier. It never trusts prose, code fences, or a caller's
`final_code`. Missing/invalid files and incomplete runs score zero. Workspace
collection failures are masked; lost sessions cannot fall back to answer text.
Close, failed-seed cleanup, and server shutdown stop lent workspaces. A failed
stop retains ownership so cleanup can retry. Serving this state requires one
resources-server worker.

## Prepare and run

From the repository root, first build the network-disabled verifier image, then
the workspace image (the default configuration pins Linux x86-64):

```bash
docker build --platform linux/amd64 -t deepcad-cadquery:2.8.0 resources_servers/deepcad
docker build --platform linux/amd64 -t deepcad-opencode:1.17.11 \
  -f resources_servers/deepcad/sandbox_image/Dockerfile .
python -m resources_servers.deepcad.prepare_opencode INPUT.jsonl OUTPUT.jsonl
```

`INPUT.jsonl` may contain prepared image tasks or validated metadata-only
`deepcad_lazy` tasks. Lazy conversion marks the row for the OpenCode harness
without rendering. The CAD dataset hook materializes selected rows and applies
workspace instructions before image processing. Unstructured corpus records
are not accepted. The conversion preserves images, metadata, and
sampling fields; it replaces single-turn output instructions with file-based
workspace instructions and routes rows to `deepcad_opencode`.

Load the deployment's model configuration first, followed by
`resources_servers/deepcad/configs/opencode.yaml`, so CAD's token-capture settings
override model-adapter defaults. Set `reference_root` to the external reference directory, and make
only the model endpoint reachable from the workspace. Then use the standard
`gym env resolve`, `gym env validate`, `gym env start`, and `gym eval run --no-serve`
commands, with `OUTPUT.jsonl` and agent name `deepcad_opencode_agent`.

The config uses Docker bridge for workspaces and a network-disabled Docker
provider for verification. `workspace_network_access: inherit` preserves Docker's
network. A provider that enforces network policies may use `model_only` instead;
Docker refuses that setting. Non-model services explicitly bind loopback. Check
the final resolved addresses and probe the actual deployment: a network policy
that permits the model's host permits every port on that host, and a host-network
sandbox is not isolated by loopback pins.

For native Linux ARM64, build both images with `--platform linux/arm64` and
set these two fields under
`deepcad_opencode_agent.responses_api_agents.opencode_agent` in the deployment
configuration (the plugin archive includes an architecture-specific dependency):

```yaml
sandbox_executable_sha256: e51b4a9c0e8dfa90c3387f0aa41e7c362d56b1939eec102ae76c1aac0d911186
sandbox_plugin_dependencies_sha256: 12ce5f79b27aa8886b64082c1f5d23ab4bc7410e193b7d98146f5df1a61f243a
```

## Limits and training compatibility

OpenCode 1.17.11 and its plugin dependencies are preinstalled and SHA-256 checked.
The stock harness owns process supervision, image delivery, model routing,
observations, and token capture. The CAD server owns task files and grading only.

- Context: 131,072 tokens. Output limit: 32,768 tokens **per model call**.
- `agent.build.steps: 10` is OpenCode's advisory step limit, not a hard ten-call
  stop. The 1,800-second agent timeout and context limit remain hard limits.
- Compaction and delegation are disabled through supported stock settings.
- A training row's `max_output_tokens` must be at least the configured per-call
  output limit; the stock harness rejects a smaller value.
- Stock saved rollout exports retain tool-result text; they need not contain the
  preview images. Validate the model-call media capture separately for training.
- Gym token capture is enabled in this config. NeMo-RL serving must implement the
  required-prefix extension; no custom model adapter is included in the PR.

These limits differ from the removed custom agent's remaining-context policy.
Report them when comparing scores. The shared geometry reward is unchanged.

## Validation

```bash
python -m pytest resources_servers/deepcad/tests/test_opencode_sessions.py \
  resources_servers/deepcad/tests/test_opencode_manifest.py -q
```

Automated tests cover resource sessions, final-file authority, retained source,
cleanup/retries, prompt/image preservation, and stock-harness input/config
compatibility. They do not establish Docker or real-model rollout success.
A fresh Linux image build, render/edit model smoke, and distributed token/media
capture validation are required before marking this implementation production-ready.
