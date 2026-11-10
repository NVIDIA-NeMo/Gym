# Multi-turn CAD with OpenCode

The agent gives OpenCode target PNGs and an isolated workspace. It writes and
edits `/workspace/model.py`, renders intermediate results with `cad_cad_render`,
and can inspect workspace images with `cad_cad_view`. Tool feedback contains
inline images, not filenames alone. The generated program must explicitly
import CadQuery and assign its final solid to `result`.

Only the final regular UTF-8 `model.py` is collected. It is evaluated by the
separate DeepCAD verifier; earlier code, prose, and agent-exported geometry are
not graded. Reference STEP files never enter the agent workspace. Missing files
and incomplete agent execution earn zero. Verifier infrastructure failures keep
their masking behavior.

## Setup

Build the verifier image described in
[`resources_servers/deepcad`](../../resources_servers/deepcad/README.md), then:

```bash
docker build -t deepcad-opencode:latest responses_api_agents/deepcad_opencode
python responses_api_agents/deepcad_opencode/prepare_manifest.py INPUT.jsonl OUTPUT.jsonl
```

Load `configs/deepcad_opencode.yaml` alongside the model-server configuration.
Set the model endpoint and its sandbox-reachable URL for your deployment. The
agent uses a separate network-enabled Docker provider to reach that endpoint;
the generated-code verifier uses a network-disabled provider. Adjust image names
and `reference_root` to your prepared assets.

The example pins OpenCode 1.17.11, disables compaction and delegation, and limits
each episode to ten model steps and a 65,536-token context. Keep the model
endpoint's context window consistent with this budget. Target views and preview
views use the same camera layout and foreground color.

For training, use the current upstream Gym token-capture path and NeMo-RL's
HTTP serving backend with `supply_prefix_token_ids` and
`return_token_id_information` enabled. The backend must implement the required
prefix extension. No CAD-specific model adapter or shared capture patches are
included here. Validate distributed token/media capture with your training
framework before using a new deployment for learning.

## Validation

```bash
python -m pytest responses_api_agents/deepcad_opencode/tests -q
```

These tests cover final-file selection, inline image feedback, workspace path
boundaries, renderer errors, termination handling, and cleanup. Earlier HSG
model rollouts exercised the full experiment branch. The reduced upstream
implementation still needs a fresh live model smoke before review readiness.
