# Nano4 OSWorld adapter

This optional adapter consumes a separately supplied prompt-format producer and runs Nano4 through Gym's existing OSWorld task loop. It keeps the producer's system prompt, tool schema, observation feedback and action-block image history together, while selecting the coordinate consumer independently. It does not change the v3 or Super 3.5 adapters or their defaults.

The direct-library entrypoint requires the shared `model_protocol_agent` runner and Python-suite execution support in [PR #5101](https://github.com/NVIDIA-NeMo/Gym/pull/5101). This package does not duplicate that PR's implementation. Until the dependency lands, registration reports an actionable error.

## Configuration

Supply a local Git checkout of the matching producer. `reference.lock.json` records its commit and the exact source-file and prompt-profile hashes; loading verifies all of them. The producer is not vendored, and no model checkpoint, prompt dataset or serving asset is included here. The deployment must provide its own compatible tokenizer, chat template, reasoning parser and tool parser; this adapter alone does not establish the checkpoint's training protocol.

```python
from responses_api_agents.osworld_agent.nano4.runtime import run_task

result = run_task(
    task_config,
    messages_model_fn,
    reference_root="/path/to/producer-checkout",
    model="policy_model",
    coordinate_policy="step1500_compat_v1",
    max_steps=100,
)
```

`messages_model_fn(messages, payload)` is supplied by the caller and should route inference through the configured Gym model server. The adapter makes no inference HTTP calls of its own. Its current execution-feedback integration requires a controller-capable DesktopEnv/SandboxDesktopEnv; the resources-server-only proxy is rejected explicitly.

The producer emits `exec_py(code_block=...)`, `computer.wait()` and `computer.terminate(status=..., answer=...)`. Both structured tool calls and function/parameter XML are decoded. Original reasoning, summaries and model commands remain in history. Execution feedback uses the original command SHA256 and the real controller return value.

## Coordinate consumers

`reference_unit` enforces the producer's literal [0,1] contract. `step1500_compat_v1` keeps [0,1] outputs working and accepts the observed grid/pixel alternatives. This named consumer is an explicit model-compatibility choice, not a universal claim about Nano4 coordinates.

| Selection | Input range | Pixel projection |
| --- | --- | --- |
| `# coordinate_space: unit` | [0,1] | `round(u*(size-1))` |
| `# coordinate_space: grid1000` | [0,1000] | `round(g/1000*(size-1))` |
| `# coordinate_space: pixel` | [0,size] | `min(size-1,round(p))` |
| No declaration, unit range | [0,1] | unit projection |
| No declaration, outside unit but inside grid | [0,1000] | grid1000 prior; ambiguity recorded |
| No declaration, outside grid but inside screenshot | [0,size] | pixel projection |

Grid and pixel coordinates overlap. The unlabelled fallback is a configured prior and cannot determine intent from numbers alone. An explicit comment resolves this overlap. Negative, nonfinite and out-of-bounds values produce execution errors instead of silently clipping an offscreen action.

Projection occurs immediately before PyAutoGUI calls in the guest, so ordinary variables, arithmetic, functions, loops, aliases, explicit from-imports and screenshot assignments work. Relative movement and cursor-only operations retain their PyAutoGUI behavior. The adapter records each source value, chosen space, ambiguity, screenshot size and projected value in `nemotron_v4_coordinate_projection`; raw controller output is retained. This audit describes projection before the device call, not task success.

The compatible consumer includes the rejected response and concrete parser error on a retry. Guest execution errors arrive through execution feedback. A task with completed verification keeps its original score, including zero; model failures are not infrastructure retry candidates.

## Validation scope

Pure device-boundary tests run without a producer checkout. Set `OSWORLD_V4_PROMPT_FORMAT_REPO` to the pinned local checkout to exercise its full serializer and feedback integration. A controlled replay of saved model responses additionally checks the parser/compiler acceptance boundary. Passing compilation does not prove a correctly grounded click.

The originating implementation was exercised on three real OSWorld desktop tasks and both unit and grid1000 actions. All20 Python commands returned success; all13 pointer projections were inspected (9 unit, 4 grid1000), and no run-owned desktop remained. These are runtime smoke checks, not a full-suite accuracy claim. The final package's test and dependency status is recorded in the PR description.
