# OSWorld Agent Design Notes

This file records the integration choices for running OSWorld through NeMo
Gym while preserving OSWorld's native agent contracts.

## Goals

- Keep the existing Gym-managed `gym_pyautogui` path stable.
- Add a native OSWorld `mm_agents.agent.PromptAgent` path without copying or
rewriting OSWorld's prompt templates.
- Make observation/action-space choices explicit and testable.
- Keep rollout execution inside the existing Gym `/run` boundary so
  `ng_run` and `ng_collect_rollouts` continue to work normally.

## Runtime Shape

`OSWorldAgent.run()` receives one Gym row, extracts
`verifier_metadata.osworld_task`, and launches one Ray task. The Ray task
runs `run_osworld_task()`, which owns:

- `DesktopEnv(...)` construction
- `env.reset(task_config)`
- per-step observe -> model -> action -> `env.step(...)`
- `env.evaluate()`
- `env.close()`

There is no separate OSWorld resource server. OSWorld's evaluator runs
inline through `DesktopEnv.evaluate()`.

## Runner Registry

`runner_registry.py` is the source of truth for OSWorld runner contracts.
Each `RunnerSpec` declares:

- runner kind: `gym_policy` or `prompt_agent`
- `DesktopEnv` import path
- action space
- observation type
- native agent class path, when applicable
- extra native agent kwargs

The default runner is still:

```text
gym_pyautogui -> screenshot observation + pyautogui action space
```

That path keeps Gym in charge of prompt construction and model calls.

The native runner:

```text
prompt_agent -> screenshot_a11y_tree observation + computer_13 action space
```

matches upstream OSWorld `PromptAgent` defaults. Explicit runners cover the
other supported upstream combinations:

- `prompt_agent_screenshot_pyautogui`
- `prompt_agent_computer_13`
- `prompt_agent_a11y_tree_pyautogui`
- `prompt_agent_a11y_tree_computer_13`
- `prompt_agent_screenshot_a11y_tree_pyautogui`
- `prompt_agent_screenshot_a11y_tree_computer_13`
- `prompt_agent_som_pyautogui`

`som + computer_13` is not registered because upstream `PromptAgent`
rejects that combination.

## Prompt Strategy

The Gym-managed `gym_pyautogui` prompt is intentionally self-contained in
`prompts.py`. It is byte-for-byte equivalent to upstream OSWorld
`SYS_PROMPT_IN_SCREENSHOT_OUT_CODE` after formatting the default password.

Native `PromptAgent` runners do not copy prompts into Gym. Instead, the
wrapper instantiates `mm_agents.agent.PromptAgent`, lets it pick the correct
upstream prompt based on `action_space` and `observation_type`, and patches
only `call_llm()`.

The patched `call_llm()` forwards PromptAgent-built OpenAI-style messages to
the Gym policy model server. It strips `<think>` / `<thinking>` blocks before
returning to `PromptAgent`, because PromptAgent parses the model output
itself and expects pure action JSON/code/sentinel text.

## Accessibility Tree Handling

Native observation modes that need accessibility data automatically construct:

```python
DesktopEnv(require_a11y_tree=True)
```

This applies to:

- `a11y_tree`
- `screenshot_a11y_tree`
- `som`

The default `gym_pyautogui` runner does not enable a11y by default. It can
still be forced with config overrides for diagnostic or research runs.

## Configuration Semantics

`runner_name`, `provider_name`, and other OSWorld agent fields are server
configuration. They must be set when launching `ng_run`, not only when
calling `ng_collect_rollouts`.

The native overlay:

```text
configs/osworld_agent_native_prompt_agent.yaml
```

sets `runner_name: prompt_agent`. Include it after the base OSWorld config in
`ng_run` config paths.

## Tests

The focused test suite is:

```bash
python -m pytest \
  responses_api_agents/osworld_agent/tests/test_runner_registry.py \
  responses_api_agents/osworld_agent/tests/test_client.py \
  responses_api_agents/osworld_agent/tests/test_app.py
```

Coverage includes:

- default `gym_pyautogui` compatibility
- native `PromptAgent` runner matrix
- native messages routed to the Gym policy model
- thinking-strip before native action parsing
- automatic a11y enablement for native a11y/SOM runners
- FastAPI `/run` success and error paths

## Smoke Script

`scripts/run_native_prompt_agent_smoke.sh` starts `ng_run` with the native
overlay, collects one rollout from `data/example.jsonl`, and prints a compact
summary. It defaults to one row and one sample in parallel to keep VM usage
small.

Use `RUNNER_NAME=...` to select a specific explicit native runner.

## Known Constraints

- `PromptAgent` is the only native `mm_agents.agent` class present in the
  current OSWorld fork installed on Colossus.
- The native path depends on upstream OSWorld prompt and parser behavior.
- `computer_13` actions are dictionaries and are preserved in
  `verifier_metadata.osworld_steps`.
- First-run OSWorld VM downloads must be prestaged for concurrent runs, or
  the docker provider can race on `Ubuntu.qcow2.zip`.
- Long a11y prompts can be large; use smaller `max_trajectory_length` or
  shorter smoke inputs when debugging.
