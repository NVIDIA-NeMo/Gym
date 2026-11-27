# OpenCode plugins for long Agentic-VBench episodes

Three OpenCode plugins for sessions that compact: the prepare script stages them into every task's step `workdir/`
under `.opencode-plugins/`, so they reach the agent's sandbox at `/workspace/.opencode-plugins/`, and the training
entry's [`compaction_plugins.yaml`](../../../environments/harbor/agentic_vbench_train/compaction_plugins.yaml) loads
them through `opencode_agent`'s `plugins` (owner-staged plugins, loaded after Gym's own). Without that overlay the
files sit unused in the workspace. They change what the model reads after a compaction, never prompts, tools or
verifiers.

| Plugin | Hook | What it does |
| --- | --- | --- |
| `compaction_order.mjs` | `experimental.chat.messages.transform` | Moves each finished compaction summary (with its marker) to where the retained tail starts, so the model reads summary, then retained messages, then what followed. Messages are moved, never changed; an unknown tail start or a failed summary is left alone; mixed sessions fail loudly. |
| `compaction_anchor.mjs` | `experimental.chat.messages.transform` | Saves a session's first real user message (the task prompt) under OpenCode's state directory and puts it back verbatim at the front once a compaction has dropped it. A prompt that changed within the session, a missing anchor or a tampered file fail loudly. |
| `compaction_summary_header.mjs` | `chat.headers` | Sends `X-NeMo-Gym-OpenCode-Agent: <agent>` with every model call. `vllm_model` reads `compaction` to bound the summary's output (`compaction_summary_max_tokens`) and to retry a summary that is empty, reasoning-only, truncated or without the sections OpenCode's compaction prompt asks for (`compaction_summary_max_attempts`), instead of handing it to the session. |

The ordering and anchor plugins were written against OpenCode's message schema of 1.14 (`info.summary`,
`parts[].type == "compaction"` with `tail_start_id`) and are inert where a field is absent. They are a port of the
internal AgenticVBench timeout-and-compaction fixes (Leili Tavabi, NVIDIA GitLab `ltavabi/avbench-gym-fixes`, branch
`ltavabi/avb-compaction-recovery`), whose process-ownership half the shared sandbox supervisor
(`nemo_gym/agents/utils/process_supervisor.sh`) already provides on this path. Tests:
`tests/unit_tests/benchmarks/test_agentic_vbench_opencode_plugins.py` (the `node` tests beside this file, skipped
without `node`).
