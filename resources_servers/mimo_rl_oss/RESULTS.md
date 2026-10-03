# Results

All runs use OpenSandbox boxes from the released Docker Hub images and `nvidia/nvidia/nemotron-3.5-lightning` on the
NVIDIA inference endpoint, graded by the real verifiers. One rollout per cell unless noted.

## Grader checks

| Subset | Check | Result |
|---|---|---|
| code | untouched repo | 0.0 (2 of 26 hidden tests fail) |
| terminal_bench | untouched task | 0.0 (2 tests fail) |
| cyber | in-box PoC server | submissions return crash and match verdicts, opencode reproduced the crash (1.0) |
| music | clean hand-written tune | 0.66, blank-line variant rejected (0.0) |
| music | two Lightning compositions | 0.0 and 0.0, both rejected by the scorer's hard gates |

## Harnesses on one code task (format-code-task-001457)

| Profile | Reward | Seconds | Notes |
|---|---|---|---|
| default | 1.0 | 38 | |
| bashonly-agent | 1.0 | 66 | |
| cc-agent | 1.0 | 166 | |
| codex-agent | 1.0 | 85 | needs `ptc: false`, with code mode the model saw no tools |
| mimocode-agent | 0.0 | 60 | patch written, tests fail |
| claude-code | 1.0 | 80 | |
| codex | 1.0 | 90 | |
| mimocode | 1.0 | 77 | after pinning 0.1.15 |
| kilocode | 1.0 | 81 | |
| kimi-code | 1.0 | 86 | |
| openclaw | 1.0 | 137 | |
| opencode | 1.0 | 113 | |
| hermes | 1.0 | 229 | |
| mini-swe-agent | 1.0 | 116 | adapter reports an error status because there is no final message |
| pi | 0.0 | 89 | patch written, tests fail |
| omp | 0.0 | 196 | patch written, tests fail |
| kimi-cli | 0.0 | 76 | Lightning emitted invalid tool-call JSON and kimi-cli aborts on it |
| grok | | | rejects `nvidia/...` model names, needs Gym's model server with an alias |
| dsh | | | rejects names outside its catalog, needs Gym's model server with an alias |

## Other subsets

| Subset | Profile | Reward | Notes |
|---|---|---|---|
| cyber (arvo_35858) | opencode | 1.0 | |
| cyber | cc-agent | 0.0 | step limit, no matching crash |
| cyber | claude-code | 0.0 | endpoint context limit |
| terminal_bench | cc-agent, claude-code, opencode | 1.0, 1.0, 1.0 | |
| webdev | claude-code | 0.73 | Qwen3.5-122B vision judge |
| webdev | opencode | 0.0 | |
| webdev | cc-agent | dropped | page loads scripts from a mistyped CDN host, MiMo masks these |

## Integration findings

- Some released code images keep commits past the task's base, so mimoagent's history check fails. Code rows strip
  history per rollout (`git_leak_prevention: strip`).
- The general_agent images ship mcp 2.x in the system python, while the task scripts need mcp 1.x in
  `/opt/openai-agents-venv`. Setup builds that venv.
- The general_agent images have no git, so the in-box runtime installs mimoagent from a source archive.
- Claude Opus refuses the cyber tasks under Anthropic's cyber safeguards.
