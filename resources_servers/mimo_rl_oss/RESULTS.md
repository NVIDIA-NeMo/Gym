# Results

All runs use OpenSandbox boxes from the released Docker Hub images and the real verifiers. The first sections use
`nvidia/nvidia/nemotron-3.5-lightning` on the NVIDIA inference endpoint with mimoagent driving the box from outside.
The last section runs the full Gym path (harness_agent, `agent: mimoagent`, in-box runtime) with Nemotron 3.5 Nano
(`nano-3p5-honest-dolphin-rlvr-v41`) self-hosted on vLLM.

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

## Gym path with self-hosted Nano (4 code tasks, 2 rollouts each)

| Profile | Mean reward | Pass@2 | Notes |
|---|---|---|---|
| default | 0.38 | 0.50 | |
| bashonly-agent | 0.25 | 0.25 | |
| cc-agent | 0.63 | 0.75 | |
| codex-agent | 0.13 | 0.25 | |
| mimocode-agent | 0.50 | 0.50 | |
| claude-code | 0.25 | 0.25 | |
| codex | 0.13 | 0.25 | |
| mimocode | 0.25 | 0.50 | |
| kilocode | 0.43 | 0.50 | |
| kimi-code | 0.50 | 0.67 | |
| kimi-cli | 0.25 | 0.25 | |
| openclaw | 0.25 | 0.25 | |
| opencode | 0.50 | 0.75 | |
| hermes | 0.25 | 0.25 | |
| omp | 0.25 | 0.25 | |
| mini-swe-agent | 0.13 | 0.25 | |
| dsh | 0.63 | 0.75 | |
| pi | 0.17 | 0.25 | |
| grok | 0.00 | 0.00 | Gym's Anthropic stream sends thinking blocks without `signature`, which grok's client requires |

## Integration findings

- Some released code images keep commits past the task's base, so mimoagent's history check fails. Code rows strip
  history per rollout (`git_leak_prevention: strip`).
- The general_agent images ship mcp 2.x in the system python, while the task scripts need mcp 1.x in
  `/opt/openai-agents-venv`. Setup builds that venv.
- The general_agent images have no git, so the in-box runtime installs mimoagent from a source archive.
- Claude Opus refuses the cyber tasks under Anthropic's cyber safeguards.
- Gym's chat schema rejects the `name` mimoagent puts on tool messages, and its Responses schema requires `strict` on
  function tools. The agent fixes both before sending.
- harness_agent on main points boxes at the policy server's backend `base_url`, so with vLLM the harness reaches vLLM
  directly and sends its profile's model name. Profiling serves those names as vLLM aliases, or sets
  `sandbox_model_base_url` to the Gym model server.
- The CLI harnesses append their own API path, so they get the bare model URL.
