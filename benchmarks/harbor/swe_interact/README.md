# SWE-Interact on the Harbor task shape

[SWE-Interact](https://arxiv.org/abs/2606.30573) turns 75 SWE tasks (25 DeepSWE, 25 SWE Atlas refactoring, 25
SWE-bench Pro) into user-driven sessions. Its [dataset](https://github.com/scaleapi/SWE-Interact) runs as published
through the generic `harbor` environment server:

```sh
git clone https://github.com/scaleapi/SWE-Interact.git
export USER_MODEL_BASE_URL=<openai-compatible endpoint> USER_MODEL_API_KEY=<key>
gym eval run SWE-Interact/data/multiturn --agent codex_agent --sandbox opensandbox \
  --config benchmarks/harbor/swe_interact/swe_interact.yaml \
  --model-type openai_model --model <model> --model-url <url>
```

Each task is a Harbor multi-step task, and each step runs as one turn on the same sandbox, in a fresh agent
session (`environment_servers/harbor/README.md`, Multi-step tasks):

| Turn | The agent | Verifier |
| --- | --- | --- |
| `01_plan` | agrees on a plan with the user through `ask_user`, saves `/tmp/plan.md` | writes 1 |
| `02_implement` | carries the plan out, commits each pass, asks for review until the user approves | writes 1 |
| `03_handoff` | exports its commit history to `/logs/agent` | writes 1 |
| `04_write_tests` | adds tests in a separate commit | writes 1 |
| `05_test_handoff` | reports its final status | scored: commit checks, then the original verifier |

`multi_step_reward_strategy = "final"`, so the last turn's verifier is the reward. The task's top-level
`instruction.md` is the full original specification; it is never sent to the agent. The simulated user's
`persona.md` holds it, and the user reveals it one missing detail at a time after inspecting the agent's commits.

`swe_interact.yaml` sets what the run needs beyond the defaults:

- `builds.force`: the tasks' declared images lack `repo_exec_server.py`, the read-only command server the
  simulated user inspects the repository through, so every Dockerfile is built (Harbor's `--force-build`).
  Prepare also builds each task's `user-server` sidecar: 150 contexts in all. Builds need buildah or podman and a
  `builds.repository` for OpenSandbox.
- `task_mcp_grants: required`: the agent is granted the task's `user-server` MCP server on every turn.
- `compose_variables`: the sidecar's `${LITELLM_*}` and `${SIM_USER_MODEL}` point at the `user_model` Gym model
  server, under the episode's rollout prefix, so provider keys stay on the Gym host.
- `default_episode_timeout_seconds`: Atlas tasks budget about 26,000 s across their turns and verifiers.

Compose groups need OpenSandbox with `networking.enabled: true` (see the harbor README). Native Codex sessions
must accept MCP tool grants; without that, `task_mcp_grants: required` fails each seed with Codex's refusal. The
published runs use GPT-5.5 (high) as the simulated user; scores are comparable only with the same user model.
