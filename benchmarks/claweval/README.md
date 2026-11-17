# ClawEval

Train on ClawEval task packages, NVIDIA-internal video and image agent tasks graded by the package's own `claw_eval`
grader with an LLM judge, through Gym's generic [`harbor` environment server](../../environment_servers/harbor/README.md)
with `opencode_agent`. The committed entry is
[`environments/harbor/claweval_train/`](../../environments/harbor/claweval_train/README.md); this folder holds the code
that prepares it.

| File | Role |
| --- | --- |
| `prepare_harbor_env.py` | Writes a root of task packages into the entry as Harbor task folders (`python -m benchmarks.claweval.prepare_harbor_env --tasks-root <root> --claw-eval-source <checkout>`): prompt and fixtures for the agent, grader material under `tests/`, `task_data.json`, the OpenCode compaction plugins |
| `judge_claweval.py` | The verifier entrypoint staged into every task's `tests/`: plays the package's snapshot commands in the agent's workspace, runs the package's grader with the judge and writes `reward.json` |
| `judge_transport.py` | The grader's judge client, with the NVIDIA Inference Hub Gemini route (full model id, inline image evidence) where the native client would strip the provider prefix |

The packages, the `claw_eval` package and the judge route are not part of this repository.
