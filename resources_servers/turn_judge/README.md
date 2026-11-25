# turn_judge: pairwise turn-level judge (sandbox + OpenCode + judge model)

Ported onto `binhu/swe-reply` (1564921c41) from Jiaqi's
`jiaqiz/turn-judge-service` (0c7d6b994). Matching RL branch: `binhu/nt4-turn-judge`.
The branch-replay environment remains unchanged.

## Running the service

Use your working Gym/OpenSandbox environment. From the Gym repository root:

```bash
uv run python -m resources_servers.turn_judge.service \
  --endpoint-file /shared/turn-judge/model.env \
  --service-env-out /shared/turn-judge/service.env \
  --opencode-install-script /mnt/s3-data/data/bxyu/opencode/install_cached_opencode.sh \
  --opencode-binary /mnt/s3-data/data/bxyu/opencode/opencode-linux-x64 \
  --opencode-musl-binary /mnt/s3-data/data/bxyu/opencode/opencode-linux-x64-musl-v1.17.11
```

The three `--opencode-*` options are the OpenCode assets as seen inside the judge sandbox; give the same paths as
the rollout agent's `remote_opencode_*` settings. Without them the upstream defaults under
`/mnt/s3-data/data/harness/opencode` are used. Rollouts with `mask_sample` (outcome not measured) are not paired.

`model.env` must provide `JUDGE_BASE_URL` and `JUDGE_MODEL_NAME` for an already
served judge model (upstream uses Qwen3.8-Flash-Next). `service.env` is written
by this process and is the file read by RL's `grpo.turn_credit.judge.endpoint_file`.
The two files are different. The labeling prompt defaults to the bundled
`LABELING_annotator.md`; use `--labeling` to override it.

Required infrastructure: OpenSandbox provider configuration, accessible task
images, S3-cached OpenCode assets used by `runner.py`, and identical shared paths
for Gym rollout artifacts in both rollout and judge processes. No model or
sandbox allocation is started by this command. `--tb-task-root` overrides
`/terminal-bench-tasks`; `TURN_JUDGE_GYM_ROOT` overrides the repository root used
for relative task folders. Use `--help` for concurrency, timeouts and work paths.

This service has no authentication: bind/expose it only inside the trusted
training network. Requests reference local artifacts and may create sandboxes.

CPU tests use a fake model/sandbox. Optional archive fidelity tests use
`TURN_JUDGE_FIXTURE_ROOT` and are skipped without those external fixtures.

## Implementation

Labels the decisive turns of agent rollouts (`good` / `harmful`, group `ill_posed`) for turn-level advantage
reshaping, built like the GAR group grader: each judgment runs in its own OpenSandbox sandbox (task image), where an
OpenCode session driving the judge model (Qwen3.8-Flash-Next) reads a `/judge` workspace and writes
`/judge/out/pair.json`.

- `pairing.py`: pairs = size of the larger outcome side; each rollout of that side gets one partner from the smaller
  side (shuffled per group, cycled so every smaller-side rollout is used). Uniform groups produce no pairs.
- `prompt.py`: the pair instructions; the rules are the turn_judge LABELING.md (annotator edition) uploaded as
  `/judge/LABELING.md`.
- `runner.py`: one pair = one sandbox: upload a tarball, install the S3-cached OpenCode, `opencode run` against the judge
  endpoint, read back `pair.json`, `notes.md` and the session export, stop the sandbox.
- `merge.py`: mechanical checks (idx snapped to the call start, verbatim quotes, one decision label per call) and the
  group merge (smaller-side rollouts keep labels given by at least half of their pairs; ill_posed by pair vote).
- `offline.py`: development driver over labeled groups (turn_judge workspaces), append-only outputs.

- `service.py`: the training judge service (HTTP: `POST/GET/DELETE /judge_group`, `GET /health`, `POST /shutdown`):
  per sealed GRPO group, the workspace is built on node-local
  disk from the trainer's payload (`workspace.py`, vendored from turn_judge `tools/group_workspace.py`), then
  offline.py's variant-D path runs (one stage-1 session, no retry; one session per pair; merge of the finished
  pairs at the request's deadline). Launch it with the command above.
- `tests/`: service tests with a fake `run_pair` (CPU only) and the workspace fidelity test against offline
  workspaces (cluster storage).
