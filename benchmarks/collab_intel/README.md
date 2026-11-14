# Collaborative Intelligence

[Collaborative Intelligence](https://github.com/zou-group/collaborative-intelligence) (Zou group, arXiv 2611.02633):
the model works in a team chat with partner seats and one scripted teammate who creates a problem (pressure to cut a
safety corner, a faked result, a hard bargain, a compromised handoff, ...). 8 dimensions, 160 of 240 scenarios released.

- `prepare.py` downloads the release at commit 3de5f7d0a652c0b393342b4fbae59b78034034ea into `data/repo/` (again
  when `data/repo/.commit` names another commit) and writes one row per released scenario of the scored set and test
  seed (157 scenarios x 3 seeds = 471 rows; the 3 encrypted GPQA-based scenarios are skipped).
- `collab_intel_agent` runs the upstream harness CLI from `data/repo/` in that repo's own uv venv: the scenario alone,
  then the test version only if the policy passed alone. Reward = the test run's pass.
- The score is the key metric `collab_intel/score`: upstream's mean test pass over rows with `alone_pass=true`, then
  the mean over dimensions. `mean/reward` is not the score: gate-failed rows are measured zeros there, since
  `mask_sample` marks only evaluation-side faults (a harness crash, a judge failure). `collab_intel/alone_pass_rate`
  and `collab_intel/unfinished` show how many rows the score skipped and why.
- Partners and referee judges are one self-hosted OpenAI-compatible endpoint (`COLLAB_INTEL_JUDGE_BASE_URL`,
  `COLLAB_INTEL_JUDGE_MODEL`, both required). `judge_extra` defaults to DeepSeek's thinking-off flag: change it with
  the judge model. The upstream leaderboard uses Claude/GPT/Qwen3 partners and Claude judges, so scores here are not
  comparable to it.
- Coding scenarios run model-written code (pytest in the policy's workspace, a few checkers' probes). By default
  (`code_execution: sandbox`) each phase gets one sandbox from Gym's sandbox providers (`sandbox_provider`: the name of
  a provider config block such as `nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml`, or an inline
  mapping; `sandbox_spec`: the image, which needs python, and `provider_options`). The sandbox must have no network: set
  a deny-by-default policy in `sandbox_spec.provider_options` (OpenSandbox: `network_policy: {defaultAction: deny}`). A
  probe when the sandbox starts (a connection to a public address on port 443) refuses one that can reach out, unless
  `sandbox_spec.require_no_egress` is false; OpenSandbox's policy still lets port 53 through. A sandbox failure ends the episode as an
  `agent_run_error`, never a run on the host. `code_execution: host` runs the code unsandboxed in a temp dir on the
  agent's host, for trusted models only: give the job a memory cap. Either way the harness gets only PATH, LANG, LC_ALL,
  TMPDIR, its own HARNESS_* settings and HOME set to the row's temp dir, never the server's tokens or keys.
- Eval only: `collab_intel_agent` is a bridge on the legacy `/run` path, wired through the `legacy_agent` relay, not an
  agent-session harness. The rollout's `response.output` is empty. The policy turns live in `collab_intel.*.messages` and in
  `ng_trajectory`, one invocation per phase. `ng_trajectory` is written only with `++rollout_correlation=true`.
- License: data CC BY 4.0 (7 scenarios CC BY-NC-SA 4.0), code MIT. Evaluation only; never train on it (some scenarios
  reuse GPQA).

```bash
python benchmarks/collab_intel/prepare.py
COLLAB_INTEL_JUDGE_BASE_URL=http://<judge host>/v1 COLLAB_INTEL_JUDGE_MODEL=<judge served name> \
gym eval run --config benchmarks/collab_intel/config.yaml \
  --split benchmark --output out/rollouts.jsonl --concurrency 16 \
  --model-type vllm_model --model <served name> --model-url http://<policy>/v1 --model-api-key dummy \
  ++rollout_correlation=true
```
