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
- Each coding-scenario pytest request runs in a fresh Gym sandbox with networking disabled and no host mounts.
  Only that episode's validated regular files cross the boundary; the trusted harness and model/judge requests stay
  on the server. Configure a named Docker or OpenSandbox provider and a reviewed, prepared image containing Python,
  pytest and the scenario dependencies. Missing isolation fails closed with `agent_run_error`, never host pytest.
  Non-coding phases that do not request pytest need neither the provider nor the image.
- Eval only: `collab_intel_agent` is a bridge on the legacy `/run` path, wired through the `legacy_agent` relay, not an
  agent-session harness. The rollout's `response.output` is empty. The policy turns live in `collab_intel.*.messages` and in
  `ng_trajectory`, one invocation per phase. `ng_trajectory` is written only with `++rollout_correlation=true`.
- License: data CC BY 4.0 (7 scenarios CC BY-NC-SA 4.0), code MIT. Evaluation only; never train on it (some scenarios
  reuse GPQA).

## Run

1. Prepare the benchmark inputs with `python benchmarks/collab_intel/prepare.py`.
2. Create `collab-intel-deployment.yaml` with the Docker example below. Docker requires a running local daemon.
   For OpenSandbox, compose its [provider configuration](../../nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml)
   under the same `sandbox` name instead.

   ```yaml
   config_paths:
     - nemo_gym/sandbox/providers/docker/configs/docker.yaml

   collab_intel_benchmark_agent:
     responses_api_agents:
       collab_intel_agent:
         pytest_sandbox_image: example.invalid/pytest-sandbox:reviewed
   ```

3. Replace the non-pullable example image with a reviewed deployment image reference. Prepare all scenario dependencies
   in that image before running: pytest cannot download packages from its offline sandbox. Keep the repository default
   unset; do not select the image through an environment variable.
4. Run the benchmark with the deployment override composed after the benchmark config:

```bash
COLLAB_INTEL_JUDGE_BASE_URL=http://<judge host>/v1 COLLAB_INTEL_JUDGE_MODEL=<judge served name> \
gym eval run --config benchmarks/collab_intel/config.yaml --config collab-intel-deployment.yaml \
  --split benchmark --output out/rollouts.jsonl --concurrency 16 \
  --model-type vllm_model --model <served name> --model-url http://<policy>/v1 --model-api-key dummy \
  ++rollout_correlation=true
```
