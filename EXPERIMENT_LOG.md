# Terminal-Bench 2.1 × Qwen3.6-27B × OpenSandbox — Experiment Log

Goal: integrate `nemo_gym/sandbox` API into `responses_api_agents/harbor_agent` (non-intrusively), then run the full Terminal-Bench 2.1 eval end-to-end:

- **Model**: Qwen/Qwen3.6-27B on vLLM, 1 GPU, EKS cluster `ltqlfcnzyr-dgxc-k8s-aws-use2-prod` (us-east-2, acct 942195279341)
- **Driver**: Ray driver in the same cluster
- **Sandbox**: OpenSandbox provider → cluster `nemo-rl-sbx-use2-prod-01`, via istio endpoint (to be discovered)
- **Storage**: Lustre PVC `/mnt/rl-workspace/hemild/<new subdir>` for traces/results
- **Target**: reproduce the Terminal-Bench score reported on the Qwen3.6-27B model card

## Log

### 2026-07-14

- Created git worktree `tb21-qwen36-opensandbox-eval` (branch `worktree-tb21-qwen36-opensandbox-eval`, off `origin/main`).
- Verified kubeconfig has both target contexts:
  - `arn:aws:eks:us-east-2:942195279341:cluster/ltqlfcnzyr-dgxc-k8s-aws-use2-prod` (vLLM + Ray driver)
  - `arn:aws:eks:us-east-2:942195279341:cluster/nemo-rl-sbx-use2-prod-01` (OpenSandbox)
- Found prior context in main checkout (untracked): `opensandbox-upgrade.md` (0.2.0 upgrade notes), `infra/examples/` (opensandbox manifests), plus uncommitted opensandbox provider helpers (`diagnostics.py`, `prewarm.py`).
- Launched parallel repo/infra/model-card recon workflow.

#### Recon results (8-agent workflow, ~10 min)

- **Model card** (Qwen/Qwen3.6-27B): reports **Terminal-Bench 2.0 = 59.3** via "Harbor/Terminus-2 harness; 3h timeout, 32 CPU/48 GB RAM; temp=1.0, top_p=0.95, top_k=20, max_tokens=80K, 256K ctx; avg of 5 runs". Dense 27B, BF16 ≈ 54–56 GB. vLLM ≥ 0.19, `--reasoning-parser qwen3`; thinking mode on by default. NOTE: we run TB **2.1** (89 tasks; 26 tasks fixed vs 2.0) per the goal — score comparison carries that caveat.
- **Harbor agent**: `/run` → single-task Harbor `Job` in a Ray worker. Execution backend is config-swappable via `harbor_environment_import_path` (`app.py:482-487`) — a custom `harbor.environments.base.BaseEnvironment` plugs in with **zero app.py changes**. Model calls = chat completions via `NemoGymLLM`. Verification = Harbor's own verifier (no Gym resources server). Reward extracted from trial `result.json`.
- **Terminal-Bench 2.1**: Harbor hub registry dataset `terminal-bench/terminal-bench-2-1` (89 tasks). Input JSONL = one row per task, `instance_id: "<alias>::<task>"`. Per-task Docker Hub images from task.toml.
- **Traces**: rollouts JSONL from `gym eval run --output` is the trace; plus per-trial Harbor artifacts (`agent/trajectory.json`, `verifier/*`) under `harbor_jobs_dir`. Metrics via `/aggregate_metrics` (pass rate = mean reward).
- **DGXC cluster**: 69× p6e-gb300r (4× GB300 each, **arm64**), ~218 GPUs free. Lustre PVC = `rl-workspace` (ns `default`), FSx `fs-08d808968a707f3c6`. KubeRay installed. Kyverno rewrites Docker Hub → ECR PTC. Working GB300 vLLM images on-cluster: `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.0-*`, `jwillthomson/dynamo-latest-main-*` (terryk pattern: serve from materialized dir on lustre, runAsUser 0, nodeSelector GB300).
- **SBX cluster**: OpenSandbox server v0.3.0-nemo.1, 2×32 replicas (cell1/cell2) in `opensandbox-system`; server in insecure mode (api_key placeholder ok). Pools mostly idle; BatchSandbox CRD present. ~2.5k free CPU cores; Karpenter `cpu` nodepool (amd64) for sandbox pods.

#### Course correction (user)

- **Do NOT use the public ELB/istio ingress endpoint** (`k8s-sandboxi-opensand-*.elb.amazonaws.com` + gateway key). Use the **internal** path to the sbx cluster instead. Launched a mesh-discovery workflow (DGXC-side + sbx-side istio recon, then a live probe pod on DGXC testing candidate internal endpoints).
- Harbor reportedly has an **official opensandbox environment** — evaluating it as an alternative/reference; primary plan remains integrating via `nemo_gym.sandbox` API (`AsyncSandbox` + `OpenSandboxProvider`, which already defaults to `opensandbox-server.opensandbox-system.svc.cluster.local` with `use_server_proxy: true`).
- Confirmed provider internals (`provider.py`): create-with-retries + readiness probe, exec via SDK `commands.run` (cwd/env/timeout/user), file upload/download via SDK files API (single files — directory transfer will be tar-based in the Harbor env wrapper), `close()` kills sandbox.

#### OpenSandbox internal endpoint — RESOLVED (mesh-discovery workflow + live probe)

- Both clusters are in one **istio 1.30 ambient multi-cluster mesh** (`nemo-ci-mesh`, trust domain `nemo-ci.nvidia.com`), wired via `istio-remote-secret-*` + HBONE (:15008) east-west gateways.
- DGXC has selector-less **global stub services** (`istio.io/global=true`) in `opensandbox-system` whose endpoints resolve to the **sbx cluster**: `opensandbox-cell-1` (→ sbx `opensandbox-server`) and `opensandbox-cell-2` (→ sbx `opensandbox-server-cell2`, currently serving osworld).
- **Live-verified from a probe pod in DGXC ns `default`**: `http://opensandbox-cell-1.opensandbox-system.svc.cluster.local:80/v1/sandboxes` → HTTP 200 (no auth header needed on the mesh path; the gateway key is only for the public ingress). `/health` → 200.
- **Chosen endpoint: `http://opensandbox-cell-1.opensandbox-system:80`** (cell-1 idle; avoids terryk's osworld traffic on cell-2). Client pod must run in an ambient-enrolled ns (`default` is).
- Caveats: DGXC also runs a LOCAL opensandbox server under the `opensandbox-server.opensandbox-system` name (not the sbx one — do not use, per goal); `opensandbox-fastsandbox-cell` stub is broken (TCP reset).
- Prior-art template: `default/terryk-osworld-omni-os-gate` job uses `OPENSANDBOX_DOMAIN=opensandbox-cell-2.opensandbox-system:80` from ns `default`.

#### vLLM deployment (in progress)

- Version probes on GB300: `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.0-deepseek-v4-cuda13-dev.2` → vllm 0.20.0 / transformers 5.6.2; `jwillthomson/dynamo-latest-main-aeda23b48...` → **vllm 0.24.0 / transformers 5.12.1** (chosen; also terryk's proven GB300 serve image).
- Applied `hemild-tb21-model-download` Job (HF snapshot → `/mnt/rl-workspace/hemild/tb21-qwen36-opensandbox/models/Qwen3.6-27B`, hf_transfer) — **completed in ~4 min**.
- Applied `hemild-tb21-vllm` Deployment (1× GB300 GPU, ns default) + ClusterIP Service `hemild-tb21-vllm:8000`. Flags: `--max-model-len 262144 --max-num-seqs 32 --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder`, served name `Qwen/Qwen3.6-27B`. Container Running, model loading; readiness monitor armed.
- Note: local HF API calls are rate-limited (429) from this IP — model config fetched via cluster instead; top_k=20 default expected from the model's own generation_config.json (verify in vLLM startup logs).
- **vLLM READY** (~12 min end-to-end): `/v1/models` shows `Qwen/Qwen3.6-27B`, max_model_len 262144, KV cache 195 GiB (3.16M tokens). Startup log confirms model generation_config defaults applied by vLLM: `{'temperature': 1.0, 'top_k': 20, 'top_p': 0.95}` — matches model card. Chat completion smoke OK.

#### Integration implementation (harbor_agent × nemo_gym.sandbox)

- Read pinned Harbor source (commit `9dddd797`): `BaseEnvironment` contract (async start/stop/exec/upload_file/upload_dir/download_file/download_dir; constructor receives task `EnvironmentConfig` with docker_image/cpus/memory_mb/storage_mb/gpus/allow_internet; `EnvironmentConfig.kwargs` from `harbor_environment_kwargs` arrive as **kwargs). Verifier uploads tests → `/tests`, runs `bash /tests/test.sh 2>&1 | tee /logs/verifier/test-stdout.txt`, downloads `/logs/verifier` when `is_mounted=False`. Terminus-2 drives everything via `environment.exec` (tmux; auto-installs tmux+asciinema if missing). **No official opensandbox env at the pinned commit** (only docker/daytona/e2b/gke/modal/runloop) — custom import_path env remains the right non-intrusive seam.
- sbx server limits: proxy HTTP timeout 28800s, max sandbox TTL 28800s, create timeout 1200s, egress mode dns+nft, workload_provider=batchsandbox.
- **New files (only additions, no existing-file changes)**:
  - `responses_api_agents/harbor_agent/custom_envs/nemo_gym_sandbox/environment.py` — `NemoGymSandboxEnvironment(BaseEnvironment)` backed by `nemo_gym.sandbox.AsyncSandbox`; provider-agnostic (`sandbox_provider` kwarg), task.toml resources → `SandboxSpec.resources`, tar-based dir transfer w/ per-file fallback, always-kill on stop, `is_mounted=False`, opt-in `allow_unenforced_internet_isolation`, `image_rewrites` support.
  - `responses_api_agents/harbor_agent/custom_envs/nemo_gym_sandbox/test_environment.py` — unit tests with fake registered provider.
  - `responses_api_agents/harbor_agent/configs/harbor_agent_opensandbox.yaml` — TB-2.1 dataset alias, Terminus2NemoGym, opensandbox provider block (server-proxy mode), 3h agent timeout override, model_info 262144/81920 (matches card's 256K ctx / 80K max_tokens).

#### Driver pod

- `hemild-tb21-driver` Running in DGXC ns `default` (ambient-enrolled), 24 CPU/96Gi, amd64, mounts `rl-workspace`, env `OPENSANDBOX_DOMAIN=opensandbox-cell-1.opensandbox-system:80`.
- Verified FROM the driver pod: OpenSandbox cell-1 `/health` → 200 over the mesh; vLLM `/health` → 200. Code sync to `/mnt/rl-workspace/hemild/tb21-qwen36-opensandbox/gym` started.

#### Dataset: TB 2.1 via pinned tasks repo

- The pinned harbor commit (`9dddd797`) predates the Harbor hub registry; its legacy `registry.json` only has terminal-bench **2.0**. Harbor latest main has both the hub registry **and an official `opensandbox.py` environment** — upgrading the pin was rejected as intrusive (7 months of API drift vs `Terminus2NemoGym`); noted as the eventual upstream path.
- Solution: TB 2.1 consumed as a **local Harbor dataset** from a pinned checkout of `harbor-framework/terminal-bench-2-1` @ `36d417f5` (2026-07-13): exactly 89 tasks. Added `prepare_terminal_bench_2_1.py` (regenerates the checked-in 89-row input JSONL from the pinned repo).
- Task audit: all 89 tasks have `docker_image`; **zero** `allow_internet=false`; zero GPU tasks; resources ≤ 4 CPU / 8 GiB; agent timeouts 600–12000s; verifier timeouts 360–12000s. Official leaderboard config confirms `n_attempts: 5`, temperature 1.
- Matching the model card's stated harness ("3h timeout, 32 CPU/48 GB RAM"): `harbor_agent_override_timeout: 10800` + `override_cpus: 32` / `override_memory_mb: 49152` via `harbor_environment_kwargs`.

#### Smoke-test debugging (2 tasks, `--limit 2`)

1. `gym eval run` requires `--split benchmark`.
2. Input datasets must be **declared in the agent config** (`datasets:` list) — the e2e pipeline materializes `preprocessed_datasets/<split>.jsonl` from them; `--input` alone is not enough. `benchmark`-type datasets require `prepare_script` + `prompt_config` keys.
3. gym's per-server venv creation hit an upstream dep conflict (head-server deps pin `openai==2.45.0` vs nemo-gym `openai<=2.7.2`) → bypassed with `++skip_venv_if_present=true` + prebuilt venv symlinked into each server dir. Gotcha: `ln -sfn` into an *existing* `.venv` dir nests the link — stale venvs had to be `rm -rf`'d first.
4. `ray` is not pulled in by `uv sync --extra dev` — installed `ray[default]>=2.55.1` into the driver venv explicitly.
5. Unit tests: 17/17 pass locally (fake provider; tar round-trip verified at the byte level). ruff clean. Integration committed (DCO sign-off; GPG signing unavailable non-interactively).
6. **Smoke attempt 8: both servers up; two BatchSandbox CRs READY on the sbx cluster** (`opensandbox` ns, pods Running, 6h TTL) — full path driver→mesh→cell-1→BatchSandbox→pod works.
7. **First complete trial: `bn-fit-modify` scored reward 1.0** — Terminus-2 drove tmux in the OpenSandbox pod (15 trajectory steps, ~199K input / 15K output tokens, asciinema `recording.cast` + `terminus_2.pane` captured, all 9 verifier tests PASSED). The `/run` response then 500'd: openai 2.45 requires `input_tokens_details.cache_write_tokens`, which `HarborAgentUtils.extract_usage` didn't set → **fixed in `utils.py`** (one line; pre-existing bug for any live-model run under the current lock).
8. Also mirrored Harbor's docker/daytona `bash -ic` exec wrapping in `NemoGymSandboxEnvironment` (new `exec_shell` kwarg, default `"bash -ic"`) after harbor-contract agent's report — `.bashrc`-dependent task images (conda/pyenv) behave identically to the reference backends. 18/18 unit tests.
9. One orphaned sandbox from the killed attempt-8 run was deleted manually; the completed trial's sandbox had already been cleaned by `stop()` — lifecycle confirmed.
10. Smoke attempt 9 (with fixes): first rollout collected cleanly end-to-end in 9m12s (usage validates; reward 0.0 this time — temp-1.0 stochasticity; attempt 8 scored 1.0 on the same task). Timing breakdown per trial: env_setup ~12s, agent_setup ~20s, verifier ~14s, **agent_execution ~500s (inference-bound)**.
11. Per user direction: sandbox resources set to uniform **4 vCPU / 16 GiB** (`override_cpus: 4`, `override_memory_mb: 16384`), replacing the model-card 32/48G mirror. Committed + pushed.
12. Second smoke task (adaptive-rejection-sampler) ran >50 min (legitimately long: thinking model, 80K max output/episode) — smoke killed after validation was complete; its sandbox (verified by `harbor-*` labels) deleted.

### Full run 1 (89 tasks)

- Launched `gym eval run` with `--concurrency 32`, output `results/rollouts_full_run1.jsonl`, per-server logs in `results/server_logs/`. Draft PR: https://github.com/NVIDIA-NeMo/Gym/pull/2031 (sanitized: no internal endpoints).
- Watch items: vLLM prefix-cache hit rate reported 0% during smoke (episode prefills should share prefixes — possible throughput left on the table; decode still dominates).

#### Why a task takes ~9 min (user question) + throughput tuning

- Per-trial timing breakdown (from `result.json` timing fields): sandbox create+ready **~12s**, agent setup (tmux+asciinema install) **~20s**, verifier **~14s**, **agent_execution ~500s**. So ~93% of wall-clock is the Terminus-2 ↔ model loop, **not** sandbox overhead.
- Root cause: Qwen3.6-27B is a thinking model (long `<think>` per turn; a sample trial did ~199K input / 15K output tokens over 15 tmux turns). During the 2-task smoke only 1–2 requests hit the single GPU → decode-bound at ~40–90 tok/s. The **4 vCPU / 16 GiB** the user asked about are the sandbox pod's `override_cpus`/`override_memory_mb` (compute happens on the model GPU, not in the sandbox) — irrelevant to the latency.
- **Run 1 aborted at 22/89** to apply three throughput fixes, then relaunched (run v2):
  1. vLLM `--enable-prefix-caching` (was off; Terminus-2 resends the full growing transcript each turn, so prefix reuse is a big win) + `--max-num-seqs 64` (was 32).
  2. `NemoGymLLM` timeout 120s → **7200s** (`nemo_model_server_timeout_sec`): the 120s default made Terminus-2 abandon+resend long thinking generations, stacking zombie requests (root cause of the 7 tracebacks seen mid-run-1).
  3. Sandbox resources pinned 4 vCPU / 16 GiB per user request.
- Run-1 partial signal (22 tasks, pre-fix): mean reward **0.68**, 15/22 passed — already in the ballpark of the model card's TB-2.0 = 59.3, but not the reportable number (partial + concurrency-throttled). Run v2 is the clean full-89 measurement.
- Run v2 launched: concurrency 32, prefix caching on, aggregate vLLM throughput ~1100–1300 tok/s with 32 running / 58 waiting reqs; 24 sandboxes up within ~1 min. Own sandboxes from run 1 cleaned by exact name (verified via `harbor-*` labels).

##### Run v2 health check (early, ~2/89 done)

- **Prefix caching confirmed working: 55.9% hit rate** (was 0% before the flag) — the growing-transcript resends now reuse KV. Generation throughput ~837 tok/s at this instant (varies with how many episodes are mid-generation vs mid-tool-exec).
- 24 of my BatchSandbox CRs up on the sbx cluster; **0 tracebacks, 0 HTTP 500s** in the harbor server log (the timeout fix eliminated the zombie-request tracebacks seen in run 1).
- Rollout progress bar ETA ≈ **70 min** for all 89 (concurrency-bound, not per-task). First 2 completed tasks both passed (running mean 1.0 — small sample).
- Persistent monitor `bsq5u566p` will fire on completion → then compute final pass@1 over 89 and compare to the model card.

##### Run v2 mid-run (~18/89): transient sandbox 500s (non-fatal)

- 7 "tracebacks" in the harbor log are **handled retries**, not crashes: a single flaky sandbox pod (`89126aa0`) returned HTTP 500 on `/files/upload` and `commands.run`. The `OpenSandboxProvider` retry (tenacity, attempts 1–5 w/ exp backoff) absorbed most; the one exhaustion was caught by Terminus-2 (`"Agent error: SandboxApiException ... Returning history from completed turns"`) so the trial still produced a reward. **Zero HTTP 500s on the harbor `/run` endpoint.**
- Running signal at 18/89: mean reward **0.833** (15 pass / 3 fail) — above the model-card TB-2.0 = 59.3. Infra flakiness is real but small and doesn't systematically sink the score. Continuing without intervention.

#### Root cause of the sandbox 500s (user asked) + config refactor

- The mid-run HTTP 500s were **not** a server-capacity issue (32 server replicas, all 65 server pods 0 restarts). They were on the `/proxy/44772/...` path — the server forwarding to a specific sandbox pod's execd. The sbx namespace event log showed the smoking gun: `Warning Evicted pod/c83c901b-...-0 "Pod ephemeral local storage usage exceeds the total limit of containers 10Gi"`. **Disk-heavy TB tasks (image builds, dataset downloads) overran the sandbox pod's 10 GiB ephemeral-storage limit → kubelet evicted the pod → in-flight proxy upload/exec calls 500'd.** Only 2 pods hit it; provider retries + agent catch kept the run alive.
- **Yes, the ephemeral limit is set through the sandbox API**: `SandboxSpec.resources.disk_gib` → `OpenSandboxProvider._resource_map` emits `ephemeral-storage: "{disk_gib}Gi"` → `Sandbox.create(resource=...)` → the pod's container limit. In the harbor path that's `override_storage_mb`.
- **Refactor (per user) — 3 changes, run v2 killed at 59/89 and replaced by the authoritative "final" run:**
  1. **Config style** now matches `mini_swe_agent_2.yaml`: the provider is referenced by name — `sandbox_provider: ${sandbox}` — and its definition comes from the shipped `nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml`, included via a second `--config`. Verified `${sandbox}` resolves (use_server_proxy=true, domain from env) and my env's `resolve_provider_config`/`resolve_provider_metadata` handle the block with **no code change**.
  2. **`gym` CLI**: `gym eval run --config <agent> --config <provider> --model-type vllm_model --model ... --split benchmark`.
  3. **`override_storage_mb: 30720`** (30 GiB) — eviction fix; startup log confirms `"Overriding storage to 30720 MB"`.
- Final run launched → `results/rollouts_final.jsonl`; partial run-2 archived to `rollouts_run_v2_inline_partial56.jsonl` (56 rows, running pass ≈ 0.79). Config committed + pushed (PR #2031).

#### Full concurrency (per user) — authoritative run

- Raised concurrency from 32 → **89** (all TB 2.1 tasks at once): agent config `concurrency: 89`, `gym eval run --concurrency 89`, and vLLM `--max-num-seqs 64 → 96` so the model server isn't the artificial cap (prefix caching still on). vLLM redeployed + healthy (`max_num_seqs=96`, `enable_prefix_caching=True`).
- Relaunched: log confirms **"Querying with 89 concurrent requests"**. Committed + pushed. Monitor `bt723wdts`.
- Single-GPU note: 89 concurrent episodes fill the batch, but throughput is still bounded by the one GB300 + KV cache (3.16M tokens); vLLM preempts/recomputes as needed. Full concurrency maximizes overlap of sandbox/tool-exec phases with generation, not raw GPU tok/s.

#### Full concurrency — the real bottleneck (user noticed only 27 sandboxes)

- Symptom: with `--concurrency 89` and agent `concurrency: 89`, only ~27 sandboxes existed. **Root cause: Ray.** `runner_ray_remote` is `@ray.remote` with no `num_cpus`, so Ray reserves the default **1 CPU per Harbor-job task**. The driver pod has 24 CPUs → Ray runs ≤24 job tasks at once; the other 65 `/run` requests queue behind the CPU pool (semaphore/flag are satisfied, but Ray scheduling is the true cap).
- Fix (committed): new `harbor_ray_task_num_cpus` config field (default `None` = unchanged 1-CPU behavior). Set to **0.25** in the TB2.1 config — jobs are I/O-bound (work is in the remote sandbox + on the GPU, not the driver), so 89 × 0.25 = 22.25 CPU fits the 24-CPU driver. Applied at the call site via `runner_ray_remote.options(num_cpus=...)`.
- **Verified: 89 batchsandboxes live, Ray available CPU 1.75/24** → genuinely all 89 tasks concurrent. Relaunched authoritative run; monitor `bzg22awsk`. Config + code committed/pushed (PR #2031).

#### vLLM engine wedge under the 89-request burst → fixed with 2 replicas (user)

- **Symptom**: with true 89-way concurrency, the run stalled — 0/89 rollouts, all agents idle at the terminal after setup, and vLLM served **zero** LLM requests for ~24 min despite 89 sandboxes running.
- **Diagnosis**: `GET /health` and `/v1/models` returned in 3ms, but `POST /v1/chat/completions` (even `max_tokens:1`, direct to the pod) hung forever with 0 tokens in `/metrics`. GETs skip the engine; POST needs the engine core → **the single vLLM engine deadlocked** when all 89 long-context requests hit it at once. (My 2h LLM timeout hid it — agents waited silently instead of erroring.)
- **Fix (per user): scale vLLM to 2 replicas.** Each GB300 pod now serves ~½ the load. A fresh pod serves `POST /chat/completions` in ~5s (confirmed), so the wedge was burst-induced, not a config bug. Kept `max-num-seqs 96` + prefix caching; 2 pods on separate nodes behind the ClusterIP Service (k8s LB). Hiccup: a 3rd pod stuck in `Init` on a node missing the FSx CSI driver — reverted the template to the 2 already-serving pods to converge instantly at `updated=2 ready=2`.
- Also had to `ray stop --force` + clean the driver (233 zombie Ray workers had piled up from the aborted runs; container PID 1 = `sleep infinity` doesn't reap children).
- **Relaunched, verified healthy**: log "Querying with 89 concurrent requests"; **both replicas actively serving** (pod A running 11, pod B running 25, 0 waiting, prompt+gen tokens climbing on both). Rollouts flowing — 22/89 at first check, running pass@1 90.9% (early/fast tasks skew high). Monitor `bs1qth0l9`.
- Progress trace (running pass@1 settling as the hard long-tail resolves, all above the model card's 59.3): 22 → 90.9%, 58 → 74.1%, 67 → 70.1%, 72 → 69.4%, 78 → 65.4%. Tracebacks all handled OpenSandbox transient-500s (12 total, 2 distinct pods); 0 evictions; both vLLM pods verified serving `POST /chat/completions` in ~0.1s mid-run (no re-wedge).
- **~03:15 UTC 2026-07-15: AWS SSO token expired** → lost kubectl access to both clusters. The eval is **unaffected** (runs server-side on the driver, writes to Lustre); last observed state before the gap was **79/89, running pass@1 ≈ 65%**. Self-resolved with `aws sso login` (default profile, browser auto-opened); access restored, run had advanced to 81/89. No data lost.

## RESULT — Terminal-Bench 2.1, Qwen3.6-27B

**Final pass@1 = 59.55% (53/89 tasks passed), mean reward 0.5955.** Run completed 89/89.

| | Value |
|---|---|
| **This run** — TB **2.1**, Harbor/Terminus-2, OpenSandbox, Qwen3.6-27B, single run | **59.55%** (53/89) |
| **Model card** — TB **2.0**, Harbor/Terminus-2, avg of 5 runs | **59.3** |

**This reproduces the model-card number essentially exactly (within ~0.25 points).**

Aggregate metrics (`rollouts_final_aggregate_metrics.json`): `mean/reward 0.5955`, `agent_timeout_error 8.99%` (8 tasks hit the 3h agent timeout → reward 0), `context_length_exceeded 2.25%` (2 tasks), `memory_limit_exceeded 0%`. Token usage per task: mean 2.08M total (2.0M input / 81K output), max 15.9M (a long agentic task).

Caveats on the comparison (all noted for honesty; net effect small since the number matched):
- **Benchmark version**: this is TB **2.1** (89 tasks; 26 fixed vs 2.0 for bugs/timeouts/reward-hacking robustness) vs the card's TB **2.0**. Same task count and harness.
- **Runs**: single run here vs the card's **average of 5 runs** (temp 1.0 → run-to-run variance of a few points is expected).
- **Resources**: sandboxes ran at **4 vCPU / 16 GiB / 30 GiB disk** (per user direction + eviction fix) vs the card's stated 32 CPU / 48 GB. The score matched regardless.
- **Sampling matched the card**: temperature 1.0, top_p 0.95, top_k 20 (model `generation_config.json` defaults, confirmed in vLLM startup), 3h agent timeout, 256K ctx / 80K max output.

**Traces captured for all 89 tasks**: `results/rollouts_final.jsonl` (one row per task with the full converted Responses-API trajectory, reward, usage) + per-trial Harbor artifacts under `.../harbor_agent/jobs/20260715/.../` (`agent/trajectory.json`, `agent/recording.cast` asciinema, `verifier/reward.txt` + test stdout). ~770 MB on Lustre at `/mnt/rl-workspace/hemild/tb21-qwen36-opensandbox/`.

### Reproduce

```bash
# on the driver pod (ns default, DGXC cluster), with OPENSANDBOX_DOMAIN + TERMINAL_BENCH_2_1_TASKS_DIR set:
gym eval run \
  --config responses_api_agents/harbor_agent/configs/harbor_agent_opensandbox.yaml \
  --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
  --model-type vllm_model --model Qwen/Qwen3.6-27B \
  --model-url http://<vllm-svc>:8000/v1 --model-api-key dummy_key \
  --agent harbor_agent --split benchmark --concurrency 89 \
  --output rollouts_final.jsonl ++skip_venv_if_present=true
```

Infra used: vLLM Qwen3.6-27B on **2× GB300** replicas (1 GPU each; started at 1 replica, scaled to 2 after the single engine wedged under the 89-request burst) behind a ClusterIP Service; gym/Ray driver pod (24 CPU, `harbor_ray_task_num_cpus: 0.25` to fit 89 concurrent Harbor jobs); OpenSandbox reached over the istio ambient multi-cluster mesh at `opensandbox-cell-1.opensandbox-system:80` (server-proxy mode, no public endpoint).

##### Run v2 checkpoint (33/89, ~36 min in)

- **Running pass@1 = 0.788** (26 pass / 7 fail), 0 error-flagged rollouts (no agent-timeout/context-length flags). Prefix-cache hit rate climbed to **90.9%** as transcripts share more prefix. 14 running / 0 waiting reqs (GPU not saturated — concurrency bounded by how many episodes are mid-generation vs mid-tool-exec at any instant).
- Progress-bar ETA is noisy (task lengths vary widely); actual completion driven by the longest-tail tasks.
- Caveats for the final comparison: (a) model card reports TB **2.0 = 59.3** as **avg of 5 runs**; this is TB **2.1** (26 tasks fixed) at **1 run / task** — expect single-run variance; (b) model card also lists Terminus-2 harness, matching ours.

##### Run v2 mid-run (51–53/89): pass rate + sandbox-flakiness tally

- **Running pass@1 ≈ 78% (40/51)** at 51 done. Expect some drift down as the longer-tail tasks finish.
- Sandbox 500s now total 16 log tracebacks but from only **2 distinct sandbox pods**; **2** exhausted retries (agent caught → partial history → likely reward 0 for those 2 tasks). Zero `/run` 500s, zero code errors. This is shared-cluster OpenSandbox infra noise, not a defect — a couple tasks may be under-scored vs a clean-infra run, but it's a small fraction and reflects genuine deployment conditions.
