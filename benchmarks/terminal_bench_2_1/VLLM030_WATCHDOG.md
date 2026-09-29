# TB 2.1 on vLLM 0.30.0 with a decode-engine watchdog and server-side TCP keepalive

This branch combines the pieces we use to run Terminal Bench 2.1 reliably with disaggregated vLLM serving and remote
sandboxes:

| Change | What | Why |
|---|---|---|
| PR #3727 | TCP keepalive on connections accepted by Gym's uvicorn servers | Some network paths silently drop flows that stay idle for several minutes. A long model call then leaves the agent's connection half-open until the rollout times out. |
| UCX settings | `UCX_TLS=rc_x,rc,cuda_copy,cuda_ipc`, `UCX_IB_ADDR_TYPE=eth`, `UCX_NET_DEVICES=mlx5_0:1` | Needed on RoCE clusters, where the previous values fail NIXL KV-cache registration with `NIXL_ERR_BACKEND`. Revert them on InfiniBand clusters. |
| vLLM 0.30.0 config | `nemotron_3.5_lightning_vllm030.sh`, with prefill `--gpu-memory-utilization 0.85` | FlashInfer MoE autotuning runs out of memory at 0.9 on vLLM 0.30.0. |
| Decode watchdog | Opt-in `VLLM_ENGINE_WATCHDOG=1` in `sbatch_external_vllm.sh` | The decode API server's event loop can block for minutes in vLLM's non-streaming tool-call parser (`_compute_arg_delta` is quadratic in the argument length for arguments containing `<`). The engine then stops serving and requests hang. |

## Keepalive configuration

- Keepalive is always on; there is no flag to enable it.
- Gym's uvicorn servers (model, agent and resources servers) enable `SO_KEEPALIVE` on every accepted TCP connection. They use the same global settings as Gym's outgoing aiohttp connections:

| Field | Default | Meaning |
|---|---|---|
| `global_aiohttp_tcp_keepalive_idle_seconds` | 60 | Seconds a connection is idle before probing starts (`TCP_KEEPIDLE`). |
| `global_aiohttp_tcp_keepalive_interval_seconds` | 10 | Seconds between probes (`TCP_KEEPINTVL`). |
| `global_aiohttp_tcp_keepalive_probes` | 3 | Unanswered probes before the kernel drops the connection (`TCP_KEEPCNT`). |

- Override any of them as a top-level config key, e.g. `++global_aiohttp_tcp_keepalive_idle_seconds=30`.
- Keep the idle time well below the shortest idle timeout on your network path.

## Run

```bash
git clone -b hemild/tb-vllm030-keepalive-watchdog https://github.com/NVIDIA-NeMo/Gym.git gym-tb-vllm030
cd gym-tb-vllm030
# env.yaml at the checkout root (chmod 600):
#   wandb_api_key, hf_token, sandbox.opensandbox.connection.{domain,api_key}
CONTAINER=/path/to/vllm-0.30.0-with-gym.sqsh SBATCH_ACCOUNT=<acct> MOUNTS="$PWD:/opt/Gym,/data:/data" \
EXPERIMENT_NAME=<you>/tb21/<name> \
  bash benchmarks/terminal_bench_2_1/run_tb_vllm030.sh /data/checkpoints/<model>/hf
```

- **Container:** a `vllm/vllm-openai:v0.30.0` image with Gym's server venvs installed. The checkout is mounted over `/opt/Gym` at runtime.
- **Defaults:** 2 prefill + 4 decode nodes (4 GPUs each) and a 4 h limit.
- **Env overrides:** `NUM_PREFILL_NODES`, `NUM_DECODE_NODES`, `SBATCH_TIME` and `SLURM_COMMENT`.
- **Dry run:** `DRY_RUN=1` prints the command without submitting.
- **Watchdog off:** `VLLM_ENGINE_WATCHDOG=0`.
- **Multiple runs from a fresh clone:** submit one run first. Wait until `benchmarks/terminal_bench_2_1/data/benchmark_prepare.jsonl` exists (about 1 min after it starts) before submitting the rest. Every job prepares the benchmark data inside the checkout, and jobs that start together race on it: one fails with `AssertionError: 0` from `prepare.py`.

## Watchdog behaviour

- It supervises each decode `vllm serve` in its own process group.
- It restarts the engine when either:
  - `/metrics` fails 3 probes in a row (30 s interval, 10 s timeout), or
  - no tokens are generated for 300 s while requests are running.
- Before killing, it captures `py-spy dump --nonblocking` and `nvidia-smi` into the job log.
- It restarts at most 3 times per node. The router health check is tightened so traffic drains from a restarting engine. Gym's model-call retries resend the requests that were in flight.
- Knobs: `VLLM_ENGINE_WATCHDOG_{INTERVAL_S,METRIC_FAILURES,STALL_S,MAX_RESTARTS}`.
- Find events with `grep -a engine-watchdog slurm-logs/<jobid>-*/*/*.log` (`FROZEN`, `restart #N`, `giving up`).
