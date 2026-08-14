# Description

OSWorld computer-use benchmark environment. Each task runs on a real Ubuntu desktop VM
allocated from an OpenSandbox KVM pool (`poolRef: osworld-kvm`) through the
**`nemo_gym.sandbox` SDK** (`AsyncSandbox` image-less pool create +
`AsyncSandbox.endpoint(5000)` for the guest control API).

Per rollout (session) this resources server:

1. `/seed_session` — allocates a desktop VM via the SDK, waits for the desktop to render
   (screenshot `> ~500KB`), and runs the task's setup with the OFFICIAL OSWorld semantics:
   an `eval_task.py --phase setup` subprocess imports the pinned `osworld` fork
   (see `requirements.txt`) and calls `DesktopEnv.reset(task_config)`.
2. Exposes two agent tools: `POST /screenshot` (returns `{"image_base64": ...}`) and
   `POST /execute` (`{"command", "shell"}` → runs in the guest — the OSWorld action modality).
3. `/verify` — scores with the COMPLETE upstream evaluator (`eval_task.py --phase evaluate`
   → `DesktopEnv.evaluate()` with the agent-provided `action_history`; the caller always
   evaluates, including at step exhaustion), then **always** releases the VM.

Setup/evaluate run in subprocesses because the fork's remote-provider addressing is
env-var-global (`OSWORLD_CONTROL_SERVER_URL` / `OSWORLD_REMOTE_ADDR`); concurrent sessions
must not share a process. In proxied mode (`use_server_proxy: true`), `local_forwarder.py`
gives the upstream harness plain `127.0.0.1:<port>` targets that map onto the path proxy and
inject route headers. With a direct (pod-IP) endpoint, all guest ports — including Chrome
CDP `:9222` and VLC `:8080` used by some evaluators — are reachable without forwarders.

The paired agent is `responses_api_agents/nemotron_osworld` (Nemotron-Omni host-side loop).

## Provenance

- The generic Context Compaction implementation is Ali Roshan Ghias's work,
  originally published in his
  [NeMo RL branch](https://gitlab-master.nvidia.com/aroshanghias/nemo-rl/-/tree/aroshanghias/context-compaction-v2-clean)
  and [matching NeMo Gym branch](https://gitlab-master.nvidia.com/aroshanghias/Gym/-/tree/aroshanghias/context-compaction-v2-clean-gym).
  The corresponding Gym implementation is preserved in
  [commit `f881d8fc`](https://github.com/NVIDIA-NeMo/Gym/commit/f881d8fc3897f0e42c10fd80298430f43c509c67).
- The initial OSWorld environment and Cell 2 OpenSandbox integration was
  developed by Terry Kong in
  [commit `275f0ae9`](https://github.com/NVIDIA-NeMo/Gym/commit/275f0ae94c98c1a484658a5c995b97dce1bb1b4b).
- This draft adds the context-compacted OSWorld agent path, training-specific
  reliability work, and end-to-end validation with NeMo RL.

## Use from NeMo RL GRPO

The end-to-end training entry point lives in the companion
[NeMo RL draft](https://github.com/NVIDIA-NeMo/RL/pull/3642):

```text
examples/nemo_gym/run_grpo_nemo_gym.py
```

Use this recipe:

```text
examples/nemo_gym/grpo_nemotron_omni_30ba3b_osworld_cc.yaml
```

It composes these Gym configs from the pinned submodule:

```text
responses_api_models/vllm_model/configs/vllm_model_for_training.yaml
responses_api_agents/nemotron_osworld/configs/nemotron_osworld_cc.yaml
resources_servers/osworld/configs/osworld.yaml
resources_servers/osworld/configs/opensandbox_osworld.yaml
```

The full data-preparation, Slurm launch, and independent checkpoint-evaluation
commands are documented in the NeMo RL
[Context Compaction guide](https://github.com/jinglinglingling/RL/blob/feature/osworld-grpo-training-eval-signed/docs/guides/context-compaction.md#run-osworld-grpo).

## OSWorld dependency

The benchmark harness is a **referenced dependency**, not vendored: `requirements.txt` pins
`osworld @ git+.../<YOUR_USER>/osworld_internal@<sha>` (internal fork of the validated
Omni-Nano-v3 baseline branch + packaging fix + a no-lifecycle `remote` provider).
Note: the fork's full dependency set installs on **Linux only** (borb 3.x wheels contain
case-colliding member paths that fail to extract on macOS); run per-server tests and live
evaluation on Linux.

## Configuration

- `sandbox_provider: sandbox` — resolved from the merged global config; compose with
  `resources_servers/osworld/configs/opensandbox_osworld.yaml` for the validated
  Cell 2 setup (`OPENSANDBOX_DOMAIN` / `OPENSANDBOX_API_KEY` env vars).
- `OSWORLD_POOL_REF` (optional, default `osworld-kvm`) — the warm VM pool.
- `OSWORLD_CACHE_DIR` (optional) — setup download cache.

## Testing

```
gym env test --resources-server osworld   # Linux (see dependency note above)
```

Unit tests fake the sandbox provider (SDK layer) and the guest `:5000` HTTP surface, and
exercise the subprocess seam with a stub script; the live end-to-end test is skipped unless
`OPENSANDBOX_DOMAIN` is set.

# Licensing information
Code: Apache 2.0
Data: Apache 2.0

Dependencies
- nemo_gym: Apache 2.0
- OSWorld (referenced git dependency): Apache 2.0
