# Description

OSWorld computer-use benchmark environment. Each task runs on a real Ubuntu desktop VM
allocated from an OpenSandbox KVM pool (`poolRef: osworld-kvm`) through the
**`nemo_gym.sandbox` SDK** (`AsyncSandbox` placeholder-image pool create +
`AsyncSandbox.endpoint(5000)` for the guest control API). The SDK requires an
image argument, but `poolRef` supplies the actual prebuilt OSWorld VM.

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

`POST /release` is an idempotent cleanup-only path used when an agent fails after
seeding but before it can submit a valid verifier request.

Setup/evaluate run in subprocesses because the fork's remote-provider addressing is
env-var-global (`OSWORLD_CONTROL_SERVER_URL` / `OSWORLD_REMOTE_ADDR`); concurrent sessions
must not share a process. In proxied mode (`use_server_proxy: true`), `local_forwarder.py`
gives the upstream harness plain `127.0.0.1:<port>` targets that map onto the path proxy and
inject route headers. With a direct (pod-IP) endpoint, all guest ports — including Chrome
CDP `:9222` and VLC `:8080` used by some evaluators — are reachable without forwarders.

The paired agent is `responses_api_agents/nemotron_osworld` (Nemotron-Omni
host-side loop).

## Context-managed GRPO

The token-free receipt path composes these Gym configs:

```text
responses_api_agents/nemotron_osworld/configs/nemotron_osworld_cc.yaml
resources_servers/osworld/configs/osworld.yaml
resources_servers/osworld/configs/opensandbox_osworld.yaml
```

The CC profile loads the ordinary
`responses_api_models/vllm_model/configs/vllm_model.yaml` with
`return_token_id_information: false`. Gym's shared context-management client
selects parent IDs and emits `<owner>_sN` segment receipts; NeMo RL owns token
capture staging and publication.

## OSWorld dependency

The benchmark harness is a **referenced dependency**, not vendored:
`requirements.txt` pins the validated public OSWorld fork revision (packaging
fixes plus a no-lifecycle `remote` provider).
Note: the fork's full dependency set installs on **Linux only** (borb 3.x wheels contain
case-colliding member paths that fail to extract on macOS); run per-server tests and live
evaluation on Linux.

## Configuration

- `sandbox_provider: sandbox` — resolved from the merged global config; compose with
  `resources_servers/osworld/configs/opensandbox_osworld.yaml` for the validated
  Cell 2 setup (`OPENSANDBOX_DOMAIN` / `OPENSANDBOX_API_KEY` env vars).
- `OSWORLD_POOL_REF` (optional, default `osworld-kvm`) — the warm VM pool.
- `OSWORLD_SANDBOX_IMAGE` (optional, default `busybox:1.36`) — SDK validation
  placeholder; it does not replace the VM selected by `poolRef`.
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
