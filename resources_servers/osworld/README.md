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

## OSWorld dependency

The benchmark harness is a **referenced dependency**, not vendored: `requirements.txt` pins
`osworld @ git+.../<YOUR_USER>/osworld_internal@<sha>` (internal fork of the validated
Omni-Nano-v3 baseline branch + packaging fix + a no-lifecycle `remote` provider).
Note: the fork's full dependency set installs on **Linux only** (borb 3.x wheels contain
case-colliding member paths that fail to extract on macOS); run per-server tests and live
evaluation on Linux.

## Configuration

- `sandbox_provider: sandbox` — resolved from the merged global config; compose with
  `nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml`
  (`OPENSANDBOX_DOMAIN` / `OPENSANDBOX_API_KEY` env vars).
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
