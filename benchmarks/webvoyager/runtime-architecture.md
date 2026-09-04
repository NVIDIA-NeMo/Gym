# Visual-browser runtime architecture

WebVoyager has one public runtime, `visual_browser`. Model-specific protocol
adapters live in the agent and normalize Nano Omni tool calls or Qwen XML calls
into the same `WebAction` contract.

```text
Gym rollout or training worker
  -> web_agent
     -> policy protocol adapter
     -> async resource-server RPC
        -> WebSessionManager
           -> async BrowserSessionProvider acquire/release/heartbeat
           -> one session-affine Playwright thread
           -> one headed Chromium/PyAutoGUI backend
        -> Gemini judge after browser evidence is retained
```

## Why the implementation is hybrid

An all-thread design is unsafe for headed coordinate control: PyAutoGUI input
is global to a DISPLAY, and two sessions can click each other's windows. An
all-process rewrite is unnecessary for Playwright and makes lifecycle APIs
harder to compose with async providers.

The selected boundary is:

- async tasks for provider acquisition, release, heartbeat, timeouts, and HTTP
  request coordination;
- one dedicated thread per live session for synchronous Playwright, preserving
  Playwright thread affinity without blocking the FastAPI event loop;
- one process or container per X display for GUI isolation and crash recovery;
- model-server batching and distributed rollout processes for horizontal
  throughput.

This means a slow browser operation does not block unrelated server requests,
while two PyAutoGUI sessions never share a display.

## Browser supply and AgentEnv

`BrowserSessionProvider` is an async supply seam below benchmark semantics. A
provider returns an opaque lease containing session identity, transport,
endpoint, metadata, and ownership. The built-in provider represents the local
resource process and DISPLAY. Third-party providers are discovered through
the `nemo_gym.browser_session_providers` entry-point group.

An AgentEnv provider can acquire a container or VM asynchronously and return a
transport-specific handle. It must be paired with a driver that implements the
same `WebEnvironmentBackend` control contract for that transport. The local
PyAutoGUI driver deliberately rejects non-local handles rather than silently
trying to control the wrong DISPLAY. A remote-CDP driver and a remote-desktop
input driver are different implementations; browser supply alone must not
pretend those control protocols are interchangeable.

Providers that wrap blocking SDKs offload them with `asyncio.to_thread`.
External providers must implement idempotent release and enforce a provider-
side TTL. Heartbeats renew the lease; repeated or timed-out heartbeats close
the Gym session. Provider-side TTL remains the backstop for process death,
SIGKILL, or node loss, when Gym cannot execute cleanup code.

## Training failure behavior

The session manager owns late acquisitions after an RPC timeout and releases
the handle if it arrives later. Failed seed and close cleanup run in shielded,
strongly referenced tasks, so rollout cancellation cannot orphan a lease.
Backend, operation-runner, browser-provider, and site-pool cleanup are all
attempted even if an earlier layer fails. Session TTL reaping handles abandoned
but still-live rollouts.

Infrastructure and configuration failures are returned as masked rollout
sidecars. They are excluded from policy reward and routed to retry/recovery.
Only a completed policy trajectory judged as unsuccessful becomes a valid
reward-zero training example. This avoids the orphan-session failure mode in
which infrastructure loss silently becomes thousands of negative samples.

## Scale-out rule

Keep one browser session per visual-browser process. Launch more isolated
processes or containers for rollout parallelism, and share only read-only
caches and policy endpoints. This rule applies whether the process is supplied
locally, by Slurm/Pyxis, or by AgentEnv.
