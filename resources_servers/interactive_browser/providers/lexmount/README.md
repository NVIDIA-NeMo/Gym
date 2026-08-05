# `lexmount` — example session provider

Supplies one isolated **cloud** browser per rollout to `backend: remote_cdp`:
the browser runs off the training node, and this environment reaches it over
CDP. Useful as a worked example of a metered, session-based provider — creation
is slow and fallible, sessions cost quota, and teardown must not be silent.

The SDK is **not** a dependency of this environment. Nothing here is imported
unless the config selects `session_provider: {lexmount: ...}`.

## Use it

1. Register at <https://browser.lexmount.com>, create a project, copy the **API
   key** and **project ID**.
2. Install the SDK into the resources server's own venv (created the first time
   `gym env start` serves this environment) and export credentials — never
   commit them:
   ```bash
   uv pip install --python resources_servers/interactive_browser/.venv/bin/python "lexmount>=0.5.13"
   export LEXMOUNT_API_KEY=<your-api-key>
   export LEXMOUNT_PROJECT_ID=<your-project-id>
   export LEXMOUNT_BASE_URL=https://api.lexmount.com   # API base shown in your dashboard
   ```
3. Start the remote flavor of the environment:
   ```bash
   gym env start --resources-server interactive_browser/lexmount \
     --model-type openai_model --model <served-model-name> \
     --model-url https://your-endpoint/v1 --model-api-key <key>
   ```
   That flavor is just `configs/lexmount.yaml`: the same environment with
   ```yaml
   backend:
     remote_cdp:
       session_provider:
         lexmount: {browser_mode: normal, create_timeout_s: 150}
   ```

Remote browsers cannot open the bundled offline `site/` tasks (local `file://`
URIs), so the flavor ships live-web tasks (`data/example_remote.jsonl`).

## Limits to know before training concurrency

| Limit | What it means for a run |
| --- | --- |
| **One session per rollout, no client-side cap** | N concurrent rollouts bid for N cloud sessions. Size the account quota **above** the rollout concurrency, with headroom for sessions still being torn down — otherwise the quota is exhausted and every later create fails. |
| **No episode TTL** | A session is released when the rollout is scored (`verify`) or when the same `session_id` is re-seeded. A rollout abandoned without either (trainer crash, client disconnect) leaks its session until the service reclaims it. |
| **Best-effort release** | A failed close/delete is logged, not retried, and the service may still hold the session. Grep the server log for `Failed to close Lexmount session` / `Failed to delete Lexmount session`. |
| **Create blocks in a worker thread** | `sessions.create` polls until the session is active (`create_timeout_s`, default 150s). It runs off the event loop, but the thread cannot be cancelled — keep `create_timeout_s` tight enough for your step budget, and note that the client-side deadline should not be shorter than what the provider may spend. |

The default `local_playwright` backend has none of these constraints: local
processes, no quota, released with the rollout.
