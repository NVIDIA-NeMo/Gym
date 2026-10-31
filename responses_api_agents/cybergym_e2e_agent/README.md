# cybergym_e2e_agent

External-agent-loop adapter for the [CyberGym-E2E](../../benchmarks/cybergym_e2e) benchmark.

The CyberGym harness owns the whole episode (hardened E2B microVM, persistent-shell agent loop, and
4-stage validation). This server is the thin boundary Gym calls:

- `run()` drives one task through the harness (as a subprocess — its CLI is the stable interface) and
  returns `reward = 1.0` if solved (stages 1–3) else `0.0`, with stage details under `cybergym`.
- `responses()` raises `NotImplementedError`: Gym does not drive the loop turn-by-turn here.

Configuration is entirely via environment (nothing hardcoded) — see the module docstring in `app.py`:
`E2B_API_URL`/`E2B_SANDBOX_URL`/`E2B_API_KEY` (and the model via Gym's `policy_model` —
`policy_base_url`/`policy_api_key`/`policy_model_name` or `--model`/`--model-url`/`--model-api-key`),
and optional `CYBERGYM_VM_TIMEOUT` / `CYBERGYM_DEADLINE`.

`gym env validate cybergym_e2e` covers the static composition. The live rollout path (`run()` against
a real E2B backend + model server) must be exercised with one `gym eval run` before scores are
trusted.
