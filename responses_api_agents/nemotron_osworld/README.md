# Description

Nemotron-Omni OSWorld agent: the validated Omni-Nano-v3 harness
(`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`, 47–48.5% on `test_nogdrive`)
running over `resources_servers/osworld`.

The step loop is driven by the REFERENCE agent — `mm_agents.nvidia.nemotron_agent.
NemotronAgent`, imported from the pinned `osworld` fork (see `requirements.txt`) — so the
prompt template, message/history serialization (3-image window + text history), free-text
`## Action/## Code` parsing, 0–1 fractional coordinate projection, parse-retry recovery,
and the internal force-`FAIL` at the step cap are the vendor-validated code, not a port.

Per step: `POST /screenshot` (resources server) → `agent.predict` in a worker thread
(its own synchronous chat-completions client with the strict `finish_reason == "stop"`
retry contract, pointed at the gym MODEL SERVER's `/v1/chat/completions` through the
`VLLM_API_ENDPOINT` env seam) → each returned pyautogui snippet runs via `POST /execute`
with the exact `PythonController` wrapping (`python -c "<pkgs_prefix + action>"`,
`shell: false`); `WAIT` sleeps, `FAIL`/`DONE` terminate. The faithful OSWorld
`action_history` is forwarded to `/verify` (the evaluator inspects the last entry for the
infeasible/`FAIL` contract).

Serving parity for the reference numbers (see the fork's docs + Jeff Peng's
serving-contract alignment): vLLM 0.23.0, the public Nano-v3 chat template + `nano_v3`
reasoning-parser plugin, checkpoint (Omni) tokenizer, xgrammar structured outputs with
`enable_in_reasoning=false`, `--mamba-ssm-cache-dtype float32`. The gate: the model's
`## Action` header must be preceded by a real newline (a template/parser mismatch shows up
as literal `\n` + mass parse failures).

Note: the pinned fork's full dependency set installs on Linux only (borb 3.x wheels fail
to extract on case-insensitive filesystems) — run this server and its tests on Linux.

## Run

```
gym env start --agent nemotron_osworld \
  +config_paths="[responses_api_agents/nemotron_osworld/configs/nemotron_osworld.yaml,
                  resources_servers/osworld/configs/osworld.yaml,
                  nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml,
                  <your policy_model config>]"
```

# Licensing information
Code: Apache 2.0
Data: Apache 2.0

Dependencies
- nemo_gym: Apache 2.0
- OSWorld (referenced git dependency): Apache 2.0
