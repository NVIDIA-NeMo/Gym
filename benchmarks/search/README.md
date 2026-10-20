# search / BrowseComp (sandboxed opencode)

BrowseComp questions run through `responses_api_agents/opencode_sandboxed_agent` and graded by
`resources_servers/opencode_search` with an LLM judge.

Same source data and same deterministic seed-42 400-row subset as `benchmarks/browsecomp`, but the
rows are reshaped for the sandboxed agent: exactly one user message, no system message (the agent
prepends its own), and no `tools` block (opencode brings its own tools).

`opencode.yaml` is the whole benchmark: 400 rows, composed on its own.

Set `BROWSECOMP_RUN_FULL=1` at prepare time for the full 1266.

To run a smaller slice, point the dataset at your own prepare script. `prepare.py` exposes
`build_rows()` and `write_rows()` for exactly that, so a 10-row plumbing smoke is a four-line
module plus an override that restates the `datasets:` entry — the list has to be restated in full
because OmegaConf replaces lists wholesale rather than merging them.

## Arms

Every behaviour switch is an environment variable read by `opencode.yaml`, so there is one config
to compose rather than one file per feature. All default to off, and "off" means the variable is
dropped rather than exported — an unset arm leaves the sandbox byte-identical to a run that never
knew the knob existed.

| variable | effect |
| --- | --- |
| `NEMO_GYM_DISABLE_BASH=1` | remove the `bash` tool outright. The model then reaches pages only through Exa, which honours robots and bot-exclusion; `curl` honours nothing. On one measured run 4 of 5 page reads bypassed that guardrail. |
| `NEMO_GYM_FIX_GREP_DESCRIPTION=1` | strip the two lines of `grep`'s stock description that point at `bash` and `Task`. Set this **with** the above: opencode spills oversized tool output to disk and tells the model to grep it, while grep's own description talks it out of doing so. |
| `OPENCODE_WEBSEARCH_TYPE=deep` | Exa REST `deep` instead of the MCP `auto` default. Not a one-knob change — it also switches transport, adds the deep-only `summary`+`outputSchema` pair, and makes `contextMaxCharacters` bind. Measured as **no meaningful change** on BrowseComp-400, even with nearly every call carrying genuine server synthesis, so treat it as a real null. |
| `NEMO_GYM_EXA_CACHE_ROOT=<dir>` | turn on the shared Exa response cache. Unset means no store, no `/mcp` proxy route, and the plugin talks straight to `mcp.exa.ai`. |
| `NEMO_GYM_EXA_CACHE_LEGACY_ROOT=<dir>` | additionally consult an existing store, read-only and exact-key only. |

Timeouts are committed defaults, not arms: `sandbox_timeout: 21600` with no per-call deadline, so
the job's wall clock is the only thing that stops a rollout. The previous 3h default sat under a
4h wall and scored its kills as reward 0 with no failure kind — indistinguishable from a wrong
answer. Generation is likewise uncapped, matching every apptainer reference run; the output cap
that used to be set here does not truncate, it kills the rollout.

The judge is hosted GLM-5.2 on inference-api and needs a key: `++inference_api_key=...`.

Scores from this path are not comparable to a dedicated search harness — stock opencode has only
`webfetch`/`websearch` (enabled by this benchmark's config overlay; they are off by default).
