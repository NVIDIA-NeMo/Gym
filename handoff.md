# ASB Modal Campaign Handoff

This is a defensive, controlled evaluation of four NVIDIA baseline models using
the public [Agent Security Bench (ASB)](https://github.com/agiresearch/ASB)
runner. The user explicitly authorized the work. Do not contact ASB authors,
send Slack, open a PR, publish results, or modify/stop/resize shared serving
deployments.

## Honest current state

**No ASB score rows or generated ASB trajectories exist yet.** The prior work
completed input and runtime preflight only. Do not call this branch or any
receipt a benchmark result.

The required target models are:

1. `moonshotai/Kimi-K3`
2. `Qwen/Qwen3.5-122B-A10B-FP8`
3. `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4`
4. `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16`

Fresh `/v1/models` receipts were obtained for all four exact model IDs on
2026-09-19/20 CDT. Kimi, Qwen, and Ultra authenticate using the local,
untracked workspace `.env` variable `MODAL_PROXY_TOKEN`. Super VL has a
separate service authentication boundary; a stopped, one-shot Modal FDR probe
mounted `nemotron-super-vl-service` and confirmed its model receipt. No secret,
endpoint URL, or credential value is committed here.

## Preserved source provenance

- Upstream source: `https://github.com/agiresearch/ASB`
- Pinned commit: `1f561dccf92d55302368fa67679b4ba9d9c8fdc4`
- Core JSONL input hashes at that commit were inspected. Normal task inputs:
  10 agent definitions / 51 tasks. Attack tools: 400 total, 200 aggressive and
  200 non-aggressive. The complete normal task-by-matching-tool expansion is
  **2,040 selectors per attack condition**.
- `data/agent_task_pot_all.jsonl` is malformed in the pinned Git object: it is
  GitHub rate-limit HTML rather than JSONL. It is not referenced by the public
  runner/configs. Keep it separately labeled as an upstream-invalid auxiliary
  artifact; do not fabricate a selector matrix from it.

## What must be built before launch

ASB is an older AIOS orchestration stack, not an existing NeMo Gym environment.
Its upstream `vLLM` path attempts to load local weights and the convenience
scripts run only one task per agent by default. Do **not** use that default or
rename AgentDyn evidence as ASB.

Implement a NeMo Gym external-benchmark agent adapter that:

1. Runs the pinned ASB task/tool/action semantics with an OpenAI-compatible
   remote-model transport.
2. Materializes all selected public tasks and conditions before collection,
   rather than using `task_num=1` or a test selector.
3. Records source revision and input hashes; endpoint/model receipt; sampling
   and seed; task, agent, normal-tool, attack-tool, attack type, aggressiveness,
   defense, and retrieval provenance; actions/messages; utility, attack success,
   retries, and classified failures.
4. Keeps provider and infrastructure failures in explicit sidecars, outside
   model-quality denominators. Do not silently retry an invalid provider row
   into a score.
5. Generates matched clean and poisoned histories for memory detector metrics.
   Legacy retrieval-match output is a compatibility diagnostic, not FPR/FNR.

The target treatment set is the full public supported matrix: clean/control,
direct-prompt injection, observation-prompt injection, memory poisoning,
mixed attacks, PoT backdoor/control, aggressive and non-aggressive attack tools,
and their public defenses. Preserve the public attack/defense names; do not
collapse methods into generic prompt injection.

## FDR constraints

Every new Modal resource and run must be explicit `FDR` (`modal ... --env=FDR`
or process-scoped `MODAL_ENVIRONMENT=FDR`). Existing shared model deployments
are inference-only dependencies. Do not stop, resize, redeploy, or otherwise
alter them.

`scripts/asb_super_receipt.py` is a bounded receipt helper. It requires
`ASB_SUPER_BASE_URL` at runtime and mounts the existing FDR
`nemotron-super-vl-service` secret. It returns only receipt status/model count
and is intentionally unsuitable for benchmarking. It was executed successfully
once in FDR before this handoff, then stopped.

## Completion bar

Do not declare completion until all planned rows for all four models reconcile.
Deliver a private internal report and BLADE-compatible package showing per-model
coverage/denominators, clean utility, ASR/security, defense treatment results,
actual clean-vs-poison detector metrics, failure accounting, source/model/runtime
provenance, and every incomplete condition.
