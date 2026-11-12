# Eval-First Environment Onboarding Playbook — Status

Started 2026-07-09. Goal: environment contributors must ship representative eval task
sets (difficulty/edge-case coverage, baseline evidence) BEFORE training use; playbooks +
skills + tooling reduce expert manual-review burden (experts still sign off on priority
evals).

## State

`.claude/skills/env-onboarding-team/SKILL.md` exists (agent-team skill: env-auditor,
paper-researcher, dataset-designer, review-curator, baseline-runner, reward-profiler,
discriminativeness-analyst, devil-advocate; provenance tags MEASURED/STRUCTURAL/PRIOR/
PLACEHOLDER mandatory).

**First live run COMPLETE (2026-07-09, TALES, design-only mode).** Full report published
in the docs site: `fern/versions/latest/pages/environment-reports/` (new nav section
"Environment Reports": index + tales.mdx + tales-onboarding-report.mdx +
tales-expert-review.mdx). The team pattern worked: two challenge rounds materially
improved the package (paper metric corrected to normalized running HIGHSCORE; GAP-8
jericho 27-vs-54 population mismatch → paper_parity tier; alfworld tuple-order challenge
denied with upstream evidence).

## TALES outcome (resources_servers/tales, upstream pinned branch → SHA ef349ba7)

- Datasets GENERATED + validated: data/{smoke 5, eval_v0 64 (provisional), eval_full 95
  (test-clean), paper_parity 122 (all 54 jericho, harness-parity only)}.jsonl via
  scripts/generate_eval_datasets.py (deterministic, imports upstream split machinery).
  Server venv exists at resources_servers/tales/.venv (uv; install must run with CWD =
  server dir — requirements.txt has a relative `-e ../../`).
- Paper: arXiv:2504.14128; metric = normalized running highscore (benchmark.py); scores
  Table 3, σ Table 5. Repro target: self-hosted Llama-3.1-8B-Instruct, avg 13.9,
  tolerance ±max(5, 2σ).
- Blocking before verified:true (details in the onboarding report page):
  1. ~6-line server metric fix (pre-clobber game_score + running highscore +
     normalized_highscore/won/framework in info) + aggregate_metrics override — HARD
     prerequisite for any sweep (rollout `reward` is inflated for jericho; twx 0-1 scale).
  2. SHA-pin upstream + pin scienceworld dep in requirements.txt.
  3. Wave-3 sweep: endpoint needed — env.yaml inference_hub_api_key EXPIRED 2026-07-05
     (everything in env.yaml aliases it); W&B unconfigured. 2× RTX 6000 Ada available
     for self-hosted vLLM.
  4. alfworld success-path test; expert sign-off on the review package.

## Wave-3 attempt (2026-07-09 PM) — ABORTED by user; state for resume

Key renewed, full mode entered, metric server fix IMPLEMENTED in working tree
(app.py: game_score/highscore/normalized_highscore/won/framework in info, no clobber;
compute_metrics/get_key_metrics per-family macro; 13 tests pass; tales.yaml caps 25→100).
Zero successful episodes — blockers hit:
- Hub qwen backend (eccn-qwen3.6-35b) FLAPPING: conc-3 = 100% failures, conc-1 could not
  complete episodes. Endpoint-only policy (user): NO self-hosting, NO HF mirrors.
- Hub catalog DOES serve paper-exact models (verified /models): eccn-llama-3.1-8b-instruct
  (parity target, probed healthy), llama-3.2-1b, llama-3.3-70b, gpt-4o-mini(azure),
  mixtral-8x22b, phi-4-mini → parity + panel can be fully endpoint-only.
- W&B: env.yaml wandb_api_key line holds a REVOKED key (rotated after argv leak; see
  [[wandb-tracking-gate-and-secret-safety]]). Needs a FRESH key value before tracked runs.
- Two contaminated wandb runs (nvidia/nemo-gym-onboarding: nlj0q4vq, h65exb3i) hold the
  old revoked key in metadata; deletion optional now that the key is dead.
- Gym core bug documented (global_config.py:715 clobbers valid WANDB_API_KEY env var with
  unresolved ${...} literal) — recommended fix, not applied.
- Game assets fully cached (~/.cache/tales ~950M). Server venv at resources_servers/tales/.venv.

## Next steps (resume checklist)

1. User: fresh wandb key into env.yaml wandb_api_key line; confirm hub backend stability.
2. Runs (endpoint-only): smoke → eval_full ×5 (user approved grind) with the qwen target
   when its backend recovers; paper-parity via eccn-llama-3.1-8b-instruct ×5 on
   paper_parity.jsonl; panel = llama-3.2-1b / 3.1-8b / 3.3-70b or gpt-4o-mini.
3. Then discriminativeness → eval_v0 finalization → fold MEASURED results into the fern
   environment-reports pages → expert sign-off.
