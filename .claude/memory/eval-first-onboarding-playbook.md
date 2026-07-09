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

## Next steps

1. Renew inference-hub key (or nvapi/build.nvidia.com key) + set wandb_* in env.yaml.
2. Land the tales server metric fix + pins (small PR; fix spec in the report).
3. Re-run `/env-onboarding-team tales <endpoint>` in full mode → Wave 3 (smoke → eval_full
   k=5 sweep → paper-parity repro → ≥3-model reward profiling) → re-select eval_v0 →
   expert sign-off.
4. Second onboarding target: pick next unverified environment; skill is proven.
