# Skill-opening agent for Gym

This entrypoint checks that Gym's `skills_ref` describes the exact catalog prepared for the requested task, then delegates execution to `harbor_agent_general`. A mismatch returns a flagged failure before dispatch. It does not replace the shared Harbor scheduler, converter, or verifier. Its `skill_opening_rate` averages valid repetitions within each skill, then weights skills equally. It reports `skills_scored` and `trials_scored` alongside that rate; missing/error counts still come from the collector. It does not add pass@k metrics.

Use the five profiles in [the benchmark](../../benchmarks/skill_trigger/README.md). Run one catalog per collection invocation: the original tasks use 43 catalogs, so a single union would change the experiment. Packaging writes separate input files under `catalogs/` without adding scoring labels to the task images.

This remains an implementation draft. Static profile checks and a synthetic Pi run across the real Ray boundary pass; the complete five-profile migration and clean-clone checks are outstanding. Keep provider errors and catalog mismatches out of scored averages, preserve failure records, and set `NEMO_GYM_MAX_ROLLOUT_ATTEMPTS=1` for the migration pilot.
