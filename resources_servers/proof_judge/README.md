# Proof Judge (verifier + meta-verifier)

Resource server for theorem-proving GRPO: scores policy outputs with a judge model (verifier + optional meta-verifier), DeepSeek Math style reward.

- **Verifier**: Judge model scores the proof (R_Y, 0/0.5/1).
- **Meta-verifier** (when `beta > 0`): Judge model scores the policy's self-evaluation; R_Z = (1 - |s' - R_Y|) * R_meta; reward = α·R_Y + β·R_Z.

Policy output must contain `## Solution` and `## Self Evaluation` with a final `\boxed{0|0.5|1}`.

Used with nemo-rl + NemoGym: add `proof_judge_model` (e.g. DeepSeek-Math-V2) with `spinup_server: true` in `env.nemo_gym`, and point `proof_simple_agent` at this resource and the policy model.
