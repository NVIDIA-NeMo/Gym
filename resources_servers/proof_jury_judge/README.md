<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Proof jury judge

Grades natural-language olympiad proofs on the IMO 0-7 scale with a jury of LLM judges, without
reference solutions. This is the evaluation protocol of
["An Open Recipe for IMO Gold"](https://arxiv.org/abs/2609.10712) (Section 6.3), which follows
MathArena's jury procedure:

1. Every judge grades the proof independently with the paper's IMO-style prompt
   (Appendix B.5, `prompts/judge.yaml`).
2. If the scores differ by more than `reconcile_spread` (2), every judge grades once more after
   seeing all first-round judgments (Appendix B.6, `prompts/reconcile.yaml`).
3. The final score is the minimum of the last round. The reward is `score / 7`.

The judged text is the policy's final answer with its reasoning removed. A generation that was
cut off (`incomplete_details` set) or has no final answer scores 0 without judge calls.

A judge reply that was cut off or has no `\boxed{}` score is a failed judge call, as in the
paper, which reruns incomplete panels: the rollout goes to the failures sidecar and is retried
on resume, and a failure in any judge fails the whole panel. This departs from the default in
`nemo_gym/verification/judge.py` (an unparseable reply scores as wrong), as AA-LCR v1.1 does for its JSON
verdicts. Failed rollouts are left out of the metrics by default; for reported numbers, retry
them on resume or count them as zero with `count_failure_classes_as_zero: [judge_failed]`.

## Judges

`configs/proof_jury_judge.yaml` pins one model per lab, served through an OpenAI-compatible
gateway (NVIDIA's inference API by default):

| Server | Model | Settings |
| --- | --- | --- |
| `proof_jury_judge_openai_model` | `openai/openai/gpt-6.1-sol` | reasoning effort high, 128k output tokens |
| `proof_jury_judge_google_model` | `gcp/google/gemini-3.1-pro-preview` | reasoning effort high, 65k output tokens |
| `proof_jury_judge_anthropic_model` | `azure/anthropic/claude-opus-5-5` | extended thinking (120k budget), 128k output tokens |

Set `JUDGE_API_KEY` (and `JUDGE_BASE_URL` for another gateway). Requests carry
`X-Inference-Priority: batch`, the gateway's lane for requests longer than 360 s.

## Task fields

- `problem` (required): the problem statement.
- `tier` (optional): difficulty tier for per-tier metrics.
- `problem_idx` (optional): stable problem id.

## Metrics

- `pass@1[avg-of-k]/score`: mean score as a percentage of 7 points (headline).
- `pass@1[avg-of-k]/full_credit`: share of proofs scored 7.
- `pass@1[avg-of-k]/truncated`: share of generations cut at the output limit.
- `<tier>/pass@1[avg-of-k]/...`: the same per tier.

Each rollout keeps `judge_score`, `truncated` and `judgments` (judge, round, score, full text).

## Licensing information

Code: Apache 2.0<br>
Data (`data/example.jsonl`): Creative Commons Attribution 4.0 International
(`nvidia/Nemotron-IMO-Bench`, `Hwilner/imo-proofbench`)<br>
Prompts: from "An Open Recipe for IMO Gold" (arXiv 2609.10712), Appendix B.5 and B.6

Dependencies
- nemo_gym: Apache 2.0

## Tests

```bash
gym env test --resources-server proof_jury_judge
```
