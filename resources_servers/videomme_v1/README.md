# VideoMME-v1

Evaluation-only support for [Video-MME](https://github.com/MME-Benchmarks/Video-MME),
not VideoMME-v2. Both single-turn VideoAnyRes and the sandboxed OpenCode video
agent use the same deterministic verifier and frozen annotation cohort.

## Inputs and configuration

The official [HF annotations](https://huggingface.co/datasets/lmms-lab/Video-MME)
have 2,700 questions on 900 videos. The adapter reads flat annotation records
(`question_id`, `videoID`, `duration`, `domain`, `sub_category`, `task_type`,
`question`, `options`, `answer`), or the official nested per-video JSON layout.
Labels on the four options are validated before being stripped. Bind media by
`videoID`; nested layouts without it require an explicit `video_id` binding.

See [shared preparation](../video_mcqa/README.md) for the CLI and media inventory.
The resulting rows are standard Gym input JSONL; gold remains verifier-only.

| Setting | Config | Required cohort variable | Agent |
| --- | --- | --- | --- |
| Single-turn | `configs/videomme_v1_single_turn.yaml` | `videomme_v1_cohort_manifest` | `simple_agent`, one step |
| Agentic | `configs/videomme_v1_agentic.yaml` | `videomme_v1_agentic_cohort_manifest` | `opencode_video_agent` |

Compose the config with a `policy_model` and the video runtime settings,
then use Gym's standard run/eval/aggregate commands. The agentic config exposes
the same runtime archive/hash, source roots, sandbox image, media profile,
context/output and artifact-directory inputs as the VideoMME-v2 adapter.

## Scores

Single-turn namespace: `Video_MME`; agentic: `Video_MME_agentic`.
Both provide `/raw_acc`, `/coverage`, `/is_complete` and explicit counts, plus
breakdowns under `/duration/short`, `/duration/medium`, `/duration/long`,
`/domain/<label>`, `/sub_category/<label>` and `/task_type/<label>`. Only categories
present in the declared cohort are reported. The Gym-standard `mean/Video_MME`
alias is a fraction; `/raw_acc` is a percentage. There is no nonlinear metric.

This adapter uses no audio or external subtitles. Its short boxed-answer prompt
and explicit answer parser are recorded separately from upstream's bare-letter
prompt. Sampling/frame count is supplied by the external VideoAnyRes processor.

Reference: official repository commit
`06c2315b892f88578f81d73205d07cf576f292b9`; reference annotation snapshot:
`ead1408f75b618502df9a1d8e0950166bf0a2a0b`. Callers must pin their
actual annotation revision rather than treating these references as a default.

The [official dataset terms](https://github.com/MME-Benchmarks/Video-MME#-dataset)
restrict use to academic research and restrict redistribution. No questions,
videos or benchmark answers are shipped with this code. Test fixtures are
invented. Dataset registration and full real-model benchmark baselining are
not implied by these configs; both are `verified: false`.
