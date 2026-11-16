# VideoMME-v1

Evaluation-only support for [Video-MME](https://github.com/MME-Benchmarks/Video-MME),
not VideoMME-v2: single-turn video multiple-choice QA with a deterministic
verifier and a frozen annotation cohort. An agentic setting is planned as a
separate change; the verifier already accepts `setting: agentic` cohorts.

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

Compose the config with a `policy_model` such as the stock `vllm_model`, then
use Gym's standard run/eval/aggregate commands. The config runs `simple_agent`
through a `single_agent_turn_legacy` Environment Server. Rows carry the video
as a `data:` or `file://` URL; see
[single-turn inputs](../video_mcqa/README.md#single-turn-inputs).

## Scores

Single-turn namespace: `Video_MME`; agentic: `Video_MME_agentic`.
Both provide `/raw_acc`, `/coverage`, `/is_complete` and explicit counts, plus
breakdowns under `/duration/short`, `/duration/medium`, `/duration/long`,
`/domain/<label>`, `/sub_category/<label>` and `/task_type/<label>`. Only categories
present in the declared cohort are reported. The Gym-standard `mean/Video_MME`
alias is a fraction; `/raw_acc` is a percentage. There is no nonlinear metric.

This adapter uses no audio or external subtitles. Its prompt asks for a boxed
letter, and the default `boxed_only_v1` policy scores an unboxed answer as 0;
upstream asks for a bare letter. These scores are therefore not comparable to
official Video-MME results. Frame sampling is left to the serving side (vLLM's
`--media-io-kwargs`, or a VideoAnyRes preparation upstream of Gym).

Reference: official repository commit
`06c2315b892f88578f81d73205d07cf576f292b9`; reference annotation snapshot:
`ead1408f75b618502df9a1d8e0950166bf0a2a0b`. Callers must pin their
actual annotation revision rather than treating these references as a default.

The [official dataset terms](https://github.com/MME-Benchmarks/Video-MME#-dataset)
restrict use to academic research and restrict redistribution. No questions,
videos or benchmark answers are shipped with this code. Test fixtures are
invented, and the five examples in `data/` are invented questions about
synthetic clips ([example data](../video_mcqa/README.md#example-data)). Dataset
registration and full real-model benchmark baselining are not implied by these
configs; they are `verified: false`.
