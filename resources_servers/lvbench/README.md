# LVBench

Evaluation-only support for [THUDM/LVBench](https://github.com/THUDM/LVBench),
**not LongVideoBench**: single-turn video multiple-choice QA without
benchmark-specific system prompts. The agentic setting runs OpenCode with
video tools in a sandbox that the server lends each episode.

## Inputs and configuration

Read the authors' `video_info.meta.jsonl` from the
[official dataset](https://huggingface.co/datasets/THUDM/LVBench). Each video has
`key`, `type` and a `qa` list containing `uid`, `question`, `answer`, and
`question_type`. The question embeds `(A)` through `(D)` options. A lossless
flattened export with the same fields is also accepted. Integer/string `uid`
values are normalized as in the official scorer. Media binds by `key`, not by
a filename or timestamp recovered from the question. `time_reference`
and video metadata/captions are ignored as model evidence.

See [shared preparation](../video_mcqa/README.md) for the CLI, pinned annotation
revision, explicit media inventory, output cohort and provenance receipts.

| Setting | Config | Required cohort variable | Agent |
| --- | --- | --- | --- |
| Single-turn | `configs/lvbench_single_turn.yaml` | `lvbench_cohort_manifest` | `simple_agent`, one step |
| Agentic | `configs/lvbench_agentic.yaml` | `lvbench_agentic_cohort_manifest` (and `lvbench_agentic_video_roots`) | `opencode_agent` in a lent sandbox |

Compose the config with a `policy_model` such as the stock `vllm_model`, then
use standard Gym run/eval/aggregate commands. The config runs `simple_agent`
through a `single_agent_turn_legacy` Environment Server. Rows carry the video
as a `data:` or `file://` URL; see
[single-turn inputs](../video_mcqa/README.md#single-turn-inputs).
The agentic config runs the stock `opencode_agent` with `watch_video` and
`view_image` through the same Environment Server; its rows bind the original
video file, and it needs a sandbox image (`VIDEO_MCQA_SANDBOX_IMAGE`). See the
shared [agentic setting](../video_mcqa/README.md#agentic-setting).

## Scores

Namespaces: `LVBench` and `LVBench_agentic`. Overall `/raw_acc` is question-weighted
accuracy on a 0-100 scale, with a 0-1 `mean/<namespace>` alias. `/coverage`,
`/is_complete`, expected/returned/missing/correct and outcome counts expose
partial or failed runs. No nonlinear score is reported.

Per-capability scores appear under `/task_type/<percent-encoded-label>/raw_acc`:
entity recognition (ER), event understanding (EU), key information retrieval
(KIR), temporal grounding (TG), reasoning (Rea), summarization (Sum). A question
with two labels contributes to both category denominators but once overall,
matching the official scorer on a complete cohort. `/domain/<label>` provides
an additional video-type breakdown. Category scores are not macro-averaged.

The prompt asks for a boxed letter, and the default `boxed_only_v1` policy
scores an unboxed answer as 0, unlike upstream's bare-letter answers, so these
scores are not comparable to official LVBench results. For bare-letter answers,
prepare the rows with `--prompt lvbench` and a `llm_extraction_v1` cohort, and
run `configs/lvbench_single_turn_llm_extraction.yaml`; see
[answer extraction](../video_mcqa/README.md#answer-extraction-by-a-judge-model).

Reference scorer: `lvbench/metrics/compute_accuracy.py` at official commit
`518df47219862534dad39fa1373b4e7c862a4cd5`. Reference annotation snapshot:
`0caedb92002cc268bad486449e551c76f0485670`. These are provenance
references, not an implicit revision selected by the loader.

LVBench's [dataset terms](https://github.com/THUDM/LVBench#license) are
CC-BY-NC-SA-4.0 with research-only/noncommercial restrictions. Do not treat this
adapter's Apache-2.0 license as permission to train on or redistribute the data.
No real benchmark text, answers, videos or upstream implementation are bundled;
tests use invented examples, and the five examples in `data/` are invented
questions about synthetic clips ([example data](../video_mcqa/README.md#example-data)).
Dataset registration and a full real-model EF baseline are not part of this
adapter; configs are `verified: false`.
