# Video-Index third-party notice

## Official prompt and answer extraction rule (Apache-2.0)

`resources_servers/videoindex/scoring.py` adapts `PROMPT`, `INTRO_VIDEO`, `ANSWER_INSTR`, `extract_letter`,
and `score_reply` from `integrations/vlmevalkit/vlmeval/dataset/utils/video_index.py` (matching the multiple-choice
branch of `video_index/scoring.py`) in
[Espere-1119-Song/Video-Index](https://github.com/Espere-1119-Song/Video-Index) at commit
`4819d9252ae177f87a134f0ceda74855847afe2a`, authored by the Video-Index authors (Song et al., arXiv 2610.00960) and
licensed under the Apache License, Version 2.0.

Local changes in `resources_servers/videoindex/scoring.py`:
- Names, type annotations, and docstrings.
- Integration with `video_mcqa`'s `VerifierMetadata`, `final_answer_text` reasoning-block stripping, and
  `QuestionScore` / fixed-cohort outcome accounting.
- Explicit validation of 2 to 10 choices (`A`–`J`).

## Upstream datasets (evaluation only, not redistributed)

Benchmark preparation downloads or reads user-staged copies of:
- `Video-Index/Video-Index` @ `fb63b6c93948e78e76aee9c87745bf78358d3011` (840 items)
- `Video-Index/Video-Index-v1-1` @ `e7404c468cd0ad83d7c96834c6eb4767fe831a72` (144 items)

No benchmark questions, answers, or videos are committed to this repository. Video-Index aggregates questions and
clips from multiple third-party video benchmarks with mixed upstream terms; benchmark manifests record
`licensing: unknown` and restrict usage to evaluation only.
