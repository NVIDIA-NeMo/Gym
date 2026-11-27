# Video-Index resources server

Thin fixed-cohort multiple-choice video verifier for [Video-Index](https://github.com/Espere-1119-Song/Video-Index)
v1.0 and v1.1, inheriting `VideoMCQAResourcesServer` (`resources_servers/video_mcqa`).

- **Evaluation only**: no benchmark items, example questions, or videos are committed here.
- **Releases**:
  - `videoindex_v1_0`: `Video-Index/Video-Index` @ `fb63b6c93948e78e76aee9c87745bf78358d3011` (840 questions, 2–10 options)
  - `videoindex_v1_1`: `Video-Index/Video-Index-v1-1` @ `e7404c468cd0ad83d7c96834c6eb4767fe831a72` (144 questions, latest cleaned release, 2–10 options)
- **Scoring policies**:
  - `videoindex_rule_v1` (default): official Apache-2.0 rule-based extractor adapted from
    `Espere-1119-Song/Video-Index` commit `4819d9252ae177f87a134f0ceda74855847afe2a` (`video_index.py` / `scoring.py`),
    applied after stripping `<think>`/`<thinking>` reasoning blocks.
  - `llm_extraction_v1`: shared `video_mcqa` judge-model option extraction (`configs/videoindex_*_llm_extraction.yaml`).
  - `boxed_only_v1` and `boxed_or_letter_v1`: shared `video_mcqa` deterministic policies for 2–10 choices.
- **Metrics**: overall accuracy (`mean/Video_Index_v1_0`, `mean/Video_Index_v1_1`, and `*_agentic` counterparts) plus
  fixed-denominator micro accuracy, coverage, completeness, and outcome counts across the four official capability groups:
  `perception`, `temporal`, `spatial_physical`, and `reasoning_knowledge`.
- **Benchmarks**:
  - `benchmarks/videoindex_v1_0` (single-turn, `simple_agent`)
  - `benchmarks/videoindex_v1_0_agentic` (agentic, `opencode_agent` + `resources_servers/video_mcqa/sandbox_image`)
  - `benchmarks/videoindex_v1_1` (single-turn, `simple_agent`)
  - `benchmarks/videoindex_v1_1_agentic` (agentic, `opencode_agent` + `resources_servers/video_mcqa/sandbox_image`)

## Configs

| Config | Setting | Answer Policy |
| --- | --- | --- |
| `configs/videoindex_single_turn.yaml` | Single-turn (`simple_agent`) | `videoindex_rule_v1` (or `boxed_*`) |
| `configs/videoindex_single_turn_llm_extraction.yaml` | Single-turn (`simple_agent`) | `llm_extraction_v1` |
| `configs/videoindex_agentic.yaml` | Agentic (`opencode_agent`) | `videoindex_rule_v1` (or `boxed_*`) |
| `configs/videoindex_agentic_llm_extraction.yaml` | Agentic judge overlay (`127.0.0.1`) | `llm_extraction_v1` |

See `third-party-notices.md` and `resources_servers/video_mcqa/README.md` for attribution, sandbox image build steps,
and protocol details.
