# CG-Bench resources server

Thin fixed-cohort full-video multiple-choice (`long_acc`, without subtitles) verifier for
[CG-Bench](https://huggingface.co/datasets/CG-Bench/CG-Bench), inheriting `VideoMCQAResourcesServer`
(`resources_servers/video_mcqa`). Clue-clip QA (`clue_acc`), temporal grounding (`mIoU`), and open-ended QA are out of
scope for this server.

- **Evaluation only**: no benchmark items, example questions, or videos are committed here.
- **Gated dataset access**: `CG-Bench/CG-Bench` on Hugging Face is `gated: auto` (declared `MIT` license in its dataset
  card; video copyrights remain with original creators for academic research use). Accept the dataset terms on Hugging
  Face and set `HF_TOKEN` (or run `huggingface-cli login`) before downloading annotations or video archives.
- **Pinned slices** (`CG-Bench/CG-Bench` @ `aa63c0b7f74b5b43b5e0c1f012ea96bc9f460c57`):
  - `cgbench` (`cgbench.json`): 12,129 questions, 1,219 videos, 5–8 choices (`A`–`H`)
  - `cgbench_mini` (`cgbench_mini.json`): 3,000 questions, 1,118 videos, 5–8 choices (`A`–`H`)
  - `cgbench_hard` (`cgbench_hard.json`): 2,417 questions, 1,010 videos, 5–8 choices (`A`–`H`)
- **Scoring policies**:
  - `cgbench_rule_v1` (default): strips `<think>`/`<thinking>` blocks, parses CG-Bench's `{"result": "<letter>"}` JSON
    format (fenced or inline, with `\boxed{...}`, standalone letter, and explicit stated-answer fallbacks), and enforces
    each question's exact choice count (`A`–`H`).
  - `boxed_only_v1` and `boxed_or_letter_v1`: shared `video_mcqa` deterministic policies (pair with `prompt="boxed"`).
  - `llm_extraction_v1`: shared `video_mcqa` judge-model option extraction (`configs/cgbench_*_llm_extraction.yaml`).
- **Metrics**: fixed-denominator overall accuracy (`mean/CG_Bench`, `mean/CG_Bench_mini`, `mean/CG_Bench_hard`, and
  `*_agentic` counterparts) plus breakdowns along CG-Bench's four official reporting axes:
  - `category`: `Perception`, `Cognition`, `Hallucination`
  - `sub_category`: all 12 official sub-categories
  - `domain`: all 14 official video domains
  - `duration`: 7 official 10-minute duration buckets (`0-10m`, `10-20m`, `20-30m`, `30-40m`, `40-50m`, `50-60m`, `60m+`)
- **Benchmarks**:
  - `benchmarks/cgbench` / `benchmarks/cgbench_agentic`
  - `benchmarks/cgbench_mini` / `benchmarks/cgbench_mini_agentic`
  - `benchmarks/cgbench_hard` / `benchmarks/cgbench_hard_agentic`

## Configs

| Config | Setting | Answer Policy |
| --- | --- | --- |
| `configs/cgbench_single_turn.yaml` | Single-turn (`simple_agent`) | `cgbench_rule_v1` (or `boxed_*`) |
| `configs/cgbench_single_turn_llm_extraction.yaml` | Single-turn (`simple_agent`) | `llm_extraction_v1` |
| `configs/cgbench_agentic.yaml` | Agentic (`opencode_agent`) | `cgbench_rule_v1` (or `boxed_*`) |
| `configs/cgbench_agentic_llm_extraction.yaml` | Agentic judge overlay (`127.0.0.1`) | `llm_extraction_v1` |

See `resources_servers/video_mcqa/README.md` for sandbox image build steps (`resources_servers/video_mcqa/sandbox_image`)
and shared single-turn / agentic evaluation protocol details.
