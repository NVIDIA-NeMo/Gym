# GPQA Diamond

[GPQA](https://arxiv.org/abs/2311.12022) (Graduate-Level Google-Proof Q&A) Diamond is a challenging multiple-choice question answering benchmark with graduate-level questions across physics, biology, and chemistry.

## Configuration

This benchmark uses the `mcqa` resource server with the `mcqa_simple_agent`.

- **Grading mode**: `lenient_answer_colon_md` (markdown-aware `Answer: X` extraction, matching NeMo-Skills evaluator behavior)
- **Prompt**: `Answer the following multiple choice question. The last line of your response should be in the following format: 'Answer: A/B/C/D' ...`

## Usage

```bash
# Prepare data
gym eval prepare --benchmark gpqa

# Start servers
gym env start \
    --benchmark gpqa \
    --model-type vllm_model

# Collect rollouts
gym eval run --no-serve \
    --benchmark gpqa \
    --model-type vllm_model \
    --output results/gpqa.jsonl
```
