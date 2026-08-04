# MMLU-ProX

[MMLU-ProX](https://arxiv.org/abs/2503.10497) is a multilingual extension of MMLU-Pro with 10 answer choices (A–J) across 29 languages. Questions are professionally translated and include language-specific answer extraction patterns.

The supported language codes are: Afrikaans (`af`), Arabic (`ar`), Bengali (`bn`), Czech (`cs`), German (`de`), English (`en`), Spanish (`es`), French (`fr`), Hindi (`hi`), Hungarian (`hu`), Indonesian (`id`), Italian (`it`), Japanese (`ja`), Korean (`ko`), Marathi (`mr`), Nepali (`ne`), Portuguese (`pt`), Russian (`ru`), Serbian (`sr`), Swahili (`sw`), Telugu (`te`), Thai (`th`), Ukrainian (`uk`), Urdu (`ur`), Vietnamese (`vi`), Wolof (`wo`), Yoruba (`yo`), Chinese (`zh`), and Zulu (`zu`).

## Configuration

This benchmark uses the `mcqa` resource server with the `mcqa_simple_agent`.

- **Grading mode**: `null` — each row supplies its own language-specific extraction regex via `template_metadata.output_regex`
- **Prompt**: Passthrough (`{question}` only) — the complete formatted question including options is baked into the data during preparation

## Usage

```bash
# Prepare data
gym eval prepare --benchmark mmlu_prox

# Start servers
gym env start \
    --benchmark mmlu_prox \
    --model-type vllm_model

# Collect rollouts
gym eval run --no-serve \
    --benchmark mmlu_prox \
    --model-type vllm_model \
    --output results/mmlu_prox.jsonl
```
