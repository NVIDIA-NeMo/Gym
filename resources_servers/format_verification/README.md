# Description
> Keywords: Instruction Following, Format Verification, Regex, Citation, Freeform Formatting

This is a resources server for verifying the ability of the model to follow text formatting and citation instructions.

It supports two verifier types via a single `/verify` endpoint:

1. **Regex** (`verifier.type == "regex"`) -- Counts lines matching regex patterns. Used for freeform formatting tasks (bullets, headings, tables, key-value, etc.).
2. **String Match** (`verifier.type == "string_match"`) -- Checks that expected citation markers appear in the response and no spurious citations exist. Used for reference/citation format tasks.

> [!IMPORTANT]
> Evaluation is based on **format adherence only**.
> Content correctness is not verified -- the reward is 1.0 if the output matches the formatting/citation pattern, 0.0 otherwise.

See [ARCHITECTURE.md](ARCHITECTURE.md) for system diagrams and detailed field mappings.

## Datasets

### ds2: Freeform Formatting (55 records)
23 formatting pattern types including bullets, numbered lists, headings, key-value pairs, tables, and mixed formats. Each record has a `verifier` dict with regex patterns and a minimum match threshold.

### ds3: Citation Format (96 records)
9 reference styles (`[ref:N]`, `<ref:N>`, `{ref:N}`, `[source:N]`, `[web:N]`, `[N]`, `<<N>>`, `(Part N)`, `(ref N)`). Each record has a `verifier` dict with expected markers and detection patterns.

## Example Usage

### Freeform Formatting
```bash
gym env start \
    --model-type openai_model \
    --resources-server format_verification/freeform_formatting
```

Collect rollouts:
```bash
gym eval run --no-serve \
    --agent freeform_formatting_simple_agent \
    --input resources_servers/format_verification/data/ds2_freeform_formatting_train.jsonl \
    --output results/freeform_formatting_rollouts.jsonl \
    --concurrency 256 \
    --resume
```

### Citation Format
```bash
gym env start \
    --model-type openai_model \
    --resources-server format_verification/citation_format
```

Collect rollouts:
```bash
gym eval run --no-serve \
    --agent citation_format_simple_agent \
    --input resources_servers/format_verification/data/ds3_citation_format_train.jsonl \
    --output results/citation_format_rollouts.jsonl \
    --concurrency 256 \
    --resume
```

## Downloading Data

### Freeform Formatting
```bash
gym dataset collate \
    --config responses_api_models/vllm_model/configs/vllm_model_for_training.yaml \
    --resources-server format_verification/freeform_formatting \
    --output-dir data/format_verification_freeform/ \
    --mode train_preparation \
    --download
```

### Citation Format
```bash
gym dataset collate \
    --config responses_api_models/vllm_model/configs/vllm_model_for_training.yaml \
    --resources-server format_verification/citation_format \
    --output-dir data/format_verification_citation/ \
    --mode train_preparation \
    --download
```

## Testing
```bash
gym env test --resources-server format_verification
```

## Licensing
Code: Apache 2.0

Data: CC BY 4.0

Dependencies:
- nemo_gym: Apache 2.0
