# COBOL Compiler Benchmark

COBOL compilation and execution benchmark for NeMo-Gym. Uses the MultiPL-E dataset (499 problems) adapted for COBOL with GnuCOBOL.

## Prerequisites

GnuCOBOL (`cobc`) must be installed:

```bash
# Linux (Ubuntu/Debian)
apt-get install gnucobol

# macOS
brew install gnucobol
```

Verify installation:
```bash
cobc --version
```

## Dataset

499 problems from MultiPL-E (HumanEval + MBPP) converted to stdin/stdout format for COBOL. Each problem includes test cases with input/expected output pairs.

## Usage

### Single-turn evaluation (simple agent)

```bash
ng_test +entrypoint=resources_servers/cobol_compiler
ng_collect_rollouts +entrypoint=resources_servers/cobol_compiler
```

### Multi-turn evaluation (error correction agent)

```bash
ng_test +entrypoint=resources_servers/cobol_compiler +config=cobol_compiler_eval_agent
```

### Data preparation

```bash
ng_prepare_data +entrypoint=resources_servers/cobol_compiler
```

## Data Conversion

To regenerate the JSONL dataset from the DomainForge source:

```bash
python scripts/convert_dataset.py \
    --input ~/projects/domainforge/datasets/cobol_multipl_eval.json \
    --output data/cobol_multipl_eval.jsonl \
    --example-output data/example.jsonl
```

## License

Apache 2.0
