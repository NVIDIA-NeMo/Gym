# MMMLU

[MMMLU](https://huggingface.co/datasets/openai/MMMLU) (multilingual MMLU) extends MMLU across 14 languages (plus optional English via ``--include_english`` in the prepare script). Data CSVs are fetched from OpenAI public blob storage, matching NeMo-Skills.

Grading uses ``mcqa`` with per-row ``template_metadata.output_regex`` (combined multilingual ``Answer:`` prefixes and a greedy final-letter fallback), ported from NeMo-Skills ``mmmlu`` ``extract_regex`` behavior.

## Usage

```bash
# Default: all supported non-English languages (same as NeMo-Skills default).
ng_prepare_benchmark "+config_paths=[benchmarks/mmmlu/config.yaml]"

# Optional: restrict languages or add English (EN-US).
python benchmarks/mmmlu/prepare.py --languages DE-DE FR-FR --include_english

ng_run "+config_paths=[benchmarks/mmmlu/config.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"

ng_collect_rollouts \
  "+config_paths=[benchmarks/mmmlu/config.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]" \
  +output_jsonl_fpath=results/mmmlu.jsonl
```
