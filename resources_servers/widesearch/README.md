# WideSearch Resources Server

Parses Markdown tables and reproduces WideSearch's official strict score,
row-level precision/recall/F1, and item-level precision/recall/F1. It applies
the per-column preprocessing and metric pipeline supplied with each task.
Semantic entity alignment and `llm_judge` columns use the configured judge
model.

The evaluator is adapted from the MIT-licensed
[ByteDance-Seed/WideSearch](https://github.com/ByteDance-Seed/WideSearch)
implementation.
