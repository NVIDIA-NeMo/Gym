# Description

CodeRabbit PR evaluation benchmark for NeMo Gym. Evaluates LLM-generated PR summaries and review classification tags.

The model receives a PR diff and must produce:
1. A summary of the changes
2. A review classification tag: `[review_needed_senior_swe]`, `[review_needed_junior_swe]`, or `[skip_review]`

Reward is 1.0 if the model predicts the correct tag AND an LLM judge scores the summary >= 4.0 (on a 1-5 scale), else 0.0. The judge evaluates factual accuracy, completeness, and clarity against a reference summary.

Data links: GitLab dataset registry (coderabbit_pr_eval)

# Licensing information
Code: Apache 2.0
Data: Apache 2.0

Dependencies
- nemo_gym: Apache 2.0
