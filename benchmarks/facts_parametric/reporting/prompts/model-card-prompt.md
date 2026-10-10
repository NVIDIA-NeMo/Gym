You are writing the prose sections of a one-to-two page evaluation report for a model run on a benchmark. The
deterministic parts of the report (tables, denominators, provenance) are rendered by code; you write only the
sentences. Your output must be a single JSON object that validates against the provided schema.

Rules:
1. Use only the grounded context below. Every number you write must appear in `allowed_numbers` or be a count/
   percentage that appears verbatim in a metric, anchor fact, or reference in the context. Do not compute new numbers.
   Keep each count with the unit the context gives it (answers, grader samples, cases, labels, conversations); never
   relabel a count of cases as labels or a count of samples as answers.
2. Every claim in `what_the_run_says`, `calibration_and_reference`, and `limitations` must cite one or more
   anchor-fact ids from `anchor_facts` in its `evidence` list. Cite only ids that exist.
3. `examples` must use only anchor facts whose category is `example`; include every id listed in
   `required_example_anchor_ids` (up to four). Describe what the target model did, sanitized, without quoting raw
   sensitive content beyond the excerpt already given. Keep every example `description` under 320 characters and end
   it with a full stop: the schema hard-caps it at 400 characters and a cut-off sentence is rejected.
4. Write plainly: short sentences, denominators stated, no marketing language, no training or purchasing advice,
   no claims about causes that the traces do not show. Do not mention the report writer or the writing process.
5. `purpose` is one sentence about what the benchmark measures. `interpretation` is two to four complete sentences
   on what the numbers do and do not mean, including the main limitation.
6. Do not imply that Anthropic, NVIDIA, Snorkel, Google, Kaggle, or the benchmark authors endorse or produced this
   report.

Grounded context (JSON):

{context_json}

Return only the JSON object.
