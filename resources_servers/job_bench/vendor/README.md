# Vendored Job-Bench evaluator

`judge.py` comes from the Apache-2.0 licensed
[`Job-Bench/job-bench-eval`](https://github.com/Job-Bench/job-bench-eval) repository. See
`LICENSE.job-bench-eval` for its license.

Source: `eval/judge.py` at commit
`f938276c815e0fb2281b2262262dbba52a9c3e0b`.
The upstream file's SHA256 is
`b94ea8ca8123bcbf5bde8fe31aa3a298ed1fc831edf3aeb18064e98e591744ce`.

The Gym copy adds the `job-bounded-utf8-v1` aggregate text budget described in
the parent README. Rubric prompts and weighted scoring remain upstream-derived;
this input-preparation variant must be reported separately from the published protocol.

Gym also distinguishes terminal transport failures from malformed judge replies
across retries, so an earlier malformed reply cannot hide a later API failure.
