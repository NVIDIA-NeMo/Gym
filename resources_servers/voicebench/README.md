# VoiceBench

One resources-server component contains explicit scoring handlers for the VoiceBench subsets.
Create separate named runtime configurations for subsets, and reuse `simple_agent` and the same
candidate model adapter. Rows retain source transcripts/references under `verifier_metadata`;
the candidate sees only `responses_create_params` and the external audio reference.

| Subset | `scoring_type` | Legacy aggregate |
|---|---|---|
| alpacaeval_full, commoneval, wildvoice | open | Actual returned judge-choice mean, multiplied by 20 |
| openbookqa, mmsu | mcq | Accuracy × 100; parsing fallback rate reported separately |
| bbh | bbh | Task-specific accuracy × 100 |
| ifeval | ifeval | Mean of strict/loose prompt/instruction ratios × 100 |
| advbench | refusal | Refusal rate × 100 |
| sd-qa | reference_qa | Reference-grounded Yes/No majority accuracy × 100 |

`subset`, `scoring_type`, and positive `expected_num_rows` are required. Unknown combinations fail
validation. `scoring_version: legacy_v1` identifies the pinned public VoiceBench parser profile
plus NVIDIA answer-format extensions for the `mcq` and `refusal` scorers: LaTeX `\boxed{}` and
Markdown-bold answers, and curly-apostrophe refusals. They recover answers the public parsers miss,
so OpenBookQA, MMSU and AdvBench can score higher than the public evaluator. See
[vendor/README.md](vendor/README.md#nvidia-extensions) for the exact changes and why. Legacy
MCQ/BBH fallback guesses share a random stream in dataset order, matching the source.
`seed: null` preserves its unseeded behavior; an explicit integer is available for parity tests.
Aggregation replays full rule rows in `_ng_task_index` order, separately for each rollout repeat.
Per-row verification rewards are provisional when random defaults or guesses are involved; the
`voicebench/` aggregate metrics are authoritative. Reaggregating unseeded runs may draw new guesses,
as rerunning the original evaluator would. Cached statistic-only rows cannot be replayed.
IFEval runs all strict checks before all loose checks and uses the source's last-response-wins lookup
for duplicate prompts. Its language detector retains its own stochastic behavior.
SD-QA lowercases every returned judgment and uses `max(set(scores), key=scores.count)`;
unexpected strings participate in voting and ties retain Python's hash-order behavior.
Exact historical random guesses and tied votes require the original RNG state and hash seed.

For judge scoring, configure `judge_model_server` and `judge_prompt_file`. The YAML prompt has
`system` (optional) and `user` containing `{prompt}` and `{response}`; SD-QA additionally needs
`{reference}`. Prompt bytes are hashed. Judge defaults are temperature .5, top-p .95, max tokens
1024, n=3, frequency/presence penalties 0 and stop null. A shared process-wide semaphore defaults
to 40 in-flight calls; judge resources require one worker. Requested versus actual choice counts,
raw outputs/scores, finish reasons, queue/call timing and observed peak concurrency are recorded.
Actual returned choice count determines the open score. A gateway returning one of three requested
choices produces a diagnostic and remains valid when the returned score is valid.

The answer is assembled from the last assistant message and parsed reasoning, then the full
prediction is stripped before extracting text after the last `</think>`. Leading newlines after
that tag are removed; other remaining answer whitespace is preserved. An empty final answer
falls back to reasoning, matching the reference audio
adapter. An entirely empty prediction is scored as `"nan"`, matching the reference evaluator's spreadsheet loading.
Original response content and `candidate_thinking` remain available. Incomplete candidates can retain
a legacy raw score but have zero completed-status-safe reward.

The legacy open parser accepts any float, including out-of-range and nonfinite values, and ignores
judge finish reasons. Nonfinite legacy metrics are encoded as JSON strings (`"NaN"`, `"Infinity"`,
`"-Infinity"`); the Gym scalar reward is zero for a nonfinite mean. `judge_numeric_valid` reports
whether the returned grades are finite and within 1–5 without changing the legacy score.
Unparseable grades and empty SD-QA choices preserve an inspectable `legacy_scoring_error` and prevent
aggregate score publication, matching the original evaluation failure rather than averaging a
partial result. Transport failures use Gym's masked `judge_failed` sidecar and can be recovered with
`gym eval reverify`; coverage diagnostics identify partial received evaluations.

Full completion requires exactly `expected_num_rows` distinct source sample IDs, valid scores and
completed candidate responses. Optional `expected_sample_ids_file` accepts a JSON array of source
sample IDs, one ID per line, or dataset JSONL with `verifier_metadata.sample_id`; its count must
match the configured expected count. Missing, unexpected and duplicate IDs prevent full completion.
Each runtime reports one subset; suite weighting belongs to the suite configuration, not a pooled
row-reward average. IFEval aggregate uses the four original denominator-aware ratios rather than
mean per-row rewards.

Rule modules and vendored source are imported only for the selected scorer. Only IFEval provisions
NLTK punkt/punkt_tab when missing; `nltk_data_dir` or `NLTK_DATA` selects the cache location,
otherwise the private `~/.voicebench_nltk_data` cache is used.
External audio is resolved by the model adapter, so scoring saved final outputs through `/verify`
does not require an audio/model call. `scorers.open_judge.score_outputs` also parses cached judge
grades without HTTP for offline score parity.

Start with `configs/voicebench.yaml` and provide subset/scorer/count, model wiring, judge settings
and versioned prompts from text-evals. This component declares no dataset downloads. The source
license/provenance for vendored rule parsers is recorded under `vendor/`.
Candidate generation sampling belongs to the selected versioned reference profile. For the
included baseline profile it is temperature .6, top-p .95, top-k -1 (disabled); judge sampling
is configured separately above.

`configs/voicebench_example.yaml` runs five repository-authored synthetic text questions through
the MCQ scorer. These are smoke examples, not official VoiceBench data or a benchmark baseline.
Their committed rollouts came from an audio-capable model through Gym's vLLM adapter.
The adapter returned `"\nA"`, which the unchanged legacy parser fails to extract and guesses;
with unseeded fallback, repeated aggregation can therefore report different accuracy.
Dataset validation was run with `gym dataset collate` in `example_validation` mode. A separate
real audio request from `openbookqa.tar` completed through the same adapter. Actual VoiceBench
inputs, audio archives, judge prompts and suite configurations remain in the external data repo.

Live judge smoke checks also used an audio-capable model with both versioned judge prompts and
`n=3`. SD-QA majority voting completed. The open judge returned one choice containing unboxed
reasoning text; verification preserved the parser error and aggregation refused a score, as the
legacy evaluator would. This checks transport and failure behavior; the local model is not the
historical GPT judge and these smoke outputs are not benchmark baseline scores.
