# Shared grading for question-bank data

The question-bank adapter emits ordinary Gym rows with a question, reference,
agent route and provenance. It does not require a contract file or a custom
grading specification for each question.

Compose `configs/question_bank.yaml`, `configs/dsv41_flash.yaml`, and the chosen
policy-model configuration. Supply `STEM_JUDGE_BASE_URL`, `STEM_JUDGE_API_KEY`,
`STEM_JUDGE_MODEL_REVISION`, and optionally `STEM_JUDGE_MODEL` through the runtime
environment. The model alias defaults to the private DSV4.1 Flash NVCF route;
other gateways require the identifier from their own authenticated model catalog.

## Routes and final-answer selection

- `question_bank_math_simple_agent` uses the existing `math_with_judge` server's
  Math-Verify implementation, including its concurrency limit and terminable
  subprocess. A library match resolves correctness; a nonmatch, parse failure or
  library timeout goes to the structured DSV4.1 Flash judge.
- `question_bank_semantic_simple_agent` sends the final answer to the structured
  DSV4.1 Flash judge, using shared equivalence rules.

Both routes select only the **last** `\boxed{...}` in the final assistant
message. The library and the judge receive that same selected answer. Nested
braces and escaped literal braces are handled, and whitespace after `\boxed`
is accepted. A missing, empty or malformed last box fails the required output
format; it never recovers an earlier box. Policy truncation is distinct from a
truncated judge response, which is an unresolved grade.

The existing `math_with_judge` server also now isolates the last box before
calling Math-Verify. Previously, its library path could parse the whole response
even though its judge path received the final box.

Unresolved grades carry `mask_sample: true`, `evaluation_completed: false`,
`correctness: null`, and failure metadata. This server behavior does not establish
that a particular NeMoRL version excludes failures from all losses and group
statistics; that integration still requires a real rollout/training test.

`unresolved` means verification did not establish correctness. It includes an
inconclusive semantic judgment (ambiguous task, conflicting reference, or missing
evidence), an unsupported verification requirement, and an operational failure
(transport, incomplete judge output, or invalid output schema). A judge can finish
generating a valid `unresolved` JSON object while `evaluation_completed` remains
false: that field describes the correctness measurement, not generation completion.
The numeric `reward=0` on these records is a transport placeholder and must be
excluded using the mask. A completed `incorrect` judgment has `correctness=false`,
`evaluation_completed=true`, and `mask_sample=false`; it remains a negative example.
`grading_status` distinguishes `judge_unresolved`, `judge_incomplete`,
`judge_unparsed`, and other causes. The recorded judge response and `failure_reason`
retain the semantic explanation. Judge failures also use Gym's `judge_failed`
sidecar classification and the nested `instance_config.mask_sample` compatibility
flag. Sample exclusion can lead a consuming trainer to discard a whole prompt group.

## Shared scientific guidelines

The default shared prompt is `prompt_templates/shared_answer_v2.txt`. It uses
`question`, `reference`, and `candidate`; it does not require a `policy_input` field
to interpret the task. Effective policy input is still recorded for provenance.
`shared_answer_v1.txt` remains available through `shared_prompt_template_fpath`
for paired replay. Provenance records `stem-shared-v3` and the actual prompt hash.

The science guidance adapts Jiacheng Xu's
[`stem_v11.txt` at `fbe250878cc317e893392a641070bb448fbd7fd9`](https://github.com/NVIDIA-NeMo/Gym/blob/fbe250878cc317e893392a641070bb448fbd7fd9/resources_servers/stem_judge/prompt_templates/stem_v11.txt)
from this Apache-2.0 repository: required parts and alternatives, physical units and
conventions, chemical identity/stoichiometry/equilibrium, and biological entities,
sequences, inheritance, and regulation. The shared route retains its strict JSON
verdict schema and final-box input. It does not adopt the binary `Judgement` parser,
the exact-match shortcut, or rules that recover information from an unseen derivation.
The separate answer/rubric and contract interfaces keep their existing prompts.

Reference conflicts, genuinely ambiguous interpretations, precision absent from
the question, and required unavailable checks produce `unresolved`, with concrete
evidence required. Candidate assertions of ambiguity alone are insufficient.
Known unsupported task metadata is still rejected before judging. These prompt
rules apply when the LLM runs; a Math-Verify success can still return before that
call. They do not independently validate references on the library-acceptance path.

### Validation of the science prompt

On 2026-11-18, a paired replay of 20 fixed synthetic answers through the actual
shared verifier and Gym VLLM model adapter used
`nvidia/deepseek-ai/deepseek-v4.1-flash-1017`, temperature 0, and an 8192-token output
budget. The local resource-to-model HTTP hop was replaced by an in-process bridge;
model inference, response conversion, parsing, reward, and mask handling were real.
The [cases and recorded outcomes](tests/fixtures/shared-science-smoke.json) contain
no original dataset questions or policy traces. Expected verdicts were specified
before the calls. This is a prompt smoke test, not a dataset accuracy estimate,
a judge-variance study, or an end-to-end NeMoRL masking test.

The final v2 prompt matched 20/20 expected verdicts; v1 matched 18/20. V1 treated
an incorrect reference as authoritative (`2x` at `x=3`, reference `8`, candidate
`6`) and accepted a reference-identical answer to an undefined-parameter question.
V2 marked both unresolved. Both prompts handled the tested scientific equivalences,
wrong answers, missing vector direction, and candidate instruction injection.
The initial v2 draft incorrectly rejected a boxed formula on a proof task; the
final prompt explicitly gives unsupported verification scope precedence over the
missing-component rule, and the full paired replay was rerun. Unresolved records
had `evaluation_completed=false`, reward 0, and the exclusion mask in that replay.
The 251 STEM-judge unit tests also passed, including completed-generation versus
completed-evaluation semantics and replay of either prompt with provenance.

The longer prompt increases judge input tokens. Before a training rollout, replay
saved answers using the intended serving configuration and measure false grades,
unresolved rate, repeatability, and latency; this gateway smoke does not establish
behavior for a differently pinned serving model or trainer.

`judge_timeout_seconds` defaults to 300 seconds per HTTP attempt. Set it to
`null` to remove the resource server's deadline when the judge's serving policy
must govern long responses. Numeric timeouts must remain positive, finite, and
at most 300 seconds. `judge_endpoint_max_concurrency: null` separately disables
the resource's judge-call semaphore. Neither setting changes completion checks
or exclusion of unresolved judgments.

## Generation and grading instructions

The dataset adapter adds this exact instruction once:

> Solve the following problem step by step. Put your final answer, and only your final answer, in \boxed{}.

Gym's `simple_agent` forwards prepared input; there is no universal solver-prompt
append in that agent. Dataset preparation is the explicit insertion point here.
The original question is retained separately for provenance and verification.
The changed prompt is for new generations, and must not be retroactively applied
as an instruction that historical answers received.

`--prompt-overrides` accepts a JSON object with `domains` and `sources` maps of
names to additional generation instructions. For example:

```json
{"domains":{"physics":"Include the units requested by the question."},"sources":{}}
```

These are shared instructions, not per-row annotations. Canonical domain/source
keys come from the exported metadata. References and hidden tests must never be
included in generation instructions. Judge customization is separate: configure
`judge_domain_instructions` and `judge_source_instructions` in the resources
server YAML. Domain additions are applied before source additions. A Math-Verify
match still goes to the judge when generation or grading overrides apply, so a
scalar-equivalence result cannot bypass added obligations.

## Preparing the selected pool

Run from the repository root, using a new output directory:

```bash
python -m resources_servers.stem_judge.prepare \
  --dataset /path/to/questions.jsonl.gz \
  --inventory /path/to/row-grading-inventory.csv.gz \
  --development-ids /path/to/development-ids.json \
  --output-dir /path/to/new-output
```

The inventory is validated against the selected task IDs, versions and row
hashes. Its rubric/construction queue is excluded. Native-test declarations are
also excluded from the text routes, including coding tasks under other domain
labels. Known native/rubric metadata is checked again by the resources server
before any library acceptance. Coding requires its real tests and sandbox runner.

The immutable original pool and family split are preserved. The output contains
separate development/heldout Gym files, a ledger for every selected row, coverage
counts and checksums. Repaired rows retain their selected repaired question and
reference instead of restoring an original reference. Hash binding establishes
which data was used; it does not establish reference correctness.

These are **candidate evaluation inputs**, not a claim of training admission or
judge accuracy. The retained historical contract mode is optional and is not
used by this adapter. Run representative live DSV4.1 judgments and the selected
NeMoRL integration before admitting these routes to optimization.
