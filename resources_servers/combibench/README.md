# CombiBench benchmark environment

[Dataset](https://huggingface.co/datasets/AI-MO/CombiBench) and
[repository](https://github.com/MoonshotAI/CombiBench) (MIT) and
[paper](https://arxiv.org/abs/2505.03171), "CombiBench: Benchmarking LLM
Capability for Combinatorial Mathematics" (Liu et al., 2025).

One hundred combinatorics problems formalized in Lean 4, from middle-school
exercises to IMO. Fifty-five are proof-only: the model replaces the `sorry`
proof of a theorem. Forty-five are fill-in-the-blank: the statement declares
`abbrev <name>_solution : <type> := sorry` and a theorem about it, and the model
must supply both the answer and the proof. This server reproduces upstream's
**one-stage Fine-Eval** verdict: a Lean 4 compile through a Kimina Lean Server
plus the syntactic checks upstream applies around it.

Pinned upstream revision: GitHub `c67e4213597b1477351d9ef5ca37fb622084cc78`
(Lean and Mathlib `v4.24.0`). The Hugging Face dataset revision
`882ba08befd0856f5364db1e53d58c7e2cf704f9` is the alternative source; the
benchmark README explains why it is not the default.

## Scope

Two benchmarks share this server:

| Benchmark | Setting | Rows |
| --- | --- | --- |
| [`benchmarks/combibench`](../../benchmarks/combibench/) | "without solution": answer and proof both withheld | 100 |
| [`benchmarks/combibench_with_solution`](../../benchmarks/combibench_with_solution/) | "with solution": published answers substituted, proof withheld | 100 |

Upstream also ships a two-stage Fine-Eval that, when the filled-in answer is not
proved equal to the ground truth by `rfl`/`norm_num`, asks the model for that
equality proof in a second turn. It is not implemented; the paper reports that
in the without-solution setting "both evaluation methods ... produced identical
results". A row that would need the second stage scores 0 here.

## Prompting

Upstream's own prompt, byte-identical, from `evaluation/config/template.json5`:
a system message ("You are an expert in mathematics and proving theorems in
Lean 4.") and a user message that shows only the formal statement inside a
```` ```lean4 ```` fence. The informal statement is carried in the data as
`natural_language` but is not shown, matching upstream. Runs are comparable to
the paper's protocol in prompt and endpoint style (chat); decoding parameters
and the token budget are not published, so no run is a reproduction.

## Dataset format

Rows are flat task fields; `prompt.yaml` builds the messages at rollout time.

```json
{
  "theorem_name": "hackmath_1",
  "formal_statement": "import Mathlib\n\nabbrev hackmath_1_solution : ℕ := sorry\n\ntheorem hackmath_1 ... := by sorry",
  "answers": ["1716"],
  "natural_language": "How many ways can a teacher select ...",
  "tag": "hackmath",
  "source": "https://www.hackmath.net/en/word-math-problems/combinatorics",
  "split": "test",
  "dataset_source": "github",
  "dataset_revision": "c67e4213597b1477351d9ef5ca37fb622084cc78"  # pragma: allowlist secret
}
```

`formal_statement` and `answers` are read by `verify()`. `tag` groups the
per-family metrics. `natural_language`, `source`, `split`, `dataset_source`
and `dataset_revision` are provenance and are never read. The tracked
`data/example.jsonl` carries `responses_create_params.input` already rendered
and a `task_source` key, because `gym dataset render` and `gym dataset collate`
rewrite the rows; the task fields are identical.

## Scoring

`reward` is 1.0 iff every step below passes, else 0.0. Steps 1–5 are upstream's
rules (`evaluation/util.py`, `evaluation/verifier/one_stage_verify.py`); the
`status` field names the first one that failed.

1. **Extract** the last ```` ```lean4 ```` block (falling back to ```` ```lean ````);
   none → `format_error`; empty output → `empty_output`; a block longer than
   `max_code_characters` (200,000; the longest pinned statement is 3,054) →
   `code_too_long`, so nothing pathological is put on the wire.
2. **Remove comments.** The paper's Appendix A.2 shows a model passing the Lean
   check by hiding the real theorem in a comment; stripping comments first is
   what defeats it.
3. **Prepend** upstream's default header (`import Mathlib`, `import Aesop`,
   `set_option maxHeartbeats 0`, `open BigOperators Real Nat Topology Rat`) only
   when the code does not start with an import.
4. **Forbid** the substrings `axiom` and `local_instance` → `forbidden_keyword`.
5. **Statement check.** Every non-header paragraph of the reference statement,
   with `sorry` removed, must appear verbatim in the code → `statement_mismatch`.
   This is the guard against proving a weaker theorem.
6. **Answer check.** For each `abbrev <name>_solution`, append
   `example : <name>_solution = (<gold> : <type>) := by try rfl; try norm_num`.
7. **Compile** through the Lean server with a 60 s timeout. Any error message →
   `proof_failed`; a `sorry` warning or REPL `sorries` entry → `has_sorry`;
   the REPL timing out on the submission → `timeout`; any other REPL error
   string → `lean_error`, which is charged to the model because upstream's
   `is_error` fails the submission on it too.

`harness_failure` is 1.0 for the three outcomes the model cannot cause, and
`failure_reason` is set only then:

| Status | Cause |
| --- | --- |
| `lean_server_error` | the Lean server is unreachable or replied malformed |
| `header_timeout` | a cold REPL could not finish `import Mathlib` inside the timeout — Kimina reports this as `Lean REPL header command timed out`, distinct from the submission timing out |
| `bad_task` | the row cannot be scored (no `formal_statement`, malformed `answers`) |

A submission `timeout` is charged to the model: a proof that does not terminate
is the model's output, and excusing it would make hanging reward-neutral.

Every response carries `lean_version`, the Lean version the server reports for
`#eval Lean.versionString`. It is probed once per process, before the first
compile (which also warms the REPL). A server built for another toolchain
otherwise scores every row `proof_failed` with nothing in the rollouts saying
why.

`compute_metrics` adds `hackmath/`, `brualdi/`, `imo/` and
`math_competitions/` pass rates keyed on `tag`. Upstream reports one pooled
figure, so the inherited `mean/reward` stays the headline and the per-family
keys are supplementary.

### Two deliberate departures from upstream

**The gold answer is elaborated at the abbreviation's declared type.** Lean's
`=` elaborates both sides before unifying them, so in upstream's form
`example : imo_2014_p2_solution = fun n => ⌈√n⌉₊ - 1` the right-hand side is
read with `n : ℝ` and fails against an `ℕ → ℕ` abbreviation, and
`fun n => n * (n + 1) / 4` fails to synthesize `HDiv ℕ ℕ ℚ`. Measured with the
published answers substituted into their own statements: upstream's form
elaborates for **40 of 45** fill-in-the-blank problems; `brualdi_ch8_6`,
`imo_2014_p2`, `imo_2019_p5`, `imo_2022_p1` and `imo_2023_p5` can never score
under it, even with the exact gold answer and a complete proof. Ascribing the
declared type, `(<gold> : <type>)`, elaborates for **45 of 45**. Set
`answer_check_ascription: false` for upstream's behaviour; both measurements are
in `data/harness_validation_github_test*.json`.

**Trailing whitespace is ignored in the statement check.** Statements carry
lines consisting only of spaces, left behind when comments were deleted from
the published copy. The count depends on which copy is measured: **12 of the
100** Hugging Face `test` statements at `882ba08b` (13 counting any trailing
whitespace), and **1 of the 100** rows `prepare.py` writes from the GitHub
files, the default source. Trailing whitespace is never significant to Lean, so
a model that reproduces the statement without those invisible characters has
not changed what it proves; upstream's byte-exact substring test would reject
it. Indentation and every visible character are still compared exactly. Set
`normalize_trailing_whitespace: false` for upstream's behaviour.

### Known blind spots, kept for fidelity

- `native_decide`, `decide` and `set_option maxHeartbeats` are allowed, as
  upstream allows them. `native_decide` trusts the compiler rather than the
  kernel.
- The `axiom` ban is a substring test: an identifier containing `axiom` fails
  a valid proof, and constructs upstream does not name (`opaque`,
  `implemented_by`, `unsafe`) are not banned.
- Extra declarations are allowed anywhere in the code; only the reference
  paragraphs are required.
- A statement paragraph that opens with `open ... in` directly above its
  `theorem` starts with a header prefix, so the whole paragraph — theorem
  included — is skipped by the statement check. Upstream has the identical
  blind spot and no paragraph in the pinned corpus is shaped that way, so the
  behaviour is kept rather than diverging.
- `REVERIFY_MODE` is `STATELESS`: the server carries nothing between calls.
  It is not a claim that Lean compilation is a pure function of the text —
  timeouts depend on the machine.

## Harness validation

Model-free checks, all run against Mathlib v4.24.0 through this server's code
path; reports are committed under `data/`.

| Check | GitHub source (default) | Hugging Face source |
| --- | --- | --- |
| Statement with its `sorry`s compiles (only `sorry` warnings) | **100 / 100** | 94 / 100 |
| "With solution" statement compiles | **100 / 100** | 94 / 100 |
| Published answer substituted + ascribed answer check elaborates | **45 / 45** | — |
| Same with upstream's unascribed check | 40 / 45 | — |

The six Hugging Face failures (`hackmath_6`, `imo_2008_p5`, `imo_2011_p2`,
`imo_2021_p5`, `imo_2022_p6`, `imo_2023_p5`) are statements upstream rewrote
in the repository for the Lean bump and never pushed to the dataset.

Negative controls through `verify()`, all 100 rows, every one scoring 0:

| Control | Denominator | Status observed |
| --- | --- | --- |
| Empty output | 100 | `empty_output` |
| Statement echoed back with `sorry` | 100 | `has_sorry` (55 proof-only), `proof_failed` (45 fill-in: the answer check cannot reduce a `sorry` abbrev) |
| `axiom cheat : False` prepended | 100 | `forbidden_keyword` |
| Main theorem's goal replaced by `True` | 99 (one statement's theorem is not the last declaration) | `statement_mismatch` |

The controls cover these named failure classes and no others. Upstream publishes
no reference proofs, so there is no gold-as-prediction check over the real
corpus; the five synthetic example problems have complete proofs in
`tests/fixtures/synthetic_solutions.json`, and all five score 1.0 through the
live server (`data/harness_validation_example.json`). One of them answers
`3 / 12` where the gold is `1 / 4`, exercising the `norm_num` path of the
answer check.

## Lean server

Upstream verifies through [Kimina Lean Server](https://github.com/project-numina/kimina-lean-server)
(MIT): a FastAPI service that pools Lean REPL processes keyed by import header,
so `import Mathlib` is paid per pool member rather than per proof. The published
image defaults to a different Lean version, so this server ships its own
Dockerfile pinned to upstream CombiBench's toolchain —
[`kimina_image/`](kimina_image/), with the Lean tarball checksummed and Mathlib,
the REPL and the server each pinned by commit:

```bash
docker build \
    --build-arg LEAN_VERSION=v4.24.0 \
    --build-arg LEAN_SHA256=b14f5e5159219dd1a1956c3b806813319f5e94ccd5bdfd56f54520609a5bb5ec \
    --build-arg MATHLIB_COMMIT=f897ebcf72cd16f89ab4577d0c826cd14afaafc7 \
    --build-arg REPL_COMMIT=8fff8552292860d349b459d6a811e6915671dc0d \
    --build-arg KIMINA_COMMIT=fb2393de3461db35eda4c714e3fd21187e92ec90 \
    -t kimina-lean-server:v4.24.0 \
    resources_servers/combibench/kimina_image
docker run -d --name kimina-combibench -p 12332:8000 \
    -e LEAN_SERVER_MAX_REPLS=8 \
    kimina-lean-server:v4.24.0
curl http://127.0.0.1:12332/health     # {"status":"ok"}
```

Leave `LEAN_SERVER_MAX_REPL_MEM` at the image's 12G. It becomes `RLIMIT_AS` on
each REPL and a REPL holding Mathlib exceeds 8G, at which point every `/verify`
returns `Failed to start REPL` — note that `/health` answers `ok` regardless,
because FastAPI is up long before a REPL is. The image's build gate starts a
REPL and requires it to load Mathlib, so that failure cannot reach a run.

The build downloads the Mathlib cache and takes a few minutes; the first proof
after start-up pays an `import Mathlib` load, later ones reuse the REPL. Point
the server at it with `COMBIBENCH_LEAN_SERVER_URL` (and
`COMBIBENCH_LEAN_SERVER_API_KEY` if the server has one).

Keep `max_concurrent_lean_requests` equal to the server's
`LEAN_SERVER_MAX_REPLS`. Rollout fan-out is otherwise unbounded, and a request
beyond that number is a connection waiting for a REPL that does not exist yet;
past the server's own queue it becomes a 429 charged to nobody. A compile that
exhausts the client timeout is not retried (`_max_connection_retries=1`):
Gym's shared client would otherwise spend three REPL jobs and three times the
wall clock to reach the same verdict.

### Why not the existing `math_formal_lean` sandbox

| | `math_formal_lean` sandbox | Kimina Lean Server |
| --- | --- | --- |
| Lean/Mathlib | v4.12.0 (v4.19.0 image also built) | pinned here to v4.24.0, upstream's toolchain |
| REPL reuse | one process per request | header-keyed REPL pool, so `import Mathlib` is paid once |
| Relation to upstream | none | the server upstream's own harness talks to |

The toolchain is the blocking difference: these statements do not compile on
v4.12.0, so reusing that path would have meant rebuilding its image at v4.24.0
anyway. Having rebuilt it, the remaining choice was between re-deriving
upstream's REPL handling on top of the sandbox's request shape and talking to
the server upstream already uses, whose `/verify` contract this client
reproduces. The Kimina pin is a commit rather than a release because the
project publishes no tags at all: `fb2393de` (2026-01-11) is the head of
`main`, and still the latest commit. Its own default is Lean v4.26.0; the
version is a build argument, so the image here is built with
`LEAN_SERVER_LEAN_VERSION=v4.24.0` to match upstream CombiBench's toolchain.
The consequence of the pin being head-of-branch is that it does not move on its
own, and re-pinning it is a one-line change plus a rebuild.

## Quickstart

All commands run from the repository root with the Lean server up.

```bash
# 1. start servers (leave running)
COMBIBENCH_LEAN_SERVER_URL=http://127.0.0.1:12332 gym env start \
    --resources-server combibench \
    --model-type openai_model \
    --model "<model id>" \
    --model-url "<openai-compatible base url>" \
    --model-api-key "$API_KEY"

# 2. collect rollouts against them
gym eval run --no-serve \
    --agent combibench_simple_agent \
    --input resources_servers/combibench/data/example.jsonl \
    --output resources_servers/combibench/data/example_rollouts.jsonl
```

Regenerating the committed example is three stages — the tracked file is the
output of the third, not the first:

```bash
python benchmarks/combibench/prepare.py \
    --source-file resources_servers/combibench/tests/fixtures/synthetic_problems.json \
    --output resources_servers/combibench/data/example_prepare.jsonl
gym dataset render \
    --input resources_servers/combibench/data/example_prepare.jsonl \
    --prompt-config benchmarks/combibench/prompt.yaml \
    --output resources_servers/combibench/data/example.jsonl
gym dataset collate \
    --config resources_servers/combibench/configs/combibench.yaml \
    --mode example_validation \
    --output-dir resources_servers/combibench/data
```

Re-run the model-free validation after any change to the scoring path:

```bash
python resources_servers/combibench/scripts/harness_validation.py \
    --input benchmarks/combibench/data/combibench_test.jsonl \
    --output resources_servers/combibench/data/harness_validation_github_test.json \
    --lean-server-url http://127.0.0.1:12332
```

### The committed example is synthetic

`data/example.jsonl` holds five hand-written problems in the upstream shape
(binomial coefficient, permutation count, pigeonhole, Gauss sum, a rational
probability), not benchmark rows. They cover a fill-in answer of each type the
verifier handles differently — natural number, function, rational — one
proof-only statement, and a statement carrying a spaces-only line to mirror the
published data. Real runs use the prepared split, which is never committed.

## Tests

```bash
gym env test --resources-server combibench
```

## Licensing

Code: Apache 2.0. `fine_eval.py` re-derives the extraction, statement and
answer rules of upstream's MIT-licensed `evaluation/util.py` and
`one_stage_verify.py` (Copyright (c) 2025 Moonshot AI and Project Numina);
`benchmarks/combibench/prompt.yaml` reproduces the prompt strings from the same
repository's `evaluation/config/template.json5`.

CombiBench data: MIT, per the repository `LICENSE` and the dataset card.
Upstream sources: hackmath.net exercises, Brualdi's *Introductory
Combinatorics*, IMO and other olympiad problems (APMO, Baltic Way, EGMO, IMO
Shortlist, IZhO, BxMO, USAMO); the IMO 2024 P3 and P5 statements are taken from
Mathlib's `Archive`. No benchmark rows are committed; `prepare.py` downloads
them at run time.
