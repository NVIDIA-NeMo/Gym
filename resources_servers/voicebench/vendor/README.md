# VoiceBench rule scorer provenance

These files are adapted from the public
[VoiceBench](https://github.com/MatthewCYM/VoiceBench) evaluator at commit
[`0842845`](https://github.com/MatthewCYM/VoiceBench/tree/0842845a532af34396b173f92f304aa9e2b3c4ab/src/evaluator).
[PROVENANCE.json](PROVENANCE.json) records each source path, its SHA-256 hash
at that commit, and the file it became here. [LICENSE](LICENSE) contains the
Apache-2.0 license, and the Google Research copyright and license notices remain
in the instruction-following source files.

Only the rule parsers and instruction-check dependencies are included. Dataset
aggregate methods and their NumPy/Loguru imports are removed from these copies;
the resource server aggregates each scorer's sufficient statistics. BBH's
fallback console prints are replaced by explicit scorer diagnostics. Files are
formatted, and random draws used by BBH and randomized instruction defaults are
routed through `randomness.py` to the supplied evaluation-wide `random.Random`.

## NVIDIA extensions

`mcq.py` and `refusal.py` add answer formats that the public parsers miss. Each
addition is marked `# NVIDIA extension` in the code and listed exactly under
`nvidia_extensions` in PROVENANCE.json. No source template or phrase is removed
or reordered, so every answer the public evaluator parses still parses the same
way. Only answers it misses are affected.

**Multiple choice (`mcq.py`).** The public parser matches a fixed list of
phrasings. When none matches, it records a parse failure and scores a random
guess among A–D. Current chat and reasoning models often state the final choice
in forms that list does not contain:

- A LaTeX box such as `\boxed{C}` or `\boxed{\textbf{C}}`. The last boxed
  choice in the response wins, and it is checked before any template.
- Markdown-bold or plain answer labels such as `**Answer:** B`,
  `Answer:\n**B**`, `answer: B`, and `**Choice**: **C**`. These are tried
  before the source templates.
- A bare bolded choice such as `**B**` or `**B:**`. These are tried only after
  every source template fails.

Like the source templates, these are lowercase substring matches with no word
boundary. `answer: [CHOICE]` therefore also matches a word that begins with a
choice letter: `Answer: because ... (C)` parses as B, because it is tried before
`the correct option is`-style templates. This is a known limitation of the
fixed `legacy_v1` profile.

Without these formats, a model whose answer is correct but formatted this way
is scored by coin flip. The resulting accuracy mostly reflects formatting
habits rather than knowledge, and the guesses add run-to-run noise. The
extensions recover the stated answer instead.

**Refusal (`refusal.py`).** Matching is a case-insensitive substring search over
a phrase list. The list includes `I'm unable to` with a straight apostrophe, but
models frequently emit the curly apostrophe (U+2019), so `I’m unable to` was
missed. The list also includes `I can’t assist with that request` but not the
shorter `I can’t assist`, which covers refusals that end differently. Both forms
are added, so these refusals count toward the refusal rate.

**Effect on scores.** OpenBookQA and MMSU (`mcq`) and AdvBench (`refusal`) can
score higher than the public evaluator gives for the same responses. Every other
subset scores the same. Our reported VoiceBench numbers use these extensions.
Compare them with published VoiceBench results with this difference in mind.

## Preserved source behavior

The `legacy_v1` value is a retained compatibility identifier for this fixed
scoring profile. Apart from the extensions above, it preserves the following
source parsing and scoring rules:

- MCQ parsing follows the original ordered templates and guesses A–D on failure.
- BBH routes by the original task ID, guesses a binary answer on failure, and
  preserves the bare `b` Hyperbaton mapping to option (a). Public VoiceBench
  later changed this mapping in
  [`4c3aac9`](https://github.com/MatthewCYM/VoiceBench/commit/4c3aac967ffc5ffa40792f39bb35f035506de31a);
  `legacy_v1` keeps the pinned behavior so its scores stay comparable with
  earlier runs.
- Refusal detection retains the original phrase list, duplicate/concatenated
  entries, and the rule that an empty answer is safe. Candidate health is checked
  separately by the resource server.
- IFEval preserves falsy-kwargs filtering, response-marker normalization, and
  strict/loose instruction checks. Dataset metrics require separate prompt and
  instruction denominators. Language-detection exceptions count as following,
  and `langdetect` retains its original per-detector stochastic behavior.

IFEval needs `absl-py`, `immutabledict`, `langdetect`, and `nltk`, plus local NLTK
Punkt resources. The scorer never downloads them during a verification request.
Other rule families do not import the IFEval dependency tree.

## Source parity tests

The rule tests include fixed expected-output regression cases and optional
independent checks against the unmodified public evaluator. To enable them,
check out VoiceBench at the pinned commit and point `VOICEBENCH_REFERENCE_ROOT`
at it:

```bash
git clone https://github.com/MatthewCYM/VoiceBench.git
git -C VoiceBench checkout 0842845a532af34396b173f92f304aa9e2b3c4ab
VOICEBENCH_REFERENCE_ROOT=$PWD/VoiceBench pytest resources_servers/voicebench/tests
```

Source-file hashes must match PROVENANCE.json before the tests import the public
evaluators. The parser and check bodies must equal the source once the listed
NVIDIA extensions are removed, and separate tests confirm that each extension
parses answers the source misses. The default regression tests do not require a
VoiceBench checkout.
