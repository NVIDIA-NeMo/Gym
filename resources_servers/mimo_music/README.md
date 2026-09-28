# MiMo music composition

A single-turn symbolic-music environment: the model writes ABC notation, an
external `abc2midi` process renders it to MIDI, and the released MiMo scorer
measures musical features against a shipped reference distribution.

This is a **graded terminal reward**, not binary correctness, not per-token
feedback, and not a listening test. It does **not** check whether the composition
follows the prompt's requested key, mood, instrumentation, or length.

## Components

- Task rows: `responses_create_params.input` with a composition request.
  Optional `verifier_metadata` is preserved for provenance, not scored.
- Agent: the existing `simple_agent`, with one generation and no tools.
- Verifier: this resources server's `/verify` endpoint; no session state.
- Model: any compatible Gym model server. Thinking models are supported:
  reasoning items and inline `<think>` / `<thinking>` blocks are not scored.

All implementation and runnable pairing configuration live in this resources
server. No changes to the shared agent, model server, or trainer are required.

## Reward

The scorer is vendored from [XiaomiMiMo/verl's music recipe](https://github.com/XiaomiMiMo/verl/tree/a2ad9f6160b03ff2d47e59832bfb6b289f37c917/recipes/design/music).
Its feature extraction, baseline, gates, and scoring formula are unchanged:

1. Extract the first `abc` / `ABC` / unlabelled fenced block containing `X:`;
   otherwise start at the last numbered `X:` header in the final answer.
2. Render ABC to MIDI in a temporary directory.
3. Reject parser errors, ten or more bar-length warnings, internal blank lines,
   or conflicting program changes on a MIDI channel. Missing/empty music scores 0.
4. For accepted music, combine 18 features in six groups (85%) with reference
   pitch-class, interval, and duration histogram agreement (15%). Divide the
   upstream 0–100 total by 100 to get a reward in `[0, 1]`.

The reference is a statistical proxy for musical structure. A positive reward
does not establish prompt compliance or human musical preference. Reward
profiling should report average reward, not label it binary task accuracy.

Bad model output receives a valid zero. A missing renderer, scorer-worker failure,
unexpected analysis exception, or the scorer wall-time limit returns `mask_sample: true`
with a `failure_kind` and `failure_reason`. Cancellation kills the scorer's
process group, including its renderer child. Up to `num_processes` scorers run
at once (default 4); each has `score_timeout_seconds` (default 30).

The scorer executes an external native parser, not arbitrary generated Python
or shell code. Subprocess isolation and timeouts are not a security sandbox:
deploy the server in an appropriately restricted container when processing
untrusted outputs.

## Install and run

From the Gym checkout, install Gym's development dependencies first.
On server startup, an existing `abc2midi` on `PATH` is used. Otherwise Linux
downloads and builds pinned abcMIDI source into the gitignored `.abcmidi/`
directory (requires `make`, a C compiler available as `cc`, and network access).
On macOS the fallback is `brew install abcmidi`. An offline deployment should
preinstall the renderer on `PATH`. Renderer versions can affect scores; record
`abc2midi -ver` with results.

abcMIDI is an external GPL-2.0-or-later executable. No abcMIDI source or binaries
are checked into Gym. The vendored Python scorer and baseline are Apache-2.0;
see [ATTRIBUTIONS.md](../../ATTRIBUTIONS.md).

Start a compatible model endpoint, then run:

```bash
gym env start \
  --resources-server mimo_music \
  --model-type vllm_model \
  --model Qwen/Qwen2.5-3B-Instruct \
  --model-url http://127.0.0.1:8000/v1 \
  --model-api-key EMPTY
```

In another terminal:

```bash
gym eval run --no-serve \
  --agent mimo_music_simple_agent \
  --input resources_servers/mimo_music/data/example.jsonl \
  --output results/mimo_music_rollouts.jsonl \
  --num-repeats 1 --concurrency 4 --max-output-tokens 2048 --temperature 0
```

## Data and validation

`data/example.jsonl` contains five **synthetic smoke tasks**, not a translated
or filtered copy of the released dataset. The full public music subset is in
[XiaomiMiMo/MiMo-V2.6-RL-oss](https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss).
Use external data preparation to put task messages under
`responses_create_params.input`; no reference answer is required by this scorer.
Do not commit training datasets or provider credentials.

```bash
gym dataset collate \
  --config resources_servers/mimo_music/configs/mimo_music.yaml \
  --output-dir /tmp/mimo_music_examples --mode example_validation
gym env test --resources-server mimo_music +should_validate_data=true
```

Tests cover real rendering, positive/failing compositions, native reward
agreement, repeatability, reasoning exclusion, concurrency, failures, and
process cleanup. Real-renderer tests skip only when installation is unavailable.

### Onboarding compatibility

This contribution uses the supported standalone resources-server layout and its
five-example validation contract. It deliberately does not declare a binary
reward or fabricate a tune scoring exactly 1.0 to satisfy the newer manifest
fixture's endpoint requirement. Migration to a workload manifest needs a
continuous-reward fixture contract (or a genuine full-reward composition).
`verified: false` remains set: the included small-model smoke is an integration
check, not a benchmark baseline or evidence of reward quality.

### Recorded smoke (2026-09-28)

`data/example_rollouts.jsonl` contains unedited model responses collected through
Gym's `simple_agent` and `vllm_model` with public `Qwen/Qwen2.5-3B-Instruct`
(served as `qwen-music-smoke`), temperature 0, 2,048 maximum output tokens,
and abc2midi 4.88. All five requests completed and none were masked.
One output scored 0.116; four scored zero. Mean reward was 0.0232.

Inspected failures include a nonnumeric tempo, 19 bar-length warnings, an
internal blank line, and a final `X:` section without a key header. The positive
sample is repetitive and does not satisfy all prompt constraints: its low
nonzero reward is expected under the released scorer, not a claim that it is
good music. This evidence intentionally retains failures as well as a positive
score.

A second live run on the final adapter also completed all five requests without
masked samples, but all five compositions scored zero. Model generation is not
assumed repeatable across inference batches. Reverification of the **same**
recorded responses preserves their rewards exactly, including the positive
sample; this is covered by a regression test. Both runs were integration
smokes, not quality baselines.
