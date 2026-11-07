# Checks a team review adds to the rubric

A team review applies the `pr-review` rubric (`.agents/skills/pr-review/SKILL.md`) plus the checks
below. They come from team reviews and are applied when a developer runs this skill. They do not
change the rubric or anything that runs in CI.

Grade findings from these checks with the rubric's BLOCKER, RISK, and NOTE.

## Compatibility and deprecation

Downstream consumers, such as NeMo RL and other training and evaluation integrations, import Gym and
call its servers. A public surface cannot disappear or change meaning in one step. The public surface
includes:

- the names each `nemo_gym` package exports from its `__init__.py`;
- server base classes, endpoints, request and response models, and wire records;
- CLI commands and flags, config keys, and their defaults;
- dataset row fields, environment variables, and on-disk formats.

Removing, renaming, or changing the meaning of one of these needs a deprecation path first:

- A moved module or name gets an entry in `nemo_gym/_moved_paths.py`, which keeps the old path
  working with a `DeprecationWarning`.
- A renamed config key, flag, or field keeps an alias that warns.
- A changed default or meaning gets a compatibility note in the PR description.

A break without a deprecation path is a BLOCKER for a documented public name and a RISK otherwise.
Find the consumers in Gym and in NeMo RL `main` (SKILL.md §1.5). When a consumer cannot be read, ask
the author which consumers they checked.

## Architectural assumptions

Work out which assumptions about the system the PR adds, removes, or changes, and check each one:

- **Lifetime.** What creates each new piece of state (a cache, map, queue, registry, file, process,
  or sandbox), what ends it, and whether that end is tied to the session, episode, or rollout that
  created it. Relying on restarts or low request rates to keep state small is a RISK on a
  long-running server.
- **Cleanup.** Whether the state is released on every path, including errors, cancellation,
  timeouts, retries, and a second close. Judge each path by how production delivers it (see
  "Production conditions" below), not by how the tests trigger it. Cleanup that must finish is
  protected from the cancellation that triggered it and bounded in time. Without a bound, a
  dependency that stops responding holds the cancelled caller forever. Before accepting a new way
  of doing this, look for how sibling code already does it.
- **Ownership.** Which component owns the state at each point, and how ownership moves between
  components. Examples are a handoff, a borrowed resource that the borrower must not destroy, and a
  record restored after a restart. Exactly one owner is responsible at every point, including while
  ownership is moving.
- **Counts and call paths.** How many of a thing exist per session, episode, or server, and which
  component may call which.
- **Contracts.** What each data model or wire contract guarantees.

A changed assumption is a finding when code relies on it without enforcing it, when existing data
models, contracts, or consumers depended on the old one (treat that as a compatibility change), or
when the PR description does not state it. A design preference without a concrete case it breaks is
not a finding.

## Package structure

Each `nemo_gym` package says what it holds in its `__init__.py` docstring, and the import-linter
contract in `pyproject.toml` fixes which packages may import which.

- New code belongs in the package whose docstring describes it. When a change makes a package's
  docstring untrue, the docstring changes in the same PR.
- An import that points up the layer order, or a new entry in the contract's `ignore_imports` list,
  is a finding.

## Payloads at boundaries

A payload sent to another library, another server, or a persisted output is built from what that
consumer reads, and nothing else.

- Dumping a whole model (`model_dump()`, `dict(body)`) ships fields the consumer never reads. Build
  the payload from the consumer's fields instead, for example
  `model_dump(include=set(ConsumerModel.model_fields))`.
- An optional field that defaults to `None` turns an absent key into a present one, which a consumer
  testing `"key" in row` misreads. Prefer `exclude_none=True` unless the consumer needs the null.
- Data that rides along with no reader, such as a full agent transcript echoed into every output
  row, costs size and memory on every call, and a later consumer may start reading it without anyone
  deciding it should.
- When a model gains a field, list each boundary that dumps the model and decide for each whether
  that consumer should get the field.

For a PR that changes dataset rows, schemas, or request models, also apply
[data-contracts.md](data-contracts.md).

## Terminology

A name that misstates what something does is a clarity NOTE. So is a synonym for a term defined in
`fern/versions/latest/pages/about/concepts/key-terminology.mdx`, or an existing name reused for
different behavior. Give the defined term with a link to it and a concrete replacement. Whitespace
and formatting stay out of scope; linters own them.

## HTTP header names

A header that Gym defines follows the naming of the header constants in `nemo_gym`. Today those are
lowercase, start with `x-ng-`, and use hyphenated words that say what the header carries, such as
`x-ng-session-owner`. Find the current constants with `git grep -n '_HEADER = ' nemo_gym`.

- A new Gym header with another prefix or casing, or a name that does not say what it carries, is
  a NOTE with the corrected name.
- The name is defined once, as a constant in the package that owns the header, and imported
  everywhere else. A header name repeated as a string literal is a NOTE.
- A new header whose meaning matches an existing one is a NOTE pointing to the existing one.
- Headers that another system defines, such as an SDK's request id or a harness's own session
  header, keep that system's name.
- Renaming an existing Gym header is a compatibility change (see above), because servers, harnesses,
  and documentation outside the PR may send or read the old name.

## Type hints

Annotations that the PR adds or changes follow current Python typing guidance for the Python version
in `pyproject.toml`, not forms that the `typing` documentation marks as deprecated or superseded.
Examples are `X | None` instead of `Optional[X]`, `A | B` instead of `Union[A, B]`, and `list[int]`
instead of `List[int]`. An outdated form in a new or changed annotation is a NOTE, with the
replacement written out. Annotations on lines the PR does not touch stay out of scope; converting
them belongs in a separate PR.

## Production conditions

Check the code against the conditions it meets in production, not the conditions the PR's tests or
the reviewer's own checks happen to create. List them first, then check the change and every fix
the review proposes against each one. A fix suggested in review gets the same scrutiny as the PR.

- **Where it runs.** Every environment a command, binary, path, or system call reaches: each
  sandbox provider and image, and the local provider, which runs on the developer's own machine.
  Behavior that only some of them have is a finding unless the PR shows the code reaches only those.
  For example, `mv -T`, `sed -i` without a suffix, `readlink -f`, and `grep -P` are GNU-only and fail
  on macOS or BusyBox.
- **How it is called and stopped.** How production callers invoke, retry, time out, and cancel the
  code. For example, Gym's request handling cancels through an anyio cancel scope
  (`ClientDisconnectCancellationMiddleware`), which stays cancelled, so every later await in a
  cleanup is cancelled too. A single asyncio `task.cancel()` does not behave that way, so a test
  that uses it can pass while production fails.

A failure that appears only under a production condition is not downgraded because tests or
another environment do not show it.

## Tests that would fail without the change

A test added for a bug fix or a new behavior must fail on the code before the PR. A test that only
checks a value is truthy, or that passes on the old code path, does not protect the change.

- When a fix adds a protection, such as a shield, lock, timeout, or retry, a test fails with only
  that protection removed, not only with the whole fix reverted. The protection under test must be
  the only thing stopping the bad outcome; if another step in the test also stops it, the test
  passes with the protection broken.
- A test double keeps the behavior of the real dependency that the test relies on. For example, a
  fake async method that never awaits cannot be interrupted, so a test using it cannot see
  cancellation or interleaving at that point; it yields where the real call would
  (`await asyncio.sleep(0)`).

## Timing assumptions in tests

A test must pass however slow the machine is. CI runs the core suite with `pytest -n 16` on shared
runners, and the server suites under the same load, so a test that depends on how fast real work
finishes fails at random on unrelated PRs. Flag a new or changed test whose result depends on
real elapsed time. For example:

- an upper bound on measured time around real work (I/O, a process, a socket, a server start),
  such as `assert elapsed < 0.15`; lower bounds are safe, because load only makes things slower;
- a real timeout or deadline the test expects not to fire, set close to the expected time. As a
  rule of thumb, use at least 10 times the expected time, and 10 s or more when it covers a process
  start;
- acting on a process or task a fixed delay after starting it, instead of after it signals that it
  is ready. This covers a deadline meant to fire, a signal, input, or a cancel: under load the
  target has not started yet, so the action hits the wrong state;
- a sleep used to wait for a state, followed by a positive assertion, with no event or polling
  loop; a negative check after a sleep cannot fail falsely, but it may prove nothing;
- two real timers with close values, where the test assumes which fires first;
- concurrency proved by timing, with a short sleep or with elapsed time below the serial time,
  instead of a barrier that every party must reach before any finishes. The barrier records any
  party that gave up waiting and the test asserts none did; a barrier that opens anyway passes
  when the work runs one at a time;
- a wait with no outer bound, so a timing failure hangs the suite instead of failing, or a wait
  that continues silently when its deadline passes, so the test fails later with a misleading
  error or passes without proving anything. A wait helper raises when its deadline passes;
- one timeout shared by cases meant to run out of time and cases meant to finish, which is too
  tight for one kind or too slow for the other. Give each case its own budget, or make the
  expiring case block on purpose instead of relying on it being slow;
- a retry test where every attempt races the same timeout. Hold the attempt that should time out
  until the retry has answered;
- a free port found by probing and bound later, which another xdist worker can take.

Safe and not flagged: generous outer guards (`wait_for(..., 10)`, poll deadlines of 30 s or more),
timeouts the test means to fire on work that never finishes, lower bounds, values passed only to a
mock, and fake or patched clocks. For a patched clock, check that the patch reaches the call site:
patching `time.sleep` does not affect `from time import sleep`.

Suggest fixes in this order: a fake clock or injected deadline, sequencing on events, firing the
deadline on purpose after a ready signal, asserting order or counts instead of duration, and only
then a wider bound (about 10 times, with the value written out).

A wider deadline on a real process or server needs cleanup to match: a test that fails early stops
the process in `finally`, or a failure leaks it for the whole deadline and slows other xdist
workers. When the same test exists in several components, a fix or new timing value in one copy is
checked against every copy, and the review names the copies left out.

## Bundled changes

When the diff carries independent changes with different risk, such as a refactor, a behavior
change, and a new feature, say which parts could land separately.

## Suggestion blocks

A suggestion block replaces exactly the lines its comment is anchored to. The file with the block
applied must still parse, keep every function and return the original lines had, and stay formatted.
Code for any other file or line goes in a plain code block. The audit script checks this before
posting.
