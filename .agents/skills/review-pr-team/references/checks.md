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
  timeouts, retries, and a second close.
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
consumer reads. Dumping a whole model (`model_dump()`, `dict(body)`) ships fields the consumer never
reads. An optional field that defaults to `None` also turns an absent key into a present one, which a
consumer testing `"key" in row` misreads.

## Terminology

A name that misstates what something does is a clarity NOTE. So is a synonym for a term defined in
`fern/versions/latest/pages/about/concepts/key-terminology.mdx`, or an existing name reused for
different behavior. Give the defined term with a link to it and a concrete replacement. Whitespace
and formatting stay out of scope; linters own them.

## Tests that would fail without the change

A test added for a bug fix or a new behavior must fail on the code before the PR. A test that only
checks a value is truthy, or that passes on the old code path, does not protect the change.

## Bundled changes

When the diff carries independent changes with different risk, such as a refactor, a behavior
change, and a new feature, say which parts could land separately.

## Suggestion blocks

A suggestion block replaces exactly the lines its comment is anchored to. The file with the block
applied must still parse, keep every function and return the original lines had, and stay formatted.
Code for any other file or line goes in a plain code block. The audit script checks this before
posting.
