# 53 configdrift: config drift across a fleet (family F)

## Task
`n` hosts, each with one of 8 roles and 3-4 config files. Find every baseline key whose effective value deviates and
is not covered by a valid exception, and give the reason: UNAPPROVED, EXPIRED (an applicable exception for that
value expired before the audit date 2026-06-30), MISMATCH (a valid applicable exception names another value) or
MISSING. Unit = one host. Score: F1 over (host, key, reason).

## Simulation (procedural; exact gold needs it)
Role baselines (12 of 18 keys: ssh, kernel, app, ntp, audit, backup settings). Files: `base.conf`,
`conf.d/50-<role>.conf`, `conf.d/90-host-<host>.conf` and sometimes `conf.d/70-experiment.conf.disabled`;
effective config = all `*.conf` in lexical path order, last line wins, `#` comments ignored. About 10% of keys drift
and 3% go missing; decoys: drift in `base.conf` restored by a later file (25%), drifted values only in comments or
the disabled file (20%), unneeded exceptions, exceptions for other hosts, role-wide exceptions (`role:<role>`) that
also reach role peers. Exceptions carry K3-written justifications; files carry K3-written header comments. Gold is
`audit()` applied to the final files and exception registry (the rules are implemented once and unit-tested).

## Tools
`list_hosts`, `list_files(host)`, `read_file(host, path)`, `get_baseline(role)`, `get_exceptions(host)`.

## Tiers (n hosts / budget s)
easy 64 / 1800, medium 256 / 1200, hard 1024 / 1200.

## Files
`type.py`, `solutions.py` (oracle; shifted hosts, grep-any-other-value, base.conf-only, ignore-exceptions,
read-comments-and-disabled, reason-blind), `tests/test_type.py`, `checks.json`.

## Provenance
All task-facing prose (prompt, rules, tool descriptions, phrase banks, unit prompts) was written by Kimi K3 (`nvidia/moonshotai/kimi-k3` on the NVIDIA inference API, served as `kimi-k3-high`) through `../../k3gen.py`, with the system prompt below, and frozen in `text/`. Claude wrote only code, schemas, verifiers and tests. Code fills ids and numbers into K3's `{placeholders}`.

### `text/k3.json` (sha256 a7ab8b1194323fb8..., created 2026-10-14 06:56:35 UTC, served model nvidia/moonshotai/kimi-k3, 1 attempt(s))

System prompt:

```
You write task text for an offline evaluation of software agents. Follow the facts you are given exactly; do not add rules, numbers or names that are not in the brief. Reply with exactly one JSON object and nothing else.
```

User prompt:

```
Write the task-facing text for a configuration-drift audit of a server fleet. Each host has a role (web, api, db, cache, worker, gateway, batch, search); each role has a baseline of required key = value settings. A host's configuration is spread over several files. Approved exceptions allow specific deviations until an expiry date. The audit date is {date}. Return one JSON object with these keys:

"intro": the opening paragraph, with the placeholders {n} (number of hosts) and {date} (the audit date). The job: for every host, find every baseline key whose effective value deviates from the role baseline and is not covered by a valid exception, and say why.
"rules": a list of bullet strings stating these exact rules, one per bullet, in this order:
 1. A host's effective configuration is built from its files whose names end in .conf, read in lexical (sorted) order of their full paths; within and across files a later line for the same key overrides an earlier one. Files with any other ending (for example .conf.disabled) are ignored.
 2. Lines starting with # are comments and are ignored; blank lines are ignored. A setting line has the form key = value; leading and trailing spaces around the value do not count.
 3. Only keys listed in the host's role baseline are audited. A baseline key with no setting line in any .conf file is MISSING.
 4. A baseline key whose effective value differs from the baseline value is a deviation. An exception applies to a host if its target is that host's name or role:<the host's role>. An exception is valid if its expiry date is on or after the audit date.
 5. A deviation is covered (not reported) if a valid exception for that key has exactly the effective value. Otherwise report it with one reason: EXPIRED if an applicable exception for that key has exactly the effective value but has expired; MISMATCH if there is no such exception but a valid applicable exception for that key names a different value; UNAPPROVED otherwise. Exceptions for other hosts or other roles never apply.
"file_headers": 12 different one-line comments that could head a fleet configuration file (managed-by notes, ownership, change-control reminders). No placeholders, no key = value text.
"justifications": 20 different short justifications an engineer might give for a configuration exception (performance testing, vendor requirement, incident mitigation and so on). No placeholders, no dates.
"tools": an object with one-sentence descriptions for list_hosts (host name | role, paged, at most 200 per call), list_files (the configuration file paths of one host), read_file (the text of one file of one host), get_baseline (the baseline key = value list of one role), get_exceptions (all exceptions whose target is the given host or its role, with id, target, key = value, expiry date and justification).
"answer": a short paragraph: the final answer is plain lines, one finding per line, in the literal form <host> | <key> | <reason>, where reason is one of UNAPPROVED, EXPIRED, MISMATCH, MISSING; for example web-use1-0007 | app.log_level | UNAPPROVED; hosts without findings are left out; lines in any other form are ignored.
"score": one sentence: the score is the F1 score of the finding lines against the true findings.
"unit_prompt": instructions for a single worker who audits exactly one host, {host}, with the same tools and the rules placed at {rules}, and ends with its finding lines followed by a last line UNIT DONE {host2} (only UNIT DONE {host2} if the host has no findings). ({host2} is the same name as {host}.)
```

