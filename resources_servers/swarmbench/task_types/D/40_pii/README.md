# 40_pii: privacy scan with context-dependent PII rules (family D, source P/R)

## Task
A library digitization program publishes excerpts of public-domain novels. Each scanned document also holds a few
inserted passages: margin notes, loan slips, reader letters, staff and catalogue notes. Before publication the
privacy office must list every reportable personal-data item in every document under a written policy. That policy is
K3-written and covers seven item types: `name`, `phone`, `email`, `address`, `dob`, `gov_id` and `card`. Two reference
lists sit next to the documents: an officials directory (name, title, agency, office phone) and a consent register
(name, status granted/withdrawn, date, reference).

**Unit** = one document, about 1,860 words of real novel text (Project Gutenberg) with 2-8 planted passages. Answer per
document: the exact set of `{type, value}` items. The unit is correct iff the normalized set is equal. Values are
normalized as follows: digits only for phone, gov_id and card; lower case for email; dates as ISO; lower-cased
alphanumeric tokens for names and addresses.

## Traps (each document mixes several)
| kind | what a careless reader does | rule |
|---|---|---|
| official (directory) | reports the official's name and office phone | officials in their official role are not private |
| official_private | drops a directory official who appears as a private borrower (home phone) | private capacity means the name and phone are reported |
| business | reports main lines, shared mailboxes and business addresses | organisations are not individuals |
| via_business | reports the company switchboard given by a private person | report the name only |
| case / part | reports `ddd-dd-dddd` docket or part numbers as gov_id (same shape as a real id) | reference numbers are not personal data |
| isbn | (shape differs) | not reported |
| sample | reports the specimen name and id on a blank form | illustrations are not personal data |
| luhn_fail | reports a card number that fails the Luhn check | not reported, but the person's name is |
| consent_granted / consent_withdrawn | ignores the register, or treats withdrawn like granted | granted means nothing is reported for that person; withdrawn means report as normal |
| source text | reports novel characters (Elizabeth Bennet, Ahab, ...) | literary figures are never reported |

Generation guarantees per document:
- at least 2 rule traps (official, business, case/part, Luhn failure, or granted consent);
- with probability 0.97, a semantic trap (via_business, official_private or sample) that no pattern or list lookup
  can resolve;
- about 12% of documents have no private persons, and some have nothing to report at all.

All planted values are fictional and code-generated:
- names come from code lists (100 first names x 120 last names), a unique bijection per (seed, slot);
- phones use the 555 exchange;
- emails use `.example` domains.

## Why one context cannot do the hard tier
- **Reading volume:** each document is about 11k characters, so 2-3 `read_document` pages, plus 1-3 register or
  directory lookups. That is about 4-8 tool calls and ~12k tokens read per unit. The hard tier (96 documents) is ~1.1M
  characters of reading plus ~200 lookups.
- **No shortcut by search:** the planted passages are K3-written in 6 phrasings per kind, so finding them requires
  reading every paragraph. Searching for `555` finds phones, but not dob, address or name-only items.

## Tiers
`TIERS = {"easy": (8, 1800), "medium": (24, 1200), "hard": (96, 1200)}`. Prefix property: document i depends only
on (seed, i). The directory and register grow with n through fillers, which never affect gold.

## Tools (bounded to 8k chars, read-only, offline)
- `list_documents(offset, limit)`
- `read_document(doc_id, start)`: numbered paragraphs, about 7.2k chars per page
- `search_document(doc_id, query)`
- `read_directory` / `search_directory`
- `read_register` / `search_register`
- `calculator`, for the Luhn arithmetic

No tool classifies PII or applies the policy.

## Answer and scoring
The answer ends with one JSON object `{"DOC-0001": [{"type": "...", "value": "..."}], "DOC-0002": [], ...}`, read
with `common.last_json`. Type synonyms are accepted (ssn, date_of_birth, credit_card, ...). Score = mean per-document
exact-set correctness. `score_partial` is the mean per-document item F1.

## Checks (offline, `checks.json`, n = 8 / 40 / 96, seeds 0, 1)
- **1 oracle:** 1.000 on all 6 instances. The oracle is tool-only: it parses the K3 templates with `common.tre` and
  looks names up through `search_register`.
- **2 plausible-wrong (max over instances):**
  - `regex` 0.00: pattern scan for phone/email/id/card/address/date, plus capitalized pairs in those paragraphs;
  - `smart_regex` 0.02: the same, plus a name gazetteer, role-mailbox, Luhn, docket/part/ISBN keyword and company
    filters, and exact-name greps of the directory and register;
  - `careless_no_lookups` 0.00.
- **Partial readers (diagnostic):**
  - skip_register 0.79
  - skip_luhn 1.00 at n=8 (0.63-0.88 at larger n)
  - report_officials 0.17
  - report_business 0.54
  - numbers_as_ids 0.50
- **3 no-op:** 0 for all format-only answers.
- **7:** 0 tool errors on bad arguments. Tools are concurrency-safe (64-thread test).
- **8:** deterministic. Instance sha256 values are in `checks.json`.

## Sources and license
`data/SOURCES.json`: six Project Gutenberg books, public domain in the USA. The PG header and footer (the PG License
and trademark text) are stripped. Raw sha256 values are listed in SOURCES.json. The snapshot is `data/chunks.json`:
239 chunks, sha256 `e281d8ed1e78ccd16a1206e0eb3e5b060445d4942caa16bad72bd56f82257d75`, built with `python3 -I
build_snapshot.py`.

## Provenance
- All task-facing prose is in `text/k3.json`: intro, policy rules, tool descriptions, unit prompt, directory and
  register headers, and 6 (4) phrasings per passage kind.
- It was written by self-hosted Kimi K3 (`policy_model`, moonshotai/Kimi-K3 rev 9f62e4e9, self-hosted) via
  `k3gen.py`, in one attempt of 334 s.
- The exact prompt is `text/request.json` ("user"). The raw reply, usage and model are in `text/k3_raw.json`.
- A second, short K3 call (`text/request_extra.json` -> `text/k3_extra.json`, 35 s) wrote `rule_private_capacity`
  and the calculator description, after the first unit check showed that K3's `rule_officials` had left out the
  private-capacity clause the brief asked for.
- Code keeps only the `p_id` phrasings that call the number a government id or social security number (3 of 6).
  The other three said only "identification number" on a library card, which reads as a member number.
- `text/dev_stub.json` is Claude-written test scaffolding only and never ships.
- Claude wrote code, name/number generators and tests only.
- The canary is the first line of every prompt and is also in `world/CANARY.txt`.

## Unit check (check 5) and audit
`unitcheck.py 40_pii --n 40 --seeds 0,1 --k 20` with self-hosted K3. Transcripts are in
`D/_units/40_pii-k3*/units.jsonl`.

**v1: 15/20.** All five failures were hand-audited:
- **Three were rule gaps (verifier-side, now fixed):** in the official_private units, K3 reported the home phone
  but not the name, because `rule_officials` did not state the private-capacity case.
- **One was a template ambiguity (fixed):** a `p_id` phrasing said only "identification", and K3 read it as a
  member number.
- **Two were model errors:** K3 got the Luhn arithmetic wrong on valid cards.

**v2 (rule added, ambiguous `p_id` phrasings dropped, calculator tool): 18/20.**
- **Both failures are model errors:**
  - DOC-0033 left out a private person's name after correctly reporting her card;
  - DOC-0032 got Luhn wrong on a valid card (it doubled the wrong digits, then summed them with the calculator).
- **Verifier FP/FN:** 0 of 20 in v2. Every graded item was checked against the passage text.
- **Unit cost:** K3 single-unit loop mean 112 s (median 99 s), 7.8 tool calls, ~18k prompt tokens per unit.

**Tier sizing:**
- Assuming OpenCode solo runs at ~40-55 s per unit, the hard tier (96 units, ~4.3k-5.3k s of work) is ~3.5-4.5x the
  1200 s budget.
- The medium tier (24 units, ~1.1k-1.3k s) sits at about the budget.
