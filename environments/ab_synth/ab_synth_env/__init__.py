# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic workflow tasks on the AutomationBench app simulator: single-day, multi-day and multi-turn rows.

Keys in each row's `info` turn on a `wait` tool with dated events and day assertions (`period_end`, `events`), a
simulated requester (`user_card`) and requester-owned apps (`requester_services`). The reward is the guardrail-gated
share of objectives. Rows that cannot be graded are refused at load and masked in a rollout. See README.md.
The dataset owns all text the policy reads: the `wait` tool's text comes from each row.
"""

import asyncio
import collections
import copy
import json
import logging
import math
import os
import re
import typing
from datetime import datetime, timezone
from pathlib import Path

from automationbench.rubric.registry import AssertionRegistry
from automationbench.runner import strip_none_values
from automationbench.schema.world import WorldState
from automationbench.tools.api.fetch import api_fetch as upstream_api_fetch


logger = logging.getLogger(__name__)

WRITE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
DAY_ASSERTIONS = {"acted_on_day", "not_acted"}
OUT_OF_SCOPE = re.compile(r'"code": 401, "message": "No (\w+) account is connected')
THINK = re.compile(r"<think>.*?(</think>|\Z)", re.S)  # an unclosed block runs to the end of the message
# Fields the simulator stamps from the wall clock on every create or update, whatever the agent sends (upstream
# routes and schema defaults call datetime.now()). A world assertion on one cannot match a fixed value from a seed
# dated in another year, so such a row is refused. Date fields the agent writes from the task (due_date,
# start.dateTime, close_date) are graded like any other field.
STAMPED_FIELDS = {
    "created_at", "created_date", "created_time", "date_created",
    "updated_at", "updated_date", "date_updated", "last_updated_time",
    "modified_at", "date_modified", "last_modified_date", "last_modified_date_time",
    "edited_at", "generated_at", "status_changed_date_time", "canceled_at", "voided_date_time", "no_show_marked_at",
    "sent_at", "sent_date_time", "timestamp", "current_time",
}  # fmt: skip
FIELD_KEYS = {"field", "property"}  # not a sheet `column`: cells hold what the agent writes, nothing is stamped
VALUE_MAPS = {"cells", "cell_contains", "fields"}  # keyed by column names (a Google Forms "Timestamp"), not fields
REQUESTER_RESULT_CHARS = 30000  # one requester message or tool result, cut: a whole sheet runs to 70k+ characters
# An A1 reference to sheet rows: "Tab!B7", "'Q4 2025'!A7:C7", "B7", a bare "7", or rows "2:5".
ROW_REF = re.compile(r"(?:'?(?P<tab>.+?)'?!)?[A-Za-z]{0,3}(?P<first>\d+)(?::[A-Za-z]{0,3}(?P<last>\d+))?")
# A write to a record of these collections also acts on its parent: (app, collection) -> [(parent collection, field)].
PARENTS = {
    ("google_calendar", "events"): [("calendars", "calendarid")],
    ("facebook_pages", "posts"): [("pages", "page_id")],
    ("slack", "messages"): [("channels", "channel_id")],
    ("google_sheets", "worksheets"): [("spreadsheets", "spreadsheet_id")],
}
# Writes an app keeps only in its action log, by (app, action): the collections and params naming the records acted on.
ACTION_TARGETS = {
    ("jira", "add_comment"): [("issues", "issueKey")],
    ("google_drive", "create_file"): [("folders", "folder")],
    ("google_drive", "copy_file"): [("files", "source_file"), ("folders", "folder")],
    ("google_drive", "move_file"): [("files", "file"), ("folders", "addParents"), ("folders", "removeParents")],
    ("google_drive", "delete_file"): [("files", "file")],
    ("pipefy", "move_card"): [("cards", "card_id")],
    ("pipefy", "update_card_field"): [("cards", "card_id")],
    ("airtable", "createRecord"): [("bases", "applicationId")],
    ("airtable", "updateRecord"): [("bases", "applicationId")],
    ("airtable", "add_comment"): [("bases", "applicationId")],
}
# The key field of collections whose records have neither `id` nor `sid`. Records without one (a Zoom participant, a
# LinkedIn mutual connection, a Slack message without an id) cannot be named by a day assertion.
PRIMARY_KEYS = {
    ("xero", c): f
    for c, f in [
        ("contacts", "contact_id"),
        ("invoices", "invoice_id"),
        ("bills", "invoice_id"),
        ("payments", "payment_id"),
        ("bank_transactions", "bank_transaction_id"),
        ("bank_transfers", "bank_transfer_id"),
        ("credit_notes", "credit_note_id"),
        ("items", "item_id"),
        ("employees", "employee_id"),
        ("accounts", "account_id"),
        ("purchase_orders", "purchase_order_id"),
        ("quotes", "quote_id"),
    ]
}
# A second field a record is named by: a Jira issue by its key ("OPS-1") as well as its id.
ALIASES = {("jira", "issues"): "key"}

# Instructions to the requester model only. The policy sees the requester's replies, never this text.
REQUESTER_SYSTEM = """You play the person who sent a work request to an assistant that operates business apps for you.
Your request, word for word:
{request}

Your private notes (facts you know but did not write in the request):
{card}

Reply to the assistant's latest message as that person, in one to three plain sentences.
- If it asks something your notes answer, give that answer exactly as your notes state it.
- If it asks something your notes do not cover, say you do not know and that it should use its judgment. Then add
  every one of your notes that the assistant has not mentioned yet, since the work needs them.
- Never invent facts beyond the request and the notes.
- If the assistant says the work is done, or asks nothing and needs nothing from you, reply with exactly: DONE"""
ACTING_REQUESTER_SYSTEM = """You are a person who sent a work request to an AI assistant that operates business apps for
you. You are not the assistant. Every message you receive comes from that assistant, and you answer it as the person.

Your request, word for word:
{request}

Your private notes (facts you know but did not write in the request):
{card}

These apps are yours, and only you can use them: {services}. The assistant cannot reach them. You use them yourself
through the api_search tool (find the endpoint) and the api_fetch tool (call it). What they hold (ids and names):
{records}
- When the assistant's latest message asks you to do or look up something in your apps, do it now with the tools,
  using the exact values the assistant gave (names, dates, amounts, recipients, wording). Do only what it asked. If
  it left out a value you need, do not guess: ask for it.
- Then reply as the person, in one to three plain sentences: what you did, or what you saw.
- If it asks you to do something in any other app, tell it to do that part itself.
- If it asks something your notes answer, give that answer exactly as your notes state it. If your notes do not cover
  it, say you do not know and that it should use its judgment.
- Never invent facts beyond the request, the notes and what your apps show.
- If the assistant says the work is done, or asks nothing and needs nothing from you, reply with exactly: DONE"""


def _time(value, zone=timezone.utc) -> datetime:
    """Naive UTC time. A time with an offset is converted to UTC, a naive one is read in `zone` (UTC by default)."""
    t = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return t.replace(tzinfo=t.tzinfo or zone).astimezone(timezone.utc).replace(tzinfo=None)


def _as_json_text(value) -> str:
    # The api_fetch schema declares params and body as JSON objects, upstream's fetch takes JSON strings. Non-ASCII
    # text stays as is, so a body_contains such as "Müller" matches an object body.
    return value if isinstance(value, str) else json.dumps(value or {}, ensure_ascii=False)


def _text(content) -> str:
    """Visible text of a message: list content joined, reasoning dropped."""
    if isinstance(content, list):
        # Parts are dicts or, once verifiers normalizes them, objects: read `text` from either.
        content = " ".join(_get(p, "text") or "" for p in content)
    return THINK.sub("", content or "").split("</think>")[-1].strip()


def _get(msg, key):
    return msg.get(key) if isinstance(msg, dict) else getattr(msg, key, None)


def _key(app: str, collection: str, record, sheets=None):
    """A record's key: `id`, else `sid`, else the collection's primary key. A worksheet is keyed (spreadsheet, id), a
    sheet row (spreadsheet, worksheet, row_id), with titles mapped to ids by the simulator's own resolvers (`sheets`)."""
    if (app, collection) == ("google_sheets", "worksheets"):
        return (_get(record, "spreadsheet_id"), str(_get(record, "id")))
    if (app, collection) == ("google_sheets", "rows"):
        sheet, tab = _get(record, "spreadsheet_id"), _get(record, "worksheet_id")
        if sheets is not None:
            sheet = sheets._resolve_spreadsheet_id(str(sheet))
            tab = sheets._resolve_worksheet_id(sheet, str(tab))
        return (sheet, tab, _get(record, "row_id"))
    for field in ("id", "sid", PRIMARY_KEYS.get((app, collection))):
        if field and (value := _get(record, field)) is not None:
            return str(value)
    return None


def _records(world: dict, sheets=None) -> dict:
    """Every record of a world dump by (app, collection, key). A list collection holds records, a dict one maps keys to
    records. Records without a key of their own share the key None."""
    found = {}
    for app, collections in world.items():
        if not isinstance(collections, dict):
            continue
        for name, items in collections.items():
            if isinstance(items, list):
                pairs = [(_key(app, name, r, sheets), r) for r in items if isinstance(r, dict)]
            elif isinstance(items, dict):
                pairs = [(str(k), r) for k, r in items.items() if isinstance(r, dict)]
            else:
                continue
            for key, record in pairs:
                found.setdefault((app, name, key), []).append(record)
    return found


def _tag(app: str, collection: str, key) -> str:
    """A record's name in the write log and in a resolved assertion: JSON of (app, collection, key)."""
    return json.dumps([app, collection, list(key) if isinstance(key, tuple) else key])


def _actions(world: dict) -> dict:
    """Every logged action of a world dump by (app, action, id): its params."""
    return {
        (app, action, a.get("id")): a.get("params") or {}
        for app, collections in world.items()
        if isinstance(collections, dict) and isinstance(collections.get("actions"), dict)
        for action, entries in collections["actions"].items()
        for a in entries
        if isinstance(a, dict)
    }


def _names(world: dict, events: list[dict], sheets=None) -> dict:
    """Every name a record answers to (its key, its alias, a sheet row's row_id and id) -> the tags of the records with
    it, over the seed and the records its events add."""
    names = {}

    def add(app, collection, key, record):
        tag = _tag(app, collection, key)
        names.setdefault(str(key[-1] if isinstance(key, tuple) else key), set()).add(tag)
        alias = "id" if isinstance(key, tuple) else ALIASES.get((app, collection))  # a sheet row also by its own id
        if alias and record.get(alias) is not None:
            names.setdefault(str(record[alias]), set()).add(tag)

    for (app, collection, key), records in _records(world, sheets).items():
        for record in records if key is not None else ():
            add(app, collection, key, record)
    for e in events:
        patch = e["patch"]
        if "add" in patch and (key := _key(patch["app"], patch["collection"], patch["add"], sheets)):
            add(patch["app"], patch["collection"], key, patch["add"])
    return names


def _sheet_rows(world: dict, ref, sheets) -> dict:
    """The rows an A1 reference names, by worksheet: by row_id, or by position in a tab of string row ids (the first row
    is 2), as upstream addresses them. A single row past a named tab's end is the row an append there would add."""
    tabs = world.get("google_sheets") or {}
    first, last = int(ref["first"]), int(ref["last"] or ref["first"])
    found = {}
    for w in tabs.get("worksheets", []):
        if ref["tab"] is not None and sheets._resolve_worksheet_id(w["spreadsheet_id"], ref["tab"]) != w["id"]:
            continue
        keys = [_key("google_sheets", "rows", r, sheets) for r in tabs.get("rows", [])]
        rows = [row_id for sheet, tab, row_id in keys if (sheet, tab) == (w["spreadsheet_id"], w["id"])]
        strings = any(not isinstance(r, int) for r in rows)
        hits = []
        for n in range(first, last + 1):
            if n in rows:
                hits.append(n)
            elif strings and 2 <= n < len(rows) + 2:
                hits.append(rows[n - 2])
        if not hits and ref["tab"] and first == last and not strings:
            hits = [first]
        if hits:
            found[(w["spreadsheet_id"], w["id"])] = {
                _tag("google_sheets", "rows", (w["spreadsheet_id"], w["id"], n)) for n in hits
            }
    return found


def row_problems(info: dict, *, requester: bool) -> list[str]:
    """What makes a row unusable before any rollout, the checks load_environment and each rollout's setup_state share:
    world assertions on wall-clock fields, a multi-day row without its wait texts or with a time that is not ISO 8601, a
    multi-turn row without a requester endpoint, a requester service that is no app, a day assertion without a record."""
    problems = []
    for a in info.get("assertions", []):
        if fields := stamped_fields(a):
            problems.append(
                f"assertion {a.get('type')!r} reads fields the simulator stamps from the wall clock {fields}, which "
                "cannot match a fixed value: grade when the agent acted with acted_on_day or not_acted"
            )
        if a.get("type") in DAY_ASSERTIONS and not isinstance(a.get("record"), (str, int)):
            problems.append(f"{a['type']} has no record (got {a.get('record')!r})")
        if a.get("type") == "acted_on_day" and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(a.get("day"))):
            problems.append(f"acted_on_day day {a.get('day')!r} is not YYYY-MM-DD")
    events = info.get("events") or []
    if not isinstance(events, list) or not all(
        isinstance(e, dict) and isinstance(e.get("patch"), dict) for e in events
    ):
        problems.append("events must be a list of {at, patch} objects")
    elif bad := [
        e
        for e in events
        if not {"app", "collection"} <= e["patch"].keys() or not ("add" in e["patch"] or "id" in e["patch"])
    ]:
        problems.append(f"an event patch needs app, collection and add or id: {str(bad[0])[:120]}")
    if info.get("period_end"):
        if missing := [k for k in ("wait_tool_description", "wait_until_description") if not info.get(k)]:
            problems.append(f"a multi-day row needs {missing}")
        for t in [info["period_end"], *(e.get("at") for e in events if isinstance(e, dict))]:
            try:
                _time(t)
            except (TypeError, ValueError):
                problems.append(f"{t!r} is not an ISO 8601 time")
        start = ((info.get("initial_state") or {}).get("meta") or {}).get("current_time")
        if not start:  # upstream would start the clock at the wall clock, after period_end
            problems.append("a multi-day row needs initial_state.meta.current_time")
        elif not problems:
            end = _time(info["period_end"])
            if _time(start) >= end:
                problems.append(f"period_end {info['period_end']!r} is not after the seed's current_time {start!r}")
            if late := [e["at"] for e in events if _time(e["at"]) > end]:
                problems.append(f"events after period_end never fire: {late[:3]}")
    elif events:
        problems.append("events need a period_end: without one there is no wait tool and no event fires")
    if (info.get("user_card") or info.get("requester_services")) and not requester:
        problems.append(
            "multi-turn rows (user_card or requester_services) need requester_base_url and requester_model"
        )
    if unknown := [s for s in info.get("requester_services") or [] if s not in WorldState.model_fields]:
        problems.append(f"requester_services names no app: {unknown}")
    return problems


def resolve_records(info: dict) -> list[str]:
    """Resolve each day assertion's `record` (a key or alias, else an A1 reference) into its `targets`. Returns a problem
    per record that names nothing or several records, and one for the first dated event that cannot apply."""
    seed = WorldState(**strip_none_values(info["initial_state"]))
    world, sheets = seed.model_dump(mode="json"), seed.google_sheets
    names, problems = _names(world, info.get("events") or [], sheets), []
    # Run the dated events in time order on a copy of the seed, as a rollout that waits to the end would: an event that
    # cannot apply (no such record, a field or an added record the schema rejects) would mask only the rollouts that
    # wait past it.
    scratch = copy.deepcopy(seed)
    for e in sorted(info.get("events") or [], key=lambda e: _time(e["at"])):
        try:
            apply_patch(scratch, e["patch"])
        except Exception as err:  # noqa: BLE001  any failure refuses the row, with the error
            problems.append(f"unresolvable: event at {e['at']} does not apply: {type(err).__name__}: {err}"[:300])
            break
    else:
        for a in info.get("assertions", []) if info.get("events") else []:
            if a.get("type") in DAY_ASSERTIONS or "excluded" in a or a.get("scored") is False:
                continue
            if AssertionRegistry.check(seed, a) != AssertionRegistry.check(scratch, a):
                problems.append(
                    f"event-dependent: a dated event changes {a['type']}: set excluded: true to skip it, "
                    "or excluded: false if the agent must make it hold at period_end"
                )
    for a in info.get("assertions", []):
        if a.get("type") not in DAY_ASSERTIONS:
            continue
        name = str(a["record"])
        found, tabs = names.get(name, set()), {}
        if not found and (ref := ROW_REF.fullmatch(name)):
            # Read rows from the world after the events, so a row an event adds resolves too.
            tabs = _sheet_rows(scratch.model_dump(mode="json"), ref, scratch.google_sheets)
            found = set().union(*tabs.values()) if len(tabs) == 1 else set()
        if len(found) == 1 or (len(tabs) == 1 and found):
            a["targets"] = sorted(found)
        else:
            what = "ambiguous" if len(found) > 1 or len(tabs) > 1 else "unresolvable"
            problems.append(f"{what}: {a['type']} record {name!r}")
    return problems


class MissingRecord(ValueError):
    """A dated event names a record the world does not hold."""


def apply_patch(world: WorldState, patch: dict) -> None:
    """Apply one dated event: set fields on a record (`app`, `collection`, `id`, `set`) or add one (`add`).

    Some collections hold plain dicts (jira.issues, google_drive.files, ...), the others typed records."""
    app = getattr(world, patch["app"])
    collection = getattr(app, patch["collection"])
    if "add" in patch:
        # Typed from the collection's declared schema, so a record added to an empty collection matches loaded ones.
        (model,) = typing.get_args(type(app).model_fields[patch["collection"]].annotation)
        add = copy.deepcopy(patch["add"])
        collection.append(model.model_validate(add) if hasattr(model, "model_validate") else add)
        return
    pid = str(patch["id"])
    record = next(
        (r for r in collection if pid in (str(_get(r, "id")), _key(patch["app"], patch["collection"], r))), None
    )
    if record is None:
        raise MissingRecord(f"event patch targets no record {patch['id']!r} in {patch['app']}.{patch['collection']}")
    for key, value in patch["set"].items():
        if isinstance(record, dict):
            record[key] = value
        else:
            setattr(record, key, value)


class Clock:
    """Per-rollout simulated time: the world, its pending events, and the agent's writes keyed by simulated time."""

    def __init__(self, world: WorldState, events: list[dict], end: str) -> None:
        self.world = world
        self.end = _time(end)
        self.events = sorted(events, key=lambda e: _time(e["at"]))
        self.log: list[dict] = []
        # acted_on_day reads the day in the seed's own time zone (UTC for a naive seed), while the clock runs in UTC.
        self.zone = world.meta.current_time.tzinfo or timezone.utc

    def now(self) -> datetime:
        return _time(self.world.meta.current_time.isoformat())

    def wait(self, until: str) -> str:
        """Move to `until` (capped at the end of the task's period), applying every event due by then."""
        target = min(_time(until, self.zone), self.end)
        if target <= self.now():
            return json.dumps({"now": self._local(self.now()), "advanced": False})
        while self.events and _time(self.events[0]["at"]) <= target:
            patch = self.events.pop(0)["patch"]
            try:
                apply_patch(self.world, patch)
            except MissingRecord:
                # Every event applies to the seed (resolve_records runs them at load and at setup), so a record it
                # cannot find now was deleted in this rollout, by the agent or the requester it asked: skip the event.
                logger.info(
                    "ab_synth: event skipped, %s.%s %s was deleted", patch["app"], patch["collection"], patch["id"]
                )
        # Keep the seed's awareness: upstream date queries read a naive time as host-local time.
        aware = self.world.meta.current_time.tzinfo is not None
        self.world.meta.current_time = target.replace(tzinfo=timezone.utc) if aware else target
        return json.dumps({"now": self._local(target), "advanced": True})

    def _local(self, t: datetime) -> str:
        """A clock time as the agent is told it: in the seed's zone with its offset, or naive UTC for a naive seed."""
        return (
            t.isoformat()
            if self.zone is timezone.utc
            else t.replace(tzinfo=timezone.utc).astimezone(self.zone).isoformat()
        )

    def fetch(self, method: str, url: str, params=None, body=None) -> str:
        """Run one API call. A write that changes the world is logged with the records it added, changed or deleted,
        their parents, and the records its logged actions name. Failed writes (whatever shape their error reply has) and
        read-only POSTs such as searches change nothing and are not acts."""
        params, body = _as_json_text(params), _as_json_text(body)
        if method.upper() not in WRITE_METHODS:
            return upstream_api_fetch(self.world, method, url, params, body)
        # The records before the write, with sheet titles resolved against the world as it was then.
        before = self.world.model_dump(mode="json")
        old = _records(before, self.world.google_sheets)
        try:
            return upstream_api_fetch(self.world, method, url, params, body)
        finally:  # a write that changed the world and then raised is still an act
            after = self.world.model_dump(mode="json")
            if after != before:
                self._log_write(method.upper(), url, body, before, old, after)

    def _log_write(self, method: str, url: str, body: str, before: dict, old: dict, after: dict) -> None:
        now = self.now()
        day = now.replace(tzinfo=timezone.utc).astimezone(self.zone).date().isoformat()
        new = _records(after, self.world.google_sheets)
        changed = [k for k in old.keys() | new.keys() if old.get(k) != new.get(k)]
        targets = {_tag(*k) for k in changed if k[2] is not None}
        for k in changed:
            # Only the records that were added, changed or deleted credit their parent, not their unchanged peers.
            o, n = old.get(k, []), new.get(k, [])
            moved = [r for r in o if r not in n] + [r for r in n if r not in o]
            for parent, field in PARENTS.get(k[:2], ()):
                targets |= {_tag(k[0], parent, str(r[field])) for r in moved if r.get(field)}
            if k[:2] == ("google_sheets", "rows"):  # a row's spreadsheet and tab, by its key's resolved ids
                targets |= {
                    _tag("google_sheets", "spreadsheets", k[2][0]),
                    _tag("google_sheets", "worksheets", k[2][:2]),
                }
        done, names = _actions(before), None
        for (app, action, id_), params in _actions(after).items():
            if (app, action, id_) in done:
                continue
            for collection, param in ACTION_TARGETS.get((app, action), []):
                names = names or _names(after, [], self.world.google_sheets)
                values = params.get(param)
                for value in values if isinstance(values, list) else [values]:
                    targets |= {t for t in names.get(str(value), ()) if json.loads(t)[:2] == [app, collection]}
        self.log.append(
            {"at": now.isoformat(), "day": day, "method": method, "url": url, "body": body, "targets": sorted(targets)}
        )


def _body_strings(body: str) -> list[str]:
    """The string values in a JSON body, unescaped (a quote, newline or backslash reads as itself)."""
    try:
        node = json.loads(body)
    except ValueError:
        return []
    found, stack = [], [node]
    while stack:
        node = stack.pop()
        if isinstance(node, str):
            found.append(node)
        elif isinstance(node, dict):
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return found


def day_check(log: list[dict], assertion: dict) -> bool:
    targets, text = set(assertion["targets"]), assertion.get("body_contains")
    hits = [
        e
        for e in log
        if targets & set(e["targets"])
        and (not text or text in e["body"] or any(text in s for s in _body_strings(e["body"])))
    ]
    if assertion["type"] == "not_acted":
        return not hits
    first = min(hits, key=lambda e: e["at"], default=None)
    return first is not None and first.get("day", first["at"][:10]) == assertion["day"]


def headline(initial_state: dict, world: WorldState, log: list[dict], assertions: list[dict]) -> float:
    """Guardrail-gated share of objectives over world and day assertions, split as AutomationBench's headline splits
    them. A `not_acted` is a guardrail unless it sets `excluded: false`, and a broken guardrail scores the task 0."""
    initial = WorldState(**strip_none_values(initial_state))
    passed = total = 0
    for a in assertions:
        if a.get("scored") is False or a.get("excluded") is True:
            continue
        if a["type"] in DAY_ASSERTIONS:
            result, guardrail = day_check(log, a), a["type"] == "not_acted" and a.get("excluded") is not False
        else:
            result = AssertionRegistry.check(world, a)
            guardrail = AssertionRegistry.check(initial, a) and a.get("excluded") is not False
        if guardrail:
            if not result:
                return 0.0
            continue
        total += 1
        passed += int(result)
    return passed / total if total else 0.0


def _is_stamped(name: str) -> bool:
    """Whether a field name (camelCase or snake_case, or a nested path such as `meta__createdAt`) is a stamped field."""
    snake = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()  # lastModifiedDateTime -> last_modified_date_time
    return any(part.strip("_") in STAMPED_FIELDS for part in snake.split("__"))


def stamped_fields(assertion: dict) -> list[str]:
    """Stamped fields a world assertion names, as a key or as the `field` or `property` it checks."""
    if assertion.get("type") in DAY_ASSERTIONS:
        return []
    found = []

    def walk(node) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in VALUE_MAPS:
                    continue
                if _is_stamped(key):
                    found.append(key)
                elif key in FIELD_KEYS and isinstance(value, str) and _is_stamped(value):
                    found.append(value)
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(assertion)
    return found


def replay(info: dict, calls: list[dict]) -> float:
    """Score a scripted plan, each call {"wait": iso time} or {"method", "url", "params", "body"}, to check that a task
    is solvable before training on it."""
    info = copy.deepcopy(info)
    if problems := resolve_records(info):
        raise ValueError(f"day-assertion records cannot be resolved: {problems}")
    world = WorldState(**strip_none_values(info["initial_state"]))
    clock = Clock(world, copy.deepcopy(info.get("events") or []), info.get("period_end") or world.meta.current_time)
    for call in calls:
        if "wait" in call:
            clock.wait(call["wait"])
        else:
            clock.fetch(call["method"], call["url"], call.get("params"), call.get("body"))
    clock.wait(clock.end.replace(tzinfo=timezone.utc).isoformat())
    return headline(info["initial_state"], world, clock.log, info["assertions"])


def requester_tool(clock: Clock, services: list[str], name: str, arguments: str) -> str:
    """Run one requester tool call: api_search, or api_fetch with only the requester's own services connected. Writes go
    through the clock, so day assertions see a step the agent handed over."""
    from automationbench.tools.api import api_search

    try:
        args = json.loads(arguments or "{}")
        if name == "api_search":
            return api_search(str(args.get("query", "")), int(args.get("top_k") or 5))
        if name != "api_fetch":
            return json.dumps({"error": f"no tool named {name}"})
        world = clock.world
        saved, world.meta.allowed_services = world.meta.allowed_services, list(services)
        try:
            return clock.fetch(
                str(args.get("method", "GET")), str(args.get("url", "")), args.get("params"), args.get("body")
            )
        finally:
            world.meta.allowed_services = saved
    except Exception as e:  # noqa: BLE001  the requester sees the error the way the agent would
        return json.dumps({"error": f"{type(e).__name__}: {e}"})


def own_records(world: WorldState, services: list[str], budget: int = 4000) -> str:
    """The requester's index of its own apps: per collection the ids and names of up to 30 records, else a count."""

    def label(record: dict) -> str:
        return next((str(record[k]) for k in ("title", "name", "subject", "email") if record.get(k)), "")

    lines = []
    for service in services:
        for name, items in getattr(world, service).model_dump(mode="json").items():
            if not isinstance(items, list) or not items:
                continue
            if len(items) > 30 or not isinstance(items[0], dict):
                lines.append(f"{service}.{name}: {len(items)} records")
                continue
            lines.append(f"{service}.{name}: " + "; ".join(f"{x.get('id')} ({label(x)})" for x in items))
    text = "\n".join(lines) or "(nothing)"
    return text if len(text) <= budget else text[:budget] + " ..."


def reward(state, **kwargs) -> float:
    """The headline, 0 after a broken unavailable-call guardrail. Synchronous, so verifiers_agent masks a raise as a
    verifier error instead of scoring it 0."""
    if state.get("violation") or state.get("env_row_invalid"):
        return 0.0
    state["clock"].wait(state["clock"].end.replace(tzinfo=timezone.utc).isoformat())
    return headline(state["initial_state"], state["world"], state["clock"].log, state["info"]["assertions"])


class RequesterUnavailable(RuntimeError):
    """The requester endpoint gave no usable answer within its tries, or a dated event could not be applied.

    Either way the episode ends with `requester_failed` set, so the rollout is masked instead of scored."""


def _state_metric(key: str):
    def metric(state, **kwargs) -> float:
        return float(state.get(key, 0))

    metric.__name__ = key
    return metric


METRICS = (
    "user_turns",
    "requester_failed",
    "env_row_invalid",
    "empty_turn",
    "unknown_tool_calls",
    "out_of_scope_calls",
    "requester_service_calls",
    "requester_tool_calls",
    "violation",
)


def load_environment(
    tasks_path: str,
    max_turns: int = 80,
    requester_base_url: str | None = None,
    requester_model: str | None = None,
    requester_body: dict | None = None,
    requester_api_key_env: str | None = None,
    requester_api_key_file: str | None = None,
    max_user_turns: int = 3,
    max_requester_tool_calls: int = 8,
    requester_max_tries: int = 4,
    requester_timeout_s: float = 300.0,
    unavailable_call_break: bool = False,
    **kwargs,
):
    """Build the environment over the Gym rows in `tasks_path`.

    `requester_base_url` and `requester_model` name the OpenAI-compatible endpoint that plays the requester on
    multi-turn rows. `requester_body` is merged over the built-in sampling values. A key is read from the variable named
    by `requester_api_key_env` or the file `requester_api_key_file` and sent as a Bearer token."""
    if not (0 < requester_timeout_s < math.inf):
        raise ValueError(f"requester_timeout_s must be positive and finite, got {requester_timeout_s!r}")
    if requester_max_tries < 1:
        raise ValueError(f"requester_max_tries must be at least 1, got {requester_max_tries!r}")
    if requester_api_key_env:
        api_key = os.environ.get(requester_api_key_env)
        if not api_key:
            raise ValueError(f"requester_api_key_env names {requester_api_key_env!r}, which is not set")
    else:
        api_key = Path(requester_api_key_file).read_text().strip() if requester_api_key_file else None
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    import verifiers as vf
    from automationbench_env import AutomationBenchEnv
    from datasets import Dataset

    class ABSynthEnv(AutomationBenchEnv):
        async def setup_state(self, state, **setup_kwargs):
            # Gym hands each rollout the row's own info, not the loaded dataset's, so the load checks run again here. A
            # row that fails them, or a setup that raises, is masked as env_row_invalid, never a server error.
            try:
                state = await super().setup_state(state, **setup_kwargs)
                info = state["info"]
                problems = row_problems(info, requester=bool(requester_base_url and requester_model))
                if problems or (problems := resolve_records(info)):
                    return self._invalid_row(state, problems)
                return self._setup(state, info)
            except Exception as e:  # noqa: BLE001  a row the environment cannot set up is masked, not a 500
                return self._invalid_row(state, [f"setup raised {type(e).__name__}: {e}"])

        def _invalid_row(self, state, problems: list[str]):
            state["env_row_invalid"] = 1
            logger.warning("ab_synth: row cannot be run, masked as env_row_invalid: %s", problems[:3])
            return state

        def _setup(self, state, info):
            end = info.get("period_end") or state["world"].meta.current_time.isoformat()
            state["clock"] = Clock(state["world"], copy.deepcopy(info.get("events") or []), end)
            if info.get("period_end"):
                until = {"type": "string", "description": info["wait_until_description"]}
                wait_tool = vf.Tool(
                    name="wait",
                    description=info["wait_tool_description"],
                    parameters={"type": "object", "properties": {"until": until}, "required": ["until"]},
                )
                state["tool_defs"] = [*state["tool_defs"], wait_tool]
            own = set(info.get("requester_services") or [])
            if own:
                meta = state["world"].meta
                meta.allowed_services = [s for s in meta.allowed_services or [] if s not in own]
            return state

        def _unavailable(self, state, key: str) -> None:
            state[key] = state.get(key, 0) + 1
            if unavailable_call_break:
                state["violation"] = 1

        def update_tool_args(self, tool_name, tool_args, messages, state, **update_kwargs):
            if tool_name not in {t.name for t in state["tool_defs"]}:
                self._unavailable(state, "unknown_tool_calls")
            args = super().update_tool_args(tool_name, tool_args, messages, state, **update_kwargs)
            if tool_name in ("wait", "api_fetch"):
                args.pop("world", None)
                args["clock"] = state["clock"]
            return args

        async def env_response(self, messages, state, **response_kwargs):
            reply = state.pop("pending_user_reply", None)
            if reply is not None:
                return [{"role": "user", "content": reply}]
            try:
                out = await super().env_response(messages, state, **response_kwargs)
            except vf.ToolCallError as e:
                # A dated event that cannot be applied is a dataset failure, not the policy's: end the rollout masked.
                if isinstance(e.__cause__, RequesterUnavailable):
                    self._requester_failed(state, e.__cause__)
                raise
            own = set(state["info"].get("requester_services") or [])
            for m in out:
                hit = OUT_OF_SCOPE.search(str(_get(m, "content") or ""))
                if hit:
                    self._unavailable(state, "out_of_scope_calls")
                    state["requester_service_calls"] = state.get("requester_service_calls", 0) + (hit.group(1) in own)
            return out

        def _requester_failed(self, state, error: Exception) -> None:
            state["requester_failed"] = 1
            logger.warning("ab_synth: episode ends unscored: %s", str(error)[:200])

        @vf.stop
        async def invalid_row(self, state) -> bool:
            """A row that failed its checks or its setup ends before the first turn."""
            return bool(state.get("env_row_invalid"))

        @vf.stop
        async def no_tools_called(self, state) -> bool:
            """The agent ended its turn: on multi-turn rows let the requester answer, and stop when it has nothing to add.
            A check that raises ends the episode masked as env_row_invalid, not as a server error."""
            try:
                return await self._turn_ended(state)
            except Exception as e:  # noqa: BLE001  masked, not a 500
                self._invalid_row(state, [f"turn-end check raised {type(e).__name__}: {e}"])
                return True

        async def _turn_ended(self, state) -> bool:
            if not await super().no_tools_called(state):
                return False
            info = state["info"]
            if not (info.get("user_card") or info.get("requester_services")):
                return True
            if state.get("requester_step") == len(state["trajectory"]):
                # No policy step since the requester's last reply (the prompt ran past the context): answering again
                # would make an acting requester repeat its writes.
                return True
            if not _text(_get(state["trajectory"][-1]["completion"][-1], "content")):
                # An empty turn ends the episode: a reply to it would make the next model request invalid.
                state["empty_turn"] = 1
                return True
            if state.get("user_turns", 0) >= max_user_turns:
                return True
            state["requester_step"] = len(state["trajectory"])
            try:
                reply = await (self._acting_reply(state) if info.get("requester_services") else self._reply(state))
            except RequesterUnavailable as e:
                self._requester_failed(state, e)
                return True
            if not reply or reply.rstrip(".").upper() == "DONE":
                return True
            state["user_turns"] = state.get("user_turns", 0) + 1
            state["pending_user_reply"] = reply
            return False

        def _conversation(self, state, system: str, **fields) -> list[dict]:
            """The requester's view: its instructions, then the visible conversation (no reasoning, no tool traffic)."""
            prompt = list(state["prompt"])
            first = next(i for i, m in enumerate(prompt) if _get(m, "role") == "user")
            msgs = [
                {"role": "system", "content": system.format(request=_text(_get(prompt[first], "content")), **fields)}
            ]
            last = state["trajectory"][-1]
            for m in prompt[first + 1 :] + list(last["prompt"])[len(prompt) :] + list(last["completion"]):
                role, text = _get(m, "role"), _text(_get(m, "content"))[:REQUESTER_RESULT_CHARS]
                if role in ("assistant", "user") and text:
                    msgs.append({"role": "user" if role == "assistant" else "assistant", "content": text})
            return msgs

        async def _chat(self, messages: list[dict], **body) -> dict:
            """One requester call through Gym's aiohttp client. Each try ends after `requester_timeout_s` so a stalled
            requester cannot hang the rollout. A reply cut at max_tokens or empty is a failed try."""
            from nemo_gym.client import request

            url = requester_base_url.rstrip("/") + "/chat/completions"
            payload = {"model": requester_model, "messages": messages, **body, **(requester_body or {})}

            async def attempt_once() -> dict:
                # One connection try: request() would otherwise retry a refused connection until wait_for fires.
                response = await request("POST", url, json=payload, headers=headers or {}, _max_num_tries=1)
                response.raise_for_status()
                return (await response.json())["choices"][0]

            error = None
            for attempt in range(requester_max_tries):
                if attempt:
                    await asyncio.sleep(min(60, 5 * 3**attempt))
                try:
                    choice = await asyncio.wait_for(attempt_once(), requester_timeout_s)
                    message = choice["message"]
                    if choice.get("finish_reason") != "length" and (
                        _text(message.get("content")) or message.get("tool_calls")
                    ):
                        return message
                    error = f"no usable answer (finish_reason={choice.get('finish_reason')})"
                except Exception as e:  # noqa: BLE001  retried, then reported as RequesterUnavailable
                    error = f"{type(e).__name__}: {e}"
            raise RequesterUnavailable(error)

        async def _reply(self, state) -> str:
            card = state["info"]["user_card"]
            message = await self._chat(
                self._conversation(state, REQUESTER_SYSTEM, card=card), temperature=0.7, max_tokens=400
            )
            return _text(message.get("content"))

        async def _acting_reply(self, state) -> str:
            """The requester does what the agent asked in its own apps, then replies."""
            info, world = state["info"], state["world"]
            services = info["requester_services"]
            msgs = self._conversation(
                state,
                ACTING_REQUESTER_SYSTEM,
                card=info.get("user_card") or "(none)",
                services=", ".join(services),
                records=own_records(world, services),
            )
            tools = [t for t in self._all_oai_tools if t["function"]["name"] in ("api_search", "api_fetch")]
            # The budget counts tool calls, not model responses: one response may carry several calls.
            remaining = max_requester_tool_calls
            for _ in range(max_requester_tool_calls + 1):
                choice = "auto" if remaining > 0 else "none"
                message = await self._chat(msgs, temperature=0.7, max_tokens=8192, tools=tools, tool_choice=choice)
                if not message.get("tool_calls"):
                    return _text(message.get("content"))
                msgs.append(
                    {"role": "assistant", "content": message.get("content") or "", "tool_calls": message["tool_calls"]}
                )
                for call in message["tool_calls"]:
                    fn = call["function"]
                    if remaining > 0:
                        result = requester_tool(state["clock"], services, fn["name"], fn.get("arguments"))
                        remaining -= 1
                        state["requester_tool_calls"] = state.get("requester_tool_calls", 0) + 1
                    else:
                        result = json.dumps({"error": "requester tool-call limit reached; reply without tools"})
                    msgs.append(
                        {"role": "tool", "tool_call_id": call["id"], "content": result[:REQUESTER_RESULT_CHARS]}
                    )
            raise RequesterUnavailable("the requester kept calling tools after tool_choice=none")

    def wait(until: str, clock: Clock) -> str:
        try:
            _time(until)
        except (TypeError, ValueError):
            # A malformed time is the policy's mistake: report it to the agent, do not mask the rollout.
            return json.dumps({"error": f"invalid until {until!r}: expected an ISO 8601 time"})
        try:
            return clock.wait(until)
        except Exception as e:  # noqa: BLE001  a broken event row, reported as a requester failure and masked
            raise RequesterUnavailable(f"dated event failed: {type(e).__name__}: {e}") from e

    def api_fetch(method: str, url: str, params=None, body=None, clock: Clock = None) -> str:
        return clock.fetch(method, url, params, body)

    with open(tasks_path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    # A row that fails its checks is refused and counted, so one bad row does not stop the rest. A multi-turn row
    # without a configured requester is a configuration error and stops the load.
    requester = bool(requester_base_url and requester_model)
    refused = {}
    for i, r in enumerate(rows):
        try:
            problems = row_problems(r["info"], requester=requester) or resolve_records(r["info"])
        except Exception as e:  # noqa: BLE001  a row the checks cannot read is refused with the error, not the whole load
            problems = [f"row check raised {type(e).__name__}: {str(e)[:200]}"]
        if problems:
            refused[i] = problems
    if not requester and any("requester_base_url" in p for problems in refused.values() for p in problems):
        raise ValueError(
            "multi-turn rows (user_card or requester_services) need requester_base_url and requester_model"
        )
    if refused:
        first = next(iter(refused))
        kinds = collections.Counter(
            p.split(":")[0] if p.startswith(("ambiguous", "unresolvable")) else "row check"
            for problems in refused.values()
            for p in problems
        )
        logger.warning(
            "ab_synth: refused %d of %d rows (%s), e.g. row %d: %s",
            len(refused),
            len(rows),
            dict(kinds.most_common(6)),
            first,
            refused[first][:3],
        )
        if len(refused) == len(rows):
            raise ValueError(f"no row can be run; row {first}: {refused[first][0]}")
        rows = [r for i, r in enumerate(rows) if i not in refused]
    dataset = Dataset.from_list(
        [
            {
                "example_id": r.get("example_id", i),
                "prompt": r["responses_create_params"]["input"],
                "answer": "",
                "info": json.dumps(r["info"]),
            }
            for i, r in enumerate(rows)
        ]
    )
    funcs = [reward, *(_state_metric(k) for k in METRICS)]
    env = ABSynthEnv(
        dataset=dataset,
        rubric=vf.Rubric(funcs=funcs, weights=[1.0] + [0.0] * len(METRICS)),
        max_turns=max_turns,
        toolset="api",
        stop_errors=[RequesterUnavailable],
        **kwargs,
    )
    env.tool_map["api_fetch"] = api_fetch
    env.tool_map["wait"] = wait
    env.skipped_args["api_fetch"] = ["world", "clock"]
    env.skipped_args["wait"] = ["clock"]
    return env
