# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import os
import re
import threading
import time

from .sched_world import (ABBR, DAYS, HOURS, NH, NS, pref_ok, render_item, render_room, sample,  # noqa: E402
                         slot_name)

MAX_OUT = 8000
LIST_MAX = 80
SOFT_SHARE = 0.2  # score = valid_frac * (1 - SOFT_SHARE + SOFT_SHARE * soft_ratio)

PROMPT = """You are building the program of a 4-day conference with {n} talks (T0000 .. T{last}) and {r} rooms \
(R001 .. R{rl}). Every talk must get one time slot and one room.

Time grid: Monday to Thursday, start times 09:00, 10:00, ... 16:00 (8 slots per day, {ns} slots). Every talk fills \
exactly one 1-hour slot, so "be over by 14:00" means the latest start is 13:00. Write a slot as "Tue 10:00".

Each talk's constraints live only in its own document (read_item); each room's seats, equipment and calendar live \
only in its document (read_resource). Hard constraints:
1. The slot is one the talk's document allows (days, hours, exceptions; later updates in the thread override earlier \
messages).
2. The room is open at that slot, seats at least the expected audience, and has every required piece of equipment.
3. At most one talk per room per slot.
4. A speaker gives at most one talk per slot (several speakers give 2-3 talks; list_items shows speakers).
5. "After X" means a strictly later slot than talk X (any room). "Not at the same time as X" means a different slot. \
"Must not overlap any talk by S" means a slot different from every talk S gives.
Soft preferences carry weights 1-5 and are stated as such; everything else in a document is hard or irrelevant.
A plan satisfying every hard constraint exists. Rooms are 70-90% full in it, so talks placed one at a time \
without looking at the rest will collide; extract constraints first, then solve and repair the whole plan.

Tools: list_items(offset, limit, query) pages ids/titles/speakers (query filters title or speaker); read_item(id); \
read_resource(id); check_partial(assignment) reports every hard violation among the talks you pass (constraints \
involving talks you leave out are not checked); scratch_write/scratch_read/scratch_list form a key-value store \
shared by all agents (use it to collect extracted constraints, e.g. one JSON record per talk: {{"id", "slots", \
"min_seats", "equipment", "after", "not_with", "not_with_speaker", "prefs"}}, and partial plans).

Score: a talk counts if it is placed and no hard constraint involving it is violated (an "after X" constraint is \
violated if X is not placed). score = (counted talks / {n}) * (0.8 + 0.2 * soft), where soft is the preference \
weight earned by counted talks divided by the weight the hidden reference plan earns (capped at 1).

Answer format: one JSON object mapping every talk id to [slot, room], e.g.
{{"T0000": ["Mon 09:00", "R003"], "T0001": ["Wed 14:00", "R001"]}}
The last JSON object in your reply is graded. Instead of (or in addition to) pasting it, you can store parts of the \
plan in the scratch store as JSON objects of the same form under keys sharing one prefix and end your reply with a \
line "SCRATCH: <prefix>"; those objects are merged in key order and any JSON object in the reply is applied on top.
"""


# ---------- generation ----------

def generate(n, seed, out_dir):
    w = sample(n, seed)
    rooms, items = w["rooms"], w["items"]
    world_dir = os.path.join(out_dir, "world")
    os.makedirs(world_dir, exist_ok=True)
    by_id = {it["id"]: it for it in items}
    room_by_id = {r["id"]: r for r in rooms}
    with open(os.path.join(world_dir, "items.jsonl"), "w") as f:
        for it in items:
            f.write(json.dumps({"id": it["id"], "title": it["title"], "speaker": it["speaker"],
                                "text": render_item(it, by_id, room_by_id, seed)}) + "\n")
    with open(os.path.join(world_dir, "resources.jsonl"), "w") as f:
        for r in rooms:
            f.write(json.dumps({"id": r["id"], "text": render_room(r, seed)}) + "\n")
    gold = {"n": n, "seed": seed,
            "rooms": [{"id": r["id"], "name": r["name"], "cap": r["cap"], "equip": r["equip"],
                       "closed": sorted(r["closed"])} for r in rooms],
            "items": [{k: it[k] for k in ("id", "title", "speaker", "slot", "room", "allowed", "min_cap", "equip",
                                          "after", "not_with", "not_with_speaker", "prefs")} for it in items]}
    hidden = {it["id"]: (it["slot"], it["room"]) for it in items}
    gold["hidden_soft"] = Instance(gold).soft(hidden, set(hidden))
    gold_path = os.path.join(out_dir, "gold.json")
    with open(gold_path, "w") as f:
        json.dump(gold, f)
    prompt = PROMPT.format(n=n, last=f"{n - 1:04d}", r=len(rooms), rl=rooms[-1]["id"][1:], ns=NS)
    return {"id": f"schedule-n{n}-s{seed}", "prompt": prompt, "world_dir": world_dir, "n_units": n,
            "gold_path": gold_path}


# ---------- constraint checking (shared by check_partial and score) ----------

class Instance:
    def __init__(self, gold):
        self.items = {it["id"]: it for it in gold["items"]}
        self.rooms = {r["id"]: dict(r, closed=set(r["closed"])) for r in gold["rooms"]}
        self.allowed = {i: set(it["allowed"]) for i, it in self.items.items()}
        by_speaker = {}
        for i, it in self.items.items():
            by_speaker.setdefault(it["speaker"], []).append(i)
        self.by_speaker = by_speaker
        # conf[i][j] = reason the two talks may not share a slot (symmetric)
        self.conf = {i: {} for i in self.items}

        def add(a, b, why):
            if a != b:
                self.conf[a].setdefault(b, why)
                self.conf[b].setdefault(a, why)
        for sp, ids in by_speaker.items():
            for a in ids:
                for b in ids:
                    add(a, b, f"same speaker {sp}")
        for i, it in self.items.items():
            for j in it["not_with"]:
                add(i, j, "must not be at the same time")
            for sp in it["not_with_speaker"]:
                for j in by_speaker.get(sp, []):
                    add(i, j, f"{i} must not overlap talks by {sp}")

    def unary(self, i, s, r):
        it, room = self.items[i], self.rooms.get(r)
        out = []
        if room is None:
            return [f"unknown room {r}"]
        if s in room["closed"]:
            out.append(f"{r} is closed at {slot_name(s)}")
        if s not in self.allowed[i]:
            out.append(f"{slot_name(s)} is outside the talk's allowed times")
        if room["cap"] < it["min_cap"]:
            out.append(f"{r} seats {room['cap']}, too few for the expected audience")
        missing = [e for e in it["equip"] if e not in room["equip"]]
        if missing:
            out.append(f"{r} lacks required equipment: {', '.join(missing)}")
        return out

    def violations(self, assign, full):
        """assign: id -> (slot, room). Returns [(message, [ids])]. In full mode an unplaced predecessor is a violation."""
        out = []
        for i in sorted(assign):
            s, r = assign[i]
            for msg in self.unary(i, s, r):
                out.append((f"{i} @ {slot_name(s)} {r}: {msg}", [i]))
        cells = {}
        for i in sorted(assign):
            cells.setdefault(assign[i], []).append(i)
        for (s, r), ids in sorted(cells.items()):
            if len(ids) > 1:
                out.append((f"{', '.join(ids)} all in {r} at {slot_name(s)}", ids))
        for i in sorted(assign):
            s = assign[i][0]
            for j, why in self.conf[i].items():
                if i < j and j in assign and assign[j][0] == s:
                    out.append((f"{i} and {j} both at {slot_name(s)} ({why})", [i, j]))
            for y in self.items[i]["after"]:
                if y in assign and assign[y][0] >= s:
                    out.append((f"{i} ({slot_name(s)}) must come after {y} ({slot_name(assign[y][0])})", [i, y]))
                elif y not in assign and full:
                    out.append((f"{i} must come after {y}, which is not placed", [i]))
        return out

    def soft(self, assign, valid):
        slot_of = {i: a[0] for i, a in assign.items()}
        return sum(p["w"] for i in valid for p in self.items[i]["prefs"]
                   if pref_ok(p, assign[i][0], assign[i][1], slot_of))


_SLOT = re.compile(r"\b(mon|tue|wed|thu)[a-z]*\.?,?\s*(\d{1,2})(?::(\d{2}))?", re.I)
_ROOM = re.compile(r"\bR(\d{1,4})\b", re.I)


def parse_slot(x):
    if isinstance(x, int) and 0 <= x < NS:
        return x
    m = _SLOT.search(str(x))
    if not m or (m.group(3) and m.group(3) != "00"):
        return None
    h = int(m.group(2))
    if h not in HOURS:
        return None
    return [a.lower() for a in ABBR].index(m.group(1).lower()[:3]) * NH + h - HOURS[0]


def parse_entry(v):
    if isinstance(v, dict):
        v = [v.get("slot", v.get("time")), v.get("room")]
    if isinstance(v, str):
        v = [v, v]
    if not isinstance(v, (list, tuple)) or len(v) != 2:
        return None
    s, m = parse_slot(v[0]), _ROOM.search(str(v[1]))
    if s is None or not m:
        return None
    return s, f"R{int(m.group(1)):03d}"


def parse_assignment(obj, n_items):
    """Returns ({id: (slot, room)}, [bad ids])."""
    if isinstance(obj, str):
        obj = last_plan(obj)
    if isinstance(obj, dict) and isinstance(obj.get("assignment"), dict):
        obj = obj["assignment"]
    if not isinstance(obj, dict):
        return {}, []
    out, bad = {}, []
    for k, v in obj.items():
        m = re.fullmatch(r"\s*T(\d{1,5})\s*", str(k), re.I)
        e = parse_entry(v)
        if not m or int(m.group(1)) >= n_items or e is None:
            bad.append(str(k)[:40])
            continue
        out[f"T{int(m.group(1)):04d}"] = e
    return out, bad


def last_plan(text):
    # The last JSON object that holds talk ids (directly or under "assignment"), so a trailing example object does
    # not replace the plan. A truncated plan keeps its complete "T0001": [slot, room] entries.
    dec, best, i = json.JSONDecoder(), None, 0
    while (i := text.find("{", i)) >= 0:
        try:
            obj, end = dec.raw_decode(text, i)
        except ValueError:
            i += 1
            continue
        inner = obj.get("assignment") if isinstance(obj, dict) and isinstance(obj.get("assignment"), dict) else obj
        if isinstance(inner, dict) and any(re.fullmatch(r"\s*T\d{1,5}\s*", str(k), re.I) for k in inner):
            best = obj
        i = end
    if best is None:
        best = {}
        for k, v in re.findall(r'"(T\d{1,5})"\s*:\s*(\[[^\[\]]*\]|\{[^{}]*\})', text):
            try:
                best[k] = json.loads(v)
            except ValueError:
                pass
    return best


def last_json(text):
    dec, best, i = json.JSONDecoder(), None, 0
    while True:
        i = text.find("{", i)
        if i < 0:
            return best
        try:
            obj, end = dec.raw_decode(text, i)
            best, i = obj, end
        except ValueError:
            i += 1


# ---------- tools ----------

class Store:
    """Shared key-value scratch store: in memory, mirrored to an append-only log with writer provenance."""

    def __init__(self, path):
        self.path, self.lock, self.data = path, threading.Lock(), {}
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue  # torn last line after a crash
                    self.data[rec["key"]] = rec

    def write(self, key, value, agent):
        rec = {"key": key, "value": value, "agent": agent, "t": round(time.time(), 3)}
        with self.lock:
            with open(self.path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            self.data[key] = rec

    def snapshot(self):
        with self.lock:
            return dict(self.data)


_STORES, _STORES_LOCK = {}, threading.Lock()


def store_for(world_dir):
    path = os.path.realpath(os.path.join(world_dir, "scratch.jsonl"))
    with _STORES_LOCK:
        if path not in _STORES:
            _STORES[path] = Store(path)
        return _STORES[path]


def _cap(s):
    return s if len(s) <= MAX_OUT else s[:MAX_OUT - 60] + f"\n...[truncated, {len(s)} chars total]"


def _int(x, default, lo, hi):
    try:
        return max(lo, min(hi, int(x)))
    except (TypeError, ValueError):
        return default


def tools(world_dir):
    items = [json.loads(line) for line in open(os.path.join(world_dir, "items.jsonl"))]
    by_id = {it["id"]: it for it in items}
    res = {r["id"]: r["text"] for r in map(json.loads, open(os.path.join(world_dir, "resources.jsonl")))}
    inst = Instance(json.load(open(os.path.join(os.path.dirname(world_dir), "gold.json"))))
    store = store_for(world_dir)

    def list_items(a):
        off, lim = _int(a.get("offset"), 0, 0, 10**6), _int(a.get("limit"), LIST_MAX, 1, LIST_MAX)
        q = str(a.get("query") or "").strip().lower()
        rows = [it for it in items if not q or q in it["title"].lower() or q in it["speaker"].lower()]
        page = rows[off:off + lim]
        head = f"{len(rows)} talks match; showing {off}-{off + len(page) - 1}\n" if page else \
            f"{len(rows)} talks match; nothing at offset {off}\n"
        return _cap(head + "\n".join(f"{it['id']} | {it['title']} | {it['speaker']}" for it in page))

    def read_item(a):
        i = str(a.get("item_id", a.get("id", ""))).strip().upper()
        if re.fullmatch(r"T\d{1,5}", i):
            i = f"T{int(i[1:]):04d}"
        return _cap(by_id[i]["text"]) if i in by_id else f"error: unknown item {i!r} (ids are T0000..T{len(items) - 1:04d})"

    def read_resource(a):
        r = str(a.get("resource_id", a.get("id", ""))).strip().upper()
        m = re.fullmatch(r"R(\d{1,4})", r)
        r = f"R{int(m.group(1)):03d}" if m else r
        return res.get(r, f"error: unknown resource {r!r} (rooms are R001..R{len(res):03d})")

    def check_partial(a):
        assign, bad = parse_assignment(a.get("assignment"), len(items))
        if not assign and not bad:
            return "error: pass assignment as an object {\"T0001\": [\"Tue 10:00\", \"R003\"], ...}"
        v = inst.violations(assign, full=False)
        hit = {i for _, ids in v for i in ids}
        lines = [f"checked {len(assign)} talks: {len(v)} violations, {len(assign) - len(hit)} talks clean"]
        if bad:
            lines.append(f"unparsable entries (need id -> [\"Day HH:00\", \"Rnnn\"]): {', '.join(bad[:30])}")
        lines += [m for m, _ in v]
        return _cap("\n".join(lines))

    def scratch_write(a):
        key, val = str(a.get("key", "")).strip(), a.get("value", "")
        if not isinstance(val, str):
            val = json.dumps(val)
        if not key or len(key) > 200:
            return "error: key must be 1-200 characters"
        if len(val) > MAX_OUT:
            return f"error: value is {len(val)} chars; max {MAX_OUT}, split it over several keys"
        store.write(key, val, str(a.get("_agent", "")))
        return f"ok: stored {len(val)} chars under {key!r}"

    def scratch_read(a):
        rec = store.snapshot().get(str(a.get("key", "")).strip())
        return rec["value"] if rec else "error: no such key"

    def scratch_list(a):
        pre, off = str(a.get("prefix") or ""), _int(a.get("offset"), 0, 0, 10**7)
        keys = sorted(k for k in store.snapshot() if k.startswith(pre))
        page = keys[off:off + 150]
        return _cap(f"{len(keys)} keys with prefix {pre!r}; showing {len(page)} from offset {off}\n" + "\n".join(page))

    def spec(name, desc, props, req, fn):
        props = dict(props, _agent={"type": "string", "description": "optional caller id (set by the harness)"})
        return {"name": name, "description": desc, "parameters": {"type": "object", "properties": props,
                                                                    "required": req}, "fn": fn}
    S, I = {"type": "string"}, {"type": "integer"}
    return [
        spec("list_items", f"List talks as 'id | title | speaker', at most {LIST_MAX} per call. query filters by a "
             "case-insensitive substring of the title or speaker.",
             {"offset": I, "limit": I, "query": S}, [], list_items),
        spec("read_item", "Full document of one talk (abstract, room needs, scheduling notes).",
             {"item_id": S}, ["item_id"], read_item),
        spec("read_resource", "Document of one room (seats, equipment, calendar), e.g. R007.",
             {"resource_id": S}, ["resource_id"], read_resource),
        spec("check_partial", "Check a partial plan: pass {talk id: [\"Tue 10:00\", \"R003\"]} for any subset of "
             "talks. Reports every hard-constraint violation among the given talks (allowed times, room closure, "
             "seats, equipment, double-booked rooms, speaker clashes, conflicts, ordering).",
             {"assignment": {"type": "object"}}, ["assignment"], check_partial),
        spec("scratch_write", "Store a string (max 8000 chars) under a key in the store shared by all agents. "
             "Overwrites.", {"key": S, "value": S}, ["key", "value"], scratch_write),
        spec("scratch_read", "Read one key from the shared store.", {"key": S}, ["key"], scratch_read),
        spec("scratch_list", "List keys in the shared store with a prefix (150 per page).",
             {"prefix": S, "offset": I}, [], scratch_list),
    ]


# ---------- scoring ----------

def score(task, answer):
    gold = json.load(open(task["gold_path"]))
    n, inst = gold["n"], Instance(gold)
    merged = {}
    m = re.findall(r"^\s*SCRATCH:\s*(\S+)\s*$", answer or "", re.M)
    if m and task.get("world_dir"):
        data = store_for(task["world_dir"]).snapshot()
        for k in sorted(k for k in data if k.startswith(m[-1])):
            merged.update(parse_assignment(data[k]["value"], n)[0])
    merged.update(parse_assignment(answer or "", n)[0])
    v = inst.violations(merged, full=True)
    bad = {i for _, ids in v for i in ids}
    valid = set(merged) - bad
    soft = inst.soft(merged, valid)
    soft_ratio = min(1.0, soft / gold["hidden_soft"]) if gold["hidden_soft"] else 1.0
    frac = len(valid) / n
    return {"score": frac * (1 - SOFT_SHARE + SOFT_SHARE * soft_ratio), "valid_frac": frac,
            "soft_ratio": soft_ratio, "feasible": len(valid) == n, "n_placed": len(merged), "n_valid": len(valid),
            "n_violations": len(v), "soft_weight": soft, "hidden_soft_weight": gold["hidden_soft"]}
